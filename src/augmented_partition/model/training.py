# Description: This file contains the functions to train the model. 

import torch
import torch.nn as nn
import torch.distributed as dist
import matplotlib.pyplot as plt
import utils
import time
import torch.optim as optim
import compute_env as env
import gc
import numpy as np
from mpi4py import MPI
import scipy
import scipy.sparse as sp

@env.only_rank_zero
def save_training_state(model, optimizer, track_loss_edge, track_loss_node, track_validation_edge, track_validation_node, save_file):
    """
    Save the training state of the model and optimizer
    """
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, save_file + '.pt')
    torch.save(model.state_dict(), save_file + '_state_dic.pt')

    with open(save_file + '_training_loss.txt', 'w') as f:
        for edge, node in zip(track_loss_edge, track_loss_node):
            f.write(f"{edge:.8f}\t{node:.8f}\n")

    with open(save_file + '_validation_loss.txt', 'w') as f:
        for edge, node in zip(track_validation_edge, track_validation_node):
            f.write(f"{edge:.8f}\t{node:.8f}\n")

    plt.figure(figsize=(4, 3))
    plt.plot(track_loss_node, label='node')
    plt.plot(track_loss_edge, label='edge')
    plt.plot(track_validation_node, label='validation node')
    plt.plot(track_validation_edge, label='validation edge')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(save_file + '_loss.png', dpi=300, bbox_inches='tight')
    plt.close()

############################################################
# Functions to compute the loss with different filtering
############################################################

def get_loss_flattened(node_output, edge_output, batch, criterion):
    """
    Process a batch of data (forward pass + loss) for labels and targets in the uncoupled basis
    """

    if hasattr(batch, 'labelled_node_size'):
        labelled_node_size = batch.labelled_node_size.item()
        labelled_edge_size = batch.labelled_edge_size.item()
    else:
        batch_size = len(batch)
        labelled_node_size = batch[0].num_nodes * batch_size
        labelled_edge_size = batch[0].num_edges * batch_size

    # Compute the loss
    loss_node = criterion(node_output[0:labelled_node_size], batch.node_y[0:labelled_node_size])            # node_y is the node label
    loss_edge = criterion(edge_output[0:labelled_edge_size], batch.y[0:labelled_edge_size])                 # y is the edge label
    output = torch.cat([node_output[0:labelled_node_size], edge_output[0:labelled_edge_size]], dim=0)
    labels = torch.cat([batch.node_y[0:labelled_node_size], batch.y[0:labelled_edge_size]], dim=0)
    loss = criterion(output, labels)     

    return loss_node, loss_edge, loss



def get_loss_unflattened(node_output, edge_output, batch, criterion, construct_kernel, equivariant_blocks, atom_orbitals, out_slices):
    """
    Process a batch of data (forward pass + loss) for labels and targets in the coupled basis
    """
     
    if hasattr(batch, 'labelled_node_size'):
        labelled_node_size = batch.labelled_node_size.item()
        labelled_edge_size = batch.labelled_edge_size.item()
    else:
        batch_size = len(batch)
        labelled_node_size = batch[0].num_nodes * batch_size
        labelled_edge_size = batch[0].num_edges * batch_size

    arange_tensor = torch.arange(labelled_node_size).unsqueeze(0)
    torch_cat_tensor = torch.cat((arange_tensor, arange_tensor), 0) # edge_index for self-loop (nodes)

    # Process node predictions
    flattened_node_labels = construct_kernel.get_H(batch.node_y[0:labelled_node_size])
    flattened_node_pred = construct_kernel.get_H(node_output[:labelled_node_size])

    node_label = utils.unflatten(flattened_node_labels, batch.x[0:labelled_node_size], torch_cat_tensor,
                                equivariant_blocks, atom_orbitals, out_slices)
    
    node_pred = utils.unflatten(flattened_node_pred, batch.x[0:labelled_node_size], torch_cat_tensor,
                                equivariant_blocks, atom_orbitals, out_slices)

    node_label_tensor = torch.cat([matrix.flatten() for matrix in node_label.values()])
    node_pred_tensor = torch.cat([matrix.flatten() for matrix in node_pred.values()])

    # Process edge predictions
    flattened_edge_labels = construct_kernel.get_H(batch.y[0:labelled_edge_size])
    flattened_edge_pred = construct_kernel.get_H(edge_output[0:labelled_edge_size])

    edge_label = utils.unflatten(flattened_edge_labels, batch.x[0:labelled_node_size],
                                    batch.edge_index[:, 0:labelled_edge_size],
                                    equivariant_blocks, atom_orbitals, out_slices)
    
    edge_pred = utils.unflatten(flattened_edge_pred, batch.x[0:labelled_node_size],
                                batch.edge_index[:, 0:labelled_edge_size],
                                equivariant_blocks, atom_orbitals, out_slices)

    edge_label_tensor = torch.cat([matrix.flatten() for matrix in edge_label.values()])
    edge_pred_tensor = torch.cat([matrix.flatten() for matrix in edge_pred.values()])

    # Compute the loss
    loss_node = criterion(node_pred_tensor, node_label_tensor)
    loss_edge = criterion(edge_pred_tensor, edge_label_tensor)
    pred_tensor = torch.cat([node_pred_tensor, edge_pred_tensor])
    label_tensor = torch.cat([node_label_tensor, edge_label_tensor])
    loss = criterion(pred_tensor, label_tensor)  

    return loss_node, loss_edge, loss

############################################################
# Training the model
############################################################

def train_and_validate_model_subgraph(model, optimizer, training_loader, validation_loader, 
                                      num_epochs=5000, loss_tol=0.0001, patience=500, threshold=1e-3, min_lr=1e-5, 
                                      save_file='model.pth', schedule=True, dtype=torch.float32,
                                      unflatten=False, construct_kernel=None, equivariant_blocks=None, atom_orbitals=None, out_slices=None, criterion = 'mse'):
    
    device = next(model.parameters()).device  


    if criterion == 'mse':
        criterion = nn.MSELoss(reduction='mean')
    elif criterion == 'mae':
        criterion = nn.L1Loss(reduction='mean')
    # criterion = nn.L1Loss(reduction='mean')

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, threshold=threshold, verbose=True)
    
    if dist.is_available() and dist.is_initialized():
        # find_unused_parameters=True handles the cases where some parameters dont recieve gradients, such as the one-way edges
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device, find_unused_parameters=True)

    track_loss_node = []
    track_loss_edge = []
    track_validation_node = []
    track_validation_edge = []
    track_training_loss = [] # node + edge
    track_validation_loss = [] # node + edge

    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):

        # model.train()
        epoch_start_time = time.time()

        for batch in training_loader:

            batch_start_time = time.time()

            optimizer.zero_grad() 

            # Forward pass
            batch = batch.to(device)
            node_output, edge_output = model(batch)
            forward_pass_time = time.time()

            # Loss computation
            if unflatten:
                loss_node, loss_edge, mse_loss = get_loss_unflattened(node_output, edge_output, batch, criterion, construct_kernel, equivariant_blocks, atom_orbitals, out_slices)
            else:
                loss_node, loss_edge, mse_loss = get_loss_flattened(node_output, edge_output, batch, criterion)

            # Backward pass
            loss = mse_loss
            # loss = loss_node+loss_edge
            loss.backward()    
            backward_pass_time = time.time()                              
                        
            # Parameter update
            optimizer.step()

            batch_end_time = time.time()
            forward_pass_duration = forward_pass_time - batch_start_time
            backward_pass_duration = backward_pass_time - forward_pass_time
            batch_duration = batch_end_time - batch_start_time

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        track_loss_node.append(loss_node.cpu().detach().numpy()) 
        track_loss_edge.append(loss_edge.cpu().detach().numpy())
        track_training_loss.append(loss.cpu().detach().numpy())
            
        @env.only_rank_zero
        def print_train_info(): 
            print(f"Epoch {epoch} - Time: {epoch_duration:.4f} seconds")
            print(f"--> Forward Pass Time: {forward_pass_duration:.4f} seconds")
            print(f"--> Backward Pass Time: {backward_pass_duration:.4f} seconds")
            print(f"--> Total Batch process time: {batch_duration:.4f} seconds")
            # print("--> Memory allocated: " + str(torch.cuda.memory_allocated(device)/1e9) + "GB")
            # print(f"--> Memory info: {torch.cuda.mem_get_info(device)}")
            print("Epoch: " + str(epoch)+ " loss: " + str(loss))
        print_train_info()

        # Validate the model
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for batch in validation_loader:
                batch = batch.to(device)

                # Forward pass
                node_output, edge_output = model(batch) 

                # Loss computation
                if unflatten:
                    loss_node, loss_edge, loss = get_loss_unflattened(node_output, edge_output, batch, criterion, construct_kernel, equivariant_blocks, atom_orbitals, out_slices)
                else:
                    loss_node, loss_edge, loss = get_loss_flattened(node_output, edge_output, batch, criterion)
                    # loss_node, loss_edge, loss = get_loss_transformed(node_output, edge_output, batch, criterion)
                
                validation_loss += loss.cpu().detach().numpy()

        track_validation_node.append(loss_node.cpu().detach().numpy())
        track_validation_edge.append(loss_edge.cpu().detach().numpy())
        track_validation_loss.append(loss.cpu().detach().numpy())

        @env.only_rank_zero
        def print_val_info():
            print("Validation loss: ", validation_loss)
            print("Validation node loss: ", loss_node.cpu().detach().numpy())
            print("Validation edge loss: ", loss_edge.cpu().detach().numpy())
        print_val_info()

        # save the model and the current training status every 100 epochs
        if epoch % 100 == 0:
            save_training_state(model, optimizer, track_loss_edge, track_loss_node, track_validation_edge, track_validation_node, save_file)
        
        if schedule == True:    
            scheduler.step(validation_loss)
            current_lr = optimizer.param_groups[0]['lr']

            print(f"Current Learning Rate: {current_lr:.8f}")
            if current_lr <= min_lr:
                print("Learning rate has reached the minimum threshold. Stopping training.")
                break

        if loss < loss_tol:
            print("Loss has reached the minimum threshold. Stopping training.")
            break
            
    print("Final loss: ", loss) 
    save_training_state(model, optimizer, track_loss_edge, track_loss_node, track_validation_edge, track_validation_node, save_file)


############################################################
# Evaluating/Testing the model
############################################################

def evaluate_slice(model, data_loader, construct_kernel, equivariant_blocks, atom_orbitals, out_slices, device, save_file='./'):
    model.eval() 
    all_node_labels = []
    all_node_preds = []
    all_edge_labels = []
    all_edge_preds = []

    # currently only testing on a single rank with 1 batch, need to fix for multiple ranks and batches
    # all examples are set up with 1 batch
    assert len(data_loader) == 1

    if dist.is_available() and dist.is_initialized():
        # find_unused_parameters=True handles the cases where some parameters dont recieve gradients, such as the directed ones
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device, find_unused_parameters=True)
    
    with torch.no_grad(): 
        MAEloss_total = 0.0

        for i, test_batch in enumerate(data_loader):
            print(f"Loading batch {i}/{len(data_loader)}...")
            test_batch = test_batch.to(device)

            # Forward pass
            test_node, test_edge = model(test_batch)
            # print("--> Memory allocated: " + str(torch.cuda.memory_allocated(device)/1e9) + "GB")
            # torch.cuda.synchronize()  
            test_node = test_node.cpu()
            test_edge = test_edge.cpu()
            
            # if test_batch.labelled_node_size.item() exists, use it, otherwise use the total number of nodes
            if hasattr(test_batch, 'labelled_node_size'):
                labelled_node_size = test_batch.labelled_node_size.item()
                labelled_edge_size = test_batch.labelled_edge_size.item()
            else:
                batch_size = len(test_batch)
                labelled_node_size = test_batch[0].num_nodes * batch_size
                labelled_edge_size = test_batch[0].num_edges * batch_size

            arange_tensor = torch.arange(labelled_node_size).unsqueeze(0)
            onsite_edges = torch.cat((arange_tensor, arange_tensor), 0)

            # Process node predictions
            flattened_node_labels = construct_kernel.get_H(test_batch.node_y[0:labelled_node_size].cpu())
            flattened_node_pred = construct_kernel.get_H(test_node[:labelled_node_size].cpu())

            node_label = utils.unflatten(flattened_node_labels, test_batch.x[0:labelled_node_size],
                                         onsite_edges, equivariant_blocks, atom_orbitals, out_slices)
            
            node_pred = utils.unflatten(flattened_node_pred, test_batch.x[0:labelled_node_size],
                                        onsite_edges, equivariant_blocks, atom_orbitals, out_slices)
                        
            H_block_node_labels = [matrix.flatten() for matrix in node_label.values()]
            node_label_tensor = torch.cat(H_block_node_labels)
            H_block_node_pred = [matrix.flatten() for matrix in node_pred.values()]
            node_pred_tensor = torch.cat(H_block_node_pred)

            # Process edge predictions
            flattened_edge_labels = construct_kernel.get_H(test_batch.y[0:labelled_edge_size].cpu())
            flattened_edge_pred = construct_kernel.get_H(test_edge[0:labelled_edge_size].cpu())

            edge_label = utils.unflatten(flattened_edge_labels, test_batch.x[0:labelled_node_size],
                                         test_batch.edge_index[:, 0:labelled_edge_size],
                                         equivariant_blocks, atom_orbitals, out_slices)
            
            edge_pred = utils.unflatten(flattened_edge_pred, test_batch.x[0:labelled_node_size],
                                        test_batch.edge_index[:, 0:labelled_edge_size],
                                        equivariant_blocks, atom_orbitals, out_slices)
                    
            H_block_edge_labels = [matrix.flatten() for matrix in edge_label.values()]
            edge_label_tensor = torch.cat(H_block_edge_labels)
            H_block_edge_pred = [matrix.flatten() for matrix in edge_pred.values()]
            edge_pred_tensor = torch.cat(H_block_edge_pred)

            # Compute the MAE
            pred_tensor = torch.cat([node_pred_tensor, edge_pred_tensor])
            label_tensor = torch.cat([node_label_tensor, edge_label_tensor])
            MAEloss_total += torch.mean(torch.abs(pred_tensor - label_tensor))

            print("Mean Absolute Node Error in mHartree: ", torch.mean(torch.abs(node_pred_tensor - node_label_tensor)) * 1e3)
            print("Mean Absolute Edge Error in mHartree: ", torch.mean(torch.abs(edge_pred_tensor - edge_label_tensor)) * 1e3)
            print("Mean Absolute Error in mHartree: ", MAEloss_total * 1e3)

            # Collect results for plotting
            all_node_labels.append(node_label_tensor)
            all_node_preds.append(node_pred_tensor)
            all_edge_labels.append(edge_label_tensor)
            all_edge_preds.append(edge_pred_tensor)

            local_rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            # with open(save_file + '_MAE_rank_' + str(local_rank) + '_batch_' + str(i) + '_size_' + str(len(data_loader)) + '.txt', 'w') as f:
            #     f.write(f"Mean Absolute Node Error in mHartree: {torch.mean(torch.abs(node_pred_tensor - node_label_tensor)) * 1e3}\n")
            #     f.write(f"Mean Absolute Edge Error in mHartree: {torch.mean(torch.abs(edge_pred_tensor - edge_label_tensor)) * 1e3}\n")
            #     f.write(f"Mean Absolute Error in mHartree: {MAEloss_total * 1e3}\n")

            # Clear cache after processing each batch
            del test_node, test_edge, test_batch, node_label, node_pred, edge_label, edge_pred #, node_label_tensor, node_pred_tensor, edge_label_tensor, edge_pred_tensor
            del flattened_node_labels, flattened_node_pred, flattened_edge_labels, flattened_edge_pred, H_block_node_labels, H_block_node_pred, H_block_edge_labels, H_block_edge_pred
            del pred_tensor, label_tensor
            # torch.cuda.empty_cache()
            gc.collect()  # Python garbage collection
            # torch.cuda.synchronize()  
            # print("--> Memory allocated (after gc): " + str(torch.cuda.memory_allocated(device)/1e9) + "GB")

        # average the loss over all batches
        MAEloss_total = MAEloss_total / len(data_loader)

    # Concatenate all results
    all_node_labels = torch.cat(all_node_labels)
    all_node_preds = torch.cat(all_node_preds)
    all_edge_labels = torch.cat(all_edge_labels)
    all_edge_preds = torch.cat(all_edge_preds)

    # * Note: testing is always done on a single rank with 1 batch so far, need to modify this for multiple ranks and batches
    # collect the loss from all ranks:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(torch.tensor(MAEloss_total, device=device), op=dist.ReduceOp.SUM)
    
    local_rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    
    # downsample for plotting:
    downsample = 100
    all_node_labels = all_node_labels[::downsample]
    all_node_preds = all_node_preds[::downsample]
    all_edge_labels = all_edge_labels[::downsample]
    all_edge_preds = all_edge_preds[::downsample]

    # Plotting
    plt.figure(figsize=(4, 3))
    plt.scatter(all_edge_labels.cpu().numpy(), all_edge_preds.cpu().numpy(), s=1, alpha=0.5, edgecolor='none', color='crimson', label='Edge')
    plt.scatter(all_node_labels.cpu().numpy(), all_node_preds.cpu().numpy(), s=1, alpha=0.5, edgecolor='none', color='blue', label='Node')
    plt.plot(all_node_labels.cpu().numpy(), all_node_labels.cpu().numpy(), c='k', linestyle='dashed', linewidth=0.1, alpha=0.3)
    plt.xlabel(r"$(H_{ij})_{\alpha \beta}^{GT}$")
    plt.ylabel(r"$(H_{ij})_{\alpha \beta}^{pred}$")
    plt.legend()
    plt.savefig(save_file+'_prediction.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_test_info(model, test_batch, construct_kernel, equivariant_blocks, atom_orbitals, out_slices, device, save_file='model_in_training.pth'):
    """
    Evaluate the model on the test set and return the mean absolute error for the node and edge predictions after reconstructing the Hamiltonian matrices from the predictions.

    """
    test_batch = test_batch.to(device)
    test_node, test_edge = model(test_batch)

    test_info = {}

    test_node = test_node
    test_edge = test_edge

    if torch.is_tensor(test_batch.labelled_node_size):
        labelled_node_size = test_batch.labelled_node_size.item()
        labelled_edge_size = test_batch.labelled_edge_size.item() #when test batch is loaded from dataloader, it is a tensor instead of a number

    else:
        labelled_node_size = test_batch.labelled_node_size
        labelled_edge_size = test_batch.labelled_edge_size

    flattened_node_labels = construct_kernel.get_H(test_batch.node_y[0:labelled_node_size]) #convert into flattened Hamiltonian form
    flattened_node_pred = construct_kernel.get_H(test_node[0:labelled_node_size])

    onsite_edge_index = torch.cat((torch.arange(labelled_node_size).unsqueeze(0),torch.arange(labelled_node_size).unsqueeze(0)),0)
    numbers = test_batch.x[0:test_batch.labelled_node_size]

    node_label = utils.unflatten(flattened_node_labels,numbers, onsite_edge_index,equivariant_blocks,atom_orbitals,out_slices)
    node_pred = utils.unflatten(flattened_node_pred,numbers, onsite_edge_index,equivariant_blocks,atom_orbitals,out_slices)

    H_block_node_labels = [matrix.flatten() for matrix in node_label.values()]
    node_label_tensor = torch.cat(H_block_node_labels)

    H_block_node_pred = [matrix.flatten() for matrix in node_pred.values()]
    node_pred_tensor = torch.cat(H_block_node_pred)


    test_info['flattened_node_labels'] = flattened_node_labels
    test_info['flattened_node_pred'] = flattened_node_pred
    test_info['node_label'] = node_label_tensor
    test_info['node_pred'] = node_pred_tensor

    flattened_edge_labels = construct_kernel.get_H(test_batch.y[0:labelled_edge_size])
    flattened_edge_pred = construct_kernel.get_H(test_edge[0:labelled_edge_size])

    edge_label = utils.unflatten(flattened_edge_labels,numbers, test_batch.edge_index[:,0:labelled_edge_size],equivariant_blocks,atom_orbitals,out_slices)
    edge_pred = utils.unflatten(flattened_edge_pred,numbers, test_batch.edge_index[:,0:labelled_edge_size],equivariant_blocks,atom_orbitals,out_slices)

    H_block_edge_labels = [matrix.flatten() for matrix in edge_label.values()]
    edge_label_tensor = torch.cat(H_block_edge_labels)

    H_block_edge_pred = [matrix.flatten() for matrix in edge_pred.values()]
    edge_pred_tensor = torch.cat(H_block_edge_pred)

    test_info['flattened_edge_labels'] = flattened_edge_labels
    test_info['flattened_edge_pred'] = flattened_edge_pred
    test_info['edge_label'] = edge_label_tensor
    test_info['edge_pred'] = edge_pred_tensor

    torch.save(test_info, save_file+'_test_info.pt')


    MAE_node = torch.mean(torch.abs(node_label_tensor - node_pred_tensor))
    MAE_edge = torch.mean(torch.abs(edge_label_tensor - edge_pred_tensor))

    pred_tensor = torch.cat([node_pred_tensor, edge_pred_tensor])
    label_tensor = torch.cat([node_label_tensor, edge_label_tensor])
    MAEloss_total = torch.mean(torch.abs(pred_tensor - label_tensor))

    print("Mean Absolute Node Error in mHartree: ", MAE_node*1000)
    print("Node Standard Deviation: ", torch.std(torch.abs(node_label_tensor - node_pred_tensor))*1000)
    print("Mean Absolute Edge Error in mHartree: ", MAE_edge*1000)
    print("Edge Standard Deviation: ", torch.std(torch.abs(edge_label_tensor - edge_pred_tensor))*1000)
    print("Mean Total Error:", MAEloss_total*1000)

    return MAE_node, MAE_edge


def evaluate_model(
    model,
    test_batch,
    construct_kernel,
    equivariant_blocks,
    atom_orbitals,
    out_slices,
    device,
    save_file="model_in_training.pth",
    reconstruct_ham=True,
    compute_total_loss=True,
    plot=True,
    lower_triangular=False,
    H_filter=False,
    S_filter_value=None
):
    """
    Evaluate the model on the test set and return the mean absolute error for the node and edge predictions. Also reconstructs the Hamiltonian matrices from the predictions.

    """
    # MPI Initialization
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    test_batch = test_batch.to(device)
    test_node, test_edge = model(test_batch)
    global_edge_index = test_batch.edge_index

    # test_node = test_node.cpu()
    # test_edge = test_edge.cpu()

    if torch.is_tensor(test_batch.labelled_node_size):
        labelled_node_size = test_batch.labelled_node_size.item()
        labelled_edge_size = (
            test_batch.labelled_edge_size.item()
        )  # when test batch is loaded from dataloader, it is a tensor instead of a number

    else:
        labelled_node_size = test_batch.labelled_node_size
        labelled_edge_size = test_batch.labelled_edge_size

    labelled_edge_index = global_edge_index[0:labelled_edge_size]

    # 🔹 Distribute work across ranks
    node_indices_split = np.array_split(np.arange(labelled_node_size), size)
    edge_indices_split = np.array_split(np.arange(labelled_edge_size), size)

    local_node_indices = node_indices_split[rank]  # Nodes assigned to this rank
    local_edge_indices = edge_indices_split[rank]  # Edges assigned to this rank

    if S_filter_value is not None and test_batch.S_input is not None:

        print("Filtering edges based on S matrix")

        local_S_matrix = test_batch.S_input[local_edge_indices] #filter out edges where S is too small

        S_means = torch.mean(torch.abs(local_S_matrix), dim=1)
        S_mask = S_means > S_filter_value
        print(local_edge_indices.shape)
        local_edge_indices = local_edge_indices[S_mask.cpu().numpy()]
        
        print(local_edge_indices.shape)


    # 🔹 Distribute labelled_atom_index and labelled_edge_index
    local_labelled_atom_index = local_node_indices
    local_labelled_edge_index = labelled_edge_index[
        :, local_edge_indices
    ]  # Shape (2, num_local_edges)
    numbers = test_batch.x[0:labelled_node_size]

    # Extract local predicted data for this rank
    local_test_node = test_node[local_node_indices]
    local_test_edge = test_edge[local_edge_indices]


    if H_filter == True:

        print("Filtering elements based on non-zero elements of label H matrix")

        local_label_node = test_batch.node_y[local_node_indices]
        local_label_edge = test_batch.y[local_edge_indices]

        local_test_node[torch.abs(local_label_node) < 1e-6] = 0
        local_test_edge[torch.abs(local_label_edge) < 1e-6] = 0


    # 🔹 Compute local flattened Hamiltonians
    local_flattened_node_pred = construct_kernel.get_H(local_test_node)
    local_flattened_edge_pred = construct_kernel.get_H(local_test_edge)
    onsite_edge_index = torch.cat(
        (
            torch.tensor(local_labelled_atom_index).unsqueeze(0),
            torch.tensor(local_labelled_atom_index).unsqueeze(0),
        ),
        0,
    )
    # 🔹 Compute local dictionaries using unflatten
    local_node_pred_dic = utils.unflatten(
        local_flattened_node_pred,
        numbers,
        onsite_edge_index,
        equivariant_blocks,
        atom_orbitals,
        out_slices,
    )

    local_edge_pred_dic = utils.unflatten(
        local_flattened_edge_pred,
        numbers,
        local_labelled_edge_index,
        equivariant_blocks,
        atom_orbitals,
        out_slices,
    )

    if compute_total_loss == True:

        # Extract local predicted data for this rank
        local_label_node = test_batch.node_y[local_node_indices]
        local_label_edge = test_batch.y[local_edge_indices]

        # 🔹 Compute local flattened Hamiltonians
        local_flattened_node_labels = construct_kernel.get_H(local_label_node)
        local_flattened_edge_labels = construct_kernel.get_H(local_label_edge)

        # 🔹 Compute local dictionaries using unflatten
        local_node_label_dic = utils.unflatten(
            local_flattened_node_labels,
            numbers,
            onsite_edge_index,
            equivariant_blocks,
            atom_orbitals,
            out_slices,
        )

        local_edge_label_dic = utils.unflatten(
            local_flattened_edge_labels,
            numbers,
            local_labelled_edge_index,  # Use local_labelled_edge_index
            equivariant_blocks,
            atom_orbitals,
            out_slices,
        )

        H_block_node_labels = [
            matrix.flatten() for matrix in local_node_label_dic.values()
        ]
        node_label_tensor = torch.cat(H_block_node_labels)

        H_block_node_pred = [
            matrix.flatten() for matrix in local_node_pred_dic.values()
        ]
        node_pred_tensor = torch.cat(H_block_node_pred)

        H_block_edge_labels = [
            matrix.flatten() for matrix in local_edge_label_dic.values()
        ]
        edge_label_tensor = torch.cat(H_block_edge_labels)

        H_block_edge_pred = [
            matrix.flatten() for matrix in local_edge_pred_dic.values()
        ]
        edge_pred_tensor = torch.cat(H_block_edge_pred)

        MAE_node = torch.mean(torch.abs(node_label_tensor - node_pred_tensor))
        MAE_edge = torch.mean(torch.abs(edge_label_tensor - edge_pred_tensor))

        print("Mean Absolute Local Node Error in mHartree: ", MAE_node*1e3)
        print("Mean Absolute Local Edge Error in mHartree: ", MAE_edge*1e3)

        all_node_labels = comm.gather(node_label_tensor, root=0)
        all_node_preds = comm.gather(node_pred_tensor, root=0)
        all_edge_labels = comm.gather(edge_label_tensor, root=0)
        all_edge_preds = comm.gather(edge_pred_tensor, root=0)

        if rank == 0:

            all_pred_tensor = torch.cat(
                [torch.cat(all_node_preds), torch.cat(all_edge_preds)]
            )
            all_label_tensor = torch.cat(
                [torch.cat(all_node_labels), torch.cat(all_edge_labels)]
            )

            node_error = torch.mean(
                torch.abs(torch.cat(all_node_labels) - torch.cat(all_node_preds))
            )
            edge_error = torch.mean(
                torch.abs(torch.cat(all_edge_labels) - torch.cat(all_edge_preds))
            )
            total_error = torch.mean(torch.abs(all_label_tensor - all_pred_tensor))

            print("Mean Absolute Node Error in mHartree: ", node_error * 1e3)
            print("Mean Absolute Edge Error in mHartree: ", edge_error * 1e3)
            print("Mean Absolute Error in mHartree: ", total_error * 1e3)

            if plot == True:
                print("Plotting")
                plt.figure(figsize=(4, 3))
                plt.scatter(
                    torch.cat(all_edge_labels).detach().numpy(),
                    torch.cat(all_edge_preds).detach().numpy(),
                    s=1,
                    alpha=0.5,
                    edgecolor="none",
                    color="crimson",
                    label="Edge",
                )
                plt.scatter(
                    torch.cat(all_node_labels).detach().numpy(),
                    torch.cat(all_node_preds).detach().numpy(),
                    s=1,
                    alpha=0.5,
                    edgecolor="none",
                    color="blue",
                    label="Node",
                )
                plt.plot(
                    torch.cat(all_node_labels).detach().numpy(),
                    torch.cat(all_node_labels).detach().numpy(),
                    c="k",
                    linestyle="dashed",
                    linewidth=0.1,
                    alpha=0.3,
                )
                plt.xlabel("Real $H_{ij}$")
                plt.ylabel("Predicted  $H_{ij}$")
                plt.legend()
                # plt.text(0.5, 0.1, 'Node loss = '+str(MAE_node.item())+', Edge loss = '+str(MAE_edge.item()), fontsize=5, transform=plt.gca().transAxes)
                plt.savefig(
                    "prediction_" + save_file + ".png", dpi=300, bbox_inches="tight"
                )
                plt.close()

    if reconstruct_ham == True:
        pred_dic = local_node_pred_dic.copy()
        pred_dic.update(local_edge_pred_dic)
        reconstruct_hamiltonian(
            pred_dic,
            numbers,
            comm,
            rank,
            atom_orbitals,
            save_file=save_file,
            lower_triangular=lower_triangular,
        )


def reconstruct_hamiltonian(
    local_pred_dic,
    numbers,
    comm,
    rank,
    atom_orbitals,
    save_file="model_in_training.pth",
    lower_triangular=False,
):
    local_keys = local_pred_dic.keys()
    filtered_local_keys = []

    if lower_triangular == True:
        for key in local_keys:
            if key[0] >= key[1]:
                filtered_local_keys.append(tuple(key))

    else:
        for key in local_keys:
            if key[0] <= key[1]:  # remove all duplicate offsite blocks
                filtered_local_keys.append(tuple(key))

    print("filtering done")

    local_positions = []
    local_values = []

    # Start timing
    start_time = time.time()

    # Convert global_atomic_numbers to NumPy array
    global_atomic_numbers = np.array(numbers.tolist())

    # Compute the number of orbitals for each atom type
    num_orbitals_per_atom = np.array(
        [
            np.sum(2 * np.array(atom_orbitals[str(atom)]) + 1)
            for atom in global_atomic_numbers
        ]
    )

    # Compute starting indices (Hamiltonian indices start from 1)
    starting_indices = (
        np.cumsum(num_orbitals_per_atom) + 1
    )  # switch from 0-based to 1-based indexing
    starting_indices = np.insert(starting_indices, 0, 1)[
        :-1
    ]  # add 1 at the beginning and remove last element

    # Extract atom indices from keys
    keys_array = np.array(filtered_local_keys)  # Shape: (N, 2)
    atom_i_indices = keys_array[:, 0]
    atom_j_indices = keys_array[:, 1]

    starting_i = starting_indices[atom_i_indices]
    starting_j = starting_indices[atom_j_indices]

    H_blocks = [local_pred_dic[tuple(k)] for k in filtered_local_keys]

    # Process each block separately due to varying sizes
    local_positions = []
    local_values = []

    for H_block, s_i, s_j in zip(H_blocks, starting_i, starting_j):
        row_idx, col_idx = np.indices(H_block.shape)

        global_i = s_i + row_idx
        global_j = s_j + col_idx

        # Apply triangular condition
        if lower_triangular:
            mask = global_i >= global_j
        else:
            mask = global_i <= global_j

        H_block = H_block.detach().numpy()
        mask &= H_block != 0

        # Collect valid positions and values
        local_positions.append(np.column_stack((global_i[mask], global_j[mask])))
        local_values.append(H_block[mask])

    # Flatten into single arrays
    local_positions = np.concatenate(local_positions, axis=0)
    local_values = np.concatenate(local_values, axis=0)

    # Step 2: Gather results at the root rank
    all_positions = comm.gather(local_positions, root=0)
    all_values = comm.gather(local_values, root=0)

    # Step 3: Root rank processes and writes
    if rank == 0:
        # Combine results from all ranks
        combined_positions = np.concatenate(all_positions, axis=0)
        combined_values = np.concatenate(all_values, axis=0)

        # Sort by positions
        paired = zip(combined_positions, combined_values)
        sorted_pairs = sorted(paired, key=lambda pair: pair[0][0])
        positions_sorted, values_sorted = zip(*sorted_pairs)

        # Write to the output file
        with open(save_file, "w") as file:
            for (i, j), value in zip(positions_sorted, values_sorted):
                file.write(f"       {i}        {j}  {value:.8e}\n")

        print(f"Hamiltonian matrix written to {save_file}")

    # End timing
    end_time = time.time()

    # Print total time taken
    reconstruct_time = end_time - start_time
    # if rank == 0:
    print(f"Reconstruct_time: {reconstruct_time:.2f} seconds")


def plot_eigenvalue_comparison(reference_path, test_path, save_file = "model"):

    plt.rcParams.update({'font.size': 14})
    w = np.load(test_path)

    w_ref = np.load(reference_path)

    print(np.linalg.norm(w - w_ref, ord=2) / np.linalg.norm(w_ref, ord=2))
    print(np.linalg.norm(w - w_ref, ord=1) / np.linalg.norm(w_ref, ord=1))

    plt.figure(figsize=(6, 4))
    plt.scatter(np.arange(len(w)), w, s=1.2, alpha=0.2, c="tomato")  # make dot size smaller
    plt.scatter(0, -10, s=10, c="tomato", label=r"$\mathbf{H}_{ij}^{pred}$") 
    plt.scatter(np.arange(len(w_ref)), w_ref, s=1.2, alpha=0.2, c="mediumslateblue")  # make dot size smaller
    plt.scatter(0, -10, s=10, c="mediumslateblue",  label=r"$\mathbf{H}_{ij}^{GT}$") 

    # y Eigenvale
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue ($\mathbf{H})\;[E_h]$")
    plt.ylim(-2.1, 1.1)

    plt.legend(frameon=False, loc='lower right', title=r"[$\alpha$=0.2]")
    plt.savefig(save_file+"_comparison_eigenvalue"+"_zoom.png", dpi=700, bbox_inches='tight')
    plt.close("all")

def compute_eigenvalues(base_path_lower, S_path, base_path_upper = None, symmetrize=False, save_file = "model"):

    # Load the data
    S = np.loadtxt(S_path)

    S_row_ind = S[:, 0].astype(np.int32) - 1
    S_col_ind = S[:, 1].astype(np.int32) - 1
    S_data = S[:, 2]

    S_matrix = sp.coo_matrix((S_data, (S_row_ind, S_col_ind)))   

    H_lower_diagonal_name = base_path_lower
    
    H = np.loadtxt(H_lower_diagonal_name)

    H_row_ind = H[:, 0].astype(np.int32) - 1
    H_col_ind = H[:, 1].astype(np.int32) - 1
    H_data = H[:, 2]

    H_matrix = sp.coo_matrix((H_data, (H_row_ind, H_col_ind)))
    H_matrix = H_matrix.toarray()
    tmp = H_matrix.conj().T.copy()
    # set diagonal to zero
    np.fill_diagonal(tmp, 0)
    H_lower_matrix = H_matrix + tmp

    assert np.allclose(H_lower_matrix, H_lower_matrix.conj().T)

    if symmetrize:
        assert base_path_upper is not None
        H_upper_diagonal_name = base_path_upper

        H = np.loadtxt(H_upper_diagonal_name)

        H_row_ind = H[:, 0].astype(np.int32) - 1
        H_col_ind = H[:, 1].astype(np.int32) - 1
        H_data = H[:, 2]

        H_matrix = sp.coo_matrix((H_data, (H_row_ind, H_col_ind)))
        H_matrix = H_matrix.toarray()
        tmp = H_matrix.conj().T.copy()
        # set diagonal to zero
        np.fill_diagonal(tmp, 0)
        H_upper_matrix = H_matrix + tmp

        assert np.allclose(H_upper_matrix, H_upper_matrix.conj().T)

        H_full_matrix = (H_lower_matrix + H_upper_matrix)/2

    else:
        H_full_matrix = H_lower_matrix


    S_matrix = S_matrix.toarray()
    tmp = S_matrix.conj().T.copy()
    # set diagonal to zero
    np.fill_diagonal(tmp, 0)
    S_matrix = S_matrix + tmp

    assert np.allclose(H_full_matrix, H_full_matrix.conj().T)
    assert np.allclose(S_matrix, S_matrix.conj().T)

    start = time.time()
    w, v = scipy.linalg.eigh(H_full_matrix, S_matrix, lower=True)
    end = time.time()

    print("Time: ", end - start)

    # save the eigenvalues and eigenvectors
    np.save(save_file+'eigenvalues.npy', w)
    np.save(save_file+'eigenvectors.npy', v)