import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
lib_root = os.path.join(project_root, 'model')
lib_equiformer_root = os.path.join(project_root, 'lib_equiformer')
sys.path.append(lib_root)
sys.path.append(lib_equiformer_root)
print(f"Added {lib_root} to the path", flush=True)
print(f"Added {lib_equiformer_root} to the path", flush=True)

import argparse
import numpy as np
import torch.distributed as dist
import torch
import random

import data, training, structure, process_irreps, network, SO3, compute_env as env, utils
from e3nn.o3 import Irreps
print("Imported libraries", flush=True)
import torch.distributed as dist
# import torch.multiprocessing as mp

def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from keys in state_dict."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[len('module.'):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def main(folder):

    if not torch.cuda.is_available():
        raise RuntimeError("No GPUs are available!")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # ************************************************************
    # Distributed training setup (if running on multiple GPUs)
    # ************************************************************

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device, flush=True)

    if 'SLURM_PROCID' in os.environ:  
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(local_rank)
        backend = 'gloo'  # Use NCCL for multi-GPU on Piz Daint (edit: uses RDMA, switching to gloo)
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    else:  
        rank = 0
        world_size = 1
        local_rank = 0
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        backend = 'gloo'  # Use Gloo for attelas (single GPU)

    if dist.is_initialized() and dist.get_rank() == 0:  
        print(f"RANK: {rank}", flush=True)
        print(f"WORLD_SIZE: {world_size}", flush=True)
        print(f"LOCAL_RANK: {local_rank}", flush=True)

    # ************************************************************
    # Input parameters and for the HfO2 dataset
    # ************************************************************

    # data_folder = os.path.join(folder, 'datasets/HfO2_2/')
    # xyz_file = data_folder + 'structure.xyz'
    # hamiltonian_file = data_folder + 'H.csr'
    # overlap_file = data_folder + 'S.csr'

    # val_data_folder = os.path.join(folder, 'datasets/HfO2_1/')
    # val_xyz_file = val_data_folder+'structure.xyz'
    # val_hamiltonian_file = val_data_folder + 'H.csr'
    # val_overlap_file = val_data_folder + 'S.csr'

    data_folder = '/usr/scratch2/tortin13/chexia/HfO2_2/'
    xyz_file = data_folder + 'snapshot.xyz'
    hamiltonian_file = data_folder + 'memrstors-KS_SPIN_1-1_0.csr'
    overlap_file = data_folder + 'memrstors-S_SPIN_1-1_0.csr'

    val_data_folder = '/usr/scratch2/tortin13/chexia/HfO2_1/'
    val_xyz_file = val_data_folder+'snapshot.xyz'
    val_hamiltonian_file = val_data_folder + 'memrstors-KS_SPIN_1-1_0.csr'
    val_overlap_file = val_data_folder + 'memrstors-S_SPIN_1-1_0.csr'


    

    # Material parameters:
    pbc = True
    orbital_basis = 'SZV'
    rcut = 4.0          
    lmax = 4 
    mmax = 4

    # Graph partitioning methods (the first three are for the slice option, the last two are for the graph partitioning option):
    partitioning = 'slice_length'                                                          # 'slice' or 'graph' partitioning
    
    # slice parameters:
                                                                 # cutoff boundary of the slice used for training (interaction radius = 2*cutoff)
    
    # graph partitioning parameters:
    num_subgraph = 1                                                                # min 10 for P100 GPU memory with attn_hidden_channels=64
    num_batch = 1                                                                   # number of subgraphs which will actually be added to the dataset for training,
                                                                                    # after dividing the graph into 'num_subgraph' subgraphs
    # Parameters:
    restart_file = None
    save_file = 'model_HfO2_'+str(world_size)+'_contact_multiple_direction_new'
    train_or_test = 'train'                                          
    num_MP_layers = 1                                                              # Number of message passing layers 
    num_epochs = 50000                                                              # Number of epochs                                                
    learning_rate = 1e-4                                                            # Initial Learning rate                 
    loss_tol = 0                                                                    # Loss tolerance for early stopping
    dtype = torch.float32

    # *** Initialize the hyperparameters of the SO2 model:
    sphere_channels = 16
    num_heads = 2
    attn_hidden_channels = 16 
    attn_alpha_channels = 16
    attn_value_channels = 16
    ffn_hidden_channels = 64

    # ************************************************************
    # Create the dataset
    # ************************************************************

    # *** Initialize the domain and electronic structure matrices:
    
    # a_HfO2s = []

    a_HfO2 = structure.Structure(xyz_file, 
                                    hamiltonian_file, 
                                    overlap_file, 
                                    pbc, 
                                    orbital_basis, 
                                    make_soap=False, 
                                    save_matrices=False,
                                    self_interaction=False,
                                    bothways=True, 
                                    rcut = rcut)
    
    print("Structure 1 created", flush=True)

    # a_HfO2s.append(a_HfO2)

    a_HfO2_val = structure.Structure(val_xyz_file, 
                                    val_hamiltonian_file, 
                                    val_overlap_file, 
                                    pbc, 
                                    orbital_basis, 
                                    make_soap=False, 
                                    save_matrices=False,
                                    self_interaction=False,
                                    bothways=True, 
                                    rcut = rcut)
    
    print("Validation structure created", flush=True)

    if dist.is_initialized():
        dist.barrier()
    
    
    # ************************************************************
    # Initialize the SO2 model
    # ************************************************************

    # *** Define irreducible representations
    irreps_in = Irreps([(sphere_channels, (0, 1)), (sphere_channels, (1, 1)), (sphere_channels, (2, 1)), (sphere_channels, (3, 1)), (sphere_channels, (4, 1))])
    edge_channels_list = [sphere_channels, sphere_channels, sphere_channels]  

    # *** Perform orbital analysis:
    atom_orbitals = {'8':[0,1], '72':[0,0,1,2]}                                           # Orbital types of each atom in the structure
    numbers = a_HfO2.atomic_numbers                                                       # Atomic numbers of each atom in the structure
    no_parity = True                                                                      # No parity symmetry          
    orbital_types = [[0,1],[0,0,1,2]]      # must be in ascending order of atomic numbers                                                 # basis rank of each atom in the structure 

    targets, net_out_irreps, net_out_irreps_simplified = process_irreps.orbital_analysis(atom_orbitals, targets=None, no_parity=no_parity)
    index_to_Z, inverse_indices = torch.unique(numbers, sorted=True, return_inverse=True)
    equivariant_blocks, out_js_list, out_slices = process_irreps.process_targets(orbital_types, index_to_Z, targets)


    construct_kernel = process_irreps.e3TensorDecomp(net_out_irreps, 
                                          out_js_list, 
                                          default_dtype_torch=torch.float32, 
                                          spinful=False,
                                          no_parity=no_parity, 
                                          if_sort=False, 
                                          device_torch='cpu') #the data is created on cpu, so the construct_kernel must be on cpu 
    print("Orbital analysis completed", flush=True)

    # *** Create the input dataloader:
    if partitioning == 'slice_center':
        slice_list = [1000,1200,1400]                                                             # slice boundaries for partitioning the structure into subgraphs                
        cutoff = 1.5    
        data_loader = data.batch_data_subgraph(a_HfO2, slice_list, cutoff, equivariant_blocks=equivariant_blocks, out_slices=out_slices, construct_kernel=construct_kernel, dtype=torch.float32)
        print("Data loader created - using " + str(len(slice_list)) + " slices", flush=True)
    
    
    elif partitioning == 'slice_length':
        # total_length = 54
        # num_slices = 18
        start = 15
        total_length = 3
        num_slices = 1
        data_loader = data.batch_data_HfO2_cartesian(a_HfO2, start, total_length, num_slices, equivariant_blocks=equivariant_blocks, out_slices=out_slices, construct_kernel=construct_kernel, dtype=torch.float32, slice_direction=0)
        print("Data loader created")

        start = 25
        total_length = 4
        num_slices = 1
        assert num_slices == 1 #only one slice for validation
        validation_loader = data.batch_data_HfO2_cartesian(a_HfO2_val, start, total_length, num_slices, equivariant_blocks=equivariant_blocks, out_slices=out_slices, construct_kernel=construct_kernel, dtype=torch.float32, slice_direction=0)
        print("Validation loader created")
        # test_data = torch.load('test_data_structures/model_HfO2x2_structure_1_training_1500_2.pt')
        # validation_loader = data.batch_data_load(test_data, equivariant_blocks, out_slices, construct_kernel, dtype=torch.float32)

    else:
        data_loader = data.batch_data_graphpartition(a_HfO2, num_subgraph, num_batch, equivariant_blocks=equivariant_blocks, out_slices=out_slices, construct_kernel=construct_kernel, dtype=torch.float32)
        print("Data loader created - using " + str(num_subgraph) + " subgraphs", flush=True)
    if dist.is_initialized():
        dist.barrier()

    # *** Initialize the model:
    mappingReduced = SO3.CoefficientMappingModule(lmax, mmax)
    irreps_out = net_out_irreps
    model = network.SO2Net(num_MP_layers, 
                                lmax, 
                                mmax, 
                                mappingReduced, 
                                sphere_channels, 
                                edge_channels_list, 
                                attn_hidden_channels, 
                                num_heads, 
                                attn_alpha_channels, 
                                attn_value_channels, 
                                ffn_hidden_channels, 
                                irreps_in, 
                                irreps_out)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    
    if restart_file is not None:
        print("Restarting training from a saved model and optimizer state...", flush=True)
        checkpoint = torch.load(restart_file)
        state_dict = checkpoint['model_state_dict']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if dist.is_available() and dist.is_initialized():
            # If the model was saved with DDP, remove the 'module' prefix that it might have (just in case)
            if 'module.' in next(iter(checkpoint['model_state_dict'].keys())):
                prefix = 'module.'
                state_dict = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}
            # with the current training setup, the module prefix is already removed
            model.load_state_dict(state_dict)
        else:
            state_dict = remove_module_prefix(checkpoint['model_state_dict'])
            model.load_state_dict(state_dict)

    print("Model initialized", flush=True)
    print("Number of parameters: ", sum(p.numel() for p in model.parameters()), flush=True)

    print("memory: " + str(torch.cuda.memory_allocated(device)/1e9) + "GB", flush=True)
    if dist.is_initialized():
        dist.barrier()

    # ************************************************************
    # Training and testing the model
    # ************************************************************

    if train_or_test == 'train':
        
        # *** Train the model parameters:
        
        print("validation loader created", flush=True)  
        print("training...", flush=True)

        training.train_and_validate_model_subgraph(model, optimizer, data_loader, validation_loader, num_epochs, loss_tol, save_file=save_file, schedule=True, dtype=dtype)
        print("Model trained, plotting fit to training data", flush=True)
        training.evaluate_model(model, data_loader, construct_kernel, equivariant_blocks, atom_orbitals, out_slices, device)

        print("testing on unseen data...", flush=True)

        training.evaluate_model(model, validation_loader, construct_kernel, equivariant_blocks, atom_orbitals, out_slices, device)


    # use with a restarted model, to test the model
    elif train_or_test == 'test':
        print("testing on unseen data...", flush=True)
        test_data = torch.load('test_data_structures/model_HfO2x2_structure_1_training_1500_2.pt')
        test_data_loader = data.batch_data_load(test_data, equivariant_blocks, out_slices, construct_kernel, dtype=torch.float32)
        # data_loader = data.batch_data_subgraph(a_HfO2, slice_list, cutoff, equivariant_blocks=equivariant_blocks, out_slices=out_slices, construct_kernel=construct_kernel, dtype=torch.float32)
        training.evaluate_model(model, test_data_loader, construct_kernel, equivariant_blocks, atom_orbitals, out_slices, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amorphous GNNs --- HfO2")
    parser.add_argument("-f", "--folder", default="", required=False)
    args = parser.parse_args()

    print(f"Starting main ... dataset folder is '{args.folder}'", flush=True)

    main(args.folder)
