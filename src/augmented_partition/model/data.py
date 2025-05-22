# This file contains the functions to process the data, create the input data object for the GNN, and batch the data for training

import torch
import numpy as np
import augmented_partition.model.utils as utils

from torch_geometric.data import Data as gnnData
from torch_geometric.data import Batch, Data
from torch.utils.data import Dataset, DataLoader
from ase.geometry import find_mic
import torch.distributed as dist

# Custom dataset class for the GNN
class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def custom_collate_fn(batch):
    return Batch.from_data_list(batch)


#First part is for molecules 

def split_data_indices(num_train, num_validate, num_test, num_total, offset=0):
    """
    Splits the data indices into training, validation, and test sets
    """
    indices = np.arange(offset, num_total)
    np.random.shuffle(indices)

    train_indices = indices[:num_train]
    validate_indices = indices[num_train:num_train+num_validate]
    test_indices = indices[num_train+num_validate:num_train+num_validate+num_test]

    return train_indices, validate_indices, test_indices


def create_input_data_molecules(structure, equivariant_blocks, out_slices, construct_kernel, device, dtype):

    # Note: for SO2 network, edge_index has two-way edges, and does not include self-connections 
    edge_index = structure.edge_matrix
    numbers = torch.tensor([utils.periodic_table[i] for i in structure.atomic_species])
    coordinates = structure.atomic_structure.get_positions()
    cell = structure.atomic_structure.get_cell()

    # Make targets:

    # off-diagonal orbital blocks for each edge (bothways)
    edge_hams = structure.get_orbital_blocks(edge_index)
    edge_index = torch.tensor(edge_index)
    H_blocks_edge = [edge_hams[(edge_index[0][i].item(), edge_index[1][i].item())] for i in range(len(edge_index[0]))]
    H_blocks_edge = np.array(H_blocks_edge, dtype=object)

    # diagonal orbital blocks (onsite Hamiltonian)
    onsite_edge_index = np.array([np.arange(len(numbers)),np.arange(len(numbers))])
    onsite_hams = structure.get_orbital_blocks(onsite_edge_index)
    onsite = [onsite_hams[(onsite_edge_index[0][i].item(), onsite_edge_index[1][i].item())] for i in range(len(numbers))]  
    onsite = np.array(onsite, dtype=object)

    # off-diagonal orbital blocks
    edge_labels = []
    for i in range(len(edge_index[0])):
        label = np.zeros(out_slices[-1])
        for index_target, equivariant_block in enumerate(equivariant_blocks):
                for N_M_str, block_slice in equivariant_block.items():
                    slice_row = slice(block_slice[0], block_slice[1])
                    slice_col = slice(block_slice[2], block_slice[3])
                    # len_row = block_slice[1] - block_slice[0]
                    # len_col = block_slice[3] - block_slice[2]
                    slice_out = slice(out_slices[index_target], out_slices[index_target + 1])
                    condition_number_i, condition_number_j = N_M_str.split()

                    if (numbers[edge_index[0][i]].item() == int(condition_number_i) and numbers[edge_index[1][i]].item() == int(condition_number_j)):
                        label[slice_out] += np.squeeze(H_blocks_edge[i][slice_row, slice_col].reshape(1,-1))

        edge_labels.append(label)

    # diagonal orbital blocks
    node_labels = []
    for i in range(len(onsite_edge_index[0])):

        label = np.zeros(out_slices[-1])
        for index_target, equivariant_block in enumerate(equivariant_blocks):
                for N_M_str, block_slice in equivariant_block.items():
                    slice_row = slice(block_slice[0], block_slice[1])
                    slice_col = slice(block_slice[2], block_slice[3])
                    # len_row = block_slice[1] - block_slice[0]
                    # len_col = block_slice[3] - block_slice[2]
                    slice_out = slice(out_slices[index_target], out_slices[index_target + 1])
                    condition_number_i, condition_number_j = N_M_str.split()
                    if (numbers[onsite_edge_index[0][i]].item() == int(condition_number_i) and numbers[onsite_edge_index[1][i]].item() == int(condition_number_j)):
                        label[slice_out] += np.squeeze(onsite[i][slice_row, slice_col].reshape(1,-1))

        node_labels.append(label)
    numbers = numbers.numpy()

    coordinates = torch.tensor(coordinates)

    edge_fea = torch.empty((len(edge_index[0]),4))
    for i in range(len(edge_index[0])):
        distance_vector, distance = find_mic(coordinates[edge_index[1][i]] - coordinates[edge_index[0][i]], cell)
        edge_fea[i,:] = torch.cat((torch.tensor([distance]), torch.tensor(distance_vector)))

    edge_fea = torch.tensor(edge_fea, dtype=dtype)
    x = torch.tensor(numbers)

    edge_labels = torch.tensor(np.array(edge_labels),dtype=dtype, device=device)
    y = construct_kernel.get_net_out(edge_labels) #convert Hamiltonian labels from uncoupled space to coupled space (to avoid conversion during training)

    node_labels = torch.tensor(node_labels,dtype=dtype, device=device)
    node_y = construct_kernel.get_net_out(node_labels)

    data = gnnData(x=x, edge_index=edge_index, edge_attr=edge_fea, y=y, node_y=node_y)

    return data

# Creates a dataloader for a dataset with a list of molecules
def batch_data_molecules(structures, device, num_graph=1, batch_size=1, equivariant_blocks=None, out_slices=None, construct_kernel=None, dtype=torch.float64):

    data_list = []

    for i in range(num_graph):
        data = create_input_data_molecules(structures[i], equivariant_blocks, out_slices, construct_kernel, device, dtype=dtype)
        data_list.append(data)
    
    dataset = CustomDataset(data_list)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=0)

    print("*** Batch properties:")
    for batch in loader:
        print("Node Features (x):", batch.x.size())
        print("Edge Index:", batch.edge_index.size())
        print("Edge Features (edge_attr):", batch.edge_attr.size())    

    return loader




#Second part is augmented partitioning for materials
def create_slice_graph(atom_index, edge_matrix, add_virtual = True, two_way = False):

    """
    Generates required data to locate atoms and edges belonging to the slice sub-structure/graph

    Note: Virtual atoms are always at the end of the atom index list.

    Inputs: atom_index: list of atom indices that are part of the slice
           edge_matrix: edge indices of the full structure 
           add_virtual: if True, virtual atoms are added to the slice atom index list and their edges are included in the slice edge index list    

    Outputs: slice_graph: dictionary containing the following keys: 
            full_atom_index: atom indices of the slice sub-structure/graph, including virtual nodes 
            full_mapped_edge_index: edge indices of the slice sub-structure/graph, follows the order of the atom index list
            full_edge_positions: numbers indicating the positions of selected edge indices within the full edge index list
            node_degree: number of edges connected to each node
            reduced_node_degree: number of non-virtual edges connected to each node
            real_node_size: number of non-virtual atoms in the full atom index list, used to separate the virtual atoms from the labelled atoms
            real_edge_size: number of non-virtual edges in the full edge index list, used to separate the virtual edges from the labelled edges
    """
    
    virtual_atom_index = [] #atom indices of the virtual atoms
    edge_positions = [] #numbers indicating the positions of selected edge indices within the full edge index list 

    mapped_edge_index = [] #edge indices of the slice sub-structure/graph, follows the order of the atom index list  e.g. if atom index list is [25, 26, 40 ...], then atom 25 is atom 0 in the sub-structure/graph
    node_degree = [] #number of edges connected to each node
    reduced_node_degree = [] #number of non-virtual edges connected to each node

    slice_graph = {}

    for i in range(len(atom_index)):
        edge_position = np.squeeze(np.where(edge_matrix[0] == atom_index[i])) #locate the positions of all edges connected to that particular atom
        node_degree.append(len(edge_position))
        count = 0
        for j in range(len(edge_position)):
            if edge_matrix[1][edge_position[j]] in atom_index:
                atom_source_index = atom_index.index(edge_matrix[0][edge_position[j]]) #find the positions of the source and target atoms that are part of the slice (to create the edge indices for the data objects)
                atom_target_index = atom_index.index(edge_matrix[1][edge_position[j]])
                mapped_edge_index.append([atom_source_index,atom_target_index])
                edge_positions.append(edge_position[j])
                count = count + 1
            else:
                if edge_matrix[1][edge_position[j]] not in virtual_atom_index: #if the target atom is not part of the slice, add it to the virtual atom index list. Avoid duplicates
                    virtual_atom_index.append(edge_matrix[1][edge_position[j]].item())
                    
        reduced_node_degree.append(count)


    if (add_virtual == True):
        full_atom_index = atom_index + virtual_atom_index #add the indices of the virtual atoms to the original slice atom index list
        virtual_edge_positions = []
        virtual_mapped_edge_index = []

        for i in range(len(virtual_atom_index)): 
            virtual_edge_position = np.squeeze(np.where(edge_matrix[0] == virtual_atom_index[i])) #find the virtual edges connected to the virtual atoms
            for j in range(len(virtual_edge_position)):
                if edge_matrix[1][virtual_edge_position[j]] in atom_index:
                    atom_i_index = full_atom_index.index(edge_matrix[0][virtual_edge_position[j]]) #only include one way edges where the source atom is a virtual atom and the target atom is part of the slice
                    atom_j_index = full_atom_index.index(edge_matrix[1][virtual_edge_position[j]])
                    virtual_mapped_edge_index.append([atom_i_index,atom_j_index])
                    virtual_edge_positions.append(virtual_edge_position[j])

        full_mapped_edge_index = mapped_edge_index + virtual_mapped_edge_index #mapped edge indices of the full graph including virtual nodes 
        full_edge_positions = edge_positions + virtual_edge_positions
        
        if (two_way == True):
            print('Using two-way edges for virtual nodes')
            for i in range(len(atom_index)): 
                virtual_edge_position = np.squeeze(np.where(edge_matrix[0] == atom_index[i])) #find the virtual edges connected to the real atoms (source is now the real atom, target is the virtual atom)
                for j in range(len(virtual_edge_position)):
                    if edge_matrix[1][virtual_edge_position[j]] in virtual_atom_index:
                        atom_i_index = full_atom_index.index(edge_matrix[0][virtual_edge_position[j]]) 
                        atom_j_index = full_atom_index.index(edge_matrix[1][virtual_edge_position[j]])
                        virtual_mapped_edge_index.append([atom_i_index,atom_j_index])
                        virtual_edge_positions.append(virtual_edge_position[j])

    else:
        full_atom_index = atom_index
        full_mapped_edge_index = mapped_edge_index
        full_edge_positions = edge_positions

    slice_graph['full_atom_index'] = torch.tensor(full_atom_index)
    slice_graph['full_mapped_edge_index'] = torch.tensor(full_mapped_edge_index).T
    slice_graph['full_edge_positions'] = torch.tensor(full_edge_positions)
    slice_graph['node_degree'] = node_degree
    slice_graph['reduced_node_degree'] = reduced_node_degree
    slice_graph['real_node_size'] = len(atom_index) #index of the labelled atoms that are part of the slice 
    slice_graph['real_edge_size'] = len(edge_positions) #index of the labelled edges that are part of the slice
    
    return slice_graph


def flatten_data(H_blocks, edge_matrix, numbers, equivariant_blocks, out_slices):
    """
    Flattens the Hamiltonian blocks H_blocks into a 1D tensor for each edge in the slice sub-structure/graph
    """

    labels = []
    for i in range(len(edge_matrix[0])):
        label = np.zeros(out_slices[-1])
        for index_target, equivariant_block in enumerate(equivariant_blocks):
                for N_M_str, block_slice in equivariant_block.items():
                    slice_row = slice(block_slice[0], block_slice[1])
                    slice_col = slice(block_slice[2], block_slice[3])
                    # len_row = block_slice[1] - block_slice[0]
                    # len_col = block_slice[3] - block_slice[2]
                    slice_out = slice(out_slices[index_target], out_slices[index_target + 1])
                    condition_number_i, condition_number_j = N_M_str.split()
                    if (numbers[edge_matrix[0][i]].item() == int(condition_number_i) and numbers[edge_matrix[1][i]].item() == int(condition_number_j)):
                        label[slice_out] = label[slice_out]+np.squeeze(H_blocks[i][slice_row, slice_col].reshape(1,-1)) #slice_out should match with slice_row x slice_row when flattened

        labels.append(label)    

    return labels


def slice_cartesian(atom_pos,start,length,slice_direction = 0):
    if atom_pos[slice_direction] >= start and atom_pos[slice_direction] < start + length:
        return True
    else:
        return False

def createdata_subgraph_cartesian(structure, start, length, equivariant_blocks, out_slices, construct_kernel, dtype=torch.float64, slice_direction = 0, add_virtual = True, two_way = False, use_overlap = False):
    
    pos = structure.atomic_structure.get_positions()
    cell = structure.atomic_structure.get_cell()
    edge_matrix = structure.edge_matrix
    numbers = structure.atomic_numbers

    atom_index = []

    for i in range(len(numbers)):
        if slice_cartesian(pos[i],start,length, slice_direction):
            atom_index.append(i)

    slice_graph = create_slice_graph(atom_index, edge_matrix, add_virtual, two_way)

    full_mapped_edge_index = slice_graph['full_mapped_edge_index']
    full_edge_positions = slice_graph['full_edge_positions']
    full_atom_index = slice_graph['full_atom_index']


    # find the off-diagonal Hamiltonian blocks of all edges that are part of the graph
    edge_matrix = torch.tensor(edge_matrix)
    edge_index = edge_matrix.T[full_edge_positions].numpy() 
    edge_index = edge_index.T
    offsite_ham = structure.get_orbital_blocks(edge_index)
    H_blocks_edge = []
    for i in range(len(edge_index[0])):
        H_blocks_edge.append(offsite_ham[(edge_index[0][i].item(), edge_index[1][i].item())])
    H_blocks_edge = np.array(H_blocks_edge, dtype=object) 
    edge_labels = flatten_data(H_blocks_edge, edge_index, numbers, equivariant_blocks, out_slices)



    # find the onsite Hamiltonian blocks for all atoms that are part of the graph
    onsite_edge_index = np.array([np.array(full_atom_index),np.array(full_atom_index)])
    onsite_ham = structure.get_orbital_blocks(onsite_edge_index)
    H_blocks_node = []
    for i in range(len(onsite_edge_index[0])):
         H_blocks_node.append(onsite_ham[(onsite_edge_index[0][i].item(),onsite_edge_index[1][i].item())])
    H_blocks_node = np.array(H_blocks_node, dtype=object) 
    node_labels = flatten_data(H_blocks_node, onsite_edge_index, numbers, equivariant_blocks, out_slices)



    #create edge features, which are the interatomic distances - including periodic boundary conditions
    edge_fea = torch.empty((len(edge_index[0]),4))
    for i in range(len(edge_index[0])):
        distance_vector, distance = find_mic(pos[edge_index[1][i]] - pos[edge_index[0][i]], cell)
        edge_fea[i,:] = torch.cat((torch.tensor([distance]), torch.tensor(distance_vector)))
    edge_fea = torch.tensor(edge_fea, dtype = dtype)

    # create the node features, which are the atomic numbers of the atoms in the slice
    atomic_numbers = numbers[full_atom_index] 
    x = torch.tensor(atomic_numbers)

    #create the label data for edges and nodes 
    edge_labels_np = np.array(edge_labels)  # Convert list of numpy arrays to a single numpy ndarray
    edge_labels = torch.tensor(edge_labels_np,dtype = dtype)
    node_labels_np = np.array(node_labels)  # Convert list of numpy arrays to a single numpy ndarray
    node_labels = torch.tensor(node_labels_np, dtype = dtype)
    
    
    # convert Hamiltonian labels from uncoupled space to coupled space (to avoid conversion during training)
    y = construct_kernel.get_net_out(edge_labels) 
    node_y = construct_kernel.get_net_out(node_labels)

    atom_indices = torch.tensor(full_atom_index)
    atom_coordinates = torch.tensor(pos[atom_indices])

    if use_overlap ==True:

        overlap_matrix = structure.get_orbital_blocks(edge_index, operator = 'overlap')
        S_blocks_edge = []
        for i in range(len(edge_index[0])):
            S_blocks_edge.append(overlap_matrix[(edge_index[0][i].item(), edge_index[1][i].item())])

        S_blocks_edge = np.array(S_blocks_edge, dtype=object)
        S_labels = flatten_data(S_blocks_edge, edge_index, numbers, equivariant_blocks, out_slices)

        S_labels_np = np.array(S_labels)  # Convert list of numpy arrays to a single numpy ndarray
        S_labels = torch.tensor(S_labels_np,dtype = dtype)
        S_input = construct_kernel.get_net_out(S_labels)

    else:

        S_input = None

    # create the data object
    data = Data(x=x, 
                edge_index=full_mapped_edge_index, 
                edge_attr=edge_fea, 
                y=y, 
                node_y=node_y, 
                labelled_edge_size=slice_graph['real_edge_size'],
                labelled_node_size=slice_graph['real_node_size'], 
                node_degree=slice_graph['node_degree'], 
                reduced_node_degree=slice_graph['reduced_node_degree'], 
                atom_indices=atom_indices, 
                atom_coordinates=atom_coordinates, S_input = S_input)    

    return data

    

# used in structures/materials/a-HfO2/
def batch_data_HfO2_cartesian(graph, start0, total_length, num_slices, test_list = None, save_file = 'None', cutoff = 2, equivariant_blocks = None, out_slices = None, construct_kernel=None, dtype = torch.float32, slice_direction = 0, add_virtual = True, two_way = False, extra_data = None, use_overlap = False): 

    data_list = []

    start = start0
    length = total_length/num_slices
    num_atoms = 0
    num_edges = 0

    print(length)

    for i in range(num_slices):
        train_data = createdata_subgraph_cartesian(graph, start, length ,equivariant_blocks, out_slices, construct_kernel, dtype, slice_direction, add_virtual, two_way, use_overlap=use_overlap)
        torch.save(train_data, save_file+'_structure_'+'_training_'+str(start)+'_'+str(start+length)+'.pt')
        data_list.append(train_data)
        start = start + length
        num_atoms += train_data.labelled_node_size
        num_edges += train_data.labelled_edge_size
        print("Number of atoms:", train_data.labelled_node_size)
        print("Number of edges:", train_data.labelled_edge_size)

    print("Total Number of Atoms: ", num_atoms)
    print("Total Number of Edges: ", num_edges)

    if extra_data is not None:
        for i in range(len(extra_data)): #extra data is a list of structure objects 
            start = start0
            num_atoms = 0
            num_edges = 0

            for j in range(num_slices):
                train_data = createdata_subgraph_cartesian(extra_data[i], start, length ,equivariant_blocks, out_slices, construct_kernel, dtype, slice_direction, add_virtual, two_way, use_overlap=use_overlap)
                torch.save(train_data, save_file+'_extra_'+str(i)+'_training_'+str(start)+'_'+str(start+length)+'.pt')
                data_list.append(train_data)
                start = start + length
                num_atoms += train_data.labelled_node_size
                num_edges += train_data.labelled_edge_size
                print("Number of atoms:", train_data.labelled_node_size)
                print("Number of edges:", train_data.labelled_edge_size)
            
            print("Total Number of Atoms for Extra Structure "+str(i)+":", num_atoms)
            print("Total Number of Edges for Extra Structure "+str(i)+":", num_edges)
            
    dataset = CustomDataset(data_list)

    if dist.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        loader = DataLoader(dataset, sampler=sampler, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    else:
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

    print("*** Batch properties:")
    for batch in loader:
        print("--> Batch: ")
        print("Node Features (x):", batch.x.size())
        print("Edge Index:", batch.edge_index.size())
        print("Edge Features (edge_attr):", batch.edge_attr.size())
        print("Average Node Degree:", np.mean(np.array(batch.node_degree)))
        print("Average Reduced Node Degree", np.mean(np.array(batch.reduced_node_degree)))     

    return loader

