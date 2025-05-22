import argparse
import numpy as np
import torch.distributed as dist
import torch
import random
import augmented_partition.model as model
import augmented_partition.lib_equiformer as lib

from torch_geometric.data import Data
from e3nn.o3 import Irreps

from augmented_partition.model.utils import rotate_vector, compute_rotation, create_rotation_matrix, rotate_irrep_vector
from augmented_partition.model.network import SO2Net, SO2Net_local
from augmented_partition.lib_equiformer.SO3 import CoefficientMappingModule


def create_toy_data(input_vector):

    x = torch.tensor([1,1])
    edge_distance_vec = torch.tensor([[0.0,  0.0, 0.0],input_vector], dtype=torch.float32)  # shape = (2,3) = [2, 3]
    edge_index = torch.tensor([[0,1],[1,0]])
    edge_fea = torch.empty((len(edge_index[0]),4))

    for i in range(len(edge_index[0])):
        distance_vector = edge_distance_vec[edge_index[1][i]] - edge_distance_vec[edge_index[0][i]]  # vector from node 0 to node 1
        distance = torch.norm(distance_vector)  # distance between node 0 and node 1
        edge_fea[i,:] = torch.cat((torch.tensor([distance]), torch.tensor(distance_vector)))

    data = Data(x=x, edge_index=edge_index.t(), edge_attr=edge_fea, y = torch.tensor([[0.0]]), node_y = torch.tensor([[0.0]]))

    return data

def test_equivariance():

    # initial_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    # final_vector = np.array([-0.5774,  0.7887, -0.2113])


    random_vector = np.random.rand(3)
    initial_vector = random_vector / np.linalg.norm(random_vector)  

    random_vector = np.random.rand(3)
    final_vector = random_vector / np.linalg.norm(random_vector)

    R = create_rotation_matrix(initial_vector, final_vector) #create rotation matrix in cartesian space

    data = create_toy_data(initial_vector)
    rotated_data = create_toy_data(final_vector)

    # define the dummy network parameters 
    sphere_channels = 8
    num_heads = 2
    attn_hidden_channels = 8 
    attn_alpha_channels = 8
    attn_value_channels = 8
    ffn_hidden_channels = 8
    lmax = 2 
    mmax = 2
    num_MP_layers = 2

    irreps_in = Irreps([(sphere_channels, (0, 1)), (sphere_channels, (1, 1)), (sphere_channels, (2, 1))])
    edge_channels_list = [sphere_channels, sphere_channels, sphere_channels] 
    irreps_out = Irreps("1x0e+1x1e+1x2e")


    mappingReduced = CoefficientMappingModule(lmax, mmax)

    model = SO2Net(num_MP_layers, 
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
                                irreps_out,
                                )



    node, edge = model(data)
    rotated_node, rotated_edge = model(rotated_data)

    D_rotated_edge = rotate_irrep_vector(R, edge[0], irreps_out)
    D_rotated_node = rotate_irrep_vector(R, node[0], irreps_out)

    # node_error = rotated_node[0] - D_rotated_node
    node_error = rotated_node[0] - D_rotated_node
    edge_error = rotated_edge[0] - D_rotated_edge

    print("Node error: ", node_error)
    print("Edge error: ", edge_error)

    assert torch.allclose(node_error, torch.zeros_like(node_error), atol=1e-5), "Node equivariance test failed!"
    assert torch.allclose(edge_error, torch.zeros_like(edge_error), atol=1e-5), "Edge equivariance test failed!"


    # assert divide(3, 2) == pytest.approx(1.5)

    # with pytest.raises(ValueError, match="Uh oh! The value for b should not be 0"):
    #     divide(10, 0)