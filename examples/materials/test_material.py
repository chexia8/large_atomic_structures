from __future__ import annotations

import os
import random
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import numpy as np
import torch
import torch.distributed as dist
from e3nn.o3 import Irreps

import augmented_partition.lib_equiformer.process_irreps as process_irreps
import augmented_partition.lib_equiformer.SO3 as SO3
import augmented_partition.model.data as data
import augmented_partition.model.network as network
import augmented_partition.model.structure as structure
import augmented_partition.model.training as training

if TYPE_CHECKING:
    from omegaconf import DictConfig

print("Imported libraries", flush=True)


def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from keys in state_dict."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module.") :]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


@hydra.main(config_path="../config", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # ************************************************************
    # Distributed training setup (if running on multiple GPUs)
    # ************************************************************

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device, flush=True)

    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)
        backend = "gloo"  # Use NCCL for multi-GPU on Piz Daint (edit: uses RDMA, switching to gloo)
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    else:
        rank = 0
        world_size = 1
        local_rank = 0
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        backend = "gloo"  # Use Gloo for attelas (single GPU)

    if dist.is_initialized() and dist.get_rank() == 0:
        print(f"RANK: {rank}", flush=True)
        print(f"WORLD_SIZE: {world_size}", flush=True)
        print(f"LOCAL_RANK: {local_rank}", flush=True)

    # ************************************************************
    # Input parameters and for the HfO2 dataset
    # ************************************************************

    data_folder = cfg.dataset.test_folder  # Path to the dataset folder
    xyz_file = data_folder + "structure.xyz"
    hamiltonian_file = data_folder + "H.csr"
    overlap_file = data_folder + "S.csr"

    dtype = torch.float32

    tag = cfg.dataset.tag  # Tag for the results

    results_dir = Path(f"results_{tag}")
    # results_dir.mkdir(parents=True, exist_ok=True)
    save_file = "test"
    save_file = str(results_dir / save_file)

    restart_file = "model"
    restart_file = str(results_dir / restart_file)

    # Material parameters:
    test_cell = cfg.dataset.test_cell  # Lattice vectors of the material, default = None
    pbc = cfg.model.pbc  # Periodic boundary conditions
    orbital_basis = cfg.dataset.orbital_basis
    rcut = cfg.dataset.rcut
    lmax = cfg.dataset.lmax
    mmax = cfg.dataset.mmax

    # after dividing the graph into 'num_subgraph' subgraphs
    # Parameters:
    num_MP_layers = cfg.model.num_MP_layers  # Number of message passing layers                                            # Loss tolerance for early stopping
    use_overlap = False

    # *** Initialize the hyperparameters of the SO2 model:
    sphere_channels = cfg.model.sphere_channels
    num_heads = cfg.model.num_heads
    attn_hidden_channels = cfg.model.attn_hidden_channels
    attn_alpha_channels = cfg.model.attn_alpha_channels
    attn_value_channels = cfg.model.attn_value_channels
    ffn_hidden_channels = cfg.model.ffn_hidden_channels

    # ************************************************************
    # Create the dataset
    # ************************************************************

    # *** Initialize the domain and electronic structure matrices:

    material = structure.Structure(
        xyz_file,
        hamiltonian_file,
        overlap_file,
        pbc,
        orbital_basis,
        make_soap=False,
        save_matrices=False,
        self_interaction=False,
        bothways=True,
        rcut=rcut,
        use_overlap=use_overlap,
        cell=test_cell,
    )

    print("Structure 1 created", flush=True)

    print("Validation structure created", flush=True)

    if dist.is_initialized():
        dist.barrier()

    # ************************************************************
    # Initialize the SO2 model
    # ************************************************************

    # *** Define irreducible representations
    irreps_list = []
    irreps_list.extend(
        (sphere_channels, (i, 1)) for i in range(lmax + 1)
    )  # Irreps for the input features
    irreps_in = Irreps(irreps_list)
    edge_channels_list = [sphere_channels, sphere_channels, sphere_channels]

    # *** Perform orbital analysis:
    atom_orbitals = (
        cfg.dataset.atom_orbitals
    )  # Orbital types of each atom in the structure
    numbers = material.atomic_numbers  # Atomic numbers of each atom in the structure
    no_parity = True  # No parity symmetry
    orbital_types = cfg.dataset.orbital_types  # must be in ascending order of atomic numbers                                                 # basis rank of each atom in the structure

    targets, net_out_irreps, net_out_irreps_simplified = (
        process_irreps.orbital_analysis(
            atom_orbitals, targets=None, no_parity=no_parity
        )
    )
    index_to_Z, inverse_indices = torch.unique(
        numbers, sorted=True, return_inverse=True
    )
    equivariant_blocks, out_js_list, out_slices = process_irreps.process_targets(
        orbital_types, index_to_Z, targets
    )

    construct_kernel = process_irreps.e3TensorDecomp(
        net_out_irreps,
        out_js_list,
        default_dtype_torch=dtype,
        spinful=False,
        no_parity=no_parity,
        if_sort=False,
        device_torch="cpu",
    )  # the data is created on cpu, so the construct_kernel must be on cpu
    print("Orbital analysis completed", flush=True)

    # *** Create the input dataloader:
    start = cfg.dataset.test_start  # Starting position along the structure
    total_length = cfg.dataset.test_total_length
    num_slices = (
        cfg.dataset.test_num_slices
    )  # Number of slices to take from the structure
    slice_direction = (
        cfg.dataset.test_slice_direction
    )  # Direction of the slices, e.g., 'x', 'y', 'z'

    data_loader = data.batch_data_material_cartesian(
        material,
        start,
        total_length,
        num_slices,
        save_file,
        equivariant_blocks=equivariant_blocks,
        out_slices=out_slices,
        construct_kernel=construct_kernel,
        dtype=dtype,
        slice_direction=slice_direction,
        use_overlap=use_overlap,
    )
    print("Data loader created")

    if dist.is_initialized():
        dist.barrier()

    # *** Initialize the model:
    mappingReduced = SO3.CoefficientMappingModule(lmax, mmax)
    irreps_out = net_out_irreps
    model = network.SO2Net(
        num_MP_layers,
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
        use_overlap=use_overlap,
    )
    model = model.to(device)

    if restart_file is not None:
        state_dict = torch.load(restart_file + "_state_dic.pt", map_location=device)
        state_dict = remove_module_prefix(state_dict)
        model.load_state_dict(state_dict)

    print("Model initialized", flush=True)
    print(
        "Number of parameters: ", sum(p.numel() for p in model.parameters()), flush=True
    )

    print(
        "memory: " + str(torch.cuda.memory_allocated(device) / 1e9) + "GB", flush=True
    )
    if dist.is_initialized():
        dist.barrier()

    # ************************************************************
    # Training and testing the model
    # ************************************************************

    for test_batch in data_loader:
        training.evaluate_model(
            model,
            test_batch,
            construct_kernel,
            equivariant_blocks,
            atom_orbitals,
            out_slices,
            device,
            save_file=save_file,
            reconstruct_ham=True,
            compute_total_loss=True,
            plot=True,
        )


if __name__ == "__main__":
    main()
