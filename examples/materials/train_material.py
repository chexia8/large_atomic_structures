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

    data_folder = cfg.dataset.train_folder  # Path to the dataset folder
    xyz_file = data_folder + "structure.xyz"
    hamiltonian_file = data_folder + "H.csr"
    overlap_file = data_folder + "S.csr"

    val_data_folder = cfg.dataset.val_folder
    val_xyz_file = val_data_folder + "structure.xyz"
    val_hamiltonian_file = val_data_folder + "H.csr"
    val_overlap_file = val_data_folder + "S.csr"

    dtype = torch.float32

    tag = cfg.dataset.tag  # Tag for the results

    results_dir = Path(f"results_{tag}")
    results_dir.mkdir(parents=True, exist_ok=True)
    save_file = "model"
    save_file = str(results_dir / save_file)

    restart_file = None
    # restart_file = str(results_dir / restart_file)

    # Material parameters:
    train_cell = (
        cfg.dataset.train_cell
    )  # Lattice vectors of the material, default = None
    val_cell = (
        cfg.dataset.val_cell
    )  # Lattice vectors of the validation material, default = None
    pbc = cfg.model.pbc  # Periodic boundary conditions
    orbital_basis = cfg.dataset.orbital_basis
    rcut = cfg.dataset.rcut
    lmax = cfg.dataset.lmax
    mmax = cfg.dataset.mmax

    # Parameters:
    num_MP_layers = cfg.model.num_MP_layers  # Number of message passing layers                                            # Loss tolerance for early stopping
    num_epochs = cfg.model.num_epochs  # Number of epochs for training
    lr = cfg.model.lr  # Learning rate for the optimizer
    patience = cfg.model.patience  # Patience for early stopping
    decay = cfg.model.decay  # Learning rate decay factor
    threshold = cfg.model.threshold  # Threshold for early stopping
    min_lr = cfg.model.min_lr  # Minimum learning rate for the scheduler
    use_overlap = False
    loss_tol = 1e-10  # Loss tolerance for early stopping

    # *** Initialize the hyperparameters of the SO2 model:
    sphere_channels = cfg.model.sphere_channels
    num_heads = cfg.model.num_heads
    attn_hidden_channels = cfg.model.attn_hidden_channels
    attn_alpha_channels = cfg.model.attn_alpha_channels
    attn_value_channels = cfg.model.attn_value_channels
    ffn_hidden_channels = cfg.model.ffn_hidden_channels

    criterion = cfg.model.criterion  # Loss function to use for training

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
        cell=train_cell,
        use_overlap=use_overlap,
    )

    print("Structure 1 created", flush=True)

    material_val = structure.Structure(
        val_xyz_file,
        val_hamiltonian_file,
        val_overlap_file,
        pbc,
        orbital_basis,
        make_soap=False,
        save_matrices=False,
        self_interaction=False,
        bothways=True,
        rcut=rcut,
        cell=val_cell,
        use_overlap=use_overlap,
    )

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
    start = cfg.dataset.train_start  # Starting position along the structure
    total_length = cfg.dataset.train_total_length
    num_slices = (
        cfg.dataset.train_num_slices
    )  # Number of slices to take from the structure
    slice_direction = (
        cfg.dataset.train_slice_direction
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

    start = cfg.dataset.val_start  # Starting position along the structure
    total_length = cfg.dataset.val_total_length
    num_slices = 1  # fixed to 1 slice for validation
    slice_direction = (
        cfg.dataset.val_slice_direction
    )  # Direction of the slices, e.g., 'x', 'y', 'z'
    assert num_slices == 1, "num_slices must be 1 for validation"

    validation_loader = data.batch_data_material_cartesian(
        material_val,
        start,
        total_length,
        num_slices,
        equivariant_blocks=equivariant_blocks,
        out_slices=out_slices,
        construct_kernel=construct_kernel,
        dtype=dtype,
        slice_direction=slice_direction,
        use_overlap=use_overlap,
    )
    print("Validation loader created")

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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if restart_file is not None:
        print(
            "Restarting training from a saved model and optimizer state...", flush=True
        )
        checkpoint = torch.load(restart_file)
        state_dict = checkpoint["model_state_dict"]
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if dist.is_available() and dist.is_initialized():
            # If the model was saved with DDP, remove the 'module' prefix that it might have (just in case)
            if "module." in next(iter(checkpoint["model_state_dict"].keys())):
                prefix = "module."
                state_dict = {
                    k[len(prefix) :] if k.startswith(prefix) else k: v
                    for k, v in state_dict.items()
                }
            # with the current training setup, the module prefix is already removed
            model.load_state_dict(state_dict)
        else:
            state_dict = remove_module_prefix(checkpoint["model_state_dict"])
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

    # *** Train the model parameters:
    print("validation loader created", flush=True)
    print("training...", flush=True)

    training.train_and_validate_model_subgraph(
        model,
        optimizer,
        data_loader,
        validation_loader,
        num_epochs,
        loss_tol,
        patience,
        decay,
        threshold,
        min_lr=min_lr,
        save_file=save_file,
        schedule=True,
        criterion=criterion,
    )
    print("Model trained, plotting fit to training data", flush=True)
    training.evaluate_slice(
        model,
        data_loader,
        construct_kernel,
        equivariant_blocks,
        atom_orbitals,
        out_slices,
        device,
    )

    print("testing on validation data...", flush=True)
    training.evaluate_slice(
        model,
        validation_loader,
        construct_kernel,
        equivariant_blocks,
        atom_orbitals,
        out_slices,
        device,
    )


if __name__ == "__main__":
    main()
