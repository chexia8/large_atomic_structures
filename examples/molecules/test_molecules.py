from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import numpy as np
import torch
from e3nn.o3 import Irreps
from schnetpack.data import ASEAtomsData

import augmented_partition.lib_equiformer.process_irreps as process_irreps
import augmented_partition.lib_equiformer.SO3 as SO3
import augmented_partition.model.data as data
import augmented_partition.model.network as network
import augmented_partition.model.structure as structure
import augmented_partition.model.training as training
import augmented_partition.model.utils as utils

print("Imported libraries", flush=True)

# SchNetPack package for database handling


if TYPE_CHECKING:
    from omegaconf import DictConfig

# Adding units to the dataset
# spkconvert --distunit Angstrom --propunit energy:Hartree,forces:Hartree/Angstrom,hamiltonian:Hartree,overlap:dimensionless /Users/manasakani/Documents/ETH/Repos/ham_predict/datasets/schnorb_hamiltonian_water.db
# Supplementary materials of QHNet: https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-019-12875-2/MediaObjects/41467_2019_12875_MOESM1_ESM.pdf


@hydra.main(config_path="../config", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    print("Configuration:\n", cfg)
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    # ************************************************************
    # Input parameters and for the H2O molecule dataset
    # ************************************************************

    db_path = cfg.dataset.db_path
    database = ASEAtomsData(db_path)
    print("Number of Molecules in the database: ", len(database))
    norbs = utils.get_number_orbitals_QM7(database)
    print("Number of orbitals: ", norbs)

    # Dataset parameters:
    num_train = cfg.dataset.num_train  # Number of training samples
    num_validate = cfg.dataset.num_validate  # Number of validation samples
    num_test = cfg.dataset.num_train

    tag = cfg.dataset.tag  # Tag for the results
    results_dir = Path(f"results_{tag}")
    restart_file = "model"
    restart_file = str(results_dir / restart_file)

    save_file = "test"  # File to save the model
    save_file = str(results_dir / save_file)  # File to save the model

    batch_size = cfg.dataset.test_batch_size  # Batch size for training
    rcut = cfg.model.rcut  # Cutoff radius for the message passing layers

    # Structure and Network parameters:
    pbc = cfg.model.pbc  # Periodic boundary conditions
    bothways = True
    orbital_basis = "def2_SVP"
    num_MP_layers = cfg.model.num_MP_layers  # Number of message passing layers
    dtype = torch.float64  # Use double precision floating point for benchmarking
    torch.set_default_dtype(dtype)
    lmax = cfg.dataset.lmax  # Maximum angular momentum for the spherical harmonics
    mmax = (
        cfg.dataset.mmax
    )  # Maximum magnetic quantum number for the spherical harmonics

    # criterion = cfg.model.criterion  # Loss function for the training

    # Hyperparameters of the SO2 model for H2O
    sphere_channels = cfg.model.sphere_channels
    num_heads = cfg.model.num_heads
    attn_hidden_channels = cfg.model.attn_hidden_channels
    attn_alpha_channels = cfg.model.attn_alpha_channels
    attn_value_channels = cfg.model.attn_value_channels
    ffn_hidden_channels = cfg.model.ffn_hidden_channels

    irreps_list = []
    irreps_list.extend(
        (sphere_channels, (i, 1)) for i in range(lmax + 1)
    )  # Irreps for the input features

    irreps_in = Irreps(irreps_list)  # Create Irreps object for input features

    print("Irreps in: ", irreps_in)
    # ************************************************************
    # Create the dataset
    # ************************************************************

    offset = 0
    training_data_indices, validation_data_indices, testing_data_indices = (
        data.split_data_indices(
            num_train, num_validate, num_test, len(database), offset
        )
    )
    num_test = num_test - 1

    # *** Prepare the dataset:
    sample_molecule = None
    testing_molecules = []

    for i in range(num_test):
        molecule_index = int(testing_data_indices[i])
        testing_molecules.append(
            structure.Structure(
                None,
                None,
                None,
                pbc,
                orbital_basis,
                dataset="schnet",
                database_props=database.__getitem__(molecule_index),
                self_interaction=False,
                bothways=bothways,
                rcut=rcut,
            )
        )
        # training_molecules.append(testing_molecules)
    sample_molecule = testing_molecules[0]

    print("Dataset initialized")

    # ************************************************************
    # Initialize the SO2 model
    # ************************************************************

    # Define irreducible representations for the SO2 model
    edge_channels_list = [sphere_channels, sphere_channels, sphere_channels]

    atom_orbitals = (
        cfg.dataset.atom_orbitals
    )  # Orbital types of each atom in the structure
    numbers = torch.tensor(
        [utils.periodic_table[i] for i in sample_molecule.atomic_species]
    )  # Atomic numbers of each atom in the structure
    no_parity = True  # No parity symmetry
    orbital_types = cfg.dataset.orbital_types

    targets, net_out_irreps, _ = process_irreps.orbital_analysis(
        atom_orbitals, targets=None, no_parity=no_parity
    )
    index_to_Z, _ = torch.unique(numbers, sorted=True, return_inverse=True)
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
        device_torch=device,
    )

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
    )
    model = model.to(device)

    if restart_file is not None:
        state_dict = torch.load(restart_file + "_state_dic.pt", map_location=device)
        model.load_state_dict(state_dict)

    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
    print("Model initialized")

    # ************************************************************
    # Run the training process
    # ************************************************************

    testing_data_loader = data.batch_data_molecules(
        testing_molecules,
        device,
        num_test,
        batch_size,
        equivariant_blocks,
        out_slices,
        construct_kernel,
        dtype,
    )
    print("training model...")
    # create new construct_kernel for the training, this time on the cpu
    construct_kernel = process_irreps.e3TensorDecomp(
        net_out_irreps,
        out_js_list,
        default_dtype_torch=dtype,
        spinful=False,
        no_parity=no_parity,
        if_sort=False,
        device_torch="cpu",
    )

    training.evaluate_batch(
        model,
        testing_data_loader,
        construct_kernel,
        equivariant_blocks,
        atom_orbitals,
        out_slices,
        device,
    )


if __name__ == "__main__":
    main()
