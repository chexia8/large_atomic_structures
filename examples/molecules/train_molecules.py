# import sys

# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
# lib_root = os.path.join(project_root, 'model')
# lib_equiformer_root = os.path.join(project_root, 'lib_equiformer')
# sys.path.append(lib_root)
# sys.path.append(lib_equiformer_root)
# print(f"Added {lib_root} to the path", flush=True)
# print(f"Added {lib_equiformer_root} to the path", flush=True)
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
import augmented_partition.model.compute_env as env
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

    # db_path = folder+'/datasets/schnorb_hamiltonian_malondialdehyde.db'
    # db_path = '/usr/scratch/attelas8/manasa/ICLR_2025/amorphous_gnns/datasets/schnorb_hamiltonian_water.db'
    db_path = cfg.dataset.db_path
    database = ASEAtomsData(db_path)
    print("Number of Molecules in the database: ", len(database))
    norbs = utils.get_number_orbitals_QM7(database)
    print("Number of orbitals: ", norbs)

    # Dataset parameters:
    num_train = cfg.dataset.num_train  # Number of training samples
    num_validate = cfg.dataset.num_validate  # Number of validation samples
    num_test = cfg.dataset.num_train
    show_fit_for = "val"  # Show fit for the training (train) or validation (val) data

    tag = cfg.dataset.tag  # Tag for the results
    restart_file = None
    save_file = "model"

    # if not os.path.exists('results_' + tag):
    #     os.makedirs('results_' + tag)
    # save_file = 'results_' + tag + '/' + save_file

    results_dir = Path(f"results_{tag}")
    results_dir.mkdir(parents=True, exist_ok=True)

    save_file = str(results_dir / save_file)

    num_epochs = cfg.model.num_epochs  # Number of epochs for training
    batch_size = cfg.dataset.batch_size  # Batch size for training
    loss_tol = 1e-10
    lr = cfg.model.lr  # Learning rate for the optimizer
    patience = cfg.model.patience  # Patience for early stopping
    threshold = cfg.model.threshold  # Threshold for early stopping
    rcut = cfg.model.rcut  # Cutoff radius for the message passing layers

    # Structure and Network parameters:
    pbc = cfg.model.pbc  # Periodic boundary conditions
    bothways = True
    orbital_basis = "def2_SVP"
    num_MP_layers = cfg.model.num_MP_layers  # Number of message passing layers
    dtype = torch.float64  # Use double precision floating point for benchmarking
    torch.set_default_dtype(dtype)
    lmax = 4
    mmax = 4

    criterion = cfg.model.criterion  # Loss function for the training

    # Hyperparameters of the SO2 model for H2O
    sphere_channels = cfg.model.sphere_channels
    num_heads = cfg.model.num_heads
    attn_hidden_channels = cfg.model.attn_hidden_channels
    attn_alpha_channels = cfg.model.attn_alpha_channels
    attn_value_channels = cfg.model.attn_value_channels
    ffn_hidden_channels = cfg.model.ffn_hidden_channels

    # irreps_in = Irreps([(sphere_channels, (0, 1)),
    #                     (sphere_channels, (1, 1)),
    #                     (sphere_channels, (2, 1)),
    #                     (sphere_channels, (3, 1)),
    #                     (sphere_channels, (4, 1))])

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
    training_molecules = []
    validation_molecules = []
    for i in range(num_train):
        molecule_index = int(training_data_indices[i])
        training_molecules.append(
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

    for i in range(num_validate):
        molecule_index = int(validation_data_indices[i])
        validation_molecules.append(
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

    sample_molecule = training_molecules[0]

    print("Dataset initialized")

    # ************************************************************
    # Initialize the SO2 model
    # ************************************************************

    # Define irreducible representations for the SO2 model
    edge_channels_list = [sphere_channels, sphere_channels, sphere_channels]

    # *** Preform orbital analysis:
    # atom_orbitals = {'1': [0, 0, 1],'8':[0, 0, 0, 1, 1, 2]}                                                 # Orbital types of each atom in the structure
    # numbers = torch.tensor([utils.periodic_table[i] for i in sample_molecule.atomic_species])               # Atomic numbers of each atom in the structure
    # no_parity = True                                                                                        # No parity symmetry
    # orbital_types = [[0,0,1],[0, 0, 0, 1, 1, 2]]

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
    # equivariant_blocks: start and end indices of the equivariant blocks in i and j direction for each target in targets
    # out_js_list: ll the l1 l2 interactions needed
    # out_slices: marks the start and end of indices belonging to a certain target. Slice 1 (0 to 1) corresponds to the first target in equivariant blocks

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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if restart_file is not None:
        model, optimizer = env.dist_restart(
            "results_" + tag + "/" + restart_file + ".pt", model, optimizer
        )

    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
    print("Model initialized")

    # ************************************************************
    # Run the training process
    # ************************************************************

    training_data_loader = data.batch_data_molecules(
        training_molecules,
        device,
        num_train,
        batch_size,
        equivariant_blocks,
        out_slices,
        construct_kernel,
        dtype,
    )
    validation_data_loader = data.batch_data_molecules(
        validation_molecules,
        device,
        num_validate,
        batch_size,
        equivariant_blocks,
        out_slices,
        construct_kernel,
        dtype,
    )

    print("training model...")
    training.train_and_validate_model_subgraph(
        model,
        optimizer,
        training_data_loader,
        validation_data_loader,
        num_epochs,
        loss_tol,
        patience,
        threshold,
        min_lr=1e-10,
        save_file=save_file,
        unflatten=False,
        construct_kernel=construct_kernel,
        equivariant_blocks=equivariant_blocks,
        atom_orbitals=atom_orbitals,
        out_slices=out_slices,
        criterion=criterion,
    )
    print("Model trained")

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

    if show_fit_for == "train":
        training.evaluate_model(
            model,
            training_data_loader,
            construct_kernel,
            equivariant_blocks,
            atom_orbitals,
            out_slices,
            device,
            save_file=save_file,
        )
    else:
        training.evaluate_slice(
            model,
            validation_data_loader,
            construct_kernel,
            equivariant_blocks,
            atom_orbitals,
            out_slices,
            device,
        )


# if __name__ == "__main__":
#     # parser = argparse.ArgumentParser(description="Amorphous GNNs --- HfO2")
#     # parser.add_argument("-f", "--folder", default="", required=False)
#     # args = parser.parse_args()

#     # print(f"Starting main ... dataset folder is '{args.folder}'", flush=True)

#     main(cfg:DictConfig)
if __name__ == "__main__":
    main()
