# Learning the Electronic Hamiltonian of Large Atomic Structures

Official code implementation of "Learning the Electronic Hamiltonian of Large Atomic Structures" (ICML 2025)

## Installation

First step is to clone the repository and install the package. 
```bash
git clone https://github.com/yourusername/yourproject.git
cd large_atomic_structures
pip install -e .

```
The environment can be created through the following command:
```bash
conda env create -f augment_partition.yml
```
Next step is to clone the dataset into a separate folder and download the files
```bash
git clone https://huggingface.co/datasets/chexia8/Amorphous-Hamiltonians
cd Amorphous-Hamiltonians
git lfs pull
```
The .db files for small molecule datasets (water, uracil and malondialdehyde) from MD17 can be downloaded from  http://www.quantum-machine.org/datasets, under the section "Molecular Hamiltonians and overlap matrices"

## Usage

There are two sets of training and testing files, one for molecular examples and one for the material examples (with augmented partitioning) 
Add the dataset path to the dataset config files and perform training through the following command, specifying which model and dataset is being trained:

```bash
#e.g. for uracil 
python train_molecules.py dataset=uracil model=molecule

#e.g. for a-HfO2
python train_material.py dataset=a-HfO2 model=material

```

For testing: 

```bash
#e.g. for uracil 
python test_molecules.py dataset=uracil model=molecule

#e.g. for a-HfO2
python test_material.py dataset=a-HfO2 model=material

```

This will print the node, edge and total Mean Absolute Error (MAE) of the prediction, and also reconstruct the Hamiltonian if needed. Note that for large structures, multiple cores can be used through the command:

```bash
mpiexec -n 4 python test_material.py model=material dataset=a-HfO2
```

