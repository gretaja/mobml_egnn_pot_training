#!/bin/bash
#SBATCH --job-name=verify_xyz_alignment
#SBATCH --account=chem
#SBATCH --partition=compute
#SBATCH --time=0-00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4G
#SBATCH --out=verify_xyz_alignment.out
#SBATCH --chdir=/gscratch/mccoy/pot_training/pca_and_fps

# Which system to check -- must match a file in pipeline_config/systems/.
# verify_xyz_alignment.py reads this via system_config.py, same as every
# other script in this project.
export PIPELINE_SYSTEM=c2h6-2p

# Bare-metal, same as parse_ccsdt_energies.py -- pure numpy + file parsing,
# no apptainer/PySCF needed.
python3 verify_xyz_alignment.py
