#!/bin/bash
#SBATCH --job-name=parse_ccsdt_energies
#SBATCH --account=chem
#SBATCH --partition=compute
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8G
#SBATCH --out=parse_ccsdt_energies.out
#SBATCH --chdir=/gscratch/mccoy/pot_training/ccsdt_energies

# Which system to parse energies for -- must match a file in
# pipeline_config/systems/. parse_ccsdt_energies.py reads this via
# system_config.py, same as every other script in this project.
export PIPELINE_SYSTEM=h2o

python3 parse_ccsdt_energies.py
