#!/bin/bash
#SBATCH --job-name=correlate_t1_diagnostic_error
#SBATCH --account=chem
#SBATCH --partition=compute
#SBATCH --time=0-00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4G
#SBATCH --out=correlate_t1_diagnostic_error.out
#SBATCH --chdir=/gscratch/mccoy/pot_training/mobml_fitting

# Which system to check -- must match a file in pipeline_config/systems/.
export PIPELINE_SYSTEM=c2h6-2p

# Bare-metal, same as parse_ccsdt_energies.py -- pure text parsing +
# numpy, no apptainer/PySCF needed.
python3 correlate_t1_diagnostic_error.py
