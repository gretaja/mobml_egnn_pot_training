#!/bin/bash
#SBATCH --job-name=h2o_geom
#SBATCH --partition=compute
#SBATCH --account=chem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=0-01:00:00
#SBATCH --mem=5G
#SBATCH --out=h2o_xyz_to_npy.out
#SBATCH --chdir=/gscratch/mccoy/pot_training/opt_geoms

# Which system to run -- must match a file in pipeline_config/systems/.
export PIPELINE_SYSTEM=h2o

python -u xyz_to_npy.py
