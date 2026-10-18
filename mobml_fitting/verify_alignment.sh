#!/bin/bash
#SBATCH --job-name=verify_alignment
#SBATCH --partition=compute
#SBATCH --account=chem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-00:30:00
#SBATCH --mem=16G
#SBATCH --out=verify_alignment.out
#SBATCH --chdir=/gscratch/mccoy/pot_training/mobml_fitting

export UCX_TLS=tcp,self

# Which system to check -- must match a file in pipeline_config/systems/.
export PIPELINE_SYSTEM=h2o

#module load apptainer
MCENV_IMAGE=/gscratch/mccoy/group/mobml-collab/current.sif
MCENV_PACKAGES_PATH=/gscratch/mccoy/group/mobml-collab/packages

PATH=/opt/intel/oneapi/intelpython/latest/bin/python:$PATH

export FI_LOG_LEVEL=none
export PYTHONWARNINGS=ignore
export PYTHONUNBUFFERED=1
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so

MOBML_DIR=/gscratch/mccoy/pot_training/mobml_fitting
PCA_DIR=/gscratch/mccoy/pot_training/pca_and_fps
HF_DIR=/gscratch/mccoy/pot_training/hf_mob_features
CCSDT_DIR=/gscratch/mccoy/pot_training/ccsdt_energies
CONFIG_DIR=/gscratch/mccoy/pot_training/pipeline_config

srun --mpi=pmi2 apptainer run \
  --bind "$MOBML_DIR":/home/mobml_fitting \
  --bind "$PCA_DIR":/home/pca_and_fps \
  --bind "$HF_DIR":/home/hf_mob_features \
  --bind "$CCSDT_DIR":/home/ccsdt_energies \
  --bind "$MCENV_PACKAGES_PATH":/home/packages \
  --bind "$CONFIG_DIR":/home/pipeline_config \
  $MCENV_IMAGE python /home/mobml_fitting/verify_alignment.py
