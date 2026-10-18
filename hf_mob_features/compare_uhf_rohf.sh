#!/bin/bash
#SBATCH --job-name=compare_uhf_rohf
#SBATCH --partition=compute
#SBATCH --account=chem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-00:30:00
#SBATCH --mem=16G
#SBATCH --out=compare_uhf_rohf.out
#SBATCH --chdir=/gscratch/mccoy/pot_training/hf_mob_features

export UCX_TLS=tcp,self

# Which system to compare -- must match a file in pipeline_config/systems/.
export PIPELINE_SYSTEM=c2h6-2p

#module load apptainer
MCENV_IMAGE=/gscratch/mccoy/group/mobml-collab/current.sif
MCENV_PACKAGES_PATH=/gscratch/mccoy/group/mobml-collab/packages

PATH=/opt/intel/oneapi/intelpython/latest/bin/python:$PATH

export FI_LOG_LEVEL=none
export PYTHONWARNINGS=ignore
export PYTHONUNBUFFERED=1
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so

HF_DIR=/gscratch/mccoy/pot_training/hf_mob_features
PCA_DIR=/gscratch/mccoy/pot_training/pca_and_fps
CONFIG_DIR=/gscratch/mccoy/pot_training/pipeline_config

# Single serial process, ~10 single-point ROHF+UHF calcs -- no MPI pool
# needed, but srun --mpi=pmi2 is still required for this container even
# for a single rank (see verify_alignment.sh).
srun --mpi=pmi2 apptainer run \
  --bind "$HF_DIR":/home/hf_mob_features \
  --bind "$PCA_DIR":/home/pca_and_fps \
  --bind "$MCENV_PACKAGES_PATH":/home/packages \
  --bind "$CONFIG_DIR":/home/pipeline_config \
  $MCENV_IMAGE python /home/hf_mob_features/compare_uhf_rohf.py
