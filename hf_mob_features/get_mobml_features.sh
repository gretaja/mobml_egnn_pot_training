#!/bin/bash
#SBATCH --job-name=mob_features
#SBATCH --partition=compute
#SBATCH --account=chem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=1-00:00:00
#SBATCH --mem=50G
#SBATCH --out=h2o_hf_mob_features.out
#SBATCH --chdir=/gscratch/mccoy/pot_training/hf_mob_features

export UCX_TLS=tcp,self

# Which system to run -- must match a file in pipeline_config/systems/.
export PIPELINE_SYSTEM=h2o

#module load apptainer
MCENV_IMAGE=/gscratch/mccoy/group/mobml-collab/current.sif
MCENV_PACKAGES_PATH=/gscratch/mccoy/group/mobml-collab/packages

PATH=/opt/intel/oneapi/intelpython/latest/bin/python:$PATH

# Quiet the run: libfabric/OFI logging off, Python warnings suppressed.
export FI_LOG_LEVEL=none
export PYTHONWARNINGS=ignore
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so

# Don't block-buffer stdout/stderr -- without this, prints sit in memory
# and never reach the .out file until the process exits.
export PYTHONUNBUFFERED=1

# get_mobml_features.py now spreads its HF/feature-extraction work across
# all --ntasks-per-node MPI ranks (one PySCF calc per rank at a time). Pin
# each rank's internal BLAS/OpenMP threading to 1 so 16 ranks don't each
# also try to fan out across all 16 cores and oversubscribe them.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PCA_DIR=/gscratch/mccoy/pot_training/pca_and_fps
DMC_DIR=/gscratch/mccoy/pot_training/dmc_sampling
HF_DIR=/gscratch/mccoy/pot_training/hf_mob_features
CONFIG_DIR=/gscratch/mccoy/pot_training/pipeline_config

srun --mpi=pmi2 apptainer run \
  --bind "$PCA_DIR":/home/pca_and_fps:rw \
  --bind "$DMC_DIR":/home/dmc_sampling \
  --bind "$HF_DIR":/home/hf_mob_features:rw \
  --bind "$MCENV_PACKAGES_PATH":/home/packages \
  --bind "$CONFIG_DIR":/home/pipeline_config \
  $MCENV_IMAGE python -m mpi4py.futures /home/hf_mob_features/get_mobml_features.py

