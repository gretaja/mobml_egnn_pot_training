#!/bin/bash
#SBATCH --job-name=mobml_fitting
#SBATCH --partition=compute
#SBATCH --account=chem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --mem=50G
#SBATCH --out=h2o_mobml_fitting.out
#SBATCH --chdir=/gscratch/mccoy/pot_training/mobml_fitting

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

# Unlike get_mobml_features.py, this script fits one GP in a single
# process -- no MPI pool, nothing to spread across ranks. Let its BLAS/MKL
# calls (the GP's Cholesky etc.) use all --cpus-per-task cores instead of
# pinning to 1, which would only be right for the many-independent-
# processes case get_mobml_features.py runs.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

PCA_DIR=/gscratch/mccoy/pot_training/pca_and_fps
DMC_DIR=/gscratch/mccoy/pot_training/dmc_sampling
HF_DIR=/gscratch/mccoy/pot_training/hf_mob_features
MOBML_DIR=/gscratch/mccoy/pot_training/mobml_fitting
CCSDT_DIR=/gscratch/mccoy/pot_training/ccsdt_energies
CONFIG_DIR=/gscratch/mccoy/pot_training/pipeline_config

srun --mpi=pmi2 apptainer run \
  --bind "$PCA_DIR":/home/pca_and_fps:rw \
  --bind "$DMC_DIR":/home/dmc_sampling \
  --bind "$HF_DIR":/home/hf_mob_features:rw \
  --bind "$MOBML_DIR":/home/mobml_fitting:rw \
  --bind "$CCSDT_DIR":/home/ccsdt_energies \
  --bind "$MCENV_PACKAGES_PATH":/home/packages \
  --bind "$CONFIG_DIR":/home/pipeline_config \
  $MCENV_IMAGE python /home/mobml_fitting/fit_mobml.py

