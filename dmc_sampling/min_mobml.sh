#!/bin/bash
#SBATCH --job-name=min_mobml
#SBATCH --partition=compute
#SBATCH --account=chem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --time=1-00:00:00
#SBATCH --mem=50G
#SBATCH --out=min_mobml.out
#SBATCH --chdir=/gscratch/mccoy/pot_training/dmc_sampling

export UCX_TLS=tcp,self

# Which system to run -- must match a file in pipeline_config/systems/.
export PIPELINE_SYSTEM=h2o

#module load apptainer
MCENV_IMAGE=/gscratch/mccoy/group/mobml-collab/current.sif
MCENV_PACKAGES_PATH=/gscratch/mccoy/group/mobml-collab/packages

PATH=/opt/intel/oneapi/intelpython/latest/bin/python:$PATH

# Quiet the run: libfabric/OFI logging off, Python warnings suppressed.
export FI_LOG_LEVEL=warn
export PYTHONWARNINGS=ignore
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so

DMC_DIR=/gscratch/mccoy/pot_training/dmc_sampling
GEOM_DIR=/gscratch/mccoy/pot_training/opt_geoms
CONFIG_DIR=/gscratch/mccoy/pot_training/pipeline_config

srun --mpi=pmi2 apptainer run \
  --bind "$DMC_DIR/scripts":/home/scripts \
  --bind "$MCENV_PACKAGES_PATH":/home/packages \
  --bind "$DMC_DIR/output":/home/output:rw \
  --bind "$GEOM_DIR":/home/opt_geoms \
  --bind "$CONFIG_DIR":/home/pipeline_config \
  $MCENV_IMAGE python -m mpi4py.futures /home/scripts/min_mobml.py

