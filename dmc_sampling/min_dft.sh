#!/bin/bash
#SBATCH --job-name=min_dft
#SBATCH --partition=compute
#SBATCH --account=stf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=7-00:00:00
#SBATCH --mem=150G
#SBATCH --out=min_dft.out
#SBATCH --chdir=/gscratch/mccoy/pot_training/dmc_sampling

export UCX_TLS=tcp,self

# Which system to run -- must match a file in pipeline_config/systems/.
export PIPELINE_SYSTEM=h3o2

#module load apptainer
MCENV_IMAGE=/gscratch/mccoy/group/mobml-collab/current.sif
MCENV_PACKAGES_PATH=/gscratch/mccoy/group/mobml-collab/packages

PATH=/opt/intel/oneapi/intelpython/latest/bin/python:$PATH

# Quiet the run: libfabric/OFI logging off, Python warnings suppressed.
export FI_LOG_LEVEL=warn
export PYTHONWARNINGS=ignore
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so

# Don't block-buffer stdout/stderr -- without this, prints sit in memory
# and never reach the .out file until the process exits. If the job gets
# killed rather than exiting cleanly (which is exactly what "runs a few
# seconds then aborts with nothing in the output file" looks like),
# whatever it already printed -- including the actual error -- is lost
# entirely rather than just delayed.
export PYTHONUNBUFFERED=1

DMC_DIR=/gscratch/mccoy/pot_training/dmc_sampling
GEOM_DIR=/gscratch/mccoy/pot_training/opt_geoms
CONFIG_DIR=/gscratch/mccoy/pot_training/pipeline_config

srun --mpi=pmi2 apptainer run \
  --bind "$DMC_DIR/scripts":/home/scripts \
  --bind "$MCENV_PACKAGES_PATH":/home/packages \
  --bind "$DMC_DIR/output":/home/output:rw \
  --bind "$GEOM_DIR":/home/opt_geoms \
  --bind "$CONFIG_DIR":/home/pipeline_config \
  $MCENV_IMAGE python -m mpi4py.futures /home/scripts/min_dft.py

