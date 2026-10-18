#!/bin/sh
#SBATCH --job-name=train_egnn
#SBATCH --time=1-00:00:00
#SBATCH --mem=50GB
#SBATCH --account=mccoy
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=6
#SBATCH --out=h2o_egnn_64_3_3_1e6decay.out
#SBATCH --chdir=.

# Which system to run -- must match a file in pipeline_config/systems/.
export PIPELINE_SYSTEM=h2o

export CUDA_HOME=/usr/local/cuda-12.7
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0  # Or any other GPU ID

python -u train_egnn.py
