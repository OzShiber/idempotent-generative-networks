#!/bin/bash
#
# SBATCH Submission Script: PyTorch Training Job
#

# --- Job Configuration ---
#SBATCH --job-name=ign_model_train        # Updated descriptive name
#SBATCH --output=slurm-%j.out         # Redirects standard output
#SBATCH --error=slurm-%j.err          # Redirects standard error
#SBATCH --nodes=1                     # Confines the job to a single node
#SBATCH --ntasks=1                    # Runs the entire job as a single task

# --- The Crucial Upgrades ---
#SBATCH --mem=32G                     
#SBATCH --time=24:00:00               # Maxed out to the 24h QoS limit so it doesn't die early

# --- Resource Allocation ---
#SBATCH --partition=l40s-public       # Submits the job to the dedicated L40S GPU partition
#SBATCH --qos=24h_4g                  # Quality of Service tier
#SBATCH --gres=gpu:nvidia_l40s:1      # Requests 1 L40S GPU
#SBATCH --cpus-per-gpu=12             # Allocates 12 CPU cores per requested GPU

# --- Container & Mounts ---
#SBATCH --container=/usr/local/apptainer/images/pytorch/latest/torch2.7-tf2.20-cu128.sif # The PyTorch container

# --- Execution Block ---
set -e # Exit immediately if a command exits with a non-zero status

echo "================================================================"
echo "Timestamp            : $(date)"
echo "Host                 : $(hostname)"
echo "SLURM Job ID         : $SLURM_JOB_ID"
echo "GPU(s) Allocated     : $CUDA_VISIBLE_DEVICES"
echo "================================================================"

echo "--- Launching IGN Training ---"

# WandB API key
export WANDB_API_KEY="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"

# The container automatically mounts your current working directory. 
# SLURM will run this exactly where you submit the script from.

python train_ign.py --mode train --dataset mnist --exp_name CNN1_L1 --batch_size 128 --n_epochs 25 

echo "Mission Complete."
