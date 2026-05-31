#!/bin/bash
#
# SBATCH Submission Script: CelebA-64 IGN Training
#

# --- Job Configuration ---
#SBATCH --job-name=ign_celeba
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# --- Resources ---
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=l40s-public
#SBATCH --qos=24h_4g
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-gpu=12

# --- Node constraint ---
#SBATCH --nodelist=n317,n18

# --- Container ---
#SBATCH --container=/usr/local/apptainer/images/pytorch/latest/torch2.7-tf2.20-cu128.sif

set -e

echo "================================================================"
echo "Timestamp            : $(date)"
echo "Host                 : $(hostname)"
echo "SLURM Job ID         : $SLURM_JOB_ID"
echo "GPU(s) Allocated     : $CUDA_VISIBLE_DEVICES"
echo "================================================================"

export WANDB_API_KEY="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"

echo "--- Launching CelebA-64 IGN Training ---"

python train_ign.py \
    --mode train \
    --dataset celeba \
    --orig_im_size 218 \
    --target_im_size 64 \
    --im_shape 3 64 64 \
    --exp_name celeba_18L_256w_rotation \
    --n_layers 18 \
    --hidden_chans 256 \
    --binarizer rotation \
    --batch_size 64 \
    --n_epochs 50 \
    --eval_classifier_path imagenet \
    --ckpt ""

echo "Mission Complete."
