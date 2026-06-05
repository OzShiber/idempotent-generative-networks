#!/bin/bash
#
# SBATCH Submission Script: CelebA-64 IGN — rotation_beta sweep, single seed
#
# Sweeps --rotation_beta over {0.3, 0.5, 0.7, 1.0} as a 4-task array, seed 0.
# Everything else is held fixed (30L, w256, overlapping convs, pure L_rec) so
# the effect of beta on A_active / FID / OOD / sharpness is isolated.
#   beta < 1  -> weaker sparsity pressure -> higher A_active (more detail)
#   beta = 1  -> current baseline
#

# --- Job Configuration ---
#SBATCH --job-name=ign_celeba_beta
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --array=0-3
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

# Map array index -> beta value
BETAS=(0.3 0.5 0.7 1.0)
BETA=${BETAS[$SLURM_ARRAY_TASK_ID]}

echo "================================================================"
echo "Timestamp            : $(date)"
echo "Host                 : $(hostname)"
echo "SLURM Job ID         : $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID)"
echo "GPU(s) Allocated     : $CUDA_VISIBLE_DEVICES"
echo "rotation_beta        : $BETA"
echo "================================================================"

export WANDB_API_KEY="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"

echo "--- Launching CelebA-64 IGN: rotation_beta=${BETA} seed=0 ---"

python train_ign.py \
    --mode train \
    --dataset celeba \
    --orig_im_size 218 \
    --target_im_size 64 \
    --im_shape 3 64 64 \
    --exp_name celeba_30L_256w_beta${BETA}_s0 \
    --n_layers 30 \
    --hidden_chans 256 \
    --binarizer rotation \
    --rotation_beta ${BETA} \
    --batch_size 128 \
    --n_epochs 50 \
    --seed 0 \
    --eval_classifier_path imagenet \
    --ckpt ""

echo "Mission Complete."
