#!/bin/bash
#
# SBATCH Submission Script: CelebA-64 IGN — 24 Layers, 5 Seeds
#

# --- Job Configuration ---
#SBATCH --job-name=ign_celeba_24L
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --array=0-4
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

SEED=$SLURM_ARRAY_TASK_ID

echo "================================================================"
echo "Timestamp            : $(date)"
echo "Host                 : $(hostname)"
echo "SLURM Job ID         : $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID)"
echo "GPU(s) Allocated     : $CUDA_VISIBLE_DEVICES"
echo "Seed                 : $SEED"
echo "================================================================"

export WANDB_API_KEY="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"

echo "--- Launching CelebA-64 IGN Training: 24L seed=${SEED} ---"

python train_ign.py \
    --mode train \
    --dataset celeba \
    --orig_im_size 218 \
    --target_im_size 64 \
    --im_shape 3 64 64 \
    --exp_name celeba_24L_256w_s${SEED} \
    --n_layers 24 \
    --hidden_chans 256 \
    --binarizer rotation \
    --batch_size 128 \
    --n_epochs 50 \
    --seed $SEED \
    --eval_classifier_path imagenet \
    --ckpt ""

echo "Mission Complete."
