#!/bin/bash
#
# SBATCH Submission Script: CelebA-64 IGN — corrupted-input test mode
#
# Runs --mode test on a trained checkpoint: feeds crafted/corrupted faces
# (clean / blur / noise / occlusion / low-res / pure noise) through f and saves
# 2-row input-vs-projection grids to <exp_dir>/free_inputs/.
#
# >>> EDIT THE CKPT LINE BELOW to point at the checkpoint you want to test. <<<
# The --n_layers / --hidden_chans / --im_shape MUST match the trained model or
# the state_dict load will fail with a shape mismatch.
#

# --- Job Configuration ---
#SBATCH --job-name=ign_celeba_test
#SBATCH --output=slurm-test-%j.out
#SBATCH --error=slurm-test-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# --- Resources (inference only — short + light) ---
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=l40s-public
#SBATCH --qos=24h_4g
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-gpu=8

# --- Node constraint ---
#SBATCH --nodelist=n317,n18

# --- Container ---
#SBATCH --container=/usr/local/apptainer/images/pytorch/latest/torch2.7-tf2.20-cu128.sif

set -e

# >>> EDIT THIS: path to the checkpoint to test <<<
CKPT="./results/celeba_20260605_100612_celeba_30L_256w_s0/ckpts/e50.pth"

echo "================================================================"
echo "Timestamp            : $(date)"
echo "Host                 : $(hostname)"
echo "SLURM Job ID         : $SLURM_JOB_ID"
echo "GPU(s) Allocated     : $CUDA_VISIBLE_DEVICES"
echo "Checkpoint           : $CKPT"
echo "================================================================"

export WANDB_API_KEY="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"

echo "--- Launching CelebA-64 IGN corrupted-input test ---"

# --no-wandb: test mode produces no training curves worth logging.
# n_layers / hidden_chans / im_shape must match the checkpoint's architecture.
python train_ign.py \
    --mode test \
    --dataset celeba \
    --orig_im_size 218 \
    --target_im_size 64 \
    --im_shape 3 64 64 \
    --exp_name celeba_test \
    --n_layers 30 \
    --hidden_chans 256 \
    --binarizer rotation \
    --no-wandb \
    --ckpt "$CKPT"

echo "Mission Complete. Grids saved under the new results/celeba_*_celeba_test/free_inputs/ dir."
