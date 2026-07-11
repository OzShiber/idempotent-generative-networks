#!/bin/bash
#
# CLEAN FINAL MODEL — g + A restoration, NO flow head, no Amit code.
#
# Byte-for-byte the June-27 combined config EXCEPT --lambda_flow 0: the flow
# module is never built, so nothing Amit-derived exists in training or in the
# checkpoint. Because the June-27 flow trained on detached latents (zero
# gradient into g/A), this run's g+A should reproduce June-27's restoration
# quality while being a genuinely flow-free model.
#
# This is THE final model: restoration via --mode test, generation via
# --mode prior_gen — both on this one checkpoint (see final_*.sbatch, which
# auto-target the newest celeba_restore_clean run).

# --- Job Configuration ---
#SBATCH --job-name=ign_restore_clean
#SBATCH --output=slurm-restoreclean-%j.out
#SBATCH --error=slurm-restoreclean-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# --- Resources (rtx6k-shocher) ---
#SBATCH --partition=rtx6k-shocher
#SBATCH --qos=24h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=24:00:00

# --- Container ---
#SBATCH --container=/apps/apptainer/images/pytorch/latest/torch-tf-cu130.sif

set -e

export APPTAINER_BIND="$PRJ_WORKSPACE,$HOME"
export WANDB_API_KEY="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"

echo "================================================================"
echo "Timestamp            : $(date)"
echo "Host                 : $(hostname)"
echo "SLURM Job ID         : $SLURM_JOB_ID"
echo "GPU(s) Allocated     : $CUDA_VISIBLE_DEVICES"
echo "================================================================"

echo "--- Launching CelebA-64 CLEAN restoration model (lambda_flow 0) ---"

python train_ign.py \
    --mode train \
    --dataset celeba \
    --orig_im_size 218 \
    --target_im_size 64 \
    --im_shape 3 64 64 \
    --exp_name celeba_restore_clean \
    --seed 0 \
    --n_epochs 50 \
    --batch_size 128 \
    --lr 0.001 \
    --n_layers 30 \
    --hidden_chans 256 \
    --g_coupling cnn \
    --binarizer rotation \
    --a_operator local_conv \
    --local_conv_stride 4 \
    --local_conv_kernel 8 \
    --local_conv_topk 16 \
    --lambda_rec 1.0 \
    --lambda_denoise 1.0 \
    --degradation_mode mixture \
    --noise_sigma 0.3 \
    --lambda_opcons 1.0 \
    --lambda_flow 0 \
    --eval_classifier_path imagenet \
    --keep_last_n_ckpts 3 \
    --ckpt "" \
    --wandb

echo "Mission Complete. Checkpoint under results/celeba_*_celeba_restore_clean/ckpts/"
