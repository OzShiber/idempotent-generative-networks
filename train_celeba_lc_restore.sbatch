#!/bin/bash
#
# SBATCH Submission Script: CelebA-64 IGN — local_conv restoration sweep
#
# 2x2 factorial over the two restoration levers, all on the local_conv operator
# trained DIRECTLY on a degradation mixture (--degradation_mode mixture):
#     topk in {8, 16}   x   stride in {4, 2}
#   task 0: topk= 8, stride=4   (capacity baseline, more compression)
#   task 1: topk=16, stride=4   (more channels)
#   task 2: topk= 8, stride=2   (less compression -> more spatial detail)
#   task 3: topk=16, stride=2   (both)
# Single seed (0), everything else fixed, so the effect of channel budget (topk)
# and compression (stride) is isolated and combinable.

# --- Job Configuration ---
#SBATCH --job-name=ign_lc_restore
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1

# --- Resources (rtx6k-shocher partition) ---
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

# Map array index -> (topk, stride)
TOPKS=(8 16 8 16)
STRIDES=(4 4 2 2)
TOPK=${TOPKS[$SLURM_ARRAY_TASK_ID]}
STRIDE=${STRIDES[$SLURM_ARRAY_TASK_ID]}

echo "================================================================"
echo "Timestamp            : $(date)"
echo "Host                 : $(hostname)"
echo "SLURM Job ID         : $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID)"
echo "GPU(s) Allocated     : $CUDA_VISIBLE_DEVICES"
echo "local_conv_topk      : $TOPK"
echo "local_conv_stride    : $STRIDE"
echo "================================================================"

export WANDB_API_KEY="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"

echo "--- Launching CelebA-64 local_conv restoration: topk=${TOPK} stride=${STRIDE} ---"

# Restoration training: --degradation_mode mixture + --lambda_denoise 1.0 makes
# the 'denoise' loss the blind-restoration objective (f(corrupt(x)) -> clean x
# over noise/blur/occlusion/lowres). Watch the 'denoise' column fall, A_active
# pinned at TOPK, and opcons -> ~0.
# NOTE: rtx6k = RTX 6000. The restoration loss adds a second forward through f
# per step; if it is the 24 GB card, drop --batch_size to 64 (or 32) on OOM.
python train_ign.py \
    --mode train \
    --dataset celeba \
    --orig_im_size 218 \
    --target_im_size 64 \
    --im_shape 3 64 64 \
    --exp_name celeba_30L_lc_restore_k${TOPK}_st${STRIDE}_s0 \
    --n_layers 30 \
    --hidden_chans 256 \
    --binarizer rotation \
    --a_operator local_conv \
    --local_conv_topk ${TOPK} \
    --local_conv_stride ${STRIDE} \
    --degradation_mode mixture \
    --lambda_denoise 1.0 \
    --batch_size 128 \
    --n_epochs 50 \
    --seed 0 \
    --eval_classifier_path imagenet \
    --ckpt ""

echo "Mission Complete."
