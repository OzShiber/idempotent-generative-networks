#!/bin/bash
#
# ABLATION — "how blind is it really?": NO corruption supervision.
#
# Identical to train_celeba_restore_clean.sbatch (same g+A, local_conv top-K 16,
# 30L/256w, no flow) EXCEPT the restoration loss is OFF: --lambda_denoise 0.
# The model only ever sees CLEAN faces (reconstruction f(x)=x + operator
# consistency). It never trains on noise/blur/low-res.
#
# Then test it with test_celeba_restore_purerec.sbatch on the SAME corrupted-
# input grid as the mixture model. Read-out:
#   - restores unseen corruptions well  -> restoration is EMERGENT from the
#     idempotent projection (the strong "blind" claim).
#   - corruptions pass through / poorly  -> "blind" restoration was really
#     SUPERVISED by the degradation mixture.
# --degradation_mode noise here is irrelevant (weight 0) — set only so no
# mixture is ever constructed.

#SBATCH --job-name=ign_purerec
#SBATCH --output=slurm-purerec-%j.out
#SBATCH --error=slurm-purerec-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=rtx6k-shocher
#SBATCH --qos=24h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --container=/apps/apptainer/images/pytorch/latest/torch-tf-cu130.sif

set -e
export APPTAINER_BIND="$PRJ_WORKSPACE,$HOME"
export WANDB_API_KEY="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"

echo "=== ABLATION: pure reconstruction, no corruption supervision ==="
echo "Timestamp : $(date)   Host: $(hostname)   Job: $SLURM_JOB_ID"

python train_ign.py \
    --mode train \
    --dataset celeba \
    --orig_im_size 218 \
    --target_im_size 64 \
    --im_shape 3 64 64 \
    --exp_name celeba_restore_purerec \
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
    --lambda_denoise 0.0 \
    --degradation_mode noise \
    --lambda_opcons 1.0 \
    --lambda_flow 0 \
    --eval_classifier_path imagenet \
    --keep_last_n_ckpts 3 \
    --ckpt "" \
    --wandb

echo "Done. Checkpoint under results/celeba_*_celeba_restore_purerec/ckpts/"
