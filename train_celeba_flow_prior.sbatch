#!/bin/bash
#
# SBATCH Submission Script: CelebA-64 IGN — combined restore + generate,
# with the FIXED flow source (step 2 of the generation fix).
#
# Identical to the June-27 combined run except --flow_source prior: the
# flow-matching head now starts from latents sampled from a running EMA
# Gaussian fitted to the data latents g(x), instead of g(pixel noise) —
# the prior_sample diagnostic showed pixel-noise latents land 5-15x outside
# the real-latent distribution, which is why the v1 flow generated noise.
# Restoration training is unchanged (flow latents stay detached).
#
# Watch flow_samples_e*.png during training: with the fixed source they
# should show faces within a few epochs (the source alone already decodes
# to faces; the flow only sharpens toward the true latent distribution).

# --- Job Configuration ---
#SBATCH --job-name=ign_flow_prior
#SBATCH --output=slurm-flowprior-%j.out
#SBATCH --error=slurm-flowprior-%j.err
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

echo "--- Launching CelebA-64 combined run with flow_source=prior ---"

python train_ign.py \
    --mode train \
    --dataset celeba \
    --orig_im_size 218 \
    --target_im_size 64 \
    --im_shape 3 64 64 \
    --exp_name celeba_localconv_restore_gen_flowprior \
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
    --lambda_opcons 1.0 \
    --lambda_flow 1.0 \
    --flow_rank 256 \
    --flow_source prior \
    --flow_temp 1.0 \
    --eval_classifier_path imagenet \
    --keep_last_n_ckpts 3 \
    --ckpt "" \
    --wandb

echo "Mission Complete."
