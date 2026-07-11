#!/bin/bash
#
# SBATCH Submission Script: CelebA-64 IGN — latent-prior GENERATION (step 1)
# on the combined restore+generate checkpoint (June 27 run).
#
# Fits the low-rank Gaussian prior to g(x) latents (the prior_sample
# diagnostic showed this decodes to faces), samples 8x8 sheets at
# T = 1.0 / 0.85 / 0.7, and scores FID vs the old f(pixel-noise) path and
# the real-vs-real floor. No training. Outputs land in
# results/celeba_<ts>_celeba_prior_gen/prior_gen/ (sheets + latent_prior.pth
# + prior_gen_stats.txt with the FID table).
#
# CRITICAL: the model is rebuilt from these CLI flags before loading the
# checkpoint — architecture flags copied from the trained run's conf.pkl.

# --- Job Configuration ---
#SBATCH --job-name=ign_prior_gen
#SBATCH --output=slurm-priorgen-%j.out
#SBATCH --error=slurm-priorgen-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# --- Resources (rtx6k-shocher; inference only) ---
#SBATCH --partition=rtx6k-shocher
#SBATCH --qos=24h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00

# --- Container ---
#SBATCH --container=/apps/apptainer/images/pytorch/latest/torch-tf-cu130.sif

set -e

export APPTAINER_BIND="$PRJ_WORKSPACE,$HOME"

RUN_DIR="./results/celeba_20260627_205237_celeba_localconv_restore_AND_generate"
CKPT="$RUN_DIR/ckpts/final_model.pth"
if [ ! -f "$CKPT" ]; then
    CKPT=$(ls -v "$RUN_DIR"/ckpts/e*.pth 2>/dev/null | tail -1)
fi
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: no checkpoint found under $RUN_DIR/ckpts" >&2
    exit 1
fi

echo "================================================================"
echo "Timestamp            : $(date)"
echo "Host                 : $(hostname)"
echo "SLURM Job ID         : $SLURM_JOB_ID"
echo "GPU(s) Allocated     : $CUDA_VISIBLE_DEVICES"
echo "Checkpoint           : $CKPT"
echo "================================================================"

echo "--- Launching CelebA-64 latent-prior generation ---"

python train_ign.py \
    --mode prior_gen \
    --dataset celeba \
    --orig_im_size 218 \
    --target_im_size 64 \
    --im_shape 3 64 64 \
    --exp_name celeba_prior_gen \
    --batch_size 128 \
    --n_layers 30 \
    --hidden_chans 256 \
    --g_coupling cnn \
    --binarizer rotation \
    --a_operator local_conv \
    --local_conv_stride 4 \
    --local_conv_kernel 8 \
    --local_conv_topk 16 \
    --lambda_flow 1.0 \
    --flow_rank 256 \
    --prior_pca_rank 256 \
    --prior_gen_n_fit 8192 \
    --prior_gen_n_fid 1024 \
    --no-wandb \
    --ckpt "$CKPT"

echo "Mission Complete. Outputs under results/celeba_*_celeba_prior_gen/prior_gen/"
