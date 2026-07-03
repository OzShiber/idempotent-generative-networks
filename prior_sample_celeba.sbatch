#!/bin/bash
#
# SBATCH Submission Script: CelebA-64 IGN — latent-prior sampling diagnostic
# ("option 1" of the generation plan) on the combined restore+generate run.
#
# Fits a Gaussian to g(x) latents of real faces and decodes samples — answers
# whether g's latent space supports generation at all, with zero training.
# Outputs land in results/celeba_<ts>_celeba_prior_sample_restore_gen/prior_samples/
# (grids + prior_stats.txt). See prior_sampling.py for how to read them.
#
# CRITICAL: prior_sample mode rebuilds the model from these CLI flags before
# loading the checkpoint — the architecture flags below are copied from the
# trained run's conf.pkl (30L, w256, local_conv k8 s4 topk16, flow head on).

# --- Job Configuration ---
#SBATCH --job-name=ign_prior_sample
#SBATCH --output=slurm-prior-%j.out
#SBATCH --error=slurm-prior-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# --- Resources (rtx6k-shocher; inference only — short + light) ---
#SBATCH --partition=rtx6k-shocher
#SBATCH --qos=24h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# --- Container ---
#SBATCH --container=/apps/apptainer/images/pytorch/latest/torch-tf-cu130.sif

set -e

export APPTAINER_BIND="$PRJ_WORKSPACE,$HOME"

# Prefer the final model of the finished combined run; fall back to the
# highest-epoch e*.pth if final_model.pth is missing.
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

echo "--- Launching CelebA-64 latent-prior sampling diagnostic ---"

python train_ign.py \
    --mode prior_sample \
    --dataset celeba \
    --orig_im_size 218 \
    --target_im_size 64 \
    --im_shape 3 64 64 \
    --exp_name celeba_prior_sample_restore_gen \
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
    --prior_n_fit 4096 \
    --prior_pca_rank 256 \
    --no-wandb \
    --ckpt "$CKPT"

echo "Mission Complete. Outputs under results/celeba_*_celeba_prior_sample_restore_gen/prior_samples/"
