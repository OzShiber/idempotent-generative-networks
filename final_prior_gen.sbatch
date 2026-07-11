#!/bin/bash
#
# FINAL TEST 2/2 — GENERATION, on the SAME clean g+A model (NO flow head, no Amit).
#
# Generation = sample g's own latent distribution and decode. This uses ONLY
# g and A (the restorer): fit a low-rank Gaussian to g(x) latents, sample it,
# decode via g^-1(A .). No separate head, no flow, no Amit code — --lambda_flow 0
# means the flow module is never even built. This is the "one head does both"
# result: the restoration-trained model generates for free from its latent.
#
# Output: results/celeba_<ts>_celeba_final_gen/prior_gen/
#   prior_gen_T{1.0,0.85,0.7}.png   8x8 generated face sheets
#   prior_gen_stats.txt             FID vs real (and vs the old f(z) path + floor)
#   latent_prior.pth                the fitted prior (mean + low-rank cov)

#SBATCH --job-name=ign_final_gen
#SBATCH --output=slurm-finalgen-%j.out
#SBATCH --error=slurm-finalgen-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=rtx6k-shocher
#SBATCH --qos=24h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --container=/apps/apptainer/images/pytorch/latest/torch-tf-cu130.sif

set -e
export APPTAINER_BIND="$PRJ_WORKSPACE,$HOME"

# Same clean no-flow model as the restoration test: newest celeba_restore_clean.
RUN_DIR=$(ls -dv ./results/*celeba_restore_clean 2>/dev/null | tail -1)
if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR/ckpts" ]; then
    echo "ERROR: no celeba_restore_clean run under ./results/ — run train_celeba_restore_clean.sbatch first (and let it sync)." >&2
    exit 1
fi
CKPT="$RUN_DIR/ckpts/final_model.pth"
if [ ! -f "$CKPT" ]; then
    CKPT=$(ls -v "$RUN_DIR"/ckpts/e*.pth 2>/dev/null | tail -1)
fi
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: no checkpoint found under $RUN_DIR/ckpts" >&2; exit 1
fi

echo "=== FINAL GENERATION TEST (latent-prior, g+A only) ==="
echo "Checkpoint : $CKPT"

python train_ign.py \
    --mode prior_gen \
    --dataset celeba \
    --orig_im_size 218 \
    --target_im_size 64 \
    --im_shape 3 64 64 \
    --exp_name celeba_final_gen \
    --batch_size 128 \
    --n_layers 30 \
    --hidden_chans 256 \
    --g_coupling cnn \
    --binarizer rotation \
    --a_operator local_conv \
    --local_conv_stride 4 \
    --local_conv_kernel 8 \
    --local_conv_topk 16 \
    --lambda_flow 0 \
    --prior_pca_rank 256 \
    --prior_gen_n_fit 8192 \
    --prior_gen_n_fid 1024 \
    --no-wandb \
    --ckpt "$CKPT"

echo "Done. Sheets + FID under results/celeba_*_celeba_final_gen/prior_gen/"
