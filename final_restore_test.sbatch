#!/bin/bash
#
# FINAL TEST 1/2 — RESTORATION, on the clean g+A model (NO flow head, no Amit).
#
# Checkpoint: the June-13 local-conv restoration run (a_operator local_conv,
# top-K 16, stride 4, degradation mixture, lambda_flow absent -> no flow head
# was ever built). --lambda_flow 0 here so the model is pure g + A; f(x) is the
# plain g^-1(A g(x)) restoration path.
#
# Output: results/celeba_<ts>_celeba_final_restore/free_inputs/ — 2-row grids,
# top = corrupted inputs (clean/blur/noise/occlusion/low-res/pure-noise),
# bottom = f(input).

#SBATCH --job-name=ign_final_restore
#SBATCH --output=slurm-finalrestore-%j.out
#SBATCH --error=slurm-finalrestore-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=rtx6k-shocher
#SBATCH --qos=24h_4g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --container=/apps/apptainer/images/pytorch/latest/torch-tf-cu130.sif

set -e
export APPTAINER_BIND="$PRJ_WORKSPACE,$HOME"

# The clean no-flow model: newest celeba_restore_clean run (train it with
# train_celeba_restore_clean.sbatch first). No silent fallback to a flow
# checkpoint — this test is specifically for the flow-free model.
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

echo "=== FINAL RESTORATION TEST ==="
echo "Checkpoint : $CKPT"

python train_ign.py \
    --mode test \
    --dataset celeba \
    --orig_im_size 218 \
    --target_im_size 64 \
    --im_shape 3 64 64 \
    --exp_name celeba_final_restore \
    --n_layers 30 \
    --hidden_chans 256 \
    --g_coupling cnn \
    --binarizer rotation \
    --a_operator local_conv \
    --local_conv_stride 4 \
    --local_conv_kernel 8 \
    --local_conv_topk 16 \
    --lambda_flow 0 \
    --no-wandb \
    --ckpt "$CKPT"

echo "Done. Grids under results/celeba_*_celeba_final_restore/free_inputs/"
