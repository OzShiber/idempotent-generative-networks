#!/bin/bash
#
# ABLATION TEST — run the no-corruption-supervision model (celeba_restore_purerec)
# through the SAME corrupted-input grid as the mixture model, so the two grids
# are directly comparable. Same architecture flags, --lambda_flow 0 (pure g+A).
#
# Compare the output grid against the mixture model's (celeba_final_restore /
# celeba_restore_clean test): if the pure-rec model restores blur/noise/low-res
# it never trained on, restoration is emergent; if not, it was supervised.

#SBATCH --job-name=ign_purerec_test
#SBATCH --output=slurm-purerectest-%j.out
#SBATCH --error=slurm-purerectest-%j.err
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

RUN_DIR=$(ls -dv ./results/*celeba_restore_purerec 2>/dev/null | tail -1)
if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR/ckpts" ]; then
    echo "ERROR: no celeba_restore_purerec run under ./results/ — run train_celeba_restore_purerec.sbatch first." >&2
    exit 1
fi
CKPT="$RUN_DIR/ckpts/final_model.pth"
if [ ! -f "$CKPT" ]; then
    CKPT=$(ls -v "$RUN_DIR"/ckpts/e*.pth 2>/dev/null | tail -1)
fi
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: no checkpoint found under $RUN_DIR/ckpts" >&2; exit 1
fi

echo "=== ABLATION RESTORATION TEST (pure-rec model) ==="
echo "Checkpoint : $CKPT"

python train_ign.py \
    --mode test \
    --dataset celeba \
    --orig_im_size 218 \
    --target_im_size 64 \
    --im_shape 3 64 64 \
    --exp_name celeba_purerec_restore \
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

echo "Done. Grid under results/celeba_*_celeba_purerec_restore/free_inputs/"
