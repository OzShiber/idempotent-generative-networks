#!/bin/bash
#
# SBATCH Submission Script: corrupted-input test for a local_conv restoration run
#
# Runs --mode test (blur / noise / occlusion / low-res / clean / pure-noise
# grids) on one of the train_celeba_lc_restore.sbatch checkpoints.
#
# >>> EDIT the three vars below to pick which run to test. <<<
# CRITICAL: --a_operator local_conv, --n_layers, --hidden_chans, --local_conv_stride
# and --local_conv_topk MUST match the trained run or the checkpoint loads into a
# differently-behaving model (stride changes shapes; topk changes which channels
# the operator keeps). --degradation_mode is NOT needed at test time.

# --- Job Configuration ---
#SBATCH --job-name=ign_test_lc
#SBATCH --output=slurm-test-%j.out
#SBATCH --error=slurm-test-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# --- Resources (rtx6k-shocher; inference only) ---
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

# >>> EDIT THESE to match the run you want to test <<<
TOPK=8
STRIDE=4
RUN_DIR="./results/celeba_30L_lc_restore_k${TOPK}_st${STRIDE}_s0"
# (run dirs are timestamped — if the glob below finds nothing, set RUN_DIR to the
#  exact ./results/celeba_<date>_<time>_30L_lc_restore_... path.)
RUN_DIR=$(ls -dv ${RUN_DIR} ./results/*_30L_lc_restore_k${TOPK}_st${STRIDE}_s0 2>/dev/null | tail -1)

CKPT=$(ls -v "$RUN_DIR"/ckpts/e*.pth "$RUN_DIR"/ckpts/final_model.pth 2>/dev/null | tail -1)
if [ -z "$CKPT" ]; then
    echo "ERROR: no checkpoint found under $RUN_DIR/ckpts (check TOPK/STRIDE/RUN_DIR)" >&2
    exit 1
fi

echo "================================================================"
echo "Host        : $(hostname)   GPU: $CUDA_VISIBLE_DEVICES"
echo "Run dir     : $RUN_DIR"
echo "Checkpoint  : $CKPT   (topk=$TOPK stride=$STRIDE)"
echo "================================================================"

export WANDB_API_KEY="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"

python train_ign.py \
    --mode test \
    --dataset celeba \
    --orig_im_size 218 \
    --target_im_size 64 \
    --im_shape 3 64 64 \
    --exp_name celeba_lc_restore_test_k${TOPK}_st${STRIDE} \
    --n_layers 30 \
    --hidden_chans 256 \
    --binarizer rotation \
    --a_operator local_conv \
    --local_conv_topk ${TOPK} \
    --local_conv_stride ${STRIDE} \
    --no-wandb \
    --ckpt "$CKPT"

echo "Mission Complete. Grids under results/celeba_*_celeba_lc_restore_test_*/free_inputs/"
