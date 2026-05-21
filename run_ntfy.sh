#!/bin/bash

START_TIME=$(date +%s)

# Notification topic
NTFY_TOPIC="farshid-echonet-2026-a9x73job"

set -o pipefail

# Tell PyTorch to use our new, clean cache folder
export TORCH_HOME="/home/AD.UNLV.EDU/farhadik/tests/dynamic/clean_torch_cache"

# Force libraries to use only 1 thread per worker
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# NCCL flags
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

OUTPUT_DIR="output/swin_with_videos_autocrop_off_no_cutout"
DATA_DIR="/home/AD.UNLV.EDU/farhadik/echo_new/EchoNet-Dynamic"

mkdir -p ${OUTPUT_DIR}

TRAIN_LOG="${OUTPUT_DIR}/training_terminal_output.txt"
TEST_LOG="${OUTPUT_DIR}/testing_terminal_output.txt"

notify_finish() {
    EXIT_CODE=$?

    END_TIME=$(date +%s)
    TOTAL_SECONDS=$((END_TIME - START_TIME))

    HOURS=$((TOTAL_SECONDS / 3600))
    MINUTES=$(((TOTAL_SECONDS % 3600) / 60))
    SECONDS=$((TOTAL_SECONDS % 60))

    if [ $EXIT_CODE -eq 0 ]; then
        STATUS="SUCCESS"
    else
        STATUS="FAILED"
    fi

    curl -d "EchoNet job ${STATUS}

Output directory:
${OUTPUT_DIR}

Training log:
${TRAIN_LOG}

Testing log:
${TEST_LOG}

Total time:
${HOURS}h ${MINUTES}m ${SECONDS}s

Exit code:
${EXIT_CODE}" https://ntfy.sh/${NTFY_TOPIC}
}

trap notify_finish EXIT

echo "=========================================================="
echo " Starting DDP Training"
echo "=========================================================="

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --master_port=29509 \
    echonet/utils/video_aug.py \
    --data_dir ${DATA_DIR} \
    --output ${OUTPUT_DIR} \
    --model_name swin3d_s \
    --num_epochs 45 \
    --batch_size 8 \
    --num_workers 4 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    # --beta 5.0 \
    # --no-augment \
    2>&1 | tee ${TRAIN_LOG}

echo "========================================="
echo " Training Complete. Starting Evaluation."
echo "========================================="

CUDA_VISIBLE_DEVICES=0 torchrun \
    --nproc_per_node=1 \
    --master_port=29510 \
    echonet/utils/video_aug.py \
    --data_dir ${DATA_DIR} \
    --output ${OUTPUT_DIR} \
    --model_name swin3d_s \
    --weights ${OUTPUT_DIR}/best.pt \
    --run_test \
    --batch_size 1 \
    --num_workers 4 \
    --num_epochs 0 \
    # --beta 5.0 \
    # --no-augment \
    2>&1 | tee ${TEST_LOG}

python3 plot_metrics.py "$OUTPUT_DIR"

END_TIME=$(date +%s)
TOTAL_SECONDS=$((END_TIME - START_TIME))

HOURS=$((TOTAL_SECONDS / 3600))
MINUTES=$(((TOTAL_SECONDS % 3600) / 60))
SECONDS=$((TOTAL_SECONDS % 60))

echo "=========================================================="
echo " Script completed in: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "=========================================================="
