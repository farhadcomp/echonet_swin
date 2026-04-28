#!/bin/bash

# Tell PyTorch to use our new, clean cache folder
export TORCH_HOME="/home/AD.UNLV.EDU/farhadik/tests/dynamic/clean_torch_cache"

# Force libraries to use only 1 thread per worker to prevent deadlocks
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --- THE BULLETPROOF NCCL FLAGS ---
# Disable unstable InfiniBand and P2P hardware bridges 
# Forces GPUs to use stable Shared Memory routing
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
# ----------------------------------

# Output directory specifically for the Baseline EF-Only run
OUTPUT_DIR="output/ef_with_aug"
DATA_DIR="/home/AD.UNLV.EDU/farhadik/echo_new/EchoNet-Dynamic"

mkdir -p ${OUTPUT_DIR}
TRAIN_LOG="${OUTPUT_DIR}/training_terminal_output.txt"
TEST_LOG="${OUTPUT_DIR}/testing_terminal_output.txt"

echo "=========================================================="
echo " Starting TRUE DDP Training: Baseline Swin3D with Data Augmentation"
echo "=========================================================="

# Training on 6 GPUs (bypassing the frozen GPU 0)
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
    --nproc_per_node=4 \
    --master_port=29509 \
    echonet/utils/video_aug.py \
    --data_dir ${DATA_DIR} \
    --output ${OUTPUT_DIR} \
    --model_name swin3d_s \
    --num_epochs 45 \
    --batch_size 8 \
    --num_workers 8 \
    --lr 1e-4 \
    2>&1 | tee ${TRAIN_LOG}

echo "========================================="
echo " Training Complete. Starting Evaluation."
echo "========================================="

# Testing the new Swin3D weights
CUDA_VISIBLE_DEVICES=4 torchrun \
    --nproc_per_node=1 \
    --master_port=29510 \
    echonet/utils/video_aug.py \
    --data_dir ${DATA_DIR} \
    --output ${OUTPUT_DIR}/ \
    --model_name swin3d_s \
    --weights ${OUTPUT_DIR}/best.pt \
    --run_test \
    --batch_size 1 \
    --num_workers 8 \
    --num_epochs 0 \
    2>&1 | tee ${TEST_LOG}

python3 plot_metrics.py "$OUTPUT_DIR"