#!/bin/bash

# 1. Define the output directory to keep paths clean and avoid typos
OUTPUT_DIR="output/video/swin_DDP_2gpu_run"

# >>> IMPORTANT: Replace this with the path containing your Tensors_224 and FileList.csv <<<
DATA_DIR="/home/AD.UNLV.EDU/farhadik/echo"

# Create the output directory immediately so we can save logs into it right away
mkdir -p ${OUTPUT_DIR}

TRAIN_LOG="${OUTPUT_DIR}/training_terminal_output.txt"
TEST_LOG="${OUTPUT_DIR}/testing_terminal_output.txt"

echo "========================================="
echo " Starting TRUE DDP Training: GPUs 0 and 1"
echo "========================================="

# 2. Run Training
# CHANGED: Using 'torchrun --nproc_per_node=2' to launch 2 DDP processes
# CHANGED: 'echonet video' to 'video.py' to use your custom Swin code
# NOTE: batch_size=8 here means 8 PER GPU (Global effective batch = 16)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 echonet/utils/video.py \
    --data_dir ${DATA_DIR} \
    --output ${OUTPUT_DIR} \
    --num_epochs 45 \
    --batch_size 8 \
    --num_workers 4 \
    --lr 1e-4 \
    --period 3 \
    2>&1 | tee ${TRAIN_LOG}

echo "========================================="
echo " Training Complete. Starting Evaluation."
echo "========================================="

# 3. Run Testing 
# We also use torchrun here since your code now expects DDP environment variables
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 echonet/utils/video.py \
    --data_dir ${DATA_DIR} \
    --output ${OUTPUT_DIR}/ \
    --weights ${OUTPUT_DIR}/best.pt \
    --run_test \
    --batch_size 1 \
    --num_epochs 0 \
    2>&1 | tee ${TEST_LOG}

echo "========================================="
echo " Pipeline Complete!"
echo " Logs saved to: ${TRAIN_LOG} and ${TEST_LOG}"
echo "========================================="

# =========================================
#  Generate Loss Curves
# =========================================
echo "Training and Testing Complete. Generating plots..."

# Run the python script and pass the output directory to it
python3 plot_metrics.py "$OUTPUT_DIR"
