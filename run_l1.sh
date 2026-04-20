#!/bin/bash

# 1. Define the output directory to keep paths clean and avoid typos
OUTPUT_DIR="output/video/swin_smooth_l1_run"

# Create the output directory immediately so we can save logs into it right away
mkdir -p ${OUTPUT_DIR}

TRAIN_LOG="${OUTPUT_DIR}/training_terminal_output.txt"
TEST_LOG="${OUTPUT_DIR}/testing_terminal_output.txt"

echo "========================================="
echo " Starting Training: Swin smooth L1"
echo "========================================="

# 2. Run Training
# '2>&1' captures both standard output and error messages.
# '| tee' prints the output to your terminal screen AND saves it to the text file simultaneously.
CUDA_VISIBLE_DEVICES=0,1 echonet video --num_epochs 45 --batch_size 16 --num_workers 4 --lr 1e-4 --output ${OUTPUT_DIR} 2>&1 | tee ${TRAIN_LOG}

echo "========================================="
echo " Training Complete. Starting Evaluation."
echo "========================================="

# 3. Run Testing 
# (Updated paths to use the weights from the training run that just finished)
CUDA_VISIBLE_DEVICES=0,1 echonet video --output ${OUTPUT_DIR}/ --weights ${OUTPUT_DIR}/best.pt --run_test --batch_size 1 --num_epochs 0 2>&1 | tee ${TEST_LOG}

echo "========================================="
echo " Pipeline Complete!"
echo " Logs saved to: ${TRAIN_LOG} and ${TEST_LOG}"
echo "========================================="
