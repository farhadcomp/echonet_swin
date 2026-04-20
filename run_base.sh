#!/bin/bash

# 1. Activate your clean, working environment FIRST
source ~/echo_env/bin/activate

# 2. Define the output directory for the evaluation results
OUTPUT_DIR="output/video/r2plus1d_pretrained_eval"

# Create the output directory immediately
mkdir -p ${OUTPUT_DIR}

echo "========================================="
echo " Running Evaluation: r2plus1d Pre-trained"
echo "========================================="

# 3. Run ONLY the testing phase using your exact command
# --num_epochs 0 ensures it doesn't accidentally try to train
CUDA_VISIBLE_DEVICES=0 echonet video \
    --weights r2plus1d_18_32_2_pretrained.pt \
    --run_test \
    --num_epochs 0 \
    --output ${OUTPUT_DIR} 2>&1 | tee ${OUTPUT_DIR}/testing_terminal_output.txt

echo "========================================="
echo " Evaluation Complete!"
echo " Results saved in: ${OUTPUT_DIR}"
echo "========================================="