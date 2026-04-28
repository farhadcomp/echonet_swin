#!/bin/bash

# 1. Activate your clean, working environment FIRST
source ~/echo_env/bin/activate

# 2. Define the directories
OUTPUT_DIR="output/r2plus1d_pretrained_eval"
DATA_DIR="/home/AD.UNLV.EDU/farhadik/echo_new/EchoNet-Dynamic" # Added your data path

# Create the output directory immediately
mkdir -p ${OUTPUT_DIR}

echo "========================================="
echo " Running Evaluation: r2plus1d Pre-trained"
echo "========================================="

# 3. Run ONLY the testing phase using torchrun (REQUIRED FOR DDP)
CUDA_VISIBLE_DEVICES=4 torchrun \
    --nproc_per_node=1 \
    --master_port=29510 \
    echonet/utils/video_r.py \
    --data_dir ${DATA_DIR} \
    --output ${OUTPUT_DIR} \
    --model_name r2plus1d_18 \
    --weights r2plus1d_18_32_2_pretrained.pt \
    --run_test \
    --num_epochs 0 \
    --batch_size 1 \
    --num_workers 8 \
    2>&1 | tee ${OUTPUT_DIR}/testing_terminal_output.txt

echo "========================================="
echo " Evaluation Complete!"
echo " Results saved in: ${OUTPUT_DIR}"
echo "========================================="

python3 plot_metrics.py "$OUTPUT_DIR"