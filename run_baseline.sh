#!/bin/bash

#!/bin/bash

# Tell PyTorch to use our new, clean cache folder
export TORCH_HOME="/home/AD.UNLV.EDU/farhadik/tests/dynamic/clean_torch_cache"

# ADD THESE TWO LINES: Force libraries to use only 1 thread per worker
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Output directory specifically for the baseline
OUTPUT_DIR="output/video/r2plus1d_fresh_cache"
DATA_DIR="/home/AD.UNLV.EDU/farhadik/echo"

mkdir -p ${OUTPUT_DIR}
TRAIN_LOG="${OUTPUT_DIR}/training_terminal_output.txt"
TEST_LOG="${OUTPUT_DIR}/testing_terminal_output.txt"

echo "========================================="
echo " Starting TRUE DDP Training: R2Plus1D on 7 GPUs"
echo "========================================="

# 1. Using all 8 GPUs
# 2. explicitly passing --model_name r2plus1d_18
# 3. We keep batch_size=8 (per GPU) and lr=1e-4 so the hyperparams match the Swin run
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 echonet/utils/video.py \
#     --data_dir ${DATA_DIR} \
#     --output ${OUTPUT_DIR} \
#     --model_name r2plus1d_18 \
#     --num_epochs 45 \
#     --batch_size 8 \
#     --num_workers 4 \
#     --lr 1e-4 \
#     2>&1 | tee ${TRAIN_LOG}
export NCCL_DEBUG=INFO
# We add --master_port with a random number like 29507
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun \
#     --nproc_per_node=7 \
#     --master_port=29507 \
#     echonet/utils/video.py \
#     --data_dir ${DATA_DIR} \
#     --output ${OUTPUT_DIR} \
#     --model_name r2plus1d_18 \
#     --num_epochs 45 \
#     --batch_size 8 \
#     --num_workers 1 \
#     --lr 1e-4 \
#     2>&1 | tee ${TRAIN_LOG}

echo "========================================="
echo " Training Complete. Starting Evaluation."
echo "========================================="

CUDA_VISIBLE_DEVICES=6,7 torchrun \
    --nproc_per_node=2 \
    --master_port=29508 \
    echonet/utils/video.py \
    --data_dir ${DATA_DIR} \
    --output ${OUTPUT_DIR}/ \
    --model_name r2plus1d_18 \
    --weights ${OUTPUT_DIR}/best.pt \
    --run_test \
    --batch_size 1 \
    --num_workers 1 \
    --num_epochs 0 \
    2>&1 | tee ${TEST_LOG}

python3 plot_metrics.py "$OUTPUT_DIR"