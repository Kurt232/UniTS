#!/usr/bin/bash

# CONFIG="$2"
# OUTPUT_DIR="$3"

LOAD_PATH="units_x128_pretrain.pth"
CONFIG='/data/wenhao/wjdu/data/realworld_thigh_2s_25hz/train_thigh_8.json'
OUTPUT_DIR="/data/wenhao/wjdu/realworld_thigh_25hz/UniTS_HEAD_8"

mkdir -p "$OUTPUT_DIR"

# CUDA_VISIBLE_DEVICES=4 python -m debugpy --wait-for-client --listen localhost:5678 \
CUDA_VISIBLE_DEVICES=4 python -u -m torch.distributed.launch --master_port=2113 --nproc_per_node=1 --use_env \
 train.py --data_config "$CONFIG" --batch_size 256 \
 --epochs 100 --warmup_epochs 1 --blr 5e-4 --weight_decay 5e-6 \
 --load_path "$LOAD_PATH" \
 --output_dir "$OUTPUT_DIR" \
#  2>&1 | tee "$OUTPUT_DIR"/output.log &