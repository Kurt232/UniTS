#!/usr/bin/bash

# CONFIG="$2"
# OUTPUT_DIR="$3"

LOAD_PATH="units_x128_pretrain.pth"
CONFIG="/data/wenhao/wjdu/location/data/train_pure_cla_ankle.json"
OUTPUT_DIR="/data/wenhao/wjdu/location/output/UniTS_HEAD/ankle"

mkdir -p "$OUTPUT_DIR"

# CUDA_VISIBLE_DEVICES=4 python -m debugpy --wait-for-client --listen localhost:5678 \
CUDA_VISIBLE_DEVICES=4,5,6,7 python -u -m torch.distributed.launch --master_port=2113 --nproc_per_node=4 --use_env \
 train.py --data_config "$CONFIG" --batch_size 128 \
 --epochs 40 --warmup_epochs 1 --blr 1e-3 --weight_decay 5e-6 \
 --load_path "$LOAD_PATH" \
 --output_dir "$OUTPUT_DIR" \
#  2>&1 | tee "$OUTPUT_DIR"/output.log &