#!/usr/bin/bash

# CONFIG="$2"
# OUTPUT_DIR="$3"

# LOAD_PATH="units_x128_pretrain.pth"
CONFIG='/data/wjdu/data4/realworld1/realworld_10_thigh_TRAIN.json /data/wjdu/data4/realworld1/realworld_10_thigh_TEST.json'
OUTPUT_DIR="/data/wjdu/TST_HEAD/"
MODEL_CONFIG="configs/model_c2_h1_d64_nh8_nl3.json"

mkdir -p "$OUTPUT_DIR"

# CUDA_VISIBLE_DEVICES=4 python -m debugpy --wait-for-client --listen localhost:5678 \
CUDA_VISIBLE_DEVICES=4 python -u -m torch.distributed.launch --master_port=2114 --nproc_per_node=1 --use_env \
 train.py --data_config $CONFIG --model_config $MODEL_CONFIG --batch_size 512 \
 --epochs 200 --warmup_epochs 1 --blr 1e-4 --min_lr 1e-6 --weight_decay 5e-6 \
 --output_dir "$OUTPUT_DIR" \
 --seed 42
#  2>&1 | tee "$OUTPUT_DIR"/output.log &