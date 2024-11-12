#!/bin/bash

GPU=0
LOAD_PATH="/home/wjdu/units_head/units_x128_pretrain.pth"
DATA_CONFIG="/data/wjdu/data4/realworld/realworld_10_thigh_TRAIN.json /data/wjdu/data4/realworld/realworld_10_thigh_TEST.json"
OUTPUT_DIR="/data/wjdu/realworld_thigh/UniTS_HEAD_p_10"
MASTER_PORT=2112

mkdir -p "$OUTPUT_DIR"

echo "Training on GPU: $GPU with MASTER_PORT: $MASTER_PORT"
echo "Data config: $DATA_CONFIG"
echo "Output directory: $OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=$GPU python -u -m torch.distributed.launch --master_port=$MASTER_PORT --nproc_per_node=1 --use_env \
 train.py --data_config $DATA_CONFIG --load_path $LOAD_PATH --batch_size 512 \
 --epochs 200 --warmup_epochs 20 --blr 1e-4 --min_lr 1e-6 --weight_decay 5e-6 \
 --output_dir "$OUTPUT_DIR" \
 --seed 42 \
#  > "$OUTPUT_DIR/output.log" 2>&1
#  > "$OUTPUT_DIR/output.log" 2>&1
