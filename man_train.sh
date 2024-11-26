#!/bin/bash

GPU=0,1,2,3
LOAD_PATH=""
DATA_CONFIG="data/s_p.yaml"
OUTPUT_DIR="/data/wjdu/unihar/a5_p/s/UniTS_HEAD"
MASTER_PORT=2233

mkdir -p "$OUTPUT_DIR"

echo "Training on GPU: $GPU with MASTER_PORT: $MASTER_PORT"
echo "Data config: $DATA_CONFIG"
echo "Output directory: $OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=$GPU python -u -m torch.distributed.launch --master_port=$MASTER_PORT --nproc_per_node=4 --use_env \
 train.py --data_config "$DATA_CONFIG" --load_path "$LOAD_PATH" --batch_size 512 \
 --epochs 40 --warmup_epochs 20 --blr 1e-4 --min_lr 1e-6 --weight_decay 5e-6 \
 --output_dir "$OUTPUT_DIR" \
 --seed 42 \
#  > "$OUTPUT_DIR/output.log" 2>&1
