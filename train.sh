#!/bin/bash

GPU=$1
DATA_CONFIG=$2
OUTPUT_DIR=$3
MASTER_PORT=$4

NNODE=$(($(echo $GPU | tr -cd , | wc -c) + 1))  # GPU="0,1,2,3" -> NNODE=4

mkdir -p "$OUTPUT_DIR"

echo "Training on GPU: $GPU with MASTER_PORT: $MASTER_PORT"
echo "Data config: $DATA_CONFIG"
echo "Output directory: $OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=$GPU python -u -m torch.distributed.launch --master_port=$MASTER_PORT --nproc_per_node=$NNODE --use_env \
 train.py --data_config $DATA_CONFIG --batch_size 512 \
 --epochs 40 --warmup_epochs 20 --blr 1e-4 --min_lr 1e-6 --weight_decay 5e-6 \
 --output_dir "$OUTPUT_DIR" \
 --seed 42 \
 > "$OUTPUT_DIR/output.log" 2>&1
