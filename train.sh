#!/usr/bin/env bash
set -e  # Exit immediately if a command exits with a non-zero status

GPU=$1
DATA_CONFIG=$2
OUTPUT_DIR=$3
MASTER_PORT=$4
SETTING_ID=$5

NNODE=$(($(echo $GPU | tr -cd , | wc -c) + 1))  # GPU="0,1,2,3" -> NNODE=4

DATA_CONFIG="data/config.yaml"
TRAIN_DIR="${ROOT}/output/${MODEL}_m"

echo "Training on GPU: $GPU with MASTER_PORT: $MASTER_PORT"
echo "Data config: $DATA_CONFIG"
echo "Output directory: $OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=$GPU python -u -m torch.distributed.launch --master_port=$MASTER_PORT --nproc_per_node=$NNODE --use_env \
 train.py --data_config $DATA_CONFIG --batch_size 512 \
 --epochs 40 --warmup_epochs 10 --blr 1e-4 --min_lr 1e-6 --weight_decay 5e-6 \
 --output_dir "$OUTPUT_DIR" \
 --seed 42 \
 --setting_id $SETTING_ID \
 > "$OUTPUT_DIR/output.log"
