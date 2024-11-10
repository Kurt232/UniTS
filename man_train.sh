#!/bin/bash

GPU=0
MODEL_CONFIG=configs/model_c2_h3_d256_nh4_nl6.json
DATA_CONFIG="/data/wjdu/data4/realworld/realworld_10_thigh_TRAIN.json /data/wjdu/data4/realworld/realworld_10_thigh_TEST.json"
OUTPUT_DIR="/data/wjdu/TST_HEAD_10"
MASTER_PORT=2112

# Create a unique output directory based on the model config filename
MODEL_NAME=$(basename "${MODEL_CONFIG%.*}")
OUTPUT_DIR="${OUTPUT_DIR}/${MODEL_NAME}"

mkdir -p "$OUTPUT_DIR"

echo "Training with model config: $MODEL_CONFIG on GPU: $GPU with MASTER_PORT: $MASTER_PORT"
echo "Data config: $DATA_CONFIG"
echo "Output directory: $OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=$GPU python -u -m torch.distributed.launch --master_port=$MASTER_PORT --nproc_per_node=1 --use_env \
 train.py --data_config $DATA_CONFIG --model_config $MODEL_CONFIG --batch_size 512 \
 --epochs 200 --warmup_epochs 20 --blr 1e-4 --min_lr 1e-6 --weight_decay 5e-6 \
 --output_dir "$OUTPUT_DIR" \
 --seed 42 \
 > "$OUTPUT_DIR/output.log" 2>&1
