#!/bin/bash

GPUS="0,1,2,3"

LOAD_PATH=""
DATA_CONFIG="data/test2.yaml"
TRAIN_DIR="/data/wjdu/multi2/test/UniTS_HEAD_2"
OUTPUT_DIR="/data/wjdu/multi2/test/res2"

MASTER_PORT=2233

NNODE=$(($(echo $GPUS | tr -cd , | wc -c) + 1))

mkdir -p "$TRAIN_DIR" "$OUTPUT_DIR"

echo "Training on GPU: $GPU with MASTER_PORT: $MASTER_PORT"
echo "Data config: $DATA_CONFIG"
echo "Output directory: $TRAIN_DIR"

CUDA_VISIBLE_DEVICES="$GPUS" torchrun --nproc_per_node=$NNODE --master_port=2111 \
 train.py --data_config "$DATA_CONFIG" --load_path "$LOAD_PATH" --batch_size 1024 \
 --epochs 40 --warmup_epochs 10 --blr 1e-4 --min_lr 1e-6 --weight_decay 5e-6 \
 --output_dir "$TRAIN_DIR" \
 --seed 42 \
 --setting_id 3 \
 > "$TRAIN_DIR/output.log"

python infer.py -l "$TRAIN_DIR" -d "$DATA_CONFIG" -o "$OUTPUT_DIR" >> "${OUTPUT_DIR}/output.log"
python /data/wjdu/multi2/eval.py "$OUTPUT_DIR" >> "${OUTPUT_DIR}/output.log"
