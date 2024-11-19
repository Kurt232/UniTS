#!/usr/bin/env bash
set -e  # Exit immediately if a command exits with a non-zero status

GPUS="4,5,6,7"

LOAD_PATH=""
MASTER_PORT=2333
NNODE=$(($(echo $GPUS | tr -cd , | wc -c) + 1))
ROOT="/data/wjdu/benchmark"
MODEL="LIMU_HEAD"

DATA_CONFIG="data/benchmark/up.yaml"
FLAG=$(basename ${DATA_CONFIG%.yaml})
TRAIN_DIR="${ROOT}/output/${MODEL}_1/${MODEL}_${FLAG}"
OUTPUT_DIR="${ROOT}/result/${MODEL}_1/${MODEL}_${FLAG}"

mkdir -p "$TRAIN_DIR" "$OUTPUT_DIR"

echo "Task: $CURRENT_IDX/$TASK_LEN"
echo "Data config: $DATA_CONFIG"
echo "Output directory: $TRAIN_DIR"

CUDA_VISIBLE_DEVICES="$GPUS" torchrun --nproc_per_node=$NNODE --master_port=$MASTER_PORT \
    train.py --data_config "$DATA_CONFIG" --load_path "$LOAD_PATH" --batch_size 1024 \
    --epochs 40 --warmup_epochs 10 --blr 1e-4 --min_lr 1e-6 --weight_decay 5e-6 \
    --output_dir "$TRAIN_DIR" \
    --seed 42 \
    --setting_id 2 \
    > "$TRAIN_DIR/output.log"

python infer.py -l "$TRAIN_DIR" -d "$DATA_CONFIG" -o "$OUTPUT_DIR" >> "${OUTPUT_DIR}/output.log"
python ${ROOT}/eval.py "$OUTPUT_DIR" >> "${OUTPUT_DIR}/output.log"