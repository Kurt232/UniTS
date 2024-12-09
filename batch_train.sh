#!/usr/bin/env bash
set -e  # Exit immediately if a command exits with a non-zero status

GPUS="0,1,2,3,4,5,6,7"
# GPUS="4,5,6,7"

LOAD_PATH=""
MASTER_PORT=2233
NNODE=$(($(echo $GPUS | tr -cd , | wc -c) + 1))
ROOT="/data/wjdu/benchmark"
MODEL="UniTS_HEAD"
SETTING_ID=3
MARK="_3"

# Count total number of tasks
TASK_LEN=$(ls data/benchmark/*.yaml | wc -l)

# Initialize loop index
CURRENT_IDX=0

# read all configs from `data/benchmark/`
for DATA_CONFIG in data/benchmark/*.yaml; do
    # Increment loop index
    CURRENT_IDX=$((CURRENT_IDX + 1))

    DATA_CONFIG=${DATA_CONFIG##*/}
    DATA_CONFIG="data/benchmark/$DATA_CONFIG"
    FLAG=$(basename ${DATA_CONFIG%.yaml})
    TRAIN_DIR="${ROOT}/output/${MODEL}${MARK}/${MODEL}_${FLAG}"
    OUTPUT_DIR="${ROOT}/result/${MODEL}${MARK}/${MODEL}_${FLAG}"
    
    mkdir -p "$TRAIN_DIR" "$OUTPUT_DIR"

    echo "Task: $CURRENT_IDX/$TASK_LEN"
    echo "Data config: $DATA_CONFIG"
    echo "Output directory: $TRAIN_DIR"

    CUDA_VISIBLE_DEVICES="$GPUS" torchrun --nproc_per_node=$NNODE --master_port=$MASTER_PORT \
    train.py --data_config "$DATA_CONFIG" --load_path "$LOAD_PATH" --batch_size 1024 \
    --epochs 40 --warmup_epochs 10 --blr 1e-4 --min_lr 1e-6 --weight_decay 5e-6 \
    --output_dir "$TRAIN_DIR" \
    --seed 42 \
    --setting_id $SETTING_ID \
    > "$TRAIN_DIR/output.log"

    python infer.py -l "$TRAIN_DIR" -d "$DATA_CONFIG" -o "$OUTPUT_DIR" >> "${OUTPUT_DIR}/output.log"
    python ${ROOT}/eval.py "$OUTPUT_DIR" >> "${OUTPUT_DIR}/output.log"
done