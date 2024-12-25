#!/usr/bin/env bash
set -e  # Exit immediately if a command exits with a non-zero status

GPUS="2,3,4,5,6,7"

ROOT="/data/wjdu/benchmark1"
MODEL=$1
SETTING_ID=$2
MARK=$3

MASTER_PORT=$4
NNODE=$(($(echo $GPUS | tr -cd , | wc -c) + 1))
CONFIGS="data/train1"

# for each dir in `$CONFIGS`
for CONFIG in $CONFIGS/*; do
    # check if the dir is a directory and not startswith `_`
    if [[ ! -d "$CONFIG" ]]; then
        continue
    fi

    CONFIG_NAME=$(basename $CONFIG)
    if [[ $CONFIG_NAME == _* ]]; then
        continue
    fi
    echo $CONFIG, $CONFIG_NAME

    # Count total number of tasks
    TASK_LEN=$(ls $CONFIG/*.yaml | wc -l)

    # Initialize loop index
    CURRENT_IDX=0

    # read all configs from `data/benchmark/`
    for DATA_CONFIG in $CONFIG/*.yaml; do
        # Increment loop index
        CURRENT_IDX=$((CURRENT_IDX + 1))

        FLAG=$(basename ${DATA_CONFIG%.yaml})
        TRAIN_DIR="${ROOT}/output/${MODEL}${MARK}/${CONFIG_NAME}_${FLAG}"
        OUTPUT_DIR="${ROOT}/result/${CONFIG_NAME}/${MODEL}${MARK}/${MODEL}_${FLAG}"
        
        # if exists TRAIN_DIR, skip
        if [ -d "$TRAIN_DIR" ]; then
            continue
        fi
        mkdir -p "$TRAIN_DIR"

        echo "Task: $CURRENT_IDX/$TASK_LEN"
        echo "Data config: $DATA_CONFIG"
        echo "Output directory: $TRAIN_DIR"

        CUDA_VISIBLE_DEVICES="$GPUS" torchrun --nproc_per_node=$NNODE --master_port=$MASTER_PORT \
        train.py --data_config "$DATA_CONFIG" --load_path "$LOAD_PATH" --batch_size 1024 \
        --epochs 40 --warmup_epochs 10 --blr 1e-4 --min_lr 1e-6 --weight_decay 5e-6 \
        --output_dir "$TRAIN_DIR" \
        --seed 42 \
        --setting_id $SETTING_ID \
        > "$TRAIN_DIR"/output.log
    done
done