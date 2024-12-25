#!/usr/bin/env bash
set -e  # Exit immediately if a command exits with a non-zero status

ROOT="/data/wjdu/benchmark1"
MODEL="UniTS_HEAD"
MARK="_1_m"
CONFIGS="data/eval1/loc"
# Count total number of tasks
TASK_LEN=$(ls $CONFIGS/*.yaml | wc -l)

# Initialize loop index
CURRENT_IDX=0

if [ -d "${ROOT}/result/${MODEL}${MARK}" ]; then
    mv "${ROOT}/result/${MODEL}${MARK}" "${ROOT}/result/${MODEL}${MARK}_$(date +%m%d%H%M%S)"
fi

# read all configs from `data/benchmark/`
for DATA_CONFIG in $CONFIGS/*.yaml; do
    # Increment loop index
    CURRENT_IDX=$((CURRENT_IDX + 1))
    
    CONFIG_NAME=$(basename ${CONFIGS})
    FLAG=$(basename ${DATA_CONFIG%.yaml})
    if [[ $FLAG == _* ]]; then
        continue
    fi
    TRAIN_DIR="/data/wjdu/benchmark1/realworld/UniTS_HEAD_1_m/checkpoint-39.pth"
    OUTPUT_DIR="${ROOT}/result/${CONFIG_NAME}/${MODEL}${MARK}"

    mkdir -p "$OUTPUT_DIR"

    echo "Task: $CURRENT_IDX/$TASK_LEN"
    echo "Data config: $DATA_CONFIG"
    echo "Output directory: $TRAIN_DIR"

    python infer1.py -l "$TRAIN_DIR" -d "$DATA_CONFIG" -o "$OUTPUT_DIR" -t "50_2" >> "${OUTPUT_DIR}/output.log"
    python ${ROOT}/eval.py "$OUTPUT_DIR" >> "${OUTPUT_DIR}/output.log"
done