#!/usr/bin/env bash
set -e  # Exit immediately if a command exits with a non-zero status

ROOT="/data/wjdu/benchmark"
MODEL="UniTS_HEAD"
MARK=""

# Count total number of tasks
TASK_LEN=$(ls data/benchmark/*.yaml | wc -l)

# Initialize loop index
CURRENT_IDX=0

if [ -d "${ROOT}/result/${MODEL}${MARK}" ]; then
    mv "${ROOT}/result/${MODEL}${MARK}" "${ROOT}/result/${MODEL}${MARK}_$(date +%m%d%H%M%S)"
fi

# read all configs from `data/benchmark/`
for DATA_CONFIG in data/benchmark/*.yaml; do
    # Increment loop index
    CURRENT_IDX=$((CURRENT_IDX + 1))

    DATA_CONFIG=${DATA_CONFIG##*/}
    DATA_CONFIG="data/benchmark/$DATA_CONFIG"
    FLAG=$(basename ${DATA_CONFIG%.yaml})
    TRAIN_DIR="${ROOT}/output/${MODEL}${MARK}/${MODEL}_${FLAG}/checkpoint-10.pth"
    OUTPUT_DIR="${ROOT}/result/${MODEL}${MARK}/${MODEL}_${FLAG}"

    mkdir -p "$OUTPUT_DIR"

    echo "Task: $CURRENT_IDX/$TASK_LEN"
    echo "Data config: $DATA_CONFIG"
    echo "Output directory: $TRAIN_DIR"

    python infer.py -l "$TRAIN_DIR" -d "$DATA_CONFIG" -o "$OUTPUT_DIR" >> "${OUTPUT_DIR}/output.log"
    python ${ROOT}/eval.py "$OUTPUT_DIR" >> "${OUTPUT_DIR}/output.log"
done