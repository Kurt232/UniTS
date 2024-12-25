#!/usr/bin/env bash
set -e  # Exit immediately if a command exits with a non-zero status

ROOT="/data/wjdu/benchmark1"

# Check if all required arguments are provided
if [ "$#" -lt 4 ]; then
    echo "Usage: $0 MODEL MARK HETER TARGETS"
    exit 1
fi

MODEL=$1
MARK=$2
HETER=$3
IFS=',' read -r -a TARGETS <<< "$4"
ROOT_FLAG=$5

# MODEL="UniTS_HEAD"
# MARK=""
# HETER="fre"
# TARGETS=("20_2" "50_2")
# HETER="dur"
# TARGETS=("50_1" "50_2" "50_4")
# HETER="loc"
# TARGETS=("50_2")

CONFIGS="data/eval/${HETER}"
# Count total number of tasks
TASK_LEN=$(ls $CONFIGS/*.yaml | wc -l)

for TARGET in "${TARGETS[@]}"; do

    TRAIN_ROOT="${ROOT}/output/${MODEL}${MARK}"
    OUTPUT_ROOT="${ROOT}/heter${ROOT_FLAG}/${MODEL}${MARK}/${HETER}/${TARGET}"

    # Initialize loop index
    CURRENT_IDX=0

    for DATA_CONFIG in $CONFIGS/*.yaml; do
        # Increment loop index
        CURRENT_IDX=$((CURRENT_IDX + 1))

        FLAG=$(basename ${DATA_CONFIG%.yaml})
        if [[ $FLAG == _* ]]; then
            continue
        fi
        TRAIN_DIR="${TRAIN_ROOT}/${TARGET}_${FLAG}"
        OUTPUT_DIR="${OUTPUT_ROOT}/${FLAG}"
        
        # if exists OUTPUT_DIR, skip
        if [ -d "$OUTPUT_DIR" ]; then
            echo "skip $OUTPUT_DIR"
            continue
        fi
        mkdir -p "$OUTPUT_DIR"

        echo "Task: $CURRENT_IDX/$TASK_LEN"
        echo "Data config: $DATA_CONFIG"
        echo "Output directory: $TRAIN_DIR"

        python infer1.py -l "$TRAIN_DIR" -d "$DATA_CONFIG" -o "$OUTPUT_DIR" -t $TARGET >> "${OUTPUT_DIR}/output.log"
        python ${ROOT}/eval.py "$OUTPUT_DIR" >> "${OUTPUT_DIR}/output.log"
    done
done