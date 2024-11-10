#!/usr/bin/env bash

# Get all JSON configuration files in the "./configs" directory
MODEL_CONFIGS=(./configs/*.json)
DATA_CONFIG="/data/wjdu/data4/realworld/realworld_80_thigh_TRAIN.json /data/wjdu/data4/realworld/realworld_10_thigh_TEST.json"
OUTPUT_DIR="/data/wjdu/TST_HEAD_80"
MASTER_PORT_BASE=2112

# Specify the GPUs to use
GPUS=(4 5 6 7)  # Adjust this based on available GPUs
NUM_GPUS=${#GPUS[@]}

# Estimated training time in hours per job (adjust as needed)
ESTIMATED_TRAINING_TIME=0.5

for i in "${!MODEL_CONFIGS[@]}"; do
    MODEL_CONFIG="${MODEL_CONFIGS[$i]}"
    MASTER_PORT=$((MASTER_PORT_BASE + i))
    GPU_INDEX=$((i % NUM_GPUS))
    GPU=${GPUS[$GPU_INDEX]}  # Get the GPU index from the GPUS array

    echo "Starting training with config: $MODEL_CONFIG on GPU: $GPU"

    # Run the training script
    bash train.sh "$GPU" "$MODEL_CONFIG" "$DATA_CONFIG" "$OUTPUT_DIR" "$MASTER_PORT" & # Run in background

    # # Optional: Limit the number of concurrent jobs
    # if (( (i + 1) % NUM_GPUS == 0 )); then
    #     wait  # Wait for current batch of jobs to finish
    # fi
done

wait  # Wait for all background jobs to finish

echo "All training jobs have completed."
