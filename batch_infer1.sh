#!/usr/bin/env bash
set -e  # Exit immediately if a command exits with a non-zero status

FLAG=""
MODEL=UniTS_HEAD
MARKS=("_1")

for MARK in "${MARKS[@]}"; do
    bash infer1.sh "$MODEL" "$MARK" fre "20_2,50_2" "$FLAG"
    bash infer1.sh "$MODEL" "$MARK" dur "50_1,50_2,50_4" "$FLAG"
    bash infer1.sh "$MODEL" "$MARK" loc "50_2" "$FLAG"
done