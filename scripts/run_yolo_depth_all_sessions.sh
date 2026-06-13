#!/usr/bin/env bash
set -e

MODEL="data/models/yolov8lFT.pt"
RGBD_ROOT="data/Synced-color-depthPNG"
OUT_ROOT="results/proximity"

mkdir -p "$OUT_ROOT"

for RGBD_DIR in "$RGBD_ROOT"/*; do
    if [ ! -d "$RGBD_DIR" ]; then
        continue
    fi

    SESSION_ID=$(basename "$RGBD_DIR")
    OUT_DIR="$OUT_ROOT/$SESSION_ID"
    OUT_CSV="$OUT_DIR/yolo_detections.csv"

    mkdir -p "$OUT_DIR"

    echo "========================================"
    echo "Session: $SESSION_ID"
    echo "RGBD dir: $RGBD_DIR"
    echo "Output: $OUT_CSV"
    echo "========================================"

    if [ -f "$OUT_CSV" ]; then
        echo "Skipping $SESSION_ID because output already exists."
        continue
    fi

    python src/proximity_mapping/processRawData.py \
        --rgbd_dir "$RGBD_DIR" \
        --yolo_model "$MODEL" \
        --output_csv "$OUT_CSV" \
        --conf 0.6 \
        --n_clusters 3
done

echo "All YOLO + depth extraction jobs completed."
