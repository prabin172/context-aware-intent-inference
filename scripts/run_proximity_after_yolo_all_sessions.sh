#!/usr/bin/env bash
set -e

TRIALS_ROOT="data/trials"
SEG_ROOT="data/xsens_segments"
OUT_ROOT="results/proximity"

for DET_CSV in "$OUT_ROOT"/*/yolo_detections.csv; do
    SESSION_DIR=$(dirname "$DET_CSV")
    SESSION_ID=$(basename "$SESSION_DIR")

    TRIALS_CSV="$TRIALS_ROOT/${SESSION_ID}_trials.csv"
    SEG_POS_CSV="$SEG_ROOT/${SESSION_ID}_Segment Position.csv"
    SEG_ORI_CSV="$SEG_ROOT/${SESSION_ID}_Segment Orientation - Euler.csv"

    OBJ_CSV="$SESSION_DIR/object_positions_by_trial.csv"
    MAPPED_CSV="$SESSION_DIR/mapped_distances_by_trial.csv"
    PROX_CSV="$SESSION_DIR/proximity.csv"

    PLOT_DIR="$SESSION_DIR/distance_plots"

    echo "========================================"
    echo "Session: $SESSION_ID"
    echo "========================================"

    if [ -f "$PROX_CSV" ]; then
        echo "SKIP: proximity.csv already exists: $PROX_CSV"
        continue
    fi

    if [ ! -f "$TRIALS_CSV" ]; then
        echo "SKIP: missing trials CSV: $TRIALS_CSV"
        continue
    fi

    if [ ! -f "$SEG_POS_CSV" ]; then
        echo "SKIP: missing segment position CSV: $SEG_POS_CSV"
        continue
    fi

    if [ ! -f "$SEG_ORI_CSV" ]; then
        echo "SKIP: missing segment orientation CSV: $SEG_ORI_CSV"
        continue
    fi

    echo "Step 1/3: Extract object depths by trial"
    python src/proximity_mapping/extractObjectPositions.py \
        --trials_csv "$TRIALS_CSV" \
        --detections_csv "$DET_CSV" \
        --output_csv "$OBJ_CSV" \
        --skip_first_n_trials 0

    echo "Step 2/3: Map human-object distances"
    python src/proximity_mapping/mapper.py \
        --trials_csv "$TRIALS_CSV" \
        --object_positions_csv "$OBJ_CSV" \
        --segment_position_csv "$SEG_POS_CSV" \
        --segment_orientation_csv "$SEG_ORI_CSV" \
        --output_csv "$MAPPED_CSV" \
        --output_plot_dir "$PLOT_DIR" \
        --skip_first_n_trials 0

    echo "Step 3/3: Normalize time to create proximity.csv"
    python src/proximity_mapping/normalizeTime.py \
        --input_csv "$MAPPED_CSV" \
        --output_csv "$PROX_CSV"

    echo "DONE: $PROX_CSV"
done

echo "All available proximity sessions processed."
