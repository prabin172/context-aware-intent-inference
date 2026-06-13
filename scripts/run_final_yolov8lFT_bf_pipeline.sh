#!/usr/bin/env bash
set -euo pipefail

RUN_ID="final_yolov8lFT_conf06_$(date +%Y%m%d_%H%M%S)"

MODEL_SRC="context-a-if/yolov8lFT.pt"
MODEL_DST="data/models/yolov8lFT.pt"
OLD_MODEL="data/models/yolov8-ex-finetuned.pt"

RGBD_ROOT="data/Synced-color-depthPNG"
TRIALS_ROOT="data/trials"
SEG_ROOT="data/xsens_segments"

PROX_ROOT="results/proximity"
BF_ROOT="results/bayesian_fusion"
EVAL_ROOT="results/evaluation"

ARCHIVE_ROOT="results/archive/${RUN_ID}"
REPORT_DIR="docs/experiment_reports/final_yolov8lFT_conf06_pipeline"

LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"

mkdir -p "$LOG_DIR"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "Final YOLOv8lFT proximity + Bayesian Fusion pipeline"
echo "Run ID: $RUN_ID"
echo "Started: $(date)"
echo "============================================================"
echo "Model source: $MODEL_SRC"
echo "Model destination: $MODEL_DST"
echo "Proximity output root: $PROX_ROOT"
echo "BF output root: $BF_ROOT"
echo "Evaluation output root: $EVAL_ROOT"
echo "Archive root: $ARCHIVE_ROOT"
echo "Report dir: $REPORT_DIR"
echo "Log file: $LOG_FILE"
echo

echo "Step 0: Model setup"
mkdir -p data/models
if [ ! -f "$MODEL_DST" ]; then
    echo "Copying YOLOv8lFT model into data/models/"
    cp "$MODEL_SRC" "$MODEL_DST"
else
    echo "Model already exists: $MODEL_DST"
fi

echo
echo "Model files:"
ls -lh "$MODEL_DST"
sha256sum "$MODEL_DST"

echo
echo "Step 1: Archive old outputs"
mkdir -p "$ARCHIVE_ROOT"

if [ -d "$PROX_ROOT" ]; then
    echo "Archiving old proximity results to $ARCHIVE_ROOT/proximity"
    mv "$PROX_ROOT" "$ARCHIVE_ROOT/proximity"
fi

if [ -d "$BF_ROOT" ]; then
    echo "Archiving old Bayesian Fusion results to $ARCHIVE_ROOT/bayesian_fusion"
    mv "$BF_ROOT" "$ARCHIVE_ROOT/bayesian_fusion"
fi

if [ -f "$OLD_MODEL" ]; then
    mkdir -p "$ARCHIVE_ROOT/models"
    echo "Archiving old YOLO ex model to $ARCHIVE_ROOT/models/"
    mv "$OLD_MODEL" "$ARCHIVE_ROOT/models/yolov8-ex-finetuned.pt"
fi

mkdir -p "$PROX_ROOT" "$BF_ROOT" "$EVAL_ROOT"

echo
echo "Step 2: Run YOLO + depth extraction for all sessions"
for RGBD_DIR in "$RGBD_ROOT"/*; do
    if [ ! -d "$RGBD_DIR" ]; then
        continue
    fi

    SESSION_ID=$(basename "$RGBD_DIR")
    OUT_DIR="$PROX_ROOT/$SESSION_ID"
    OUT_CSV="$OUT_DIR/yolo_detections.csv"

    mkdir -p "$OUT_DIR"

    echo
    echo "------------------------------------------------------------"
    echo "YOLO session: $SESSION_ID"
    echo "RGBD dir: $RGBD_DIR"
    echo "Output: $OUT_CSV"
    echo "------------------------------------------------------------"

    python src/proximity_mapping/processRawData.py \
        --rgbd_dir "$RGBD_DIR" \
        --yolo_model "$MODEL_DST" \
        --output_csv "$OUT_CSV" \
        --conf 0.6 \
        --n_clusters 3
done

echo
echo "Step 3: Run proximity mapping for all sessions"
for DET_CSV in "$PROX_ROOT"/*/yolo_detections.csv; do
    SESSION_DIR=$(dirname "$DET_CSV")
    SESSION_ID=$(basename "$SESSION_DIR")

    TRIALS_CSV="$TRIALS_ROOT/${SESSION_ID}_trials.csv"
    SEG_POS_CSV="$SEG_ROOT/${SESSION_ID}_Segment Position.csv"
    SEG_ORI_CSV="$SEG_ROOT/${SESSION_ID}_Segment Orientation - Euler.csv"

    OBJ_CSV="$SESSION_DIR/object_positions_by_trial.csv"
    MAPPED_CSV="$SESSION_DIR/mapped_distances_by_trial.csv"
    PROX_CSV="$SESSION_DIR/proximity.csv"
    PLOT_DIR="$SESSION_DIR/distance_plots"

    echo
    echo "------------------------------------------------------------"
    echo "Proximity session: $SESSION_ID"
    echo "------------------------------------------------------------"

    if [ ! -f "$TRIALS_CSV" ]; then
        echo "ERROR: missing trials CSV: $TRIALS_CSV"
        exit 1
    fi

    if [ ! -f "$SEG_POS_CSV" ]; then
        echo "ERROR: missing segment position CSV: $SEG_POS_CSV"
        exit 1
    fi

    if [ ! -f "$SEG_ORI_CSV" ]; then
        echo "ERROR: missing segment orientation CSV: $SEG_ORI_CSV"
        exit 1
    fi

    echo "Extract object depths by trial"
    python src/proximity_mapping/extractObjectPositions.py \
        --trials_csv "$TRIALS_CSV" \
        --detections_csv "$DET_CSV" \
        --output_csv "$OBJ_CSV" \
        --skip_first_n_trials 0

    echo "Map human-object distances"
    python src/proximity_mapping/mapper.py \
        --trials_csv "$TRIALS_CSV" \
        --object_positions_csv "$OBJ_CSV" \
        --segment_position_csv "$SEG_POS_CSV" \
        --segment_orientation_csv "$SEG_ORI_CSV" \
        --output_csv "$MAPPED_CSV" \
        --output_plot_dir "$PLOT_DIR" \
        --skip_first_n_trials 0

    echo "Normalize time"
    python src/proximity_mapping/normalizeTime.py \
        --input_csv "$MAPPED_CSV" \
        --output_csv "$PROX_CSV"

    echo "DONE: $PROX_CSV"
done

echo
echo "Step 4: Proximity coverage summary"
python - <<'PY'
import pandas as pd
from pathlib import Path

root = Path("results/proximity")
dist_cols = [
    "Distance Chair",
    "Distance Couch",
    "Distance Dining Table",
    "Distance Backpack",
]

rows = []
for prox_path in sorted(root.glob("*/proximity.csv")):
    session = prox_path.parent.name
    df = pd.read_csv(prox_path)

    row = {
        "session": session,
        "rows": len(df),
        "trials": df["Trial Number"].nunique() if "Trial Number" in df.columns else None,
        "time_min": df["Time (ms)"].min() if "Time (ms)" in df.columns else None,
        "time_max": df["Time (ms)"].max() if "Time (ms)" in df.columns else None,
    }

    for c in dist_cols:
        short = c.replace("Distance ", "").replace("Dining Table", "Table").replace(" ", "_")
        row[f"{short}_non_null"] = df[c].notna().mean()
        row[f"{short}_nan"] = df[c].isna().mean()

    rows.append(row)

summary = pd.DataFrame(rows)
out = Path("results/proximity/proximity_coverage_summary.csv")
summary.to_csv(out, index=False)

print(summary.to_string(index=False))
print()
print("Saved:", out)
PY

echo
echo "Step 5: Run Bayesian Fusion for all subjects"
rm -rf "$BF_ROOT"

python -m src.bayesian_fusion.run_bayesian_fusion \
    --subjects sub2b sub3 sub4 sub5 sub6 \
    --gr_root results/gr \
    --proximity_root "$PROX_ROOT" \
    --output_root "$BF_ROOT" \
    --save_merged_inputs

echo
echo "Step 6: Evaluate BF"
python -m src.evaluation.evaluate_predictions \
    --input_glob "results/bayesian_fusion/inference_*/bf_predictions_all_test_trials.csv" \
    --method_name bf_dynamic_yolov8lFT_conf06 \
    --output_dir "$EVAL_ROOT" \
    --label_mode dynamic \
    --onset_time_ms 2000 \
    --max_time_ms 2500

echo
echo "Step 7: Build lightweight GitHub report"
mkdir -p "$REPORT_DIR"

cp results/proximity/proximity_coverage_summary.csv \
   "$REPORT_DIR/proximity_coverage_summary.csv"

cp results/evaluation/bf_dynamic_yolov8lFT_conf06_overall_metrics.csv \
   "$REPORT_DIR/bf_overall_metrics.csv"

cp results/evaluation/bf_dynamic_yolov8lFT_conf06_subject_metrics.csv \
   "$REPORT_DIR/bf_subject_metrics.csv"

cp results/evaluation/bf_dynamic_yolov8lFT_conf06_per_class_metrics.csv \
   "$REPORT_DIR/bf_per_class_metrics.csv"

cp results/evaluation/bf_dynamic_yolov8lFT_conf06_fixed_time_metrics.csv \
   "$REPORT_DIR/bf_fixed_time_metrics.csv"

tail -n 500 "$LOG_FILE" > "$REPORT_DIR/log_tail.txt"

cat > "$REPORT_DIR/README.md" <<EOF
# Final YOLOv8lFT confidence 0.6 proximity + Bayesian Fusion run

Generated automatically by:

\`\`\`bash
scripts/run_final_yolov8lFT_bf_pipeline.sh
\`\`\`

Run ID:

\`\`\`text
$RUN_ID
\`\`\`

This run archives the previous proximity/BF outputs, uses:

\`\`\`text
data/models/yolov8lFT.pt
confidence = 0.6
\`\`\`

and reruns:

1. YOLO + depth detection
2. object-position extraction
3. 2D proximity mapping
4. Bayesian Fusion
5. dynamic BF evaluation from 0–2500 ms, with 0–2000 ms as Nothing and 2000–2500 ms as action

Tracked report files:

- \`proximity_coverage_summary.csv\`
- \`bf_overall_metrics.csv\`
- \`bf_subject_metrics.csv\`
- \`bf_per_class_metrics.csv\`
- \`bf_fixed_time_metrics.csv\`
- \`log_tail.txt\`

Full generated outputs are local and ignored under:

- \`results/proximity/\`
- \`results/bayesian_fusion/\`
- \`results/evaluation/\`
- \`results/archive/$RUN_ID/\`
EOF

echo
echo "Step 8: Commit and push lightweight report"
git add "$REPORT_DIR" scripts/run_yolo_depth_all_sessions.sh scripts/run_final_yolov8lFT_bf_pipeline.sh .gitignore

if git diff --cached --quiet; then
    echo "No tracked report/script changes to commit."
else
    git commit -m "Add final YOLOv8lFT Bayesian fusion pipeline report"
    git push
fi

echo
echo "============================================================"
echo "Finished final pipeline"
echo "Finished: $(date)"
echo "Run ID: $RUN_ID"
echo "Log file: $LOG_FILE"
echo "GitHub report dir: $REPORT_DIR"
echo "============================================================"
