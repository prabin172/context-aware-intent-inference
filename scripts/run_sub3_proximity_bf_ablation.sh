#!/usr/bin/env bash
set -euo pipefail

SESSION_ID="sub3"

RGBD_DIR="data/Synced-color-depthPNG/${SESSION_ID}"
TRIALS_CSV="data/trials/${SESSION_ID}_trials.csv"
SEG_POS_CSV="data/xsens_segments/${SESSION_ID}_Segment Position.csv"
SEG_ORI_CSV="data/xsens_segments/${SESSION_ID}_Segment Orientation - Euler.csv"

OLD_PROX="context-a-if/scripts/BF/old_proximity.csv"
BASELINE_PROX="results/proximity/${SESSION_ID}/proximity.csv"

PROX_ABL_ROOT="results/proximity_ablation"
BF_ABL_ROOT="results/bayesian_fusion_ablation"
EVAL_ABL_ROOT="results/evaluation_ablation"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/sub3_proximity_bf_ablation.log"

mkdir -p "$LOG_DIR"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "Sub3 proximity + BF ablation"
echo "Started: $(date)"
echo "============================================================"
echo "Session: $SESSION_ID"
echo "RGBD dir: $RGBD_DIR"
echo "Trials CSV: $TRIALS_CSV"
echo "Segment position CSV: $SEG_POS_CSV"
echo "Segment orientation CSV: $SEG_ORI_CSV"
echo "Log file: $LOG_FILE"
echo

run_variant () {
    local VARIANT_NAME="$1"
    local MODEL_PATH="$2"
    local CONF="$3"

    local OUT_ROOT="${PROX_ABL_ROOT}/${VARIANT_NAME}"
    local OUT_DIR="${OUT_ROOT}/${SESSION_ID}"

    local DET_CSV="${OUT_DIR}/yolo_detections.csv"
    local OBJ_CSV="${OUT_DIR}/object_positions_by_trial.csv"
    local MAPPED_CSV="${OUT_DIR}/mapped_distances_by_trial.csv"
    local PROX_CSV="${OUT_DIR}/proximity.csv"
    local PLOT_DIR="${OUT_DIR}/distance_plots"
    local COVERAGE_CSV="${OUT_DIR}/coverage_report.csv"

    local BF_OUT_ROOT="${BF_ABL_ROOT}/${VARIANT_NAME}"
    local EVAL_OUT_DIR="${EVAL_ABL_ROOT}/${VARIANT_NAME}"

    echo
    echo "============================================================"
    echo "Variant: $VARIANT_NAME"
    echo "Model:   $MODEL_PATH"
    echo "Conf:    $CONF"
    echo "Output:  $OUT_DIR"
    echo "Time:    $(date)"
    echo "============================================================"

    mkdir -p "$OUT_DIR" "$BF_OUT_ROOT" "$EVAL_OUT_DIR"

    if [ ! -f "$MODEL_PATH" ]; then
        echo "ERROR: missing model: $MODEL_PATH"
        exit 1
    fi

    if [ ! -f "$DET_CSV" ]; then
        echo "Step 1/6: YOLO + depth detections"
        python src/proximity_mapping/processRawData.py \
            --rgbd_dir "$RGBD_DIR" \
            --yolo_model "$MODEL_PATH" \
            --output_csv "$DET_CSV" \
            --conf "$CONF" \
            --n_clusters 3
    else
        echo "Step 1/6: SKIP existing detections: $DET_CSV"
    fi

    if [ ! -f "$OBJ_CSV" ]; then
        echo "Step 2/6: Extract object positions by trial"
        python src/proximity_mapping/extractObjectPositions.py \
            --trials_csv "$TRIALS_CSV" \
            --detections_csv "$DET_CSV" \
            --output_csv "$OBJ_CSV" \
            --skip_first_n_trials 0
    else
        echo "Step 2/6: SKIP existing object positions: $OBJ_CSV"
    fi

    if [ ! -f "$MAPPED_CSV" ]; then
        echo "Step 3/6: Map human-object distances"
        python src/proximity_mapping/mapper.py \
            --trials_csv "$TRIALS_CSV" \
            --object_positions_csv "$OBJ_CSV" \
            --segment_position_csv "$SEG_POS_CSV" \
            --segment_orientation_csv "$SEG_ORI_CSV" \
            --output_csv "$MAPPED_CSV" \
            --output_plot_dir "$PLOT_DIR" \
            --skip_first_n_trials 0
    else
        echo "Step 3/6: SKIP existing mapped distances: $MAPPED_CSV"
    fi

    if [ ! -f "$PROX_CSV" ]; then
        echo "Step 4/6: Normalize time to proximity.csv"
        python src/proximity_mapping/normalizeTime.py \
            --input_csv "$MAPPED_CSV" \
            --output_csv "$PROX_CSV"
    else
        echo "Step 4/6: SKIP existing proximity: $PROX_CSV"
    fi

    echo "Step 5/6: Coverage report"
    python - <<PY
import pandas as pd
from pathlib import Path

variant = "${VARIANT_NAME}"
prox_path = Path("${PROX_CSV}")
det_path = Path("${DET_CSV}")
coverage_path = Path("${COVERAGE_CSV}")

dist_cols = [
    "Distance Chair",
    "Distance Couch",
    "Distance Dining Table",
    "Distance Backpack",
]

df = pd.read_csv(prox_path)
det = pd.read_csv(det_path)

rows = []
for c in dist_cols:
    rows.append({
        "variant": variant,
        "file": str(prox_path),
        "rows": len(df),
        "trials": df["Trial Number"].nunique() if "Trial Number" in df.columns else None,
        "distance_col": c,
        "non_null_rate": df[c].notna().mean() if c in df.columns else None,
        "nan_rate": df[c].isna().mean() if c in df.columns else None,
        "min": df[c].min() if c in df.columns else None,
        "median": df[c].median() if c in df.columns else None,
        "max": df[c].max() if c in df.columns else None,
    })

report = pd.DataFrame(rows)
report.to_csv(coverage_path, index=False)

print("Detection rows:", len(det))
if "YOLO Class" in det.columns:
    print("YOLO class counts:")
    print(det["YOLO Class"].value_counts(dropna=False).to_string())

print()
print("Coverage report:")
print(report.to_string(index=False))
PY

    echo "Step 6/6: Bayesian Fusion + evaluation"
    rm -rf "$BF_OUT_ROOT"

    python -m src.bayesian_fusion.run_bayesian_fusion \
        --subjects "$SESSION_ID" \
        --gr_root results/gr \
        --proximity_root "$OUT_ROOT" \
        --output_root "$BF_OUT_ROOT" \
        --save_merged_inputs

    python -m src.evaluation.evaluate_predictions \
        --input_glob "${BF_OUT_ROOT}/inference_*/bf_predictions_all_test_trials.csv" \
        --method_name "${VARIANT_NAME}_bf_dynamic" \
        --output_dir "$EVAL_OUT_DIR" \
        --label_mode dynamic \
        --onset_time_ms 2000 \
        --max_time_ms 2500

    echo
    echo "Variant complete: $VARIANT_NAME"
    echo "Coverage: $COVERAGE_CSV"
    echo "BF output: ${BF_OUT_ROOT}/inference_${SESSION_ID}/bf_predictions_all_test_trials.csv"
    echo "Eval output: $EVAL_OUT_DIR"
}

echo
echo "Reference coverage: old proximity vs current baseline"
python - <<'PY'
import pandas as pd
from pathlib import Path

files = [
    ("old_proximity", Path("context-a-if/scripts/BF/old_proximity.csv")),
    ("current_baseline", Path("results/proximity/sub3/proximity.csv")),
]

dist_cols = [
    "Distance Chair",
    "Distance Couch",
    "Distance Dining Table",
    "Distance Backpack",
]

for name, path in files:
    if not path.exists():
        print(name, "missing:", path)
        continue

    df = pd.read_csv(path)
    print()
    print("=" * 80)
    print(name, path)
    print("shape:", df.shape)
    if "Trial Number" in df.columns:
        print("trials:", df["Trial Number"].nunique(), "min/max:", df["Trial Number"].min(), df["Trial Number"].max())
    if "Time (ms)" in df.columns:
        print("time min/max:", df["Time (ms)"].min(), df["Time (ms)"].max())

    for c in dist_cols:
        print(f"{c}: non-null={df[c].notna().mean():.3f}, nan={df[c].isna().mean():.3f}")
PY

run_variant "sub3_yolov8ex_conf04" "data/models/yolov8-ex-finetuned.pt" "0.4"
run_variant "sub3_yolov8lFT_conf06" "context-a-if/yolov8lFT.pt" "0.6"
run_variant "sub3_yolov8lFT_conf04" "context-a-if/yolov8lFT.pt" "0.4"

echo
echo "============================================================"
echo "Summary across variants"
echo "============================================================"

python - <<'PY'
import glob
from pathlib import Path
import pandas as pd

rows = []

for cov_path in sorted(glob.glob("results/proximity_ablation/*/sub3/coverage_report.csv")):
    cov = pd.read_csv(cov_path)
    variant = cov["variant"].iloc[0]
    row = {"variant": variant}
    for _, r in cov.iterrows():
        key = r["distance_col"].replace("Distance ", "").replace("Dining Table", "Table").replace(" ", "_")
        row[f"{key}_non_null"] = r["non_null_rate"]
    rows.append(row)

coverage_summary = pd.DataFrame(rows)

metric_rows = []
for overall_path in sorted(glob.glob("results/evaluation_ablation/*/*_overall_metrics.csv")):
    p = Path(overall_path)
    variant = p.parent.name
    df = pd.read_csv(p)
    row = df.iloc[0].to_dict()
    row["variant"] = variant
    metric_rows.append(row)

metrics_summary = pd.DataFrame(metric_rows)

print()
print("Coverage summary:")
if not coverage_summary.empty:
    print(coverage_summary.to_string(index=False))
else:
    print("No coverage reports found.")

print()
print("BF evaluation summary:")
if not metrics_summary.empty:
    cols = ["variant", "accuracy", "macro_f1", "weighted_f1", "num_rows", "num_trials"]
    print(metrics_summary[cols].to_string(index=False))
else:
    print("No metric files found.")

Path("results/evaluation_ablation").mkdir(parents=True, exist_ok=True)
coverage_summary.to_csv("results/evaluation_ablation/sub3_ablation_coverage_summary.csv", index=False)
metrics_summary.to_csv("results/evaluation_ablation/sub3_ablation_metric_summary.csv", index=False)
PY

echo
echo "Finished: $(date)"
echo "Log file: $LOG_FILE"
