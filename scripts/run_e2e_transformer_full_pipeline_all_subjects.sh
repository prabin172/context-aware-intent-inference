#!/usr/bin/env bash
set -euo pipefail

RUN_ID="e2e_transformer_loso_$(date +%Y%m%d_%H%M%S)"

OUTPUT_ROOT="${OUTPUT_ROOT:-results/e2e_transformer}"
EVAL_ROOT="${EVAL_ROOT:-results/evaluation}"
REPORT_DIR="${REPORT_DIR:-docs/experiment_reports/e2e_transformer_loso_pipeline}"

LOG_DIR="${LOG_DIR:-logs}"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"

SUBJECTS=("sub2b" "sub3" "sub4" "sub5" "sub6")

BASE_EPOCHS="${BASE_EPOCHS:-10}"
FT_EPOCHS="${FT_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEQUENCE_LEN="${SEQUENCE_LEN:-16}"

# Optional debug controls:
#   MAX_BATCHES=10 scripts/run_e2e_transformer_full_pipeline_all_subjects.sh
#   MAX_WINDOWS=1000 scripts/run_e2e_transformer_full_pipeline_all_subjects.sh
MAX_BATCHES="${MAX_BATCHES:-}"
MAX_WINDOWS="${MAX_WINDOWS:-}"

# Optional:
#   NO_PRETRAINED_CNN=1 scripts/run_e2e_transformer_full_pipeline_all_subjects.sh
NO_PRETRAINED_CNN="${NO_PRETRAINED_CNN:-0}"

mkdir -p "$LOG_DIR"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "E2E Transformer LOSO full pipeline"
echo "Run ID: $RUN_ID"
echo "Started: $(date)"
echo "Output root: $OUTPUT_ROOT"
echo "Evaluation root: $EVAL_ROOT"
echo "Report dir: $REPORT_DIR"
echo "Log file: $LOG_FILE"
echo "Base epochs: $BASE_EPOCHS"
echo "Fine-tune epochs: $FT_EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Num workers: $NUM_WORKERS"
echo "Sequence len: $SEQUENCE_LEN"
echo "No pretrained CNN: $NO_PRETRAINED_CNN"
echo "============================================================"

COMMON_ARGS=(
  --output_root "$OUTPUT_ROOT"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --sequence_len "$SEQUENCE_LEN"
)

if [ "$NO_PRETRAINED_CNN" = "1" ]; then
  COMMON_ARGS+=(--no_pretrained_cnn)
fi

TRAIN_DEBUG_ARGS=()
if [ -n "$MAX_BATCHES" ]; then
  TRAIN_DEBUG_ARGS+=(--max_batches "$MAX_BATCHES")
fi

INFER_DEBUG_ARGS=()
if [ -n "$MAX_WINDOWS" ]; then
  INFER_DEBUG_ARGS+=(--max_windows "$MAX_WINDOWS")
fi

echo
echo "Step 1: Run base training, fine-tuning, and inference for each LOSO subject"
for SUBJECT in "${SUBJECTS[@]}"; do
  echo
  echo "------------------------------------------------------------"
  echo "Subject: $SUBJECT"
  echo "------------------------------------------------------------"

  echo
  echo "Base training for target subject $SUBJECT"
  python -m src.e2e_transformer.train \
    --target_subject "$SUBJECT" \
    --epochs "$BASE_EPOCHS" \
    "${COMMON_ARGS[@]}" \
    "${TRAIN_DEBUG_ARGS[@]}"

  echo
  echo "Fine-tuning for target subject $SUBJECT"
  python -m src.e2e_transformer.fineTune \
    --target_subject "$SUBJECT" \
    --epochs "$FT_EPOCHS" \
    "${COMMON_ARGS[@]}" \
    "${TRAIN_DEBUG_ARGS[@]}"

  echo
  echo "Inference for target subject $SUBJECT"
  python -m src.e2e_transformer.predict_and_compare \
    --target_subject "$SUBJECT" \
    "${COMMON_ARGS[@]}" \
    "${INFER_DEBUG_ARGS[@]}"
done

echo
echo "Step 2: Evaluate E2E Transformer dynamic predictions"
python -m src.evaluation.evaluate_predictions \
  --input_glob "${OUTPUT_ROOT}/inference_*/e2e_predictions_all_test_trials.csv" \
  --method_name e2e_transformer_dynamic \
  --output_dir "$EVAL_ROOT" \
  --label_mode dynamic \
  --onset_time_ms 2000 \
  --max_time_ms 2500

echo
if [ -n "$MAX_BATCHES" ] || [ -n "$MAX_WINDOWS" ] || [ "$NO_PRETRAINED_CNN" = "1" ]; then
  echo
  echo "Debug mode detected because MAX_BATCHES/MAX_WINDOWS/NO_PRETRAINED_CNN is set."
  echo "Skipping GitHub report creation and commit for this debug run."
  echo "============================================================"
  echo "Finished E2E Transformer LOSO debug pipeline"
  echo "Finished: $(date)"
  echo "Run ID: $RUN_ID"
  echo "Log file: $LOG_FILE"
  echo "============================================================"
  exit 0
fi

echo "Step 3: Build lightweight GitHub report"
mkdir -p "$REPORT_DIR"

cp "${EVAL_ROOT}/e2e_transformer_dynamic_overall_metrics.csv" \
   "$REPORT_DIR/e2e_overall_metrics.csv"

cp "${EVAL_ROOT}/e2e_transformer_dynamic_subject_metrics.csv" \
   "$REPORT_DIR/e2e_subject_metrics.csv"

cp "${EVAL_ROOT}/e2e_transformer_dynamic_per_class_metrics.csv" \
   "$REPORT_DIR/e2e_per_class_metrics.csv"

cp "${EVAL_ROOT}/e2e_transformer_dynamic_fixed_time_metrics.csv" \
   "$REPORT_DIR/e2e_fixed_time_metrics.csv"

tail -n 500 "$LOG_FILE" > "$REPORT_DIR/log_tail.txt"

cat > "$REPORT_DIR/README.md" <<EOF
# E2E Transformer LOSO pipeline

Generated automatically by:

\`\`\`bash
scripts/run_e2e_transformer_full_pipeline_all_subjects.sh
\`\`\`

Run ID:

\`\`\`text
$RUN_ID
\`\`\`

This run performs the cleaned E2E Transformer pipeline:

1. Base LOSO training for each target subject
2. Target-subject calibration/fine-tuning using the same GR split CSVs
3. Test-set inference using the same GR split CSVs
4. Dynamic evaluation using \`src/evaluation/evaluate_predictions.py\`

Configuration:

\`\`\`text
BASE_EPOCHS=$BASE_EPOCHS
FT_EPOCHS=$FT_EPOCHS
BATCH_SIZE=$BATCH_SIZE
NUM_WORKERS=$NUM_WORKERS
SEQUENCE_LEN=$SEQUENCE_LEN
NO_PRETRAINED_CNN=$NO_PRETRAINED_CNN
\`\`\`

Tracked report files:

- \`e2e_overall_metrics.csv\`
- \`e2e_subject_metrics.csv\`
- \`e2e_per_class_metrics.csv\`
- \`e2e_fixed_time_metrics.csv\`
- \`log_tail.txt\`

Generated model/prediction outputs are local and ignored under:

- \`results/e2e_transformer/\`
EOF

echo
echo "Step 4: Commit and push lightweight report"
git add "$REPORT_DIR" scripts/run_e2e_transformer_full_pipeline_all_subjects.sh

if git diff --cached --quiet; then
  echo "No tracked report/script changes to commit."
else
  git commit -m "Add E2E Transformer LOSO pipeline report"
  git push
fi

echo
echo "============================================================"
echo "Finished E2E Transformer LOSO pipeline"
echo "Finished: $(date)"
echo "Run ID: $RUN_ID"
echo "Log file: $LOG_FILE"
echo "Report dir: $REPORT_DIR"
echo "============================================================"
