#!/usr/bin/env bash
set -e

SUBJECTS=("sub2b" "sub3" "sub4" "sub5" "sub6")
EPOCHS_BASE=100
EPOCHS_FINETUNE=25

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0

mkdir -p logs

echo "Creating/updating GR subject split manifests..."
python scripts/create_gr_subject_splits.py

for SUBJECT in "${SUBJECTS[@]}"; do
  echo
  echo "============================================================"
  echo "GR pipeline for target subject: ${SUBJECT}"
  echo "============================================================"

  BASE_DIR="results/gr/base_${SUBJECT}"
  FINETUNE_DIR="results/gr/finetuned_${SUBJECT}"
  INFER_DIR="results/gr/inference_${SUBJECT}"
  SPLIT_CSV="results/gr/splits/${SUBJECT}_trial_split.csv"

  BASE_MODEL="${BASE_DIR}/transformer_gr_leaveout_${SUBJECT}.keras"
  FINETUNED_MODEL="${FINETUNE_DIR}/transformer_gr_finetuned_${SUBJECT}.keras"
  INFER_COMBINED="${INFER_DIR}/gr_predictions_all_test_trials.csv"

  if [ -f "${BASE_MODEL}" ]; then
    echo "Skipping base training; found ${BASE_MODEL}"
  else
    echo "Running base training for ${SUBJECT}..."
    python -m src.gr.train \
      --target_subject "${SUBJECT}" \
      --epochs "${EPOCHS_BASE}" \
      --output_dir "${BASE_DIR}" \
      --no_merge_sit \
      2>&1 | tee "logs/gr_base_${SUBJECT}.log"
  fi

  if [ -f "${FINETUNED_MODEL}" ]; then
    echo "Skipping fine-tuning; found ${FINETUNED_MODEL}"
  else
    echo "Running fine-tuning for ${SUBJECT}..."
    python -m src.gr.fine_tune \
      --target_subject "${SUBJECT}" \
      --base_model_dir "${BASE_DIR}" \
      --split_csv "${SPLIT_CSV}" \
      --output_dir "${FINETUNE_DIR}" \
      --epochs "${EPOCHS_FINETUNE}" \
      --no_merge_sit \
      2>&1 | tee "logs/gr_finetune_${SUBJECT}.log"
  fi

  if [ -f "${INFER_COMBINED}" ]; then
    echo "Skipping inference; found ${INFER_COMBINED}"
  elif [ -d "${INFER_DIR}" ] && [ -n "$(find "${INFER_DIR}" -type f -name '*.csv' -print -quit 2>/dev/null)" ]; then
    echo "Inference output folder exists but combined CSV is missing."
    echo "Assuming partial/in-progress inference. Skipping ${SUBJECT} to avoid duplicate work."
    echo "Folder: ${INFER_DIR}"
  else
    echo "Running inference for ${SUBJECT}..."
    python -m src.gr.inference \
      --target_subject "${SUBJECT}" \
      --model_dir "${FINETUNE_DIR}" \
      --base_model_dir "${BASE_DIR}" \
      --split_csv "${SPLIT_CSV}" \
      --output_dir "${INFER_DIR}" \
      2>&1 | tee "logs/gr_inference_${SUBJECT}.log"
  fi
done

echo
echo "All requested GR subject pipelines finished or skipped."
