#!/usr/bin/env bash
set -euo pipefail

MATCHED_ROOT="results/e2e_transformer_matched_gr"
EVAL_ROOT="results/evaluation"
REPORT_DIR="docs/experiment_reports/e2e_transformer_loso_pipeline"

echo "Matching E2E predictions to GR test-trial set..."

rm -rf "$MATCHED_ROOT"
mkdir -p "$MATCHED_ROOT"

python - <<'PY'
from pathlib import Path
import pandas as pd

subjects = ["sub2b", "sub3", "sub4", "sub5", "sub6"]

e2e_root = Path("results/e2e_transformer")
gr_root = Path("results/gr")
matched_root = Path("results/e2e_transformer_matched_gr")

summary_rows = []

for subject in subjects:
    e2e_path = e2e_root / f"inference_{subject}" / "e2e_predictions_all_test_trials.csv"
    gr_path = gr_root / f"inference_{subject}" / "gr_predictions_all_test_trials.csv"

    out_dir = matched_root / f"inference_{subject}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "e2e_predictions_all_test_trials.csv"

    e2e = pd.read_csv(e2e_path)
    gr = pd.read_csv(gr_path)

    key_cols = ["Session", "Global Trial"]
    gr_keys = gr[key_cols].drop_duplicates()
    filtered = e2e.merge(gr_keys, on=key_cols, how="inner")

    filtered.to_csv(out_path, index=False)

    summary_rows.append({
        "Subject": subject,
        "e2e_before_rows": len(e2e),
        "e2e_before_trials": e2e["Global Trial"].nunique(),
        "gr_trials": gr["Global Trial"].nunique(),
        "e2e_after_rows": len(filtered),
        "e2e_after_trials": filtered["Global Trial"].nunique(),
    })

    print()
    print("=" * 80)
    print(subject)
    print("E2E before rows/trials:", len(e2e), e2e["Global Trial"].nunique())
    print("GR trials:", gr["Global Trial"].nunique())
    print("E2E after rows/trials:", len(filtered), filtered["Global Trial"].nunique())
    print("by session after:")
    print(filtered.groupby("Session")["Global Trial"].nunique().sort_index())

summary = pd.DataFrame(summary_rows)
summary.to_csv(matched_root / "matching_summary.csv", index=False)

print()
print("Saved matching summary:", matched_root / "matching_summary.csv")
PY

echo
echo "Evaluating matched E2E predictions..."

python -m src.evaluation.evaluate_predictions \
  --input_glob "${MATCHED_ROOT}/inference_*/e2e_predictions_all_test_trials.csv" \
  --method_name e2e_transformer_dynamic_matched_gr \
  --output_dir "$EVAL_ROOT" \
  --label_mode dynamic \
  --onset_time_ms 2000 \
  --max_time_ms 2500

echo
echo "Updating report files..."

mkdir -p "$REPORT_DIR"

cp "${EVAL_ROOT}/e2e_transformer_dynamic_matched_gr_overall_metrics.csv" "$REPORT_DIR/e2e_overall_metrics.csv"
cp "${EVAL_ROOT}/e2e_transformer_dynamic_matched_gr_subject_metrics.csv" "$REPORT_DIR/e2e_subject_metrics.csv"
cp "${EVAL_ROOT}/e2e_transformer_dynamic_matched_gr_per_class_metrics.csv" "$REPORT_DIR/e2e_per_class_metrics.csv"
cp "${EVAL_ROOT}/e2e_transformer_dynamic_matched_gr_fixed_time_metrics.csv" "$REPORT_DIR/e2e_fixed_time_metrics.csv"
cp "${MATCHED_ROOT}/matching_summary.csv" "$REPORT_DIR/e2e_matching_summary.csv"

python - <<'PY'
from pathlib import Path

p = Path("docs/experiment_reports/e2e_transformer_loso_pipeline/README.md")
text = p.read_text()

marker = "## Matched-GR evaluation note"
if marker in text:
    text = text.split(marker)[0].rstrip()

note = """## Matched-GR evaluation note

The final reported E2E metrics in this folder are filtered to the exact same subject/session/global-trial set used by the GR and Bayesian Fusion outputs.

This removes 33 extra sub4-002 trials that were present in the raw E2E inference output but not present in the GR/BF test outputs.

Final matched trial count: 1029 trials.

Use these matched metrics for fair GR vs BF vs E2E comparison.
"""

p.write_text(text.rstrip() + "\n\n" + note)
print("Updated README:", p)
PY

echo
echo "Done."
