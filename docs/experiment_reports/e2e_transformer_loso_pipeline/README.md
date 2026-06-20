# E2E Transformer LOSO pipeline

Generated automatically by:

```bash
scripts/run_e2e_transformer_full_pipeline_all_subjects.sh
```

Run ID:

```text
e2e_transformer_loso_20260613_131715
```

This run performs the cleaned E2E Transformer pipeline:

1. Base LOSO training for each target subject
2. Target-subject calibration/fine-tuning using the same GR split CSVs
3. Test-set inference using the same GR split CSVs
4. Dynamic evaluation using `src/evaluation/evaluate_predictions.py`

Configuration:

```text
BASE_EPOCHS=10
FT_EPOCHS=15
BATCH_SIZE=4
NUM_WORKERS=4
SEQUENCE_LEN=16
NO_PRETRAINED_CNN=0
```

Tracked report files:

- `e2e_overall_metrics.csv`
- `e2e_subject_metrics.csv`
- `e2e_per_class_metrics.csv`
- `e2e_fixed_time_metrics.csv`
- `log_tail.txt`

Generated model/prediction outputs are local and ignored under:

- `results/e2e_transformer/`

## Evaluation note

The final E2E metrics in this folder are evaluated on the same 1029 test trials used for the GR and Bayesian Fusion results.

The matching is done by subject/session/global-trial so that GR, Bayesian Fusion, and E2E can be compared on the same trial set.

Use these metrics for the final GR vs BF vs E2E comparison.
