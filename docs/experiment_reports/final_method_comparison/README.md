# Final method comparison report

This report compares GR, Bayesian Fusion, and E2E Transformer on the cleaned 9-class split-sit evaluation.

The E2E predictions are evaluated on the same 1029 test trials used for GR and Bayesian Fusion.

Generated files:

- final_overall_metrics.csv
- final_subject_metrics.csv
- final_per_class_f1.csv
- final_early_prediction_lead_time.csv
- final_early_prediction_summary.csv
- final_ambiguity_metrics.csv
- final_conditional_bf_threshold_sweep.csv
- final_common_grid_summary.csv

Notes:

- Dynamic ground truth uses Nothing before the onset threshold and the action label after onset.
- Early prediction lead time is computed as onset time minus the first time the true action remains the top predicted class for the stability window.
- Ambiguous timestamps are defined using the GR confidence gap between the top two predictions.
- Conditional BF uses GR by default and switches to BF when the GR confidence gap is below the threshold.
