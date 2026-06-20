# Final method comparison report

This folder contains the final paper-facing evaluation summary and figures comparing:

- Gesture Recognition (GR)
- Bayesian Fusion (BF)
- End-to-End Transformer (E2E)

The comparison uses the cleaned 9-class split-sit label space under the full leave-one-subject-out (LOSO) evaluation.

## Main files

- `final_evaluation_results.txt`  
  Human-readable summary of the values reported in the paper, including:
  - subject-level accuracy, macro-F1, and weighted-F1
  - statistical tests for overall metrics
  - early prediction lead-time results and paired tests
  - per-class F1 values
  - ambiguity analysis
  - conditional switching analysis

- `figure7_early_prediction_boxplot.pdf`  
  Final early prediction lead-time plot.

- `per_class_f1_full_loso.pdf`  
  Final per-class F1 plot.

- `figure9_ambiguity_macro_f1.pdf`  
  Final ambiguity macro-F1 plot.

## Supporting CSV files

Some CSV files are generated as intermediate/supporting outputs by the final evaluation pipeline and are retained to make the reported values traceable.

## Notes

- Dynamic ground truth uses `Nothing` before the onset threshold and the action label after onset.
- Early prediction lead time is computed as onset time minus the first time the true action remains the top predicted class for the stability window.
- Lead time is evaluated only on matched successful trials where all compared methods produced sustained correct predictions.
- Ambiguous timestamps are defined using the GR confidence gap between the top two predictions.
- Conditional switching uses GR by default and invokes a multimodal fallback only when GR is ambiguous.
