#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "Running final paper plot scripts..."

python scripts/plots/plot_early_prediction_boxplot.py
python scripts/plots/plot_per_class_f1_all_methods.py
python scripts/plots/plot_ambiguity_macro_f1_bar.py

echo
echo "Final plots saved under:"
echo "  docs/experiment_reports/final_method_comparison/"
