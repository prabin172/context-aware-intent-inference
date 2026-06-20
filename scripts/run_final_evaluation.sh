#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "Running final paper evaluation..."

python src/evaluation/final_paper_evaluation.py

echo
echo "Final evaluation summary saved to:"
echo "  docs/experiment_reports/final_method_comparison/final_evaluation_results.txt"
