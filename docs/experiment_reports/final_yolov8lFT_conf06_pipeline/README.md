# Final YOLOv8lFT confidence 0.6 proximity + Bayesian Fusion run

Generated automatically by:

```bash
scripts/run_final_yolov8lFT_bf_pipeline.sh
```

Run ID:

```text
final_yolov8lFT_conf06_20260612_202106
```

This run archives the previous proximity/BF outputs, uses:

```text
data/models/yolov8lFT.pt
confidence = 0.6
```

and reruns:

1. YOLO + depth detection
2. object-position extraction
3. 2D proximity mapping
4. Bayesian Fusion
5. dynamic BF evaluation from 0–2500 ms, with 0–2000 ms as Nothing and 2000–2500 ms as action

Tracked report files:

- `proximity_coverage_summary.csv`
- `bf_overall_metrics.csv`
- `bf_subject_metrics.csv`
- `bf_per_class_metrics.csv`
- `bf_fixed_time_metrics.csv`
- `log_tail.txt`

Full generated outputs are local and ignored under:

- `results/proximity/`
- `results/bayesian_fusion/`
- `results/evaluation/`
- `results/archive/final_yolov8lFT_conf06_20260612_202106/`
