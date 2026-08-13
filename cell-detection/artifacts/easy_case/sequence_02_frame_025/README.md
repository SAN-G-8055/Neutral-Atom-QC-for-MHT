# Baseline result: sequence 02, frame 025

This directory contains the versioned output of:

```powershell
$env:PYTHONPATH = "src"
python -m cell_detection_pipeline easy-case
```

- `detections.csv`: one row per predicted cell for downstream point-pattern or hypothesis tests.
- `labels.tif`: predicted instance segmentation (`0` background, positive object IDs).
- `comparison_metrics.json`: human-gold, tracking-marker, and silver comparisons kept separate.
- `gold_comparison_overlay.png`: cyan prediction boundaries/dots; magenta gold boundaries;
  yellow crosses at human-gold centroids.
- `manifest.json`: input hash, parameters, thresholds, count, and runtime.

The raw input and source annotations remain local and are intentionally ignored by Git.
