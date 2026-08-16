# Sequence-01 detection benchmark

This directory is the versioned output of:

```powershell
cell-detect run
```

The run evaluates every raw frame in `PhC-C2DL-PSC/01` against the corresponding
human `01_GT/TRA` marker mask. Gold is never an input to detection. Events are
matched independently per frame with maximum-cardinality, minimum-distance
one-to-one assignment inside the fixed 10 px gate.

Headline result: **micro centroid F1 = 0.778** (precision 0.862, recall 0.709;
50,622 TP, 8,104 FP, 20,781 FN). Exact parameters, input fingerprints, metric
definitions, and 5/10/15 px sensitivity are in `summary.json`.

The overview deliberately includes early and crowded late frames. Green circles
are matched predictions, orange circles are false positives, and magenta crosses
are missed human-gold events.
