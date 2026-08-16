# Neutral-atom quantum computing for multiple-hypothesis tracking

This repository studies whether a neutral-atom maximum-weight-independent-set
solver can help multiple-hypothesis tracking (MHT). The project is being
refactored from two disconnected code trees into one reproducible pipeline. The
first completed stage is cell detection: raw microscopy frames now produce one
strict event format, one evaluation, and one set of inspectable artifacts.

## Repository layout

```text
src/neutral_atom_mht/   reusable detection, evaluation, data, and plotting code
tests/                  fast unit tests (not challenge Test Data)
scripts/                retained MHT/quantum experiment scripts
artifacts/detection/    versioned sequence-01 predictions, scores, and figures
figures/                report/poster figures from the wider experiment
report/                 paper source and PDF
poster/                 poster source and PDF
data/                   local source data; ignored by Git
```

The former `cell-detection/` and `NBQSS2026_MHT_NeutralAtoms/` wrappers have
been removed. The MHT/quantum scripts remain monolithic for now; later stages
can extract tracking and neutral-atom solving without reintroducing a second
detector.

## The one retained dataset

Detection uses only **PhC-C2DL-PSC sequence 01** from the
[Cell Tracking Challenge](https://celltrackingchallenge.net/2d-datasets/):

- `01/`: 300 raw phase-contrast frames;
- `01_GT/TRA/`: human gold tracking markers on all 300 frames.

The challenge Test Data is omitted because it has no public gold annotations.
Sequence 02, computer-generated `ST` silver masks, and deliberately relabelled
`ERR_SEG` masks are also unused. `GT/TRA` is the correct reference for detection
events; sparse `GT/SEG` contours answer a different segmentation-shape question.

The downloader temporarily fetches the official training archive, extracts only
the two directories above, validates all 300 frame pairs, and discards the ZIP.
Source images remain local and ignored; compact derived results are versioned.

## Strict detection method

A detection event is one positive final instance label in one frame:

```text
(sequence, frame, detection_id, x_px, y_px, area_px, source)
```

`x_px` is the zero-based image column, `y_px` is the zero-based row, and the
origin is the upper-left pixel. The position is the geometric centroid.
Predicted `detection_id` values are unique only within a frame and are **not**
track identities. Human-gold IDs retain the source `GT/TRA` labels, but this
stage matches events independently per frame and never uses their continuity.

For each raw frame, the frozen deterministic method performs:

```text
Gaussian denoise
  -> broad Gaussian background subtraction
  -> Otsu-derived high-confidence seeds
  -> morphology and seed-area filtering
  -> connected low-threshold support
  -> nearest-seed assignment within each support component
  -> final-area filtering
  -> geometric centroids
```

Preflight hashes the gold files for provenance, but gold labels are decoded only
after each prediction and never enter the detector. Evaluation is independent in
every frame:

1. Admit predicted/gold centroid pairs at Euclidean distance at most 10 px.
2. Find a maximum-cardinality one-to-one matching.
3. Among matchings of that size, minimize total distance.
4. Count matched pairs as TP, unmatched predictions as FP, and unmatched gold
   events as FN.

The headline figure of merit is pooled (micro) centroid F1:

```text
F1 = 2 * sum(TP) / (sum(predictions) + sum(gold events))
```

The numeric detector defaults came unchanged from the earlier sequence-02/frame-025
baseline, and the 10 px gate was fixed before full sequence-01 evaluation.
Results at 5 and 15 px are sensitivity checks, not replacements for the primary
score.

## Detection result

The held-out, full-sequence run contains 58,726 predicted and 71,403 gold events:

| Metric (10 px gate) | Result |
| --- | ---: |
| TP / FP / FN | 50,622 / 8,104 / 20,781 |
| Precision | 0.862 |
| Recall | 0.709 |
| **Micro F1** | **0.778** |
| Macro mean frame F1 | 0.819 |
| Matched-centroid RMSE | 3.69 px |

Frame 0 scores F1 0.915; frame 299 scores F1 0.681. The decline is useful rather
than hidden: as cells crowd and touch, the simple seed detector retains good
precision but increasingly merges or misses events. See
[`detections_overview.png`](artifacts/detection/sequence_01/detections_overview.png)
and [`performance_over_time.png`](artifacts/detection/sequence_01/performance_over_time.png).

## Reproduce detection

Python 3.11 or newer is required.

```powershell
python -m pip install -e ".[test]"
cell-detect prepare-data
cell-detect run
pytest
```

For the exact package versions recorded by the curated run, use the compact
`requirements-detection-lock.txt` first, then install this project with
`python -m pip install -e . --no-deps`.

`cell-detect run --frames 0-9` is a quick smoke run; subsets default to the
Git-ignored `outputs/` tree so they cannot overwrite the curated full run. The
complete command writes:

- `detections.csv` and `gold_events.csv`: the exact event tables;
- `matches.csv`: auditable one-to-one correspondences at 10 px;
- `per_frame_metrics.csv`: counts, scores, and detector diagnostics;
- `summary.json`: data hashes, frozen method, primary score, and gate sensitivity;
- two PNG diagnostic figures.

## Wider MHT/quantum experiment

The original experiment scripts are retained under `scripts/` and now consume
the shared detector instead of defining their own threshold and greedy matcher.
Their full quantum environment is pinned in `requirements-quantum-lock.txt`.
This stage did **not** rerun the costly neutral-atom simulation: the existing
MHT/quantum tables and figures are retained historical outputs and will need to
be regenerated when the tracking/solver stages are refactored.

```powershell
python -m pip install -e .
python -m pip install -r requirements-quantum-lock.txt
python scripts/NielsBohrProject.py
python scripts/Route1_HypothesisDiscovery.py
```

Build the paper/poster in place with `pdflatex`; their figure paths remain
relative to the root `figures/` directory.
