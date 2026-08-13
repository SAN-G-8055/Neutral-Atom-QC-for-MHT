# Cell detection pipeline

This repository turns a raw phase-contrast microscopy TIFF into an instance-label image and a
flat detection table of object centroids. It is a deterministic, CPU-only baseline designed to feed
later multiple-hypothesis-testing work without coupling that analysis to image processing.

## Chosen case

The simplest defensible case is **PhC-C2DL-PSC sequence 02, frame 025**:

| Human-gold frame | Objects | Foreground | Median nearest-centroid spacing |
| --- | ---: | ---: | ---: |
| **02 / 025** | **68** | **3.43%** | **35.02 px** |
| 01 / 098 | 100 | 4.16% | 26.10 px |
| 01 / 122 | 142 | 5.24% | 21.07 px |
| 02 / 182 | 186 | 6.07% | 16.86 px |

Frame 025 has the fewest human-segmented cells, the lowest density, the greatest separation, no
gold objects touching the border, and strong cell/background contrast. Earlier sequence-02 frames
are marginally sparser, but they have tracking markers only—not a full human segmentation mask.

The data dictionary distinguishes three references:

- `GT/SEG`: sparse, human-origin cell segmentations. This is the canonical gold standard.
- `GT/TRA`: human-origin tracking markers. These support detection validation on every frame but
  are not full cell outlines.
- `ST/SEG`: dense, computer-origin silver segmentations. These are diagnostic, not gold.

The deliberately relabeled `ERR_SEG` masks have the same foreground geometry as `ST/SEG`; their
temporal identities are corrupted. They are not used as gold.

## Pipeline

```text
raw TIFF -> Gaussian denoise -> smooth-background subtraction -> Otsu seeds
         -> morphology/area filter -> low-threshold cell support
         -> nearest-seed partition -> instance labels -> centroids + features
```

The two-threshold growth step captures more of each cell than a hard bright-core threshold while
retaining separate IDs. The current baseline is deliberately simple and fast. Crowded late frames
will need a gradient-aware or marker-controlled watershed to recover cells that merge.

### Selected-case result

Against the 68 human-gold cells at sequence 02, frame 025, the versioned run produced:

| Metric (10 px centroid gate) | Result |
| --- | ---: |
| Predicted detections | 74 |
| TP / FP / FN | 66 / 8 / 2 |
| Precision / recall / F1 | 0.892 / 0.971 / **0.930** |
| Localization RMSE | 2.02 px |
| Foreground Dice / IoU | 0.788 / 0.650 |
| Instance F1 at IoU 0.5 | 0.873 |
| Cold raw-to-detections runtime on this machine | 0.34 s |

These are baseline measurements on the chosen easy frame, not estimates of performance on the
whole time series. The deliberately simple method loses recall as cells crowd and touch late in the
sequence.

## Quick start

Python 3.11+ is required. The pipeline uses NumPy, SciPy, and Pillow only.

```powershell
python -m pip install -e ".[test]"

# Any raw image
cell-detect detect "Train Data/02/t025.tif" `
  --output-dir outputs/t025 `
  --config configs/easy_case.json `
  --dataset PhC-C2DL-PSC --sequence 02 --frame 25

# Rebuild the selected result and all comparison tables
cell-detect easy-case --data-root .

# Tests
pytest
```

Without installing the package, prepend `src` to `PYTHONPATH` and use
`python -m cell_detection_pipeline` in place of `cell-detect`.

## Outputs

The curated prediction is in
[`artifacts/easy_case/sequence_02_frame_025`](artifacts/easy_case/sequence_02_frame_025), and the
immutable human-gold centroid table is in
[`artifacts/gold_standard/sequence_02_frame_025`](artifacts/gold_standard/sequence_02_frame_025).

Detection CSVs include:

```text
dataset, sequence, frame, detection_id, x, y, area_px,
mean_intensity, std_intensity, max_intensity, integrated_intensity,
intensity_weighted_x, intensity_weighted_y, bounding box, source, image
```

Coordinates are zero-based pixels: `x` is image column and `y` is image row. Geometric centroids
are the primary `x,y`; intensity-weighted centroids are additional features. Every row carries a
`source` value so prediction, human gold, tracking markers, and silver data cannot be silently mixed.

Comparison uses gated one-to-one assignment and reports TP/FP/FN, precision, recall, F1, and
localization error at 10 px. A 15 px sensitivity analysis is included because full-cell gold
centroids and sparse tracking-marker centroids are not identical. Full segmentation comparisons
also report foreground Dice/IoU and instance matches at IoU 0.5.

## Data and Git policy

The supplied train/test directories, dictionary PDF, and ZIP archives stay local and are ignored.
Together they occupy about 0.5 GB, and both ZIP files exceed GitHub's ordinary 100 MB per-file
limit. Versioned artifacts contain the compact detection data, source hashes, configuration, and
evaluation needed for comparison. A clone can reproduce them once the original challenge data is
placed back in the documented paths.
