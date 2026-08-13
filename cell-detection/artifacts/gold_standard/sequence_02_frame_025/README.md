# Human gold standard: sequence 02, frame 025

`gold_detections.csv` is the canonical comparison table. It is derived from the human-origin
instance mask `Train Data/02_GT/SEG/man_seg025.tif`, not from the computer-generated silver
segmentation. The label-mask SHA-256 in `manifest.json` makes the derivation auditable.

`tracking_marker_detections.csv` contains human tracking-marker centroids for the same frame.
Those sparse blobs are useful for full-series detection validation but are not cell outlines.

`silver_detections.csv` is retained only as a diagnostic. It must not be described or analyzed as
gold truth.

All coordinates are zero-based pixels with `x = column` and `y = row`.
