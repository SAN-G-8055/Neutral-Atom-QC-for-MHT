from __future__ import annotations

from pathlib import Path

import pytest

from cell_detection_pipeline.config import SegmentationConfig
from cell_detection_pipeline.evaluation import match_centroids
from cell_detection_pipeline.features import detections_from_labels
from cell_detection_pipeline.io import load_image
from cell_detection_pipeline.segmentation import segment_cells


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Train Data" / "02" / "t025.tif"
GOLD = ROOT / "Train Data" / "02_GT" / "SEG" / "man_seg025.tif"


@pytest.mark.skipif(not (RAW.exists() and GOLD.exists()), reason="local challenge data is not versioned")
def test_canonical_easy_case_centroid_f1_exceeds_0_85() -> None:
    raw = load_image(RAW)
    gold_labels = load_image(GOLD)
    result = segment_cells(raw, SegmentationConfig.from_json(ROOT / "configs" / "easy_case.json"))
    predicted = detections_from_labels(result.labels, raw)
    gold = detections_from_labels(gold_labels, raw)

    metrics = match_centroids(predicted, gold, max_distance_px=10.0)

    assert len(gold) == 68
    assert 60 <= len(predicted) <= 85
    assert metrics["f1"] >= 0.85
