"""Check local sequence-01 path, loading, and verification utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from cell_data import (
    DATASET_NAME,
    CANONICAL_GOLD_SHA256,
    CANONICAL_RAW_SHA256,
    FRAME_COUNT,
    gold_tracking_path,
    load_tiff,
    paired_frame_paths,
    raw_frame_path,
    verify_sequence_01,
)
from detection import detect_frame, detections_from_label_image
from evaluation import evaluate_frame


def test_sequence_01_path_convention() -> None:
    root = Path("data") / "PhC-C2DL-PSC"

    assert raw_frame_path(root, 7) == root / "01" / "t007.tif"
    assert gold_tracking_path(root, 7) == root / "01_GT" / "TRA" / "man_track007.tif"


def test_frame_range_is_exactly_zero_through_299() -> None:
    assert FRAME_COUNT == 300
    with pytest.raises(ValueError):
        raw_frame_path("data", -1)
    with pytest.raises(ValueError):
        raw_frame_path("data", 300)


def test_incomplete_sequence_fails_before_processing(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="sequence 01 is incomplete"):
        tuple(paired_frame_paths(tmp_path))


def test_load_tiff_preserves_local_pixel_values(tmp_path) -> None:
    pixels = np.array([[0, 17], [255, 42]], dtype=np.uint8)
    image_path = tmp_path / "frame.tif"
    Image.fromarray(pixels).save(image_path)

    loaded = load_tiff(image_path)

    assert np.array_equal(loaded, pixels)
    assert loaded.dtype == pixels.dtype


LOCAL_DATASET = Path(__file__).resolve().parents[1] / "data" / DATASET_NAME


@pytest.mark.skipif(not LOCAL_DATASET.exists(), reason="source dataset is intentionally not versioned")
def test_local_sequence_contract_and_frame_zero_regression() -> None:
    pairs = tuple(paired_frame_paths(LOCAL_DATASET))
    raw_hash, gold_hash = verify_sequence_01(LOCAL_DATASET)
    frame, raw_path, gold_path = pairs[0]
    raw = load_tiff(raw_path)
    gold_labels = load_tiff(gold_path)
    prediction = detect_frame(raw, sequence="01", frame=frame)
    gold = detections_from_label_image(
        gold_labels,
        sequence="01",
        frame=frame,
        source="human_tracking_gold",
    )
    score = evaluate_frame(prediction.detections, gold, max_distance_px=10.0)

    assert len(pairs) == 300
    assert raw_hash == CANONICAL_RAW_SHA256
    assert gold_hash == CANONICAL_GOLD_SHA256
    assert raw.shape == gold_labels.shape == (576, 720)
    assert raw.dtype.name == "uint8"
    assert gold_labels.dtype.name == "uint16"
    assert len(gold) == 74
    assert score.f1 >= 0.85
