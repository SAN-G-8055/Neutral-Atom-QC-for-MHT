"""Check lossless, schema-validated CSV and JSON project formats."""

from __future__ import annotations

import csv

import pytest

from neutral_atom_mht.detection import Detection
from neutral_atom_mht.io import DETECTION_COLUMNS, write_detections


def test_detection_csv_uses_the_declared_lossless_minimal_schema(tmp_path) -> None:
    events = (
        Detection("01", 2, 1, 1 / 3, 7.5, 12, "prediction"),
        Detection("01", 2, 2, 8.0, 9.75, 15, "prediction"),
    )
    output = tmp_path / "detections.csv"

    write_detections(events, output)

    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert tuple(rows[0]) == DETECTION_COLUMNS
    assert rows[0]["x_px"] == repr(1 / 3)
    assert rows[1]["area_px"] == "15"


def test_duplicate_event_keys_are_rejected_before_writing(tmp_path) -> None:
    event = Detection("01", 2, 1, 1.0, 2.0, 3, "prediction")
    output = tmp_path / "detections.csv"

    with pytest.raises(ValueError, match="keys must be unique"):
        write_detections((event, event), output)
    assert not output.exists()
