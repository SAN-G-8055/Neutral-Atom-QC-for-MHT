"""Check lossless, schema-validated CSV and JSON project formats."""

from __future__ import annotations

import pytest

from neutral_atom_mht.detection import Detection
from neutral_atom_mht.io import DETECTION_COLUMNS, read_detections, write_detections


def test_detection_csv_round_trip_uses_the_declared_minimal_schema(tmp_path) -> None:
    events = (
        Detection("01", 2, 1, 1 / 3, 7.5, 12, "prediction"),
        Detection("01", 2, 2, 8.0, 9.75, 15, "prediction"),
    )
    output = tmp_path / "detections.csv"

    write_detections(events, output)

    assert tuple(output.read_text(encoding="utf-8").splitlines()[0].split(",")) == DETECTION_COLUMNS
    assert read_detections(output) == events


def test_duplicate_event_keys_are_rejected_on_write_and_read(tmp_path) -> None:
    event = Detection("01", 2, 1, 1.0, 2.0, 3, "prediction")
    output = tmp_path / "detections.csv"

    with pytest.raises(ValueError, match="keys must be unique"):
        write_detections((event, event), output)
    assert not output.exists()

    output.write_text(
        ",".join(DETECTION_COLUMNS)
        + "\n01,2,1,1.0,2.0,3,prediction\n01,2,1,1.0,2.0,3,prediction\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="keys must be unique"):
        read_detections(output)
