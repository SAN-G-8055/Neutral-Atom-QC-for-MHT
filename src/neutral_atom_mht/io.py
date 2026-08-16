"""Small, explicit file formats used by the detection stage."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import csv
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .detection import Detection


DETECTION_COLUMNS = (
    "sequence",
    "frame",
    "detection_id",
    "x_px",
    "y_px",
    "area_px",
    "source",
)


def write_detections(events: Iterable[Detection], path: str | Path) -> None:
    """Write the complete and intentionally minimal detection-event schema."""

    materialized = tuple(events)
    keys = [event.key for event in materialized]
    if len(keys) != len(set(keys)):
        raise ValueError("detection event keys must be unique within the table")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETECTION_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for event in materialized:
            writer.writerow(
                {
                    "sequence": event.sequence,
                    "frame": event.frame,
                    "detection_id": event.detection_id,
                    "x_px": repr(event.x_px),
                    "y_px": repr(event.y_px),
                    "area_px": event.area_px,
                    "source": event.source,
                }
            )


def read_detections(path: str | Path) -> tuple[Detection, ...]:
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != DETECTION_COLUMNS:
            raise ValueError(
                f"unexpected detection columns in {input_path}: {reader.fieldnames}"
            )
        events = tuple(
            Detection(
                sequence=row["sequence"],
                frame=int(row["frame"]),
                detection_id=int(row["detection_id"]),
                x_px=float(row["x_px"]),
                y_px=float(row["y_px"]),
                area_px=int(row["area_px"]),
                source=row["source"],
            )
            for row in reader
        )
    keys = [event.key for event in events]
    if len(keys) != len(set(keys)):
        raise ValueError("detection event keys must be unique within the table")
    return events


def write_rows(
    rows: Iterable[Mapping[str, Any]],
    columns: Sequence[str],
    path: str | Path,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(value: Any, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def sha256_files(paths: Iterable[str | Path]) -> str:
    """Hash ordered path names and contents as one reproducibility fingerprint."""

    digest = hashlib.sha256()
    for raw_path in paths:
        path = Path(raw_path)
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()
