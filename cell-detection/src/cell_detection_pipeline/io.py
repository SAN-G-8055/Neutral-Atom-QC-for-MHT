"""TIFF, CSV, JSON, hashing, and visualization helpers."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi


DETECTION_COLUMNS = (
    "dataset",
    "sequence",
    "frame",
    "detection_id",
    "x",
    "y",
    "area_px",
    "mean_intensity",
    "std_intensity",
    "max_intensity",
    "integrated_intensity",
    "intensity_weighted_x",
    "intensity_weighted_y",
    "bbox_x_min",
    "bbox_y_min",
    "bbox_x_max_exclusive",
    "bbox_y_max_exclusive",
    "source",
    "image",
)


def load_image(path: str | Path) -> np.ndarray:
    """Load a single-page two-dimensional TIFF without silently changing its values."""

    image_path = Path(path)
    with Image.open(image_path) as opened:
        array = np.asarray(opened)
    if array.ndim != 2:
        raise ValueError(f"Expected one two-dimensional grayscale image, got shape {array.shape}: {image_path}")
    return array


def save_label_image(labels: np.ndarray, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    maximum = int(labels.max(initial=0))
    if maximum > np.iinfo(np.uint16).max:
        raise ValueError(f"Label value {maximum} exceeds uint16 TIFF capacity")
    Image.fromarray(labels.astype(np.uint16, copy=False)).save(output, compression="tiff_deflate")


def _format_csv_value(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_detection_csv(records: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETECTION_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({column: _format_csv_value(record.get(column, "")) for column in DETECTION_COLUMNS})


def read_detection_csv(path: str | Path) -> list[dict[str, Any]]:
    numeric_float = {
        "x",
        "y",
        "mean_intensity",
        "std_intensity",
        "max_intensity",
        "integrated_intensity",
        "intensity_weighted_x",
        "intensity_weighted_y",
    }
    numeric_int = {
        "frame",
        "detection_id",
        "area_px",
        "bbox_x_min",
        "bbox_y_min",
        "bbox_x_max_exclusive",
        "bbox_y_max_exclusive",
    }
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            converted: dict[str, Any] = dict(row)
            for key in numeric_float:
                if converted.get(key, "") != "":
                    converted[key] = float(converted[key])
            for key in numeric_int:
                if converted.get(key, "") != "":
                    converted[key] = int(converted[key])
            records.append(converted)
    return records


def write_json(data: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _boundaries(labels: np.ndarray) -> np.ndarray:
    foreground = labels > 0
    if not foreground.any():
        return foreground
    eroded = ndi.binary_erosion(foreground, structure=np.ones((3, 3), dtype=bool), border_value=0)
    return foreground & ~eroded


def save_overlay(
    raw: np.ndarray,
    predicted_labels: np.ndarray,
    path: str | Path,
    *,
    reference_labels: np.ndarray | None = None,
    predicted_centroids: Sequence[tuple[float, float]] = (),
    reference_centroids: Sequence[tuple[float, float]] = (),
) -> None:
    """Save a compact visual check: cyan predictions and magenta human gold."""

    low, high = np.percentile(raw, (0.5, 99.5))
    if high <= low:
        base = np.zeros(raw.shape, dtype=np.uint8)
    else:
        base = np.clip((raw.astype(np.float32) - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
    rgb = np.repeat(base[..., None], 3, axis=2)
    rgb[_boundaries(predicted_labels)] = (0, 255, 255)
    if reference_labels is not None:
        rgb[_boundaries(reference_labels)] = (255, 0, 255)

    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)
    for x, y in predicted_centroids:
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline=(0, 255, 255), width=1)
    for x, y in reference_centroids:
        draw.line((x - 3, y, x + 3, y), fill=(255, 255, 0), width=1)
        draw.line((x, y - 3, x, y + 3), fill=(255, 255, 0), width=1)

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)
