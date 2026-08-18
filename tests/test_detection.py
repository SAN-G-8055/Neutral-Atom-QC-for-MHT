"""Check deterministic cell segmentation and strict detection-event records."""

from __future__ import annotations

import json
import numpy as np
import pytest

from detection import (
    Detection,
    DetectionConfig,
    detect_frame,
    detections_from_label_image,
)


def test_three_bright_cells_are_detected_near_their_centres() -> None:
    height, width = 128, 160
    y, x = np.mgrid[:height, :width]
    image = 80.0 + 0.08 * x + 0.04 * y
    expected = [(35.0, 30.0), (95.0, 55.0), (62.0, 98.0)]
    for centre_x, centre_y in expected:
        image += 150.0 * np.exp(
            -(((x - centre_x) / 7.0) ** 2 + ((y - centre_y) / 5.0) ** 2) / 2.0
        )
    image = np.clip(image, 0, 255).astype(np.uint8)
    config = DetectionConfig(
        min_seed_area_px=12,
        max_seed_area_px=500,
        min_detection_area_px=20,
        max_detection_area_px=800,
    )

    result = detect_frame(image, sequence="synthetic", frame=4, config=config)

    actual = np.asarray([(event.x_px, event.y_px) for event in result.detections])
    distance = np.linalg.norm(actual[:, None, :] - np.asarray(expected)[None, :, :], axis=2)
    assert len(result.detections) == 3
    assert np.all(np.min(distance, axis=0) < 2.0)
    assert all(event.key[:2] == ("synthetic", 4) for event in result.detections)


def test_blank_frame_has_no_detections() -> None:
    result = detect_frame(np.zeros((64, 64), dtype=np.uint8), sequence="01", frame=0)

    assert result.detections == ()
    assert result.diagnostics.detection_count == 0
    assert not result.labels.any()


@pytest.mark.parametrize(
    ("sequence", "frame"),
    [("", 0), ("01", -1)],
)
def test_blank_frame_still_validates_event_scope(sequence: str, frame: int) -> None:
    with pytest.raises(ValueError):
        detect_frame(np.zeros((8, 8), dtype=np.uint8), sequence=sequence, frame=frame)


def test_empty_label_image_still_validates_event_scope() -> None:
    with pytest.raises(ValueError, match="source"):
        detections_from_label_image(
            np.zeros((8, 8), dtype=np.uint16),
            sequence="01",
            frame=0,
            source="",
        )


def test_label_centroid_coordinates_are_x_column_and_y_row() -> None:
    labels = np.zeros((8, 10), dtype=np.uint16)
    labels[2:5, 4:8] = 17

    event = detections_from_label_image(
        labels,
        sequence="01",
        frame=3,
        source="human_tracking_gold",
    )[0]

    assert event.detection_id == 17
    assert event.x_px == 5.5
    assert event.y_px == 3.0
    assert event.area_px == 12


@pytest.mark.parametrize(
    ("field", "value"),
    [("x_px", np.nan), ("y_px", np.inf), ("area_px", 0), ("detection_id", 0)],
)
def test_detection_rejects_invalid_events(field: str, value: float) -> None:
    values = {
        "sequence": "01",
        "frame": 0,
        "detection_id": 1,
        "x_px": 1.0,
        "y_px": 2.0,
        "area_px": 3,
        "source": "prediction",
    }
    values[field] = value

    with pytest.raises(ValueError):
        Detection(**values)


def test_config_rejects_background_scale_smaller_than_noise_scale() -> None:
    with pytest.raises(ValueError, match="background_sigma_px"):
        DetectionConfig(gaussian_sigma_px=3.0, background_sigma_px=2.0)


@pytest.mark.parametrize(
    "values",
    (
        {"gaussian_sigma_px": np.nan},
        {"background_sigma_px": np.inf},
    ),
)
def test_config_rejects_non_finite_parameters(values: dict) -> None:
    with pytest.raises(ValueError, match="finite real"):
        DetectionConfig(**values)


def test_numpy_scalars_are_normalized_to_json_safe_schema_values() -> None:
    config = DetectionConfig(
        gaussian_sigma_px=np.float32(1.0),
        opening_size_px=np.int32(2),
    )
    event = Detection(
        sequence="01",
        frame=np.int32(1),
        detection_id=np.int32(2),
        x_px=np.float32(3.0),
        y_px=np.float64(4.0),
        area_px=np.int64(5),
        source="prediction",
    )

    assert type(config.gaussian_sigma_px) is float
    assert type(config.opening_size_px) is int
    assert type(event.frame) is int
    assert type(event.x_px) is float
    json.dumps(config.to_dict(), allow_nan=False)
