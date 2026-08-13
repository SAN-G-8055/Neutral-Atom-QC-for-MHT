from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from cell_detection_pipeline.config import SegmentationConfig
from cell_detection_pipeline.features import detections_from_labels
from cell_detection_pipeline.segmentation import segment_cells


def test_three_bright_cells_are_detected_near_their_centers() -> None:
    height, width = 128, 160
    y, x = np.mgrid[:height, :width]
    image = 80.0 + 0.08 * x + 0.04 * y
    expected = [(35.0, 30.0), (95.0, 55.0), (62.0, 98.0)]
    for center_x, center_y in expected:
        image += 150.0 * np.exp(-(((x - center_x) / 7.0) ** 2 + ((y - center_y) / 5.0) ** 2) / 2.0)
    image = np.clip(image, 0, 255).astype(np.uint8)

    config = SegmentationConfig(
        min_seed_area_px=12,
        max_seed_area_px=500,
        min_final_area_px=20,
        max_final_area_px=800,
    )
    result = segment_cells(image, config)
    records = detections_from_labels(result.labels, image)

    assert len(records) == 3
    actual = np.asarray([(row["x"], row["y"]) for row in records])
    expected_array = np.asarray(expected)
    distances = np.linalg.norm(actual[:, None, :] - expected_array[None, :, :], axis=2)
    assert np.all(np.min(distances, axis=0) < 2.0)


def test_blank_image_has_no_detections() -> None:
    result = segment_cells(np.zeros((64, 64), dtype=np.uint8))
    assert result.detection_count == 0
    assert not result.labels.any()


def test_feature_coordinates_are_x_column_y_row_and_bbox_max_is_exclusive() -> None:
    labels = np.zeros((8, 10), dtype=np.uint16)
    labels[2:5, 4:8] = 7
    image = np.arange(labels.size, dtype=np.float32).reshape(labels.shape)

    record = detections_from_labels(labels, image, frame=3, source="test")[0]

    assert record["detection_id"] == 7
    assert record["x"] == 5.5
    assert record["y"] == 3.0
    assert record["area_px"] == 12
    assert record["bbox_x_min"] == 4
    assert record["bbox_x_max_exclusive"] == 8
    assert record["bbox_y_min"] == 2
    assert record["bbox_y_max_exclusive"] == 5
