"""Fast conversion of microscopy images into centroid detections."""

from .config import SegmentationConfig
from .features import detections_from_labels
from .segmentation import SegmentationResult, segment_cells

__all__ = [
    "SegmentationConfig",
    "SegmentationResult",
    "detections_from_labels",
    "segment_cells",
]

__version__ = "0.1.0"
