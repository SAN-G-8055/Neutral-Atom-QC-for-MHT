"""End-to-end sequence-01 detection benchmark.

This module is orchestration only: preflight hashes gold bytes for provenance,
but gold labels are decoded only after each prediction and never enter detection.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from numbers import Integral
from pathlib import Path
import platform
from typing import Any

import matplotlib
import numpy as np

from . import __version__
from PIL import __version__ as pillow_version
import scipy

from .data import (
    DATASET_NAME,
    FRAME_COUNT,
    IMAGE_SHAPE,
    SEQUENCE,
    TRAINING_ARCHIVE_URL,
    gold_tracking_path,
    load_tiff,
    raw_frame_path,
    verify_sequence_01,
)
from .detection import (
    Detection,
    DetectionConfig,
    detect_frame,
    detections_from_label_image,
)
from .evaluation import DEFAULT_MAX_DISTANCE_PX, FrameEvaluation, SequenceEvaluation, evaluate_frame, evaluate_sequence
from .io import write_detections, write_json, write_rows
from .visualization import save_detection_overview, save_per_frame_performance


OVERVIEW_FRAMES = (0, 60, 120, 180, 240, 299)
SENSITIVITY_GATES_PX = (5.0, DEFAULT_MAX_DISTANCE_PX, 15.0)


def _validate_frame_arrays(raw: Any, gold_labels: Any, frame: int) -> None:
    if raw.shape != gold_labels.shape:
        raise ValueError(
            f"frame {frame} raw/gold shapes differ: {raw.shape} versus {gold_labels.shape}"
        )
    if raw.shape != IMAGE_SHAPE:
        raise ValueError(f"frame {frame} must have shape {IMAGE_SHAPE}, got {raw.shape}")
    if raw.dtype.name != "uint8":
        raise ValueError(f"frame {frame} raw image must be uint8, got {raw.dtype}")
    if gold_labels.dtype.name != "uint16":
        raise ValueError(f"frame {frame} gold tracking mask must be uint16, got {gold_labels.dtype}")


def _metric_summary(evaluation: SequenceEvaluation) -> dict[str, Any]:
    return {
        "gate_px": evaluation.max_distance_px,
        "predicted_count": evaluation.predicted_count,
        "gold_count": evaluation.reference_count,
        "true_positives": evaluation.true_positives,
        "false_positives": evaluation.false_positives,
        "false_negatives": evaluation.false_negatives,
        "precision": evaluation.precision,
        "recall": evaluation.recall,
        "f1": evaluation.f1,
        "macro_precision": evaluation.macro_precision,
        "macro_recall": evaluation.macro_recall,
        "macro_f1": evaluation.macro_f1,
        "localization_rmse_px": evaluation.localization_rmse_px,
    }


def _representative_frames(frames: tuple[int, ...]) -> tuple[int, ...]:
    preferred = tuple(frame for frame in OVERVIEW_FRAMES if frame in frames)
    if len(preferred) >= min(3, len(frames)):
        return preferred
    if len(frames) <= 6:
        return frames
    indices = [round(index * (len(frames) - 1) / 5) for index in range(6)]
    return tuple(frames[index] for index in indices)


def run_detection_benchmark(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    frames: Iterable[int] = range(FRAME_COUNT),
    config: DetectionConfig | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Detect, evaluate, visualize, and version one deterministic run."""

    root = Path(dataset_root)
    output = Path(output_dir)
    supplied_frames = tuple(frames)
    invalid_frames = [
        frame
        for frame in supplied_frames
        if not isinstance(frame, Integral) or isinstance(frame, bool)
    ]
    if invalid_frames:
        raise ValueError(f"frames must be integers, got {invalid_frames[:3]}")
    selected_frames = tuple(sorted(set(int(frame) for frame in supplied_frames)))
    if not selected_frames:
        raise ValueError("at least one frame is required")
    if selected_frames[0] < 0 or selected_frames[-1] >= FRAME_COUNT:
        raise ValueError(f"frames must lie in [0, {FRAME_COUNT - 1}]")
    cfg = config or DetectionConfig()
    overview_frames = _representative_frames(selected_frames)
    notify = progress or (lambda _: None)

    raw_paths = [raw_frame_path(root, frame) for frame in selected_frames]
    gold_paths = [gold_tracking_path(root, frame) for frame in selected_frames]
    missing = [path for path in raw_paths + gold_paths if not path.is_file()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:4])
        raise FileNotFoundError(f"missing {len(missing)} sequence-01 input files: {preview}")
    raw_sequence_hash, gold_sequence_hash = verify_sequence_01(root)

    predictions: list[Detection] = []
    references: list[Detection] = []
    frame_evaluations: dict[int, FrameEvaluation] = {}
    diagnostics: dict[int, Any] = {}
    overview_images: dict[int, Any] = {}
    overview_predictions: dict[int, tuple[Detection, ...]] = {}
    overview_references: dict[int, tuple[Detection, ...]] = {}
    for index, (frame, raw_path, gold_path) in enumerate(
        zip(selected_frames, raw_paths, gold_paths, strict=True),
        start=1,
    ):
        raw = load_tiff(raw_path)
        result = detect_frame(raw, sequence=SEQUENCE, frame=frame, config=cfg)
        # Gold labels are deliberately decoded only after prediction is fixed.
        gold_labels = load_tiff(gold_path)
        _validate_frame_arrays(raw, gold_labels, frame)
        gold = detections_from_label_image(
            gold_labels,
            sequence=SEQUENCE,
            frame=frame,
            source="human_tracking_gold",
        )
        evaluation = evaluate_frame(
            result.detections,
            gold,
            max_distance_px=DEFAULT_MAX_DISTANCE_PX,
        )
        predictions.extend(result.detections)
        references.extend(gold)
        frame_evaluations[frame] = evaluation
        diagnostics[frame] = result.diagnostics
        if frame in overview_frames:
            overview_images[frame] = raw
            overview_predictions[frame] = result.detections
            overview_references[frame] = gold
        if index == 1 or index % 25 == 0 or index == len(selected_frames):
            notify(f"processed {index}/{len(selected_frames)} frames")

    primary = evaluate_sequence(
        predictions,
        references,
        max_distance_px=DEFAULT_MAX_DISTANCE_PX,
        sequence=SEQUENCE,
    )
    sensitivity = {
        f"{gate:g}_px": _metric_summary(
            evaluate_sequence(predictions, references, max_distance_px=gate, sequence=SEQUENCE)
        )
        for gate in SENSITIVITY_GATES_PX
    }

    # ``summary.json`` is the completion manifest and is always published last.
    # Removing an older copy first makes an interrupted rerun visibly incomplete
    # instead of pairing a stale summary with partially refreshed tables.
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    summary_path.unlink(missing_ok=True)
    write_detections(predictions, output / "detections.csv")
    write_detections(references, output / "gold_events.csv")
    per_frame_columns = (
        "frame",
        "predicted_count",
        "gold_count",
        "true_positives",
        "false_positives",
        "false_negatives",
        "precision",
        "recall",
        "f1",
        "localization_rmse_px",
        "seed_count",
        "otsu_threshold",
        "high_threshold",
        "low_threshold",
    )
    write_rows(
        (
            {
                "frame": frame,
                "predicted_count": evaluation.predicted_count,
                "gold_count": evaluation.reference_count,
                "true_positives": evaluation.true_positives,
                "false_positives": evaluation.false_positives,
                "false_negatives": evaluation.false_negatives,
                "precision": f"{evaluation.precision:.6f}",
                "recall": f"{evaluation.recall:.6f}",
                "f1": f"{evaluation.f1:.6f}",
                "localization_rmse_px": (
                    "" if evaluation.localization_rmse_px is None else f"{evaluation.localization_rmse_px:.6f}"
                ),
                "seed_count": diagnostics[frame].seed_count,
                "otsu_threshold": f"{diagnostics[frame].otsu_threshold:.6f}",
                "high_threshold": f"{diagnostics[frame].high_threshold:.6f}",
                "low_threshold": f"{diagnostics[frame].low_threshold:.6f}",
            }
            for frame, evaluation in sorted(frame_evaluations.items())
        ),
        per_frame_columns,
        output / "per_frame_metrics.csv",
    )
    match_columns = ("sequence", "frame", "predicted_id", "gold_id", "distance_px")
    write_rows(
        (
            {
                "sequence": match.sequence,
                "frame": match.frame,
                "predicted_id": match.predicted_id,
                "gold_id": match.reference_id,
                "distance_px": repr(match.distance_px),
            }
            for match in primary.matches
        ),
        match_columns,
        output / "matches.csv",
    )

    save_detection_overview(
        overview_images,
        overview_predictions,
        overview_references,
        frame_evaluations,
        output / "detections_overview.png",
        frames=overview_frames,
        columns=3,
    )
    save_per_frame_performance(
        frame_evaluations,
        output / "performance_over_time.png",
    )

    summary = {
        "dataset": {
            "name": DATASET_NAME,
            "sequence": SEQUENCE,
            "frame_selection": {
                "count": len(selected_frames),
                "first": selected_frames[0],
                "last": selected_frames[-1],
                "continuous": selected_frames == tuple(range(selected_frames[0], selected_frames[-1] + 1)),
                **(
                    {}
                    if selected_frames == tuple(range(selected_frames[0], selected_frames[-1] + 1))
                    else {"frames": list(selected_frames)}
                ),
            },
            "official_training_archive": TRAINING_ARCHIVE_URL,
            "raw_frames_sha256": raw_sequence_hash,
            "gold_tracking_masks_sha256": gold_sequence_hash,
            "fingerprint_scope": "all 300 sequence-01 raw/gold frame pairs",
            "gold_source": "human GT/TRA tracking markers",
        },
        "method": {
            "event_definition": (
                "One positive final instance label in one frame, observed at its geometric centroid; "
                "prediction IDs are frame-local; gold IDs retain source labels, but evaluation does "
                "not use either as cross-frame track identity."
            ),
            "coordinates": "zero-based pixels; x_px is column, y_px is row; origin is top-left",
            "detector": (
                "Gaussian denoise -> broad-background subtraction -> Otsu high-confidence seeds -> "
                "morphology and seed-area filter -> connected low-threshold support -> within-component "
                "nearest-seed assignment -> final-area filter"
            ),
            "config": cfg.to_dict(),
            "gold_is_not_detector_input": True,
            "matching": (
                "Independently per frame: maximum-cardinality one-to-one centroid matching inside an "
                "inclusive Euclidean gate; minimum total distance breaks cardinality ties."
            ),
            "parameter_provenance": (
                "Numeric detector defaults were fixed by the earlier sequence-02/frame-025 baseline "
                "before this full sequence-01 evaluation."
            ),
            "primary_figure_of_merit": "micro-averaged centroid F1 at a fixed 10 px gate",
            "primary_f1_formula": "2 * sum(TP) / (sum(predicted) + sum(gold))",
        },
        "primary_metrics": _metric_summary(primary),
        "gate_sensitivity": sensitivity,
        "overview_frames": list(overview_frames),
        "software": {
            "matplotlib": matplotlib.__version__,
            "neutral_atom_mht": __version__,
            "numpy": np.__version__,
            "operating_system": platform.platform(),
            "pillow": pillow_version,
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "scipy": scipy.__version__,
        },
    }
    write_json(summary, summary_path)
    return summary
