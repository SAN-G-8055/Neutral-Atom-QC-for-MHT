"""Command-line interface for raw images, references, and the canonical easy case."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from time import perf_counter
from typing import Any, Sequence

from .config import SegmentationConfig
from .evaluation import binary_mask_metrics, instance_iou_metrics, match_centroids
from .features import detections_from_labels
from .io import (
    load_image,
    read_detection_csv,
    save_label_image,
    save_overlay,
    sha256_file,
    write_detection_csv,
    write_json,
)
from .segmentation import segment_cells


DATASET_NAME = "PhC-C2DL-PSC"


def _config(path: str | None) -> SegmentationConfig:
    return SegmentationConfig.from_json(path) if path else SegmentationConfig()


def _infer_frame(path: Path, explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else 0


def _relative_display(path: Path, root: Path | None = None) -> str:
    try:
        return path.resolve().relative_to((root or Path.cwd()).resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _detect_one(
    input_path: Path,
    *,
    output_dir: Path,
    config: SegmentationConfig,
    dataset: str,
    sequence: str,
    frame: int,
    display_root: Path | None = None,
) -> tuple[list[dict[str, Any]], Any, float]:
    started = perf_counter()
    raw = load_image(input_path)
    result = segment_cells(raw, config)
    records = detections_from_labels(
        result.labels,
        raw,
        dataset=dataset,
        sequence=sequence,
        frame=frame,
        source="prediction",
        image_name=_relative_display(input_path, display_root),
    )
    elapsed = perf_counter() - started
    output_dir.mkdir(parents=True, exist_ok=True)
    save_label_image(result.labels, output_dir / "labels.tif")
    write_detection_csv(records, output_dir / "detections.csv")
    save_overlay(
        raw,
        result.labels,
        output_dir / "overlay.png",
        predicted_centroids=[(row["x"], row["y"]) for row in records],
    )
    write_json(
        {
            "algorithm": "gaussian_background_otsu_hysteresis_voronoi",
            "config": config.to_dict(),
            "coordinate_convention": "zero-based pixels; x=column, y=row",
            "input": _relative_display(input_path, display_root),
            "input_sha256": sha256_file(input_path),
            "image_shape_yx": list(raw.shape),
            "image_dtype": str(raw.dtype),
            "otsu_threshold": result.otsu_threshold,
            "high_threshold": result.high_threshold,
            "low_threshold": result.low_threshold,
            "seed_count": result.seed_count,
            "detection_count": result.detection_count,
            "processing_seconds": elapsed,
        },
        output_dir / "manifest.json",
    )
    return records, result, elapsed


def command_detect(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    frame = _infer_frame(input_path, args.frame)
    records, _, elapsed = _detect_one(
        input_path,
        output_dir=output_dir,
        config=_config(args.config),
        dataset=args.dataset,
        sequence=args.sequence,
        frame=frame,
    )
    print(f"Detected {len(records)} objects in {elapsed:.3f}s -> {output_dir / 'detections.csv'}")
    return 0


def _write_reference(
    mask_path: Path,
    image_path: Path,
    output_csv: Path,
    *,
    dataset: str,
    sequence: str,
    frame: int,
    source: str,
    display_root: Path | None = None,
) -> list[dict[str, Any]]:
    mask = load_image(mask_path)
    raw = load_image(image_path)
    records = detections_from_labels(
        mask,
        raw,
        dataset=dataset,
        sequence=sequence,
        frame=frame,
        source=source,
        image_name=_relative_display(image_path, display_root),
    )
    write_detection_csv(records, output_csv)
    return records


def command_reference(args: argparse.Namespace) -> int:
    mask_path = Path(args.mask)
    image_path = Path(args.image)
    records = _write_reference(
        mask_path,
        image_path,
        Path(args.output_csv),
        dataset=args.dataset,
        sequence=args.sequence,
        frame=args.frame,
        source=args.source,
    )
    print(f"Extracted {len(records)} reference detections -> {args.output_csv}")
    return 0


def command_compare(args: argparse.Namespace) -> int:
    predicted = read_detection_csv(args.predicted)
    reference = read_detection_csv(args.reference)
    metrics = match_centroids(predicted, reference, max_distance_px=args.max_distance)
    write_json(metrics, args.output_json)
    print(f"F1={metrics['f1']:.3f} ({metrics['true_positive']} matches) -> {args.output_json}")
    return 0


def command_easy_case(args: argparse.Namespace) -> int:
    root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    gold_dir = Path(args.gold_dir)
    raw_path = root / "Train Data" / "02" / "t025.tif"
    gold_seg_path = root / "Train Data" / "02_GT" / "SEG" / "man_seg025.tif"
    gold_track_path = root / "Train Data" / "02_GT" / "TRA" / "man_track025.tif"
    silver_path = root / "Train Data" / "02_ST" / "SEG" / "man_seg025.tif"
    required = (raw_path, gold_seg_path, gold_track_path, silver_path)
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing easy-case inputs: " + ", ".join(str(path) for path in missing))

    predicted, result, elapsed = _detect_one(
        raw_path,
        output_dir=output_dir,
        config=_config(args.config),
        dataset=DATASET_NAME,
        sequence="02",
        frame=25,
        display_root=root,
    )
    gold_dir.mkdir(parents=True, exist_ok=True)
    gold = _write_reference(
        gold_seg_path,
        raw_path,
        gold_dir / "gold_detections.csv",
        dataset=DATASET_NAME,
        sequence="02",
        frame=25,
        source="human_gold_segmentation",
        display_root=root,
    )
    tracking = _write_reference(
        gold_track_path,
        raw_path,
        gold_dir / "tracking_marker_detections.csv",
        dataset=DATASET_NAME,
        sequence="02",
        frame=25,
        source="human_gold_tracking_marker",
        display_root=root,
    )
    silver = _write_reference(
        silver_path,
        raw_path,
        gold_dir / "silver_detections.csv",
        dataset=DATASET_NAME,
        sequence="02",
        frame=25,
        source="computer_silver_segmentation",
        display_root=root,
    )
    gold_labels = load_image(gold_seg_path)
    tracking_labels = load_image(gold_track_path)
    silver_labels = load_image(silver_path)

    metrics = {
        "case": {"dataset": DATASET_NAME, "sequence": "02", "frame": 25},
        "coordinate_convention": "zero-based pixels; x=column, y=row",
        "processing_seconds": elapsed,
        "human_gold_segmentation": {
            "centroids_10px": match_centroids(predicted, gold, max_distance_px=10.0),
            "centroids_15px_sensitivity": match_centroids(predicted, gold, max_distance_px=15.0),
            "binary_mask": binary_mask_metrics(result.labels, gold_labels),
            "instances_iou_0_5": instance_iou_metrics(result.labels, gold_labels, iou_threshold=0.5),
        },
        "human_gold_tracking_markers": {
            "note": "Marker masks validate detections, not full-cell segmentation area.",
            "centroids_10px": match_centroids(predicted, tracking, max_distance_px=10.0),
            "centroids_15px_sensitivity": match_centroids(predicted, tracking, max_distance_px=15.0),
        },
        "computer_silver_segmentation": {
            "note": "Diagnostic only; this is algorithm-origin silver truth, not gold.",
            "centroids_10px": match_centroids(predicted, silver, max_distance_px=10.0),
            "binary_mask": binary_mask_metrics(result.labels, silver_labels),
        },
    }
    write_json(metrics, output_dir / "comparison_metrics.json")
    write_json(
        {
            "canonical_reference": "gold_detections.csv",
            "coordinate_convention": "zero-based pixels; x=column, y=row",
            "dataset": DATASET_NAME,
            "sequence": "02",
            "frame": 25,
            "references": {
                "human_gold_segmentation": {
                    "path": _relative_display(gold_seg_path, root),
                    "sha256": sha256_file(gold_seg_path),
                    "object_count": len(gold),
                },
                "human_gold_tracking_markers": {
                    "path": _relative_display(gold_track_path, root),
                    "sha256": sha256_file(gold_track_path),
                    "object_count": len(tracking),
                },
                "computer_silver_segmentation": {
                    "path": _relative_display(silver_path, root),
                    "sha256": sha256_file(silver_path),
                    "object_count": len(silver),
                    "status": "diagnostic_only_not_gold",
                },
            },
        },
        gold_dir / "manifest.json",
    )
    save_overlay(
        load_image(raw_path),
        result.labels,
        output_dir / "gold_comparison_overlay.png",
        reference_labels=gold_labels,
        predicted_centroids=[(row["x"], row["y"]) for row in predicted],
        reference_centroids=[(row["x"], row["y"]) for row in gold],
    )
    f1 = metrics["human_gold_segmentation"]["centroids_10px"]["f1"]
    print(
        f"Easy case complete: {len(predicted)} predictions, {len(gold)} human-gold objects, "
        f"centroid F1={f1:.3f}, {elapsed:.3f}s"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cell-detect",
        description="Turn a raw phase-contrast image into cell labels and centroid detections.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect = subparsers.add_parser("detect", help="Segment one raw image and write detections")
    detect.add_argument("input")
    detect.add_argument("--output-dir", required=True)
    detect.add_argument("--config")
    detect.add_argument("--dataset", default="")
    detect.add_argument("--sequence", default="")
    detect.add_argument("--frame", type=int)
    detect.set_defaults(handler=command_detect)

    reference = subparsers.add_parser("reference", help="Extract centroids from a reference label mask")
    reference.add_argument("mask")
    reference.add_argument("--image", required=True)
    reference.add_argument("--output-csv", required=True)
    reference.add_argument("--dataset", default="")
    reference.add_argument("--sequence", default="")
    reference.add_argument("--frame", type=int, default=0)
    reference.add_argument("--source", default="reference")
    reference.set_defaults(handler=command_reference)

    compare = subparsers.add_parser("compare", help="Compare predicted and reference centroid CSV files")
    compare.add_argument("predicted")
    compare.add_argument("reference")
    compare.add_argument("--output-json", required=True)
    compare.add_argument("--max-distance", type=float, default=10.0)
    compare.set_defaults(handler=command_compare)

    easy = subparsers.add_parser("easy-case", help="Reproduce sequence 02, frame 025 and its references")
    easy.add_argument("--data-root", default=".")
    easy.add_argument("--config", default="configs/easy_case.json")
    easy.add_argument("--output-dir", default="artifacts/easy_case/sequence_02_frame_025")
    easy.add_argument("--gold-dir", default="artifacts/gold_standard/sequence_02_frame_025")
    easy.set_defaults(handler=command_easy_case)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    sys.exit(main())
