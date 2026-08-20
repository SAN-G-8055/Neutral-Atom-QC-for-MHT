"""Exercise association quality metrics and balanced quantum sampling regression cases."""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path

import pytest

from overnight_benchmark import (
    OvernightBenchmarkConfig,
    _add_tracking_quality_metrics,
    _select_quantum_candidates,
    build_synthetic_scenarios,
    run_overnight_benchmark,
)


def _metric_row(
    scenario: str,
    frame: int,
    truth_x: tuple[float, ...],
    tracks: tuple[tuple[int, float], ...],
    *,
    status: str = "optimal",
) -> dict[str, object]:
    return {
        "scenario": scenario,
        "frame": frame,
        "exact_status": status,
        "ground_truth_count": len(truth_x),
        "active_tracks": len(tracks),
        "ground_truth_json": json.dumps(
            [
                {"object_id": index + 1, "x_px": x, "y_px": 0.0}
                for index, x in enumerate(truth_x)
            ]
        ),
        "track_positions_json": json.dumps(
            [
                {"track_id": track_id, "x_px": x, "y_px": 0.0}
                for track_id, x in tracks
            ]
        ),
    }


def test_tracking_metrics_count_identity_switches_and_reacquisition_fragments() -> None:
    rows = [
        _metric_row("swap", 0, (0.0, 10.0), ((1, 0.0), (2, 10.0))),
        _metric_row("swap", 1, (0.0, 10.0), ((1, 10.0), (2, 0.0))),
        _metric_row("swap", 2, (0.0, 10.0), ()),
        _metric_row("swap", 3, (0.0, 10.0), ((1, 0.0), (2, 10.0))),
    ]

    _add_tracking_quality_metrics(rows, maximum_distance_px=1.0)

    assert rows[0]["matched_track_gt_identity_correctness"] == pytest.approx(1.0)
    assert rows[0]["id_switch_count"] == 0
    assert rows[1]["matched_track_gt_identity_correctness"] == pytest.approx(0.0)
    assert rows[1]["id_switch_count"] == 2
    assert rows[2]["tracking_recall"] == pytest.approx(0.0)
    # Fragmentation is charged on reacquisition, not on the first missed frame.
    assert rows[2]["fragmentation_count"] == 0
    assert rows[3]["fragmentation_count"] == 2
    assert rows[3]["id_switch_count"] == 2
    assert rows[3]["cumulative_matched_track_gt_identity_correctness"] == pytest.approx(
        4 / 6
    )
    assert rows[3]["cumulative_id_switch_count"] == 4
    assert rows[3]["cumulative_fragmentation_count"] == 2


def test_exact_error_gap_does_not_invent_a_fragmentation() -> None:
    rows = [
        _metric_row("error-gap", 0, (0.0,), ((1, 0.0),)),
        _metric_row("error-gap", 1, (0.0,), (), status="error"),
        _metric_row("error-gap", 2, (0.0,), ((1, 0.0),)),
    ]

    _add_tracking_quality_metrics(rows, maximum_distance_px=1.0)

    assert rows[1]["tracking_metrics_status"] == "not_available_exact_failure"
    assert rows[1]["fragmentation_count"] is None
    assert rows[2]["fragmentation_count"] == 0
    assert rows[2]["id_switch_count"] == 0


def test_quantum_quota_is_balanced_by_axis_severity_and_component_size() -> None:
    scenarios = build_synthetic_scenarios(
        axes=("baseline",),
        severity_levels=(0.5,),
        object_counts=(1,),
        seeds=(0,),
        frame_count=1,
        image_shape=(32, 32),
    )
    config = OvernightBenchmarkConfig(
        scenarios=scenarios,
        run_quantum=False,
        quantum_quota_per_stratum=1,
    )
    rows = OrderedDict()
    for axis, severity in (("motion", 0.25), ("motion", 0.75), ("dropout", 0.25)):
        for frame in range(3):
            scenario = f"{axis}-{severity}-{frame}"
            rows[(scenario, frame)] = {
                "axis": axis,
                "severity": severity,
                "exact_status": "optimal",
                "maximum_nonclique_component_nodes": 4,
            }

    selected = _select_quantum_candidates(rows, config)
    reversed_selected = _select_quantum_candidates(
        OrderedDict(reversed(tuple(rows.items()))), config
    )

    assert selected == reversed_selected
    assert len(selected) == 3
    selected_strata = {
        (
            rows[key]["axis"],
            rows[key]["severity"],
            rows[key]["maximum_nonclique_component_nodes"],
        )
        for key in selected
    }
    assert selected_strata == {
        ("motion", 0.25, 4),
        ("motion", 0.75, 4),
        ("dropout", 0.25, 4),
    }


def test_legacy_quantum_quota_alias_normalizes_for_resume(tmp_path: Path) -> None:
    scenarios = build_synthetic_scenarios(
        axes=("baseline", "motion"),
        severity_levels=(0.5,),
        object_counts=(2,),
        seeds=(4,),
        frame_count=2,
        image_shape=(64, 80),
    )
    legacy = OvernightBenchmarkConfig(
        scenarios=scenarios,
        output_directory=tmp_path,
        run_quantum=False,
        quantum_quota_per_size=1,
    )
    assert legacy.quantum_quota_per_stratum == 1
    assert legacy.quantum_quota_per_size is None

    first = run_overnight_benchmark(legacy)
    assert {
        "matched_track_gt_identity_correctness",
        "id_switch_count",
        "fragmentation_count",
        "cumulative_id_switch_count",
        "cumulative_fragmentation_count",
    } <= first.records[0].keys()
    assert {
        "matched_track_gt_identity_correctness",
        "id_switch_count",
        "fragmentation_count",
        "tracking_recall",
        "tracking_precision",
    } <= first.summary.keys()
    resumed = run_overnight_benchmark(
        OvernightBenchmarkConfig(
            scenarios=scenarios,
            output_directory=tmp_path,
            run_quantum=False,
            quantum_quota_per_stratum=1,
        )
    )

    first_candidates = {
        (row["scenario"], row["frame"])
        for row in first.records
        if row["quantum_candidate"]
    }
    resumed_candidates = {
        (row["scenario"], row["frame"])
        for row in resumed.records
        if row["quantum_candidate"]
    }
    assert resumed_candidates == first_candidates

    with pytest.raises(ValueError, match="alias disagree"):
        OvernightBenchmarkConfig(
            scenarios=scenarios,
            quantum_quota_per_stratum=2,
            quantum_quota_per_size=1,
        )
