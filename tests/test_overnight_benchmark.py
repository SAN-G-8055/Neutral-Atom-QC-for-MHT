"""Exercise the small public surface of the resumable overnight benchmark."""

from __future__ import annotations

from dataclasses import replace
import csv
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

import overnight_benchmark as benchmark_module
from classical_solver import ClassicalSolver
from neutral_atom import NeutralAtomRun, QuantumSolver
from overnight_benchmark import (
    BenchmarkScenario,
    OvernightBenchmarkConfig,
    _match_detections,
    build_synthetic_scenarios,
    run_overnight_benchmark,
)
from solver import SolverInput, SolverSelection
from synthetic_data import QUANTUM_DEMO_DATA_CONFIG


def test_thresholded_hungarian_matching_maximizes_valid_cardinality_first() -> None:
    labels = np.zeros((40, 40), dtype=np.uint16)
    labels[20, 20] = 1
    labels[20, 29] = 2
    cosine = 41.0 / 162.0
    detections = (
        SimpleNamespace(detection_id=1, x_px=20.0, y_px=20.0, area_px=1),
        SimpleNamespace(
            detection_id=2,
            x_px=20.0 + 9.0 * cosine,
            y_px=20.0 + 9.0 * np.sqrt(1.0 - cosine**2),
            area_px=1,
        ),
    )

    metrics = _match_detections(
        detections,
        labels,
        maximum_distance_px=10.0,
    )

    # Pure minimum-total-distance assignment chooses distances 0 + 11 and
    # loses one match after thresholding.  The cardinality-first optimum is
    # the crossed assignment with distances 9 + 9.
    assert metrics["matched_detection_count"] == 2
    assert metrics["detection_recall"] == 1.0
    assert metrics["detection_precision"] == 1.0
    assert {
        (match["object_id"], match["detection_id"])
        for match in json.loads(str(metrics["detection_matches_json"]))
    } == {(1, 2), (2, 1)}


def _smoke_scenarios():
    return build_synthetic_scenarios(
        axes=("baseline", "motion"),
        severity_levels=(0.5,),
        object_counts=(2,),
        seeds=(1,),
        frame_count=2,
        image_shape=(64, 80),
    )


def test_scenario_builder_is_factorial_and_changes_only_requested_controls() -> None:
    baseline, motion, dropout, clutter, sensor, combined = (
        build_synthetic_scenarios(
            axes=(
                "baseline",
                "motion",
                "dropout",
                "clutter",
                "sensor_noise",
                "combined",
            ),
            severity_levels=(0.5,),
            object_counts=(4,),
            seeds=(7,),
            frame_count=3,
            image_shape=(72, 96),
        )
    )

    assert baseline.severity == 0.0
    assert motion.config.speed_px_per_frame > baseline.config.speed_px_per_frame
    assert motion.config.detection_probability == baseline.config.detection_probability
    assert dropout.config.detection_probability < baseline.config.detection_probability
    assert dropout.config.clutter_per_frame == baseline.config.clutter_per_frame
    assert clutter.config.clutter_per_frame > baseline.config.clutter_per_frame
    assert sensor.config.pixel_noise_sigma > baseline.config.pixel_noise_sigma
    assert combined.config.speed_px_per_frame == motion.config.speed_px_per_frame
    assert combined.config.detection_probability == dropout.config.detection_probability
    assert combined.config.clutter_per_frame == clutter.config.clutter_per_frame
    assert combined.config.pixel_noise_sigma == sensor.config.pixel_noise_sigma
    assert all(item.config.frame_count == 3 for item in (baseline, motion, combined))


def test_scenario_order_finishes_broad_seed_replicates_before_next_seed() -> None:
    scenarios = build_synthetic_scenarios(
        axes=("baseline", "motion", "dropout"),
        severity_levels=(0.2, 0.4),
        object_counts=(3,),
        seeds=(0, 1),
        frame_count=1,
        image_shape=(32, 40),
    )

    expected_conditions = [
        ("baseline", 0.0),
        ("motion", 0.2),
        ("dropout", 0.2),
        ("motion", 0.4),
        ("dropout", 0.4),
    ]
    assert [scenario.config.seed for scenario in scenarios] == [0] * 5 + [1] * 5
    assert [
        (scenario.axis, scenario.severity) for scenario in scenarios[:5]
    ] == expected_conditions
    assert all(
        scenario.config.speed_px_per_frame_override is not None
        and scenario.config.detection_probability_override is not None
        and scenario.config.clutter_per_frame_override is not None
        and scenario.config.pixel_noise_sigma_override is not None
        for scenario in scenarios
    )


def test_exact_campaign_checkpoints_exports_and_resumes(tmp_path: Path) -> None:
    events: list[dict[str, object]] = []
    config = OvernightBenchmarkConfig(
        scenarios=_smoke_scenarios(),
        output_directory=tmp_path,
        run_quantum=False,
        progress_every_frames=1,
    )

    result = run_overnight_benchmark(config, progress=events.append)

    assert result.stopped_reason is None
    assert len(result.records) == 4
    assert result.summary["checkpointed_exact_frames"] == 4
    assert result.summary["exact_complete"] is True
    assert result.summary["campaign_complete"] is True
    assert result.summary["exact_failures"] == 0
    assert result.database_path.is_file()
    assert result.csv_path.is_file()
    assert len(events) == 4
    assert {
        "scenario",
        "detection_recall",
        "graph_nodes",
        "maximum_nonclique_component_nodes",
        "active_tracks",
        "exact_objective",
        "quantum_status",
    } <= result.records[0].keys()
    assert not any(row["quantum_attempted"] for row in result.records)
    assert result.summary["quantum_attempts"] == 0
    assert all(row["tracks_json"] is None for row in result.records)
    assert all(row["graph_nodes_json"] is None for row in result.records)
    assert all(row["track_ids_json"].startswith("[") for row in result.records)
    assert {row["quantum_status"] for row in result.records} <= {
        "pending",
        "not_run_no_nonclique",
        "not_run_quota",
    }

    with sqlite3.connect(result.database_path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM exact_frames").fetchone()[0]
    assert count == 4
    with result.csv_path.open(newline="", encoding="utf-8") as handle:
        assert len(tuple(csv.DictReader(handle))) == 4

    resumed = run_overnight_benchmark(
        replace(config, wall_time_seconds=60.0, progress_every_frames=0)
    )
    assert len(resumed.records) == 4
    assert resumed.summary["checkpointed_exact_frames"] == 4


def test_compatible_exact_checkpoints_can_seed_a_reduced_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_config = OvernightBenchmarkConfig(
        scenarios=_smoke_scenarios(),
        output_directory=tmp_path / "source",
        run_quantum=False,
    )
    source_result = run_overnight_benchmark(source_config)
    target_config = OvernightBenchmarkConfig(
        scenarios=_smoke_scenarios()[:1],
        output_directory=tmp_path / "target",
        exact_checkpoint_source=source_result.database_path,
        run_quantum=False,
    )

    def unexpected_exact_recomputation(*args, **kwargs):
        raise AssertionError("imported exact frames must not be recomputed")

    monkeypatch.setattr(
        benchmark_module,
        "_process_exact_frame",
        unexpected_exact_recomputation,
    )
    result = run_overnight_benchmark(target_config)

    assert result.stopped_reason is None
    assert len(result.records) == 2
    assert result.summary["imported_exact_frames"] == 2
    assert result.summary["forward_work_seconds"] == 0.0
    assert result.summary["campaign_complete"] is True
    with sqlite3.connect(source_result.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM exact_frames").fetchone()[0] == 4


def test_short_budget_resume_replays_for_free_and_always_advances(
    tmp_path: Path,
) -> None:
    scenarios = build_synthetic_scenarios(
        axes=("baseline",),
        severity_levels=(0.5,),
        object_counts=(2,),
        seeds=(3,),
        frame_count=3,
        image_shape=(64, 80),
    )
    config = OvernightBenchmarkConfig(
        scenarios=scenarios,
        output_directory=tmp_path,
        run_quantum=False,
        forward_work_budget_seconds=1e-9,
    )

    first = run_overnight_benchmark(config)
    second = run_overnight_benchmark(config)
    third = run_overnight_benchmark(config)

    assert first.stopped_reason == "forward_work_budget_exhausted"
    assert second.stopped_reason == "forward_work_budget_exhausted"
    assert len(first.records) == 1
    assert len(second.records) == 2
    assert len(third.records) == 3
    assert third.stopped_reason is None
    assert third.summary["exact_complete"] is True
    assert third.summary["exact_pending_frames"] == 0


class _FailOnceAtFrameOne(ClassicalSolver):
    def __init__(self) -> None:
        super().__init__(maximum_component_nodes=64)
        self.failed = False

    def _select(self, solver_input: SolverInput) -> SolverSelection:
        if solver_input.frame == 1 and not self.failed:
            self.failed = True
            raise RuntimeError("transient exact failure")
        return super()._select(solver_input)


class _UnsupportedAtFrameOne(ClassicalSolver):
    def _select(self, solver_input: SolverInput) -> SolverSelection:
        if solver_input.frame == 1:
            return SolverSelection(
                status="unsupported_size",
                diagnostics={"message": "intentional terminal test failure"},
            )
        return super()._select(solver_input)


def test_generic_exact_error_stops_scenario_then_retries_on_resume(
    tmp_path: Path,
) -> None:
    scenario = build_synthetic_scenarios(
        axes=("baseline",),
        severity_levels=(0.2,),
        object_counts=(2,),
        seeds=(4,),
        frame_count=3,
        image_shape=(64, 80),
    )
    config = OvernightBenchmarkConfig(
        scenarios=scenario,
        output_directory=tmp_path,
        run_quantum=False,
    )
    solver = _FailOnceAtFrameOne()

    failed = run_overnight_benchmark(config, exact_solver=solver)
    resumed = run_overnight_benchmark(config, exact_solver=solver)

    assert [(row["frame"], row["exact_status"]) for row in failed.records] == [
        (0, "optimal"),
        (1, "error"),
    ]
    assert failed.stopped_reason == "exact_errors_pending_retry"
    assert failed.summary["exact_screen_finished"] is False
    assert failed.summary["exact_complete"] is False
    assert failed.summary["exact_pending_frames"] == 2
    assert len(resumed.records) == 3
    assert all(row["exact_status"] == "optimal" for row in resumed.records)
    assert resumed.stopped_reason is None
    assert resumed.summary["exact_complete"] is True


def test_terminal_exact_failure_blocks_tail_but_not_other_scenarios(
    tmp_path: Path,
) -> None:
    scenarios = build_synthetic_scenarios(
        axes=("baseline",),
        severity_levels=(0.2,),
        object_counts=(2,),
        seeds=(5, 6),
        frame_count=3,
        image_shape=(64, 80),
    )
    config = OvernightBenchmarkConfig(
        scenarios=scenarios,
        output_directory=tmp_path,
        run_quantum=False,
    )

    result = run_overnight_benchmark(
        config,
        exact_solver=_UnsupportedAtFrameOne(maximum_component_nodes=64),
    )

    assert len(result.records) == 4
    assert {(row["data_seed"], row["frame"]) for row in result.records} == {
        (5, 0),
        (5, 1),
        (6, 0),
        (6, 1),
    }
    assert sum(row["exact_status"] == "unsupported_size" for row in result.records) == 2
    assert result.stopped_reason == "exact_failures_recorded"
    assert result.summary["exact_screen_finished"] is True
    assert result.summary["exact_complete"] is False
    assert result.summary["exact_pending_frames"] == 0
    assert result.summary["exact_blocked_frames"] == 2
    assert result.summary["campaign_complete"] is True


def test_manifest_allows_new_time_budget_but_rejects_a_different_grid(
    tmp_path: Path,
) -> None:
    config = OvernightBenchmarkConfig(
        scenarios=_smoke_scenarios()[:1],
        output_directory=tmp_path,
        run_quantum=False,
    )
    run_overnight_benchmark(config)

    changed_scenario = replace(
        config.scenarios[0],
        config=replace(config.scenarios[0].config, seed=99),
    )
    with pytest.raises(ValueError, match="manifest is incompatible"):
        run_overnight_benchmark(
            replace(config, scenarios=(changed_scenario,), wall_time_seconds=30.0)
        )


class _ExactFakeRunner:
    """Return one exact sample without importing Pulser."""

    backend_name = "test_exact_enumerator"

    def execute(self, component) -> NeutralAtomRun:
        positions = {node_id: index for index, node_id in enumerate(component.node_ids)}
        best_weight = float("-inf")
        best_bits = "0" * len(component.node_ids)
        for mask in range(1 << len(component.node_ids)):
            selected = {
                node_id
                for index, node_id in enumerate(component.node_ids)
                if mask & (1 << index)
            }
            if any(left in selected and right in selected for left, right in component.edges):
                continue
            weight = sum(
                component.weights[positions[node_id]] for node_id in selected
            )
            bits = "".join(
                "1" if node_id in selected else "0" for node_id in component.node_ids
            )
            if weight > best_weight or (weight == best_weight and bits < best_bits):
                best_weight, best_bits = weight, bits
        return NeutralAtomRun(
            component_id=component.component_id,
            node_ids=component.node_ids,
            atom_order=component.qubit_ids,
            bitstring_counts=((best_bits, 1),),
            coordinates=tuple(
                (float(index), float(index % 2))
                for index in range(len(component.node_ids))
            ),
            mapping_cost=0.125,
            mapping_success=True,
            execution_mode="test",
        )


class _CountingClassicalSolver(ClassicalSolver):
    def __init__(self) -> None:
        super().__init__(maximum_component_nodes=64)
        self.selection_calls = 0

    def _select(self, solver_input: SolverInput) -> SolverSelection:
        self.selection_calls += 1
        return super()._select(solver_input)


def test_quantum_resume_replays_exact_selections_without_reoptimizing(
    tmp_path: Path,
) -> None:
    config = OvernightBenchmarkConfig(
        scenarios=(
            BenchmarkScenario(
                name="quantum-demo-replay",
                config=QUANTUM_DEMO_DATA_CONFIG,
                axis="baseline",
                severity=0.0,
            ),
        ),
        output_directory=tmp_path,
        run_quantum=False,
        quantum_max_nonclique_component_nodes=8,
        quantum_quota_per_stratum=1,
    )
    classical = _CountingClassicalSolver()
    quantum = QuantumSolver(maximum_component_nodes=8, runner=_ExactFakeRunner())
    screened = run_overnight_benchmark(
        config,
        exact_solver=classical,
        quantum_solver=quantum,
    )
    assert screened.summary["exact_complete"] is True
    assert classical.selection_calls == QUANTUM_DEMO_DATA_CONFIG.frame_count

    classical.selection_calls = 0
    completed = run_overnight_benchmark(
        replace(config, run_quantum=True),
        exact_solver=classical,
        quantum_solver=quantum,
    )

    assert completed.summary["quantum_attempts"] > 0
    assert classical.selection_calls == 0


def test_quantum_candidates_use_an_injected_runner_and_join_exact_metrics(
    tmp_path: Path,
) -> None:
    config = OvernightBenchmarkConfig(
        scenarios=(
            BenchmarkScenario(
                name="quantum-demo",
                config=QUANTUM_DEMO_DATA_CONFIG,
                axis="baseline",
                severity=0.0,
            ),
        ),
        output_directory=tmp_path,
        quantum_max_nonclique_component_nodes=8,
        quantum_quota_per_size=1,
    )
    quantum = QuantumSolver(maximum_component_nodes=8, runner=_ExactFakeRunner())

    result = run_overnight_benchmark(config, quantum_solver=quantum)

    candidates = [row for row in result.records if row["quantum_candidate"]]
    assert candidates
    assert all(row["quantum_attempted"] is True for row in candidates)
    assert all(row["quantum_status"] == "completed" for row in candidates)
    assert all(row["relative_objective"] == pytest.approx(1.0) for row in candidates)
    assert all(row["objective_gap"] == pytest.approx(0.0) for row in candidates)
    assert all(row["selection_agrees"] is True for row in candidates)
    assert all(row["maximum_mapping_cost"] is not None for row in candidates)
    assert result.summary["quantum_successes"] == len(candidates)
