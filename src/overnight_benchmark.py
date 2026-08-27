"""Resumable synthetic benchmarks for classical and neutral-atom tracking.

The public surface is deliberately small: build an interpretable scenario grid
with :func:`build_synthetic_scenarios`, put it in an
:class:`OvernightBenchmarkConfig`, and pass that configuration to
:func:`run_overnight_benchmark`.  Frames are streamed rather than written as
TIFF files, and every completed frame is committed to a stdlib SQLite database
before the next one begins.

The classical pass screens the complete grid.  A second, optional pass chooses
a deterministic quota of frames in every difficulty-axis, severity, and
frame-maximum non-clique component-size stratum, replays the same exact
tracking trajectory to those frames, and invokes ``QuantumSolver.execute`` on
the immutable frame graph.  No optional quantum dependency is imported until
such a candidate is actually executed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import closing
from dataclasses import asdict, dataclass, field, is_dataclass, replace
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import fsum, isfinite, sqrt
from pathlib import Path
import sqlite3
from time import perf_counter
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from classical_solver import ClassicalSolver
from graph import cluster_graph
from hpc import HPC, HPCConfig, PreparedFrame
from neutral_atom import QuantumSolver
from solver import SUCCESS_STATUSES, SolverResult
from synthetic_data import SyntheticDataConfig, SyntheticDataGenerator


BENCHMARK_SCHEMA_VERSION = "2.2"
DEFAULT_BENCHMARK_OUTPUT = Path("outputs") / "overnight_benchmark"
DEFAULT_AXES = (
    "baseline",
    "motion",
    "dropout",
    "clutter",
    "sensor_noise",
    "combined",
)
DEFAULT_SEVERITY_LEVELS = (0.2, 0.4, 0.6, 0.8, 1.0)
DEFAULT_OBJECT_COUNTS = (4, 12, 30, 55)
DEFAULT_SEEDS = tuple(range(5))
DEFAULT_QUANTUM_QUOTA_PER_STRATUM = 3

_QUANTUM_COLUMNS = (
    "quantum_attempted",
    "quantum_status",
    "quantum_objective",
    "relative_objective",
    "objective_gap",
    "selection_agrees",
    "selection_jaccard",
    "quantum_runtime_seconds",
    "simulated_component_count",
    "maximum_mapping_cost",
    "quantum_selected_ids_json",
    "quantum_diagnostics_json",
    "mapping_diagnostics_json",
    "quantum_error",
)


@dataclass(frozen=True, slots=True)
class BenchmarkScenario:
    """One named member of a synthetic benchmark grid."""

    name: str
    config: SyntheticDataConfig
    axis: str = "custom"
    severity: float = 0.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("scenario name must be non-empty")
        if not self.axis.strip():
            raise ValueError("scenario axis must be non-empty")
        severity = float(self.severity)
        if not isfinite(severity) or not 0.0 <= severity <= 1.0:
            raise ValueError("scenario severity must lie in [0, 1]")
        object.__setattr__(self, "severity", severity)

    @property
    def synthetic_config(self) -> SyntheticDataConfig:
        """Alias that reads naturally in exploratory notebook cells."""

        return self.config

    @property
    def family(self) -> str:
        """Return the varied difficulty axis."""

        return self.axis

    @property
    def level(self) -> float:
        """Return the normalized difficulty level."""

        return self.severity


def build_synthetic_scenarios(
    base_config: SyntheticDataConfig | None = None,
    *,
    axes: Iterable[str] = DEFAULT_AXES,
    severity_levels: Iterable[float] = DEFAULT_SEVERITY_LEVELS,
    object_counts: Iterable[int] = DEFAULT_OBJECT_COUNTS,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    frame_count: int = 40,
    image_shape: tuple[int, int] = (576, 720),
) -> tuple[BenchmarkScenario, ...]:
    """Build a factorial but interpretable overnight scenario grid.

    ``baseline`` is emitted once per object-count/seed pair.  Every other axis
    is emitted at each severity.  Isolated axes change exactly one synthetic
    override; ``combined`` changes all four.  The same seed is deliberately
    reused across axes so differences are attributable to the requested
    difficulty rather than a new random scene.
    """

    normalized_axes = tuple(str(axis).strip().lower() for axis in axes)
    unknown = set(normalized_axes) - set(DEFAULT_AXES)
    if unknown:
        raise ValueError(f"unknown benchmark axes: {', '.join(sorted(unknown))}")
    if len(normalized_axes) != len(set(normalized_axes)):
        raise ValueError("benchmark axes must be unique")

    levels = tuple(float(value) for value in severity_levels)
    if not levels or any(not isfinite(value) or not 0.0 < value <= 1.0 for value in levels):
        raise ValueError("severity_levels must contain values in (0, 1]")
    if len(levels) != len(set(levels)):
        raise ValueError("severity_levels must be unique")

    counts = tuple(int(value) for value in object_counts)
    seed_values = tuple(int(value) for value in seeds)
    if not counts or any(value < 1 for value in counts):
        raise ValueError("object_counts must contain positive integers")
    if not seed_values or any(value < 0 for value in seed_values):
        raise ValueError("seeds must contain non-negative integers")
    if len(counts) != len(set(counts)) or len(seed_values) != len(set(seed_values)):
        raise ValueError("object_counts and seeds must each be unique")
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if len(image_shape) != 2 or min(image_shape) < 1:
        raise ValueError("image_shape must contain two positive dimensions")

    base = base_config or SyntheticDataConfig(noise=0.0)
    baseline_values = {
        "speed_px_per_frame_override": base.speed_px_per_frame,
        "detection_probability_override": base.detection_probability,
        "clutter_per_frame_override": base.clutter_per_frame,
        "pixel_noise_sigma_override": base.pixel_noise_sigma,
    }
    scenarios: list[BenchmarkScenario] = []
    for axis in normalized_axes:
        axis_levels = (0.0,) if axis == "baseline" else levels
        for severity in axis_levels:
            overrides = dict(baseline_values)
            if axis in {"motion", "combined"}:
                overrides["speed_px_per_frame_override"] = (
                    base.speed_px_per_frame + 14.0 * severity
                )
            if axis in {"dropout", "combined"}:
                overrides["detection_probability_override"] = max(
                    0.05,
                    base.detection_probability - 0.50 * severity,
                )
            if axis in {"clutter", "combined"}:
                overrides["clutter_per_frame_override"] = (
                    base.clutter_per_frame + 36.0 * severity
                )
            if axis in {"sensor_noise", "combined"}:
                overrides["pixel_noise_sigma_override"] = (
                    base.pixel_noise_sigma + 6.0 * severity
                )

            for object_count in counts:
                for seed in seed_values:
                    severity_token = f"{round(100 * severity):03d}"
                    name = (
                        f"{axis}__severity-{severity_token}"
                        f"__objects-{object_count:03d}__seed-{seed:03d}"
                    )
                    dataset_name = "SYN-MHT-BENCH-" + name.replace("_", "-")
                    synthetic_config = replace(
                        base,
                        frame_count=int(frame_count),
                        object_count=object_count,
                        seed=seed,
                        dataset_name=dataset_name,
                        image_shape=(int(image_shape[0]), int(image_shape[1])),
                        **overrides,
                    )
                    scenarios.append(
                        BenchmarkScenario(
                            name=name,
                            config=synthetic_config,
                            axis=axis,
                            severity=severity,
                        )
                    )
    # Finish one broad replicate at a time.  Within a seed, severity rounds
    # interleave the requested axes before moving to the next seed, so an
    # interrupted overnight campaign is not an axis-biased prefix.
    nonbaseline_axes = tuple(axis for axis in normalized_axes if axis != "baseline")
    axis_rank = {axis: index for index, axis in enumerate(nonbaseline_axes)}
    level_rank = {level: index for index, level in enumerate(levels)}

    def condition_rank(scenario: BenchmarkScenario) -> int:
        if scenario.axis == "baseline":
            return 0
        return (
            1
            + level_rank[scenario.severity] * max(1, len(nonbaseline_axes))
            + axis_rank[scenario.axis]
        )

    scenarios.sort(
        key=lambda scenario: (
            scenario.config.seed,
            condition_rank(scenario),
            scenario.config.object_count,
        )
    )
    return tuple(scenarios)


@dataclass(frozen=True, slots=True)
class OvernightBenchmarkConfig:
    """Controls for a checkpointed exact screen and quantum subsample."""

    scenarios: tuple[BenchmarkScenario, ...] = field(
        default_factory=build_synthetic_scenarios
    )
    output_directory: Path = DEFAULT_BENCHMARK_OUTPUT
    database_filename: str = "benchmark.sqlite3"
    csv_filename: str = "benchmark_results.csv"
    exact_checkpoint_source: Path | None = None
    hpc_config: HPCConfig = field(default_factory=HPCConfig)
    match_distance_px: float = 8.0
    exact_maximum_component_nodes: int = 64
    run_quantum: bool = True
    quantum_max_nonclique_component_nodes: int = 8
    quantum_quota_per_stratum: int | None = None
    quantum_quota_per_size: int | None = None
    store_detailed_records: bool = False
    forward_work_budget_seconds: float | None = None
    wall_time_seconds: float | None = None
    progress_every_frames: int = 25
    resume: bool = True

    def __post_init__(self) -> None:
        scenarios = tuple(self.scenarios)
        if not scenarios:
            raise ValueError("scenarios must not be empty")
        names = tuple(scenario.name for scenario in scenarios)
        if len(names) != len(set(names)):
            raise ValueError("scenario names must be unique")
        object.__setattr__(self, "scenarios", scenarios)
        object.__setattr__(self, "output_directory", Path(self.output_directory))
        if self.exact_checkpoint_source is not None:
            object.__setattr__(
                self,
                "exact_checkpoint_source",
                Path(self.exact_checkpoint_source),
            )
        for name in ("database_filename", "csv_filename"):
            value = str(getattr(self, name))
            if not value or value in {".", ".."} or Path(value).name != value:
                raise ValueError(f"{name} must be one file name")
        distance = float(self.match_distance_px)
        if not isfinite(distance) or distance <= 0.0:
            raise ValueError("match_distance_px must be finite and positive")
        object.__setattr__(self, "match_distance_px", distance)
        if self.exact_maximum_component_nodes < 1:
            raise ValueError("exact_maximum_component_nodes must be positive")
        if self.quantum_max_nonclique_component_nodes < 1:
            raise ValueError(
                "quantum_max_nonclique_component_nodes must be positive"
            )
        requested_quota = self.quantum_quota_per_stratum
        legacy_quota = self.quantum_quota_per_size
        if (
            requested_quota is not None
            and legacy_quota is not None
            and int(requested_quota) != int(legacy_quota)
        ):
            raise ValueError(
                "quantum_quota_per_stratum and the legacy "
                "quantum_quota_per_size alias disagree"
            )
        quota = int(
            requested_quota
            if requested_quota is not None
            else (
                legacy_quota
                if legacy_quota is not None
                else DEFAULT_QUANTUM_QUOTA_PER_STRATUM
            )
        )
        if quota < 0:
            raise ValueError("quantum_quota_per_stratum must be non-negative")
        object.__setattr__(self, "quantum_quota_per_stratum", quota)
        # Normalize legacy construction so old and new spellings produce the
        # same scientific manifest and can resume the same checkpoint.
        object.__setattr__(self, "quantum_quota_per_size", None)
        if not isinstance(self.store_detailed_records, bool):
            raise ValueError("store_detailed_records must be a boolean")
        normalized_budgets: dict[str, float] = {}
        for name in ("forward_work_budget_seconds", "wall_time_seconds"):
            supplied = getattr(self, name)
            if supplied is None:
                continue
            value = float(supplied)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
            normalized_budgets[name] = value
        if (
            len(normalized_budgets) == 2
            and self.forward_work_budget_seconds != self.wall_time_seconds
        ):
            raise ValueError(
                "forward_work_budget_seconds and legacy wall_time_seconds "
                "must agree when both are supplied"
            )
        if self.progress_every_frames < 0:
            raise ValueError("progress_every_frames must be non-negative")

    @property
    def database_path(self) -> Path:
        return self.output_directory / self.database_filename

    @property
    def csv_path(self) -> Path:
        return self.output_directory / self.csv_filename

    @property
    def effective_forward_work_budget_seconds(self) -> float | None:
        """Budget for new checkpoints, excluding deterministic replay.

        ``wall_time_seconds`` remains a compatibility alias.  This is not a
        hard timeout: an individual solver call is uninterruptible and is
        allowed to finish atomically after it starts.
        """

        if self.forward_work_budget_seconds is not None:
            return self.forward_work_budget_seconds
        return self.wall_time_seconds


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Notebook-friendly result returned after a complete or paused campaign."""

    records: tuple[dict[str, object], ...]
    summary: dict[str, object]
    database_path: Path
    csv_path: Path
    stopped_reason: str | None = None


ProgressCallback = Callable[[Mapping[str, object]], None]


def run_overnight_benchmark(
    config: OvernightBenchmarkConfig | None = None,
    *,
    exact_solver: ClassicalSolver | None = None,
    quantum_solver: QuantumSolver | None = None,
    progress: ProgressCallback | None = None,
) -> BenchmarkResult:
    """Run or resume a benchmark and export one joined tidy CSV.

    ``progress`` receives small dictionaries at the configured interval.  A
    compatible ``exact_checkpoint_source`` is opened read-only and contributes
    only successful classical records; quantum records are never imported.
    Durable exact selections restore tracker state without re-running the
    classical optimizer.  A forward-work-budget stop is normal: the current
    frame is already durable, the CSV is refreshed, and a later call with the
    same scientific grid resumes it.  Replay of durable prefixes does not
    consume this budget, and an individual solver call cannot be interrupted.
    Budgets, progress settings, and ``run_quantum`` may change between calls;
    scientific settings may not.
    """

    benchmark = config or OvernightBenchmarkConfig()
    benchmark.output_directory.mkdir(parents=True, exist_ok=True)
    database_path = benchmark.database_path
    if database_path.exists() and not benchmark.resume:
        raise FileExistsError(
            f"refusing to overwrite existing benchmark database: {database_path}"
        )

    started = perf_counter()
    # Reconstruction of an already-checkpointed prefix is deterministic
    # overhead, not forward campaign work.  Budget only new frame checkpoints
    # so even a very short resumed run can always make progress.
    forward_work_seconds = 0.0
    classical = exact_solver or ClassicalSolver(
        maximum_component_nodes=benchmark.exact_maximum_component_nodes
    )
    neutral_atom = quantum_solver or QuantumSolver(
        maximum_component_nodes=(
            benchmark.quantum_max_nonclique_component_nodes
        )
    )
    stopped_reason: str | None = None
    imported_exact_frames = 0

    with closing(sqlite3.connect(database_path, timeout=30.0)) as connection:
        _initialize_database(
            connection,
            benchmark,
            exact_solver=classical,
            quantum_solver=neutral_atom,
        )
        if benchmark.exact_checkpoint_source is not None:
            imported_exact_frames = _import_exact_checkpoints(
                connection,
                benchmark.exact_checkpoint_source,
                benchmark,
                exact_solver=classical,
            )
        existing_exact = _load_table(connection, "exact_frames")
        processed_events = 0

        for scenario in benchmark.scenarios:
            scenario_existing = {
                frame: existing_exact[(scenario.name, frame)]
                for frame in range(scenario.config.frame_count)
                if (scenario.name, frame) in existing_exact
            }
            scenario_state = _exact_scenario_state(scenario, scenario_existing)
            if scenario_state in {"complete", "terminal_failure"}:
                continue

            tracker = HPC(benchmark.hpc_config, sequence=scenario.config.sequence)
            generator = SyntheticDataGenerator(scenario.config)
            for frame, (image, labels) in enumerate(generator.iter_frames()):
                checkpoint = scenario_existing.get(frame)
                replaying_success = (
                    checkpoint is not None
                    and checkpoint.get("exact_status") in SUCCESS_STATUSES
                )
                if checkpoint is not None and not (
                    replaying_success or checkpoint.get("exact_status") == "error"
                ):
                    # A declared solver status such as unsupported_size is a
                    # terminal methodological failure for this scenario.
                    break
                if not replaying_success and _budget_exhausted(
                    forward_work_seconds,
                    benchmark.effective_forward_work_budget_seconds,
                ):
                    stopped_reason = "forward_work_budget_exhausted"
                    break
                if replaying_success:
                    _replay_exact_checkpoint(
                        tracker,
                        image,
                        checkpoint,
                        frame=frame,
                    )
                    continue
                frame_started = perf_counter()
                record, _, _ = _process_exact_frame(
                    scenario,
                    tracker,
                    classical,
                    image,
                    labels,
                    frame=frame,
                    match_distance_px=benchmark.match_distance_px,
                    store_detailed_records=benchmark.store_detailed_records,
                )
                forward_work_seconds += perf_counter() - frame_started
                _store_record(connection, "exact_frames", record)
                existing_exact[(scenario.name, frame)] = record
                processed_events += 1
                _report_progress(
                    progress,
                    benchmark,
                    processed_events,
                    phase="exact",
                    scenario=scenario.name,
                    frame=frame,
                    completed=len(existing_exact),
                    total=sum(item.config.frame_count for item in benchmark.scenarios),
                )
                if record.get("exact_status") not in SUCCESS_STATUSES:
                    # State after a failed exact handoff is undefined.  Leave
                    # all later frames absent and continue with the next
                    # independent scenario.  Generic errors are retried by a
                    # later invocation; declared solver failures are terminal.
                    break
            if stopped_reason is not None:
                break

        expected_exact = sum(
            scenario.config.frame_count for scenario in benchmark.scenarios
        )
        exact_states = {
            scenario.name: _exact_scenario_state(
                scenario,
                {
                    frame: existing_exact[(scenario.name, frame)]
                    for frame in range(scenario.config.frame_count)
                    if (scenario.name, frame) in existing_exact
                },
            )
            for scenario in benchmark.scenarios
        }
        exact_screen_finished = all(
            state != "pending" for state in exact_states.values()
        )
        exact_complete = all(
            state == "complete" for state in exact_states.values()
        )
        candidates: tuple[tuple[str, int], ...] = ()
        if exact_screen_finished:
            candidates = _select_quantum_candidates(existing_exact, benchmark)

        if (
            stopped_reason is None
            and exact_screen_finished
            and benchmark.run_quantum
            and candidates
        ):
            existing_quantum = _load_table(connection, "quantum_frames")
            by_scenario: dict[str, set[int]] = {}
            for scenario_name, frame in candidates:
                by_scenario.setdefault(scenario_name, set()).add(frame)

            for scenario in benchmark.scenarios:
                target_frames = by_scenario.get(scenario.name)
                if not target_frames:
                    continue
                if all(
                    _quantum_checkpoint_is_terminal(
                        existing_quantum.get((scenario.name, frame))
                    )
                    for frame in target_frames
                ):
                    continue

                tracker = HPC(benchmark.hpc_config, sequence=scenario.config.sequence)
                generator = SyntheticDataGenerator(scenario.config)
                final_target = max(target_frames)
                for frame, (image, _) in enumerate(generator.iter_frames()):
                    if frame > final_target:
                        break
                    prepared, exact_result = _replay_exact_checkpoint(
                        tracker,
                        image,
                        existing_exact[(scenario.name, frame)],
                        frame=frame,
                    )
                    if frame not in target_frames:
                        continue
                    key = (scenario.name, frame)
                    if _quantum_checkpoint_is_terminal(existing_quantum.get(key)):
                        continue
                    if _budget_exhausted(
                        forward_work_seconds,
                        benchmark.effective_forward_work_budget_seconds,
                    ):
                        stopped_reason = "forward_work_budget_exhausted"
                        break
                    quantum_started = perf_counter()
                    quantum_record = _process_quantum_frame(
                        scenario,
                        frame,
                        prepared,
                        exact_result,
                        existing_exact[key],
                        neutral_atom,
                    )
                    forward_work_seconds += perf_counter() - quantum_started
                    _store_record(connection, "quantum_frames", quantum_record)
                    existing_quantum[key] = quantum_record
                    processed_events += 1
                    _report_progress(
                        progress,
                        benchmark,
                        processed_events,
                        phase="quantum",
                        scenario=scenario.name,
                        frame=frame,
                        completed=sum(
                            _quantum_checkpoint_is_terminal(
                                existing_quantum.get(candidate)
                            )
                            for candidate in candidates
                        ),
                        total=len(candidates),
                    )
                    if quantum_record["quantum_status"] == "dependency_missing":
                        stopped_reason = "dependency_missing"
                        break
                if stopped_reason is not None:
                    break

        exact_rows = _load_table(connection, "exact_frames")
        quantum_rows = _load_table(connection, "quantum_frames")

    records = _join_records(
        exact_rows,
        quantum_rows,
        candidates,
        benchmark.scenarios,
        benchmark,
        candidate_selection_final=exact_screen_finished,
    )
    if stopped_reason is None and any(
        row.get("quantum_status") == "error" for row in records
    ):
        stopped_reason = "quantum_errors_pending_retry"
    if stopped_reason is None and any(
        row.get("exact_status") == "error" for row in records
    ):
        stopped_reason = "exact_errors_pending_retry"
    if stopped_reason is None and not exact_complete:
        stopped_reason = "exact_failures_recorded"
    _write_csv(benchmark.csv_path, records)
    summary = _summarize(
        records,
        benchmark,
        expected_exact=sum(
            scenario.config.frame_count for scenario in benchmark.scenarios
        ),
        candidate_count=len(candidates),
        elapsed_seconds=perf_counter() - started,
        forward_work_seconds=forward_work_seconds,
        imported_exact_frames=imported_exact_frames,
    )
    return BenchmarkResult(
        records=records,
        summary=summary,
        database_path=database_path,
        csv_path=benchmark.csv_path,
        stopped_reason=stopped_reason,
    )


def _process_exact_frame(
    scenario: BenchmarkScenario,
    tracker: HPC,
    solver: ClassicalSolver,
    image: np.ndarray,
    labels: np.ndarray,
    *,
    frame: int,
    match_distance_px: float,
    store_detailed_records: bool,
) -> tuple[dict[str, object], PreparedFrame | None, SolverResult | None]:
    synthetic = scenario.config
    record: dict[str, object] = {
        "scenario": scenario.name,
        "axis": scenario.axis,
        "severity": scenario.severity,
        "object_count": synthetic.object_count,
        "data_seed": synthetic.seed,
        "frame": frame,
        "synthetic_config_json": _canonical_json(asdict(synthetic)),
        "speed_px_per_frame": synthetic.speed_px_per_frame,
        "detection_probability": synthetic.detection_probability,
        "clutter_per_frame": synthetic.clutter_per_frame,
        "pixel_noise_sigma": synthetic.pixel_noise_sigma,
        "ground_truth_count": 0,
        "detection_count": 0,
        "matched_detection_count": 0,
        "detection_recall": 0.0,
        "detection_precision": 0.0,
        "localization_rmse_px": None,
        "detections_json": "[]",
        "ground_truth_json": "[]",
        "detection_matches_json": "[]",
        "graph_nodes": 0,
        "graph_edges": 0,
        "graph_nodes_json": None,
        "graph_edges_json": None,
        "component_count": 0,
        "component_sizes_json": "[]",
        "component_edge_counts_json": "[]",
        "maximum_component_nodes": 0,
        "maximum_nonclique_component_nodes": 0,
        "nonclique_component_sizes_json": "[]",
        "active_tracks": len(tracker.tracks),
        "assigned_observations": 0,
        "track_ids_json": _canonical_json(
            [track.track_id for track in tracker.tracks]
        ),
        "track_positions_json": _canonical_json(
            [
                {
                    "track_id": track.track_id,
                    "x_px": track.position[0],
                    "y_px": track.position[1],
                }
                for track in tracker.tracks
            ]
        ),
        "tracks_json": (
            _canonical_json([asdict(track) for track in tracker.tracks])
            if store_detailed_records
            else None
        ),
        "assignments_json": "[]",
        "assigned_observation_ids_json": "[]",
        "exact_status": "error",
        "exact_objective": None,
        "exact_runtime_seconds": None,
        "exact_selected_ids_json": "[]",
        "exact_diagnostics_json": "{}",
        "input_fingerprint": None,
        "exact_error": None,
    }
    prepared: PreparedFrame | None = None
    solver_result: SolverResult | None = None
    try:
        prepared = tracker.prepare_frame(image, frame=frame)
        detections = prepared.observed_frame.detection.detections  # type: ignore[union-attr]
        detection_metrics = _match_detections(
            detections,
            labels,
            maximum_distance_px=match_distance_px,
        )
        record.update(detection_metrics)
        record.update(
            _graph_metrics(
                prepared,
                store_detailed_records=store_detailed_records,
            )
        )

        solver_result = tracker.solve(prepared, solver)
        record.update(
            {
                "exact_status": solver_result.status,
                "exact_objective": solver_result.objective,
                "exact_runtime_seconds": solver_result.runtime_seconds,
                "exact_selected_ids_json": _canonical_json(
                    solver_result.selected_ids
                ),
                "exact_diagnostics_json": _canonical_json(
                    solver_result.diagnostics
                ),
                "input_fingerprint": solver_result.input_fingerprint,
            }
        )
        if solver_result.successful:
            frame_result = tracker.advance(prepared, solver_result)
            selected = set(solver_result.selected_ids)
            assignments = [
                {
                    "node_id": node.node_id,
                    "track_id": node.track_id,
                    "observation_id": node.observation_id,
                    "weight": node.weight,
                }
                for node in prepared.graph.nodes
                if node.node_id in selected
            ]
            record.update(
                {
                    "active_tracks": len(frame_result.tracks),
                    "assigned_observations": len(
                        frame_result.assigned_observation_ids
                    ),
                    "tracks_json": _canonical_json(
                        [asdict(track) for track in frame_result.tracks]
                    ) if store_detailed_records else None,
                    "track_ids_json": _canonical_json(
                        [track.track_id for track in frame_result.tracks]
                    ),
                    "track_positions_json": _canonical_json(
                        [
                            {
                                "track_id": track.track_id,
                                "x_px": track.position[0],
                                "y_px": track.position[1],
                            }
                            for track in frame_result.tracks
                        ]
                    ),
                    "assignments_json": _canonical_json(assignments),
                    "assigned_observation_ids_json": _canonical_json(
                        frame_result.assigned_observation_ids
                    ),
                }
            )
    except Exception as exc:  # campaign failures are frame-local checkpoints
        record["exact_status"] = "error"
        record["exact_error"] = f"{type(exc).__name__}: {exc}"
        record["active_tracks"] = len(tracker.tracks)
        record["track_ids_json"] = _canonical_json(
            [track.track_id for track in tracker.tracks]
        )
        record["track_positions_json"] = _canonical_json(
            [
                {
                    "track_id": track.track_id,
                    "x_px": track.position[0],
                    "y_px": track.position[1],
                }
                for track in tracker.tracks
            ]
        )
        record["tracks_json"] = (
            _canonical_json([asdict(track) for track in tracker.tracks])
            if store_detailed_records
            else None
        )
    return record, prepared, solver_result


def _match_detections(
    detections: Sequence[Any],
    labels: np.ndarray,
    *,
    maximum_distance_px: float,
) -> dict[str, object]:
    ground_truth: list[dict[str, object]] = []
    for label_id in (int(value) for value in np.unique(labels) if value != 0):
        rows, columns = np.nonzero(labels == label_id)
        ground_truth.append(
            {
                "object_id": label_id,
                "x_px": float(np.mean(columns)),
                "y_px": float(np.mean(rows)),
            }
        )
    observed = [
        {
            "detection_id": int(item.detection_id),
            "x_px": float(item.x_px),
            "y_px": float(item.y_px),
            "area_px": int(item.area_px),
        }
        for item in detections
    ]

    matches: list[dict[str, object]] = []
    distances: list[float] = []
    if observed and ground_truth:
        cost = np.empty((len(ground_truth), len(observed)), dtype=float)
        for truth_index, truth in enumerate(ground_truth):
            for detection_index, detection in enumerate(observed):
                cost[truth_index, detection_index] = float(
                    np.hypot(
                        float(truth["x_px"]) - float(detection["x_px"]),
                        float(truth["y_px"]) - float(detection["y_px"]),
                    )
                )
        # A plain minimum-distance assignment can sacrifice a valid match to
        # reduce the total length of an invalid (> threshold) pair.  Give every
        # invalid edge a penalty larger than the total possible valid distance:
        # Hungarian assignment then maximizes valid cardinality first and, only
        # among those assignments, minimizes localization distance.
        assignment_size = min(len(ground_truth), len(observed))
        invalid_penalty = (
            (assignment_size + 1) * maximum_distance_px + 1.0
        )
        thresholded_cost = np.where(
            cost <= maximum_distance_px,
            cost,
            invalid_penalty,
        )
        truth_indices, detection_indices = linear_sum_assignment(thresholded_cost)
        for truth_index, detection_index in zip(
            truth_indices, detection_indices, strict=True
        ):
            distance = float(cost[truth_index, detection_index])
            if distance > maximum_distance_px:
                continue
            distances.append(distance)
            matches.append(
                {
                    "object_id": ground_truth[truth_index]["object_id"],
                    "detection_id": observed[detection_index]["detection_id"],
                    "distance_px": distance,
                }
            )

    match_count = len(matches)
    truth_count = len(ground_truth)
    detection_count = len(observed)
    return {
        "ground_truth_count": truth_count,
        "detection_count": detection_count,
        "matched_detection_count": match_count,
        "detection_recall": match_count / truth_count if truth_count else 1.0,
        "detection_precision": (
            match_count / detection_count
            if detection_count
            else (1.0 if not truth_count else 0.0)
        ),
        "localization_rmse_px": (
            sqrt(fsum(distance * distance for distance in distances) / match_count)
            if match_count
            else None
        ),
        "detections_json": _canonical_json(observed),
        "ground_truth_json": _canonical_json(ground_truth),
        "detection_matches_json": _canonical_json(matches),
    }


def _graph_metrics(
    prepared: PreparedFrame,
    *,
    store_detailed_records: bool,
) -> dict[str, object]:
    components = cluster_graph(prepared.graph)
    edge_set = set(prepared.graph.edges)
    sizes: list[int] = []
    edge_counts: list[int] = []
    nonclique_sizes: list[int] = []
    for component in components:
        members = set(component.node_ids)
        node_count = len(members)
        edge_count = sum(
            left in members and right in members for left, right in edge_set
        )
        sizes.append(node_count)
        edge_counts.append(edge_count)
        if edge_count != node_count * (node_count - 1) // 2:
            nonclique_sizes.append(node_count)
    return {
        "graph_nodes": len(prepared.graph.nodes),
        "graph_edges": len(prepared.graph.edges),
        "graph_nodes_json": (
            _canonical_json([node.to_dict() for node in prepared.graph.nodes])
            if store_detailed_records
            else None
        ),
        "graph_edges_json": (
            _canonical_json(prepared.graph.edges)
            if store_detailed_records
            else None
        ),
        "component_count": len(components),
        "component_sizes_json": _canonical_json(sizes),
        "component_edge_counts_json": _canonical_json(edge_counts),
        "maximum_component_nodes": max(sizes, default=0),
        "maximum_nonclique_component_nodes": max(nonclique_sizes, default=0),
        "nonclique_component_sizes_json": _canonical_json(nonclique_sizes),
    }


def _select_quantum_candidates(
    exact_rows: Mapping[tuple[str, int], Mapping[str, object]],
    config: OvernightBenchmarkConfig,
) -> tuple[tuple[str, int], ...]:
    by_stratum: dict[tuple[str, float, int], list[tuple[str, int]]] = {}
    for key, row in exact_rows.items():
        stratum = _quantum_stratum(row, config)
        if stratum is not None:
            by_stratum.setdefault(stratum, []).append(key)

    selected: list[tuple[str, int]] = []
    for stratum in sorted(by_stratum):
        stable = sorted(
            by_stratum[stratum],
            key=lambda key: (
                sha256(f"{key[0]}:{key[1]}".encode("utf-8")).hexdigest(),
                key,
            ),
        )
        selected.extend(stable[: config.quantum_quota_per_stratum])
    return tuple(sorted(selected))


def _quantum_stratum(
    row: Mapping[str, object],
    config: OvernightBenchmarkConfig,
) -> tuple[str, float, int] | None:
    """Return an eligible frame's balanced sampling stratum.

    Object count and seed deliberately remain outside the stratum.  They can
    therefore vary within the deterministic sample while every requested
    difficulty axis, severity, and supported component size receives its own
    quota.
    """

    size = int(row.get("maximum_nonclique_component_nodes", 0))
    if (
        row.get("exact_status") not in SUCCESS_STATUSES
        or not 1 <= size <= config.quantum_max_nonclique_component_nodes
    ):
        return None
    return (str(row.get("axis", "custom")), float(row.get("severity", 0.0)), size)


def _process_quantum_frame(
    scenario: BenchmarkScenario,
    frame: int,
    prepared: PreparedFrame | None,
    exact_result: SolverResult | None,
    exact_record: Mapping[str, object],
    solver: QuantumSolver,
) -> dict[str, object]:
    record: dict[str, object] = {
        "scenario": scenario.name,
        "frame": frame,
        "quantum_status": "error",
        "quantum_objective": None,
        "relative_objective": None,
        "objective_gap": None,
        "selection_agrees": None,
        "selection_jaccard": None,
        "quantum_runtime_seconds": None,
        "simulated_component_count": None,
        "maximum_mapping_cost": None,
        "quantum_selected_ids_json": "[]",
        "quantum_diagnostics_json": "{}",
        "mapping_diagnostics_json": "[]",
        "quantum_error": None,
    }
    started = perf_counter()
    try:
        if prepared is None or exact_result is None or not exact_result.successful:
            raise RuntimeError("exact trajectory did not produce a candidate input")
        execution = solver.execute(prepared.solver_input())
        runtime = perf_counter() - started
        diagnostics = getattr(execution, "diagnostics", {})
        runs = tuple(getattr(execution, "runs", ()))
        mapping_diagnostics = [
            {
                "component_id": int(run.component_id),
                "mapping_cost": float(run.mapping_cost),
                "mapping_success": bool(run.mapping_success),
                "execution_mode": str(run.execution_mode),
                "coordinates": run.coordinates,
            }
            for run in runs
        ]
        status = str(execution.status)
        selected_ids = tuple(int(value) for value in execution.selected_ids)
        record.update(
            {
                "quantum_status": status,
                "quantum_runtime_seconds": runtime,
                "quantum_selected_ids_json": _canonical_json(selected_ids),
                "quantum_diagnostics_json": _canonical_json(diagnostics),
                "mapping_diagnostics_json": _canonical_json(mapping_diagnostics),
                "simulated_component_count": _mapping_get(
                    diagnostics, "simulated_component_count"
                ),
                "maximum_mapping_cost": max(
                    (float(run.mapping_cost) for run in runs), default=None
                ),
            }
        )
        if status in SUCCESS_STATUSES:
            selected = set(selected_ids)
            quantum_objective = fsum(
                node.weight for node in prepared.graph.nodes if node.node_id in selected
            )
            exact_objective = float(exact_record["exact_objective"])
            exact_ids = set(exact_result.selected_ids)
            union = exact_ids | selected
            record.update(
                {
                    "quantum_objective": quantum_objective,
                    "relative_objective": (
                        quantum_objective / exact_objective
                        if exact_objective != 0.0
                        else (1.0 if quantum_objective == 0.0 else None)
                    ),
                    "objective_gap": exact_objective - quantum_objective,
                    "selection_agrees": selected == exact_ids,
                    "selection_jaccard": (
                        len(selected & exact_ids) / len(union) if union else 1.0
                    ),
                }
            )
        else:
            record["quantum_error"] = _mapping_get(diagnostics, "message")
    except Exception as exc:
        record["quantum_status"] = str(getattr(exc, "status", "error"))
        record["quantum_runtime_seconds"] = perf_counter() - started
        record["quantum_error"] = f"{type(exc).__name__}: {exc}"
    return record


def _initialize_database(
    connection: sqlite3.Connection,
    config: OvernightBenchmarkConfig,
    *,
    exact_solver: object,
    quantum_solver: object,
) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS manifest (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            manifest_json TEXT NOT NULL,
            created_utc TEXT NOT NULL
        )
        """
    )
    for table in ("exact_frames", "quantum_frames"):
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                scenario TEXT NOT NULL,
                frame INTEGER NOT NULL,
                record_json TEXT NOT NULL,
                committed_utc TEXT NOT NULL,
                PRIMARY KEY (scenario, frame)
            )
            """
        )

    manifest = _manifest(
        config,
        exact_solver=exact_solver,
        quantum_solver=quantum_solver,
    )
    manifest_json = _canonical_json(manifest)
    existing = connection.execute(
        "SELECT manifest_json FROM manifest WHERE singleton = 1"
    ).fetchone()
    if existing is None:
        connection.execute(
            "INSERT INTO manifest VALUES (1, ?, ?)",
            (manifest_json, _utc_now()),
        )
    elif existing[0] != manifest_json:
        raise ValueError(
            "benchmark database manifest is incompatible with this scientific grid; "
            "choose a new output directory or restore the original configuration"
        )
    connection.commit()


def _import_exact_checkpoints(
    connection: sqlite3.Connection,
    source_path: Path,
    config: OvernightBenchmarkConfig,
    *,
    exact_solver: object,
) -> int:
    """Import compatible successful exact records into a fresh campaign.

    The source database remains read-only.  Quantum records are deliberately
    ignored so a changed quantum method cannot silently reuse stale results.
    """

    source = source_path.resolve()
    target = config.database_path.resolve()
    if source == target:
        return 0
    if not source.is_file():
        raise FileNotFoundError(f"exact checkpoint source does not exist: {source}")

    target_scenarios = {scenario.name: scenario for scenario in config.scenarios}
    expected_methodology = {
        "hpc_config": asdict(config.hpc_config),
        "match_distance_px": config.match_distance_px,
        "exact_maximum_component_nodes": config.exact_maximum_component_nodes,
        "store_detailed_records": config.store_detailed_records,
        "exact_solver": _solver_descriptor(exact_solver),
    }
    imported = 0
    source_uri = f"{source.as_uri()}?mode=ro"
    with closing(sqlite3.connect(source_uri, uri=True, timeout=30.0)) as source_db:
        manifest_row = source_db.execute(
            "SELECT manifest_json FROM manifest WHERE singleton = 1"
        ).fetchone()
        if manifest_row is None:
            raise ValueError("exact checkpoint source has no benchmark manifest")
        source_manifest = json.loads(str(manifest_row[0]))
        for key, expected in expected_methodology.items():
            if source_manifest.get(key) != expected:
                raise ValueError(
                    "exact checkpoint source is incompatible with the current "
                    f"classical methodology ({key})"
                )

        for scenario_name, frame, record_json, committed_utc in source_db.execute(
            """
            SELECT scenario, frame, record_json, committed_utc
            FROM exact_frames
            """
        ):
            scenario = target_scenarios.get(str(scenario_name))
            frame_number = int(frame)
            if scenario is None or not 0 <= frame_number < scenario.config.frame_count:
                continue
            record = json.loads(str(record_json))
            if (
                record.get("scenario") != scenario.name
                or int(record.get("frame", -1)) != frame_number
                or record.get("synthetic_config_json")
                != _canonical_json(asdict(scenario.config))
            ):
                raise ValueError(
                    "exact checkpoint source contains a scenario-name collision "
                    f"for {scenario.name!r} frame {frame_number}"
                )
            if record.get("exact_status") not in SUCCESS_STATUSES:
                continue
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO exact_frames
                    (scenario, frame, record_json, committed_utc)
                VALUES (?, ?, ?, ?)
                """,
                (scenario.name, frame_number, str(record_json), str(committed_utc)),
            )
            imported += cursor.rowcount
    connection.commit()
    return imported


def _manifest(
    config: OvernightBenchmarkConfig,
    *,
    exact_solver: object,
    quantum_solver: object,
) -> dict[str, object]:
    return {
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "scenarios": [
            {
                "name": scenario.name,
                "axis": scenario.axis,
                "severity": scenario.severity,
                "config": asdict(scenario.config),
            }
            for scenario in config.scenarios
        ],
        "hpc_config": asdict(config.hpc_config),
        "match_distance_px": config.match_distance_px,
        "exact_maximum_component_nodes": config.exact_maximum_component_nodes,
        "quantum_max_nonclique_component_nodes": (
            config.quantum_max_nonclique_component_nodes
        ),
        "quantum_quota_per_stratum": config.quantum_quota_per_stratum,
        "store_detailed_records": config.store_detailed_records,
        "exact_solver": _solver_descriptor(exact_solver),
        "quantum_solver": _solver_descriptor(quantum_solver),
    }


def _solver_descriptor(solver: object) -> dict[str, object]:
    """Describe methodology without serializing live backend objects."""

    descriptor: dict[str, object] = {
        "class": f"{type(solver).__module__}.{type(solver).__qualname__}",
    }
    for name in ("solver_name", "maximum_component_nodes"):
        if hasattr(solver, name):
            descriptor[name] = getattr(solver, name)
    solver_config = getattr(solver, "config", None)
    if solver_config is not None and is_dataclass(solver_config):
        descriptor["config"] = asdict(solver_config)
    runner = getattr(solver, "runner", None)
    if runner is not None:
        runner_descriptor: dict[str, object] = {
            "class": f"{type(runner).__module__}.{type(runner).__qualname__}",
            "backend_name": getattr(runner, "backend_name", "injected_runner"),
        }
        runner_config = getattr(runner, "config", None)
        if runner_config is not None and is_dataclass(runner_config):
            runner_descriptor["config"] = asdict(runner_config)
        descriptor["runner"] = runner_descriptor
    return descriptor


def _store_record(
    connection: sqlite3.Connection,
    table: str,
    record: Mapping[str, object],
) -> None:
    if table not in {"exact_frames", "quantum_frames"}:
        raise ValueError("unknown checkpoint table")
    connection.execute(
        f"""
        INSERT INTO {table} (scenario, frame, record_json, committed_utc)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (scenario, frame) DO UPDATE SET
            record_json = excluded.record_json,
            committed_utc = excluded.committed_utc
        """,
        (
            record["scenario"],
            record["frame"],
            _canonical_json(record),
            _utc_now(),
        ),
    )
    connection.commit()


def _load_table(
    connection: sqlite3.Connection,
    table: str,
) -> dict[tuple[str, int], dict[str, object]]:
    if table not in {"exact_frames", "quantum_frames"}:
        raise ValueError("unknown checkpoint table")
    rows: dict[tuple[str, int], dict[str, object]] = {}
    for scenario, frame, record_json in connection.execute(
        f"SELECT scenario, frame, record_json FROM {table}"
    ):
        rows[(str(scenario), int(frame))] = json.loads(record_json)
    return rows


def _replay_exact_checkpoint(
    tracker: HPC,
    image: np.ndarray,
    checkpoint: Mapping[str, object],
    *,
    frame: int,
) -> tuple[PreparedFrame, SolverResult]:
    """Advance from a durable exact selection without optimizing it again."""

    status = str(checkpoint.get("exact_status", ""))
    if status not in SUCCESS_STATUSES:
        raise ValueError("only successful exact checkpoints can be replayed")
    prepared = tracker.prepare_frame(image, frame=frame)
    solver_input = prepared.solver_input()
    if checkpoint.get("input_fingerprint") != solver_input.fingerprint:
        raise RuntimeError(
            "checkpoint replay diverged from the recorded exact frame graph"
        )
    result = SolverResult(
        problem_id=solver_input.problem_id,
        input_fingerprint=solver_input.fingerprint,
        solver_name="classical_exact_checkpoint",
        selected_ids=tuple(
            int(node_id)
            for node_id in json.loads(
                str(checkpoint.get("exact_selected_ids_json", "[]"))
            )
        ),
        objective=float(checkpoint["exact_objective"]),
        feasible=True,
        status=status,
        runtime_seconds=float(checkpoint.get("exact_runtime_seconds") or 0.0),
        diagnostics=json.loads(
            str(checkpoint.get("exact_diagnostics_json") or "{}")
        ),
    )
    frame_result = tracker.advance(prepared, result)
    replayed_state = {
        "track_ids_json": _canonical_json(
            [track.track_id for track in frame_result.tracks]
        ),
        "track_positions_json": _canonical_json(
            [
                {
                    "track_id": track.track_id,
                    "x_px": track.position[0],
                    "y_px": track.position[1],
                }
                for track in frame_result.tracks
            ]
        ),
        "assigned_observation_ids_json": _canonical_json(
            frame_result.assigned_observation_ids
        ),
    }
    if any(
        checkpoint.get(key) != value for key, value in replayed_state.items()
    ):
        raise RuntimeError(
            "checkpoint replay diverged from the recorded exact trajectory"
        )
    return prepared, result


def _exact_scenario_state(
    scenario: BenchmarkScenario,
    rows: Mapping[int, Mapping[str, object]],
) -> str:
    """Classify a scenario checkpoint prefix for safe resume decisions."""

    ordered = [rows.get(frame) for frame in range(scenario.config.frame_count)]
    if all(
        row is not None and row.get("exact_status") in SUCCESS_STATUSES
        for row in ordered
    ):
        return "complete"
    if any(
        row is not None
        and row.get("exact_status") not in SUCCESS_STATUSES | {"error"}
        for row in ordered
    ):
        return "terminal_failure"
    return "pending"


def _quantum_checkpoint_is_terminal(
    checkpoint: Mapping[str, object] | None,
) -> bool:
    if checkpoint is None:
        return False
    return checkpoint.get("quantum_status") not in {"error", "dependency_missing"}


def _join_records(
    exact_rows: Mapping[tuple[str, int], Mapping[str, object]],
    quantum_rows: Mapping[tuple[str, int], Mapping[str, object]],
    candidates: Sequence[tuple[str, int]],
    scenarios: Sequence[BenchmarkScenario],
    config: OvernightBenchmarkConfig,
    *,
    candidate_selection_final: bool,
) -> tuple[dict[str, object], ...]:
    scenario_order = {scenario.name: index for index, scenario in enumerate(scenarios)}
    candidate_set = set(candidates)
    records: list[dict[str, object]] = []
    for key in sorted(
        exact_rows,
        key=lambda item: (scenario_order.get(item[0], len(scenario_order)), item[1]),
    ):
        record = dict(exact_rows[key])
        record["quantum_candidate"] = key in candidate_set
        stratum = _quantum_stratum(record, config)
        record["quantum_stratum_axis"] = stratum[0] if stratum else None
        record["quantum_stratum_severity"] = stratum[1] if stratum else None
        record["quantum_stratum_maximum_nonclique_nodes"] = (
            stratum[2] if stratum else None
        )
        for column in _QUANTUM_COLUMNS:
            record[column] = None
        quantum = quantum_rows.get(key)
        if quantum is not None:
            record.update(
                {column: quantum.get(column) for column in _QUANTUM_COLUMNS}
            )
            record["quantum_attempted"] = True
        else:
            record["quantum_attempted"] = False
            size = int(record.get("maximum_nonclique_component_nodes", 0))
            if record.get("exact_status") not in SUCCESS_STATUSES:
                record["quantum_status"] = "not_run_exact_failure"
            elif size == 0:
                record["quantum_status"] = "not_run_no_nonclique"
            elif size > config.quantum_max_nonclique_component_nodes:
                record["quantum_status"] = "not_run_unsupported_size"
            elif key in candidate_set or not candidate_selection_final:
                record["quantum_status"] = "pending"
            else:
                record["quantum_status"] = "not_run_quota"
        records.append(record)
    _add_tracking_quality_metrics(
        records,
        maximum_distance_px=config.match_distance_px,
    )
    return tuple(records)


@dataclass(slots=True)
class _TrackingQualityState:
    """Online identity state for one synthetic sequence.

    A retained tracker ID takes the identity of the first ground-truth object
    it is spatially matched to.  That fixed identity makes later cross-object
    associations objectively classifiable rather than redefining correctness
    after every frame.
    """

    canonical_gt_by_track: dict[int, int] = field(default_factory=dict)
    most_recent_track_by_gt: dict[int, int] = field(default_factory=dict)
    ever_matched_gt: set[int] = field(default_factory=set)
    interrupted_gt: set[int] = field(default_factory=set)
    cumulative_matches: int = 0
    cumulative_correct_matches: int = 0
    cumulative_id_switches: int = 0
    cumulative_fragmentations: int = 0


def _add_tracking_quality_metrics(
    records: Sequence[dict[str, object]],
    *,
    maximum_distance_px: float,
) -> None:
    """Add deterministic CLEAR-MOT-style association columns in place.

    Active post-update track positions and persistent synthetic labels are
    paired one-to-one with a maximum-cardinality, minimum-distance assignment
    inside ``maximum_distance_px``.  An ID switch is emitted when a ground-
    truth object is paired with a different tracker ID than at its most recent
    evaluated match (including after a gap).  A fragmentation is emitted only
    when an object is reacquired after at least one *successfully evaluated*
    frame in which it was unmatched.  Solver-error frames are marked
    unavailable and do not invent interruptions.
    """

    states: dict[str, _TrackingQualityState] = {}
    for record in records:
        state = states.setdefault(str(record.get("scenario", "")), _TrackingQualityState())
        record.update(
            {
                "tracking_metrics_status": "not_available_exact_failure",
                "matched_track_gt_count": None,
                "tracking_recall": None,
                "tracking_precision": None,
                "tracking_localization_rmse_px": None,
                "identity_correct_match_count": None,
                "matched_track_gt_identity_correctness": None,
                "id_switch_count": None,
                "fragmentation_count": None,
                "cumulative_matched_track_gt_count": state.cumulative_matches,
                "cumulative_identity_correct_match_count": (
                    state.cumulative_correct_matches
                ),
                "cumulative_matched_track_gt_identity_correctness": (
                    state.cumulative_correct_matches / state.cumulative_matches
                    if state.cumulative_matches
                    else None
                ),
                "cumulative_id_switch_count": state.cumulative_id_switches,
                "cumulative_fragmentation_count": state.cumulative_fragmentations,
                "track_gt_matches_json": "[]",
            }
        )
        if record.get("exact_status") not in SUCCESS_STATUSES:
            continue

        ground_truth = _json_object_list(record.get("ground_truth_json"))
        tracks = _json_object_list(record.get("track_positions_json"))
        matched_indices = _distance_limited_point_matches(
            ground_truth,
            tracks,
            maximum_distance_px=maximum_distance_px,
        )
        matched_gt_ids = {
            int(ground_truth[truth_index]["object_id"])
            for truth_index, _, _ in matched_indices
        }
        id_switches = 0
        fragmentations = 0
        correct_matches = 0
        match_records: list[dict[str, object]] = []
        for truth_index, track_index, distance in matched_indices:
            object_id = int(ground_truth[truth_index]["object_id"])
            track_id = int(tracks[track_index]["track_id"])
            canonical_object_id = state.canonical_gt_by_track.setdefault(
                track_id, object_id
            )
            identity_correct = canonical_object_id == object_id
            previous_track_id = state.most_recent_track_by_gt.get(object_id)
            id_switch = (
                previous_track_id is not None and previous_track_id != track_id
            )
            fragmentation = object_id in state.interrupted_gt
            correct_matches += int(identity_correct)
            id_switches += int(id_switch)
            fragmentations += int(fragmentation)
            state.most_recent_track_by_gt[object_id] = track_id
            match_records.append(
                {
                    "object_id": object_id,
                    "track_id": track_id,
                    "distance_px": distance,
                    "canonical_object_id": canonical_object_id,
                    "identity_correct": identity_correct,
                    "id_switch": id_switch,
                    "fragmentation": fragmentation,
                }
            )

        truth_ids = {int(item["object_id"]) for item in ground_truth}
        for object_id in truth_ids:
            if object_id in matched_gt_ids:
                state.interrupted_gt.discard(object_id)
            elif object_id in state.ever_matched_gt:
                state.interrupted_gt.add(object_id)
        state.ever_matched_gt.update(matched_gt_ids)

        match_count = len(matched_indices)
        truth_count = len(ground_truth)
        track_count = len(tracks)
        state.cumulative_matches += match_count
        state.cumulative_correct_matches += correct_matches
        state.cumulative_id_switches += id_switches
        state.cumulative_fragmentations += fragmentations
        record.update(
            {
                "tracking_metrics_status": "completed",
                "matched_track_gt_count": match_count,
                "tracking_recall": match_count / truth_count if truth_count else 1.0,
                "tracking_precision": (
                    match_count / track_count
                    if track_count
                    else (1.0 if not truth_count else 0.0)
                ),
                "tracking_localization_rmse_px": (
                    sqrt(
                        fsum(distance * distance for _, _, distance in matched_indices)
                        / match_count
                    )
                    if match_count
                    else None
                ),
                "identity_correct_match_count": correct_matches,
                "matched_track_gt_identity_correctness": (
                    correct_matches / match_count if match_count else None
                ),
                "id_switch_count": id_switches,
                "fragmentation_count": fragmentations,
                "cumulative_matched_track_gt_count": state.cumulative_matches,
                "cumulative_identity_correct_match_count": (
                    state.cumulative_correct_matches
                ),
                "cumulative_matched_track_gt_identity_correctness": (
                    state.cumulative_correct_matches / state.cumulative_matches
                    if state.cumulative_matches
                    else None
                ),
                "cumulative_id_switch_count": state.cumulative_id_switches,
                "cumulative_fragmentation_count": state.cumulative_fragmentations,
                "track_gt_matches_json": _canonical_json(match_records),
            }
        )


def _json_object_list(value: object) -> list[dict[str, object]]:
    if value is None:
        return []
    decoded = json.loads(str(value)) if isinstance(value, str) else value
    if not isinstance(decoded, list) or any(not isinstance(item, dict) for item in decoded):
        raise ValueError("benchmark point records must be JSON object lists")
    return decoded


def _distance_limited_point_matches(
    truth: Sequence[Mapping[str, object]],
    tracks: Sequence[Mapping[str, object]],
    *,
    maximum_distance_px: float,
) -> tuple[tuple[int, int, float], ...]:
    """Maximize match count, then minimize distance, under a hard radius."""

    truth_count = len(truth)
    track_count = len(tracks)
    if not truth_count or not track_count:
        return ()

    distances = np.empty((truth_count, track_count), dtype=float)
    for truth_index, truth_point in enumerate(truth):
        for track_index, track_point in enumerate(tracks):
            distances[truth_index, track_index] = float(
                np.hypot(
                    float(truth_point["x_px"]) - float(track_point["x_px"]),
                    float(truth_point["y_px"]) - float(track_point["y_px"]),
                )
            )

    # Dummy rows/columns represent unmatched points.  A valid real match costs
    # less than leaving both endpoints unmatched, while the finite forbidden
    # cost is larger than every possible all-dummy solution.  This avoids the
    # cardinality loss caused by running Hungarian first and thresholding later.
    unmatched_cost = maximum_distance_px + 1.0
    matrix_size = truth_count + track_count
    forbidden_cost = 4.0 * unmatched_cost * (matrix_size + 1)
    cost = np.full((matrix_size, matrix_size), forbidden_cost, dtype=float)
    valid = distances <= maximum_distance_px
    cost[:truth_count, :track_count] = np.where(
        valid, distances, forbidden_cost
    )
    for truth_index in range(truth_count):
        cost[truth_index, track_count + truth_index] = unmatched_cost
    for track_index in range(track_count):
        cost[truth_count + track_index, track_index] = unmatched_cost
    cost[truth_count:, track_count:] = 0.0

    row_indices, column_indices = linear_sum_assignment(cost)
    matches = [
        (int(row), int(column), float(distances[row, column]))
        for row, column in zip(row_indices, column_indices, strict=True)
        if row < truth_count and column < track_count and valid[row, column]
    ]
    return tuple(sorted(matches))


def _write_csv(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if records:
        fieldnames = list(records[0])
        extras = sorted(
            set().union(*(set(record) for record in records)) - set(fieldnames)
        )
        fieldnames.extend(extras)
    else:
        fieldnames = ["scenario", "frame", "exact_status", "quantum_status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _tracking_quality_summary(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    evaluated = [
        row for row in records if row.get("tracking_metrics_status") == "completed"
    ]
    matched = sum(int(row.get("matched_track_gt_count", 0)) for row in evaluated)
    correct = sum(int(row.get("identity_correct_match_count", 0)) for row in evaluated)
    truth_instances = sum(int(row.get("ground_truth_count", 0)) for row in evaluated)
    track_instances = sum(int(row.get("active_tracks", 0)) for row in evaluated)
    return {
        "tracking_evaluated_frames": len(evaluated),
        "matched_track_gt_count": matched,
        "identity_correct_match_count": correct,
        "matched_track_gt_identity_correctness": (
            correct / matched if matched else None
        ),
        "id_switch_count": sum(
            int(row.get("id_switch_count", 0)) for row in evaluated
        ),
        "fragmentation_count": sum(
            int(row.get("fragmentation_count", 0)) for row in evaluated
        ),
        "tracking_recall": matched / truth_instances if truth_instances else None,
        "tracking_precision": matched / track_instances if track_instances else None,
    }


def _summarize(
    records: Sequence[Mapping[str, object]],
    config: OvernightBenchmarkConfig,
    *,
    expected_exact: int,
    candidate_count: int,
    elapsed_seconds: float,
    forward_work_seconds: float,
    imported_exact_frames: int,
) -> dict[str, object]:
    exact_successes = sum(row.get("exact_status") in SUCCESS_STATUSES for row in records)
    quantum_attempts = sum(bool(row.get("quantum_attempted")) for row in records)
    quantum_successes = sum(
        row.get("quantum_status") in SUCCESS_STATUSES for row in records
    )
    rows_by_scenario: dict[str, dict[int, Mapping[str, object]]] = {}
    for row in records:
        rows_by_scenario.setdefault(str(row.get("scenario", "")), {})[
            int(row.get("frame", -1))
        ] = row
    exact_states = {
        scenario.name: _exact_scenario_state(
            scenario,
            rows_by_scenario.get(scenario.name, {}),
        )
        for scenario in config.scenarios
    }
    exact_screen_finished = all(state != "pending" for state in exact_states.values())
    exact_all_successful = all(state == "complete" for state in exact_states.values())
    exact_pending = sum(
        row is None or row.get("exact_status") == "error"
        for scenario in config.scenarios
        if exact_states[scenario.name] == "pending"
        for row in (
            rows_by_scenario.get(scenario.name, {}).get(frame)
            for frame in range(scenario.config.frame_count)
        )
    )
    exact_blocked = sum(
        frame not in rows_by_scenario.get(scenario.name, {})
        for scenario in config.scenarios
        if exact_states[scenario.name] == "terminal_failure"
        for frame in range(scenario.config.frame_count)
    )
    retryable_quantum_statuses = {"pending", "error", "dependency_missing"}
    quantum_pending = sum(
        bool(row.get("quantum_candidate"))
        and row.get("quantum_status") in retryable_quantum_statuses
        for row in records
    )
    # ``exact_complete`` is retained as the strict all-successful compatibility
    # flag.  ``exact_screen_finished`` also accepts terminal solver failures,
    # because those scenarios have no retryable work left.
    exact_complete = exact_all_successful
    quantum_complete = exact_screen_finished and quantum_pending == 0
    recalls = [
        float(row["detection_recall"])
        for row in records
        if row.get("detection_recall") is not None
    ]
    precisions = [
        float(row["detection_precision"])
        for row in records
        if row.get("detection_precision") is not None
    ]
    by_axis: dict[str, dict[str, object]] = {}
    for axis in dict.fromkeys(scenario.axis for scenario in config.scenarios):
        axis_rows = [row for row in records if row.get("axis") == axis]
        axis_recalls = [
            float(row["detection_recall"])
            for row in axis_rows
            if row.get("detection_recall") is not None
        ]
        by_axis[axis] = {
            "frames": len(axis_rows),
            "exact_successes": sum(
                row.get("exact_status") in SUCCESS_STATUSES for row in axis_rows
            ),
            "mean_detection_recall": (
                fsum(axis_recalls) / len(axis_recalls) if axis_recalls else None
            ),
            **_tracking_quality_summary(axis_rows),
        }
    tracking_summary = _tracking_quality_summary(records)
    candidate_strata = {
        (
            str(row["quantum_stratum_axis"]),
            float(row["quantum_stratum_severity"]),
            int(row["quantum_stratum_maximum_nonclique_nodes"]),
        )
        for row in records
        if row.get("quantum_candidate")
    }
    return {
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "scenario_count": len(config.scenarios),
        "expected_exact_frames": expected_exact,
        "checkpointed_exact_frames": len(records),
        "imported_exact_frames": imported_exact_frames,
        "exact_screen_finished": exact_screen_finished,
        "exact_all_successful": exact_all_successful,
        "exact_complete": exact_complete,
        "exact_pending_frames": exact_pending,
        "exact_blocked_frames": exact_blocked,
        "exact_successes": exact_successes,
        "exact_failures": len(records) - exact_successes,
        "quantum_candidates": candidate_count,
        "quantum_candidate_strata": len(candidate_strata),
        "quantum_quota_per_stratum": config.quantum_quota_per_stratum,
        "quantum_complete": quantum_complete,
        "quantum_pending_frames": quantum_pending,
        "quantum_attempts": quantum_attempts,
        "quantum_successes": quantum_successes,
        "quantum_failures": quantum_attempts - quantum_successes,
        "mean_detection_recall": fsum(recalls) / len(recalls) if recalls else None,
        "mean_detection_precision": (
            fsum(precisions) / len(precisions) if precisions else None
        ),
        **tracking_summary,
        "elapsed_seconds": elapsed_seconds,
        "forward_work_seconds": forward_work_seconds,
        "campaign_complete": exact_screen_finished and (
            not config.run_quantum or quantum_complete
        ),
        "by_axis": by_axis,
    }


def _report_progress(
    callback: ProgressCallback | None,
    config: OvernightBenchmarkConfig,
    processed_events: int,
    **event: object,
) -> None:
    if (
        callback is not None
        and config.progress_every_frames
        and processed_events % config.progress_every_frames == 0
    ):
        callback(dict(event))


def _budget_exhausted(spent_seconds: float, budget_seconds: float | None) -> bool:
    return budget_seconds is not None and spent_seconds >= budget_seconds


def _mapping_get(mapping: object, key: str) -> object | None:
    return mapping.get(key) if isinstance(mapping, Mapping) else None


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"cannot serialize benchmark value {type(value).__name__}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        _jsonable(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "DEFAULT_AXES",
    "DEFAULT_BENCHMARK_OUTPUT",
    "DEFAULT_OBJECT_COUNTS",
    "DEFAULT_QUANTUM_QUOTA_PER_STRATUM",
    "DEFAULT_SEEDS",
    "DEFAULT_SEVERITY_LEVELS",
    "BenchmarkResult",
    "BenchmarkScenario",
    "OvernightBenchmarkConfig",
    "build_synthetic_scenarios",
    "run_overnight_benchmark",
]
