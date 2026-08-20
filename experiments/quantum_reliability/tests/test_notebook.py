"""Keep the root notebook minimal, self-contained, and executable out of the box."""

from __future__ import annotations

import json
from pathlib import Path

from nbclient import NotebookClient
import nbformat
import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "user_notebook.ipynb"


def _payload() -> dict[str, object]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(cell: dict[str, object]) -> str:
    return "".join(cell["source"])


def _force_tiny_synthetic_run(
    cell: dict[str, object],
    output_root: Path,
    *,
    frame_count: int,
) -> None:
    """Point the data cell at a tiny synthetic scene below ``output_root``."""

    data = _source(cell)
    data = data.replace(
        "synthetic_data_config = QUANTUM_DEMO_DATA_CONFIG",
        "synthetic_data_config = SyntheticDataConfig(\n"
        "    noise=0.1,\n"
        f"    frame_count={frame_count},\n"
        "    object_count=2,\n"
        "    seed=0,\n"
        "    dataset_name='TINY-MHT',\n"
        "    image_shape=(64, 80),\n"
        ")",
    )
    data = data.replace(
        "project_root / DEFAULT_SYNTHETIC_DATA_ROOT",
        f"Path({output_root.as_posix()!r}) / DEFAULT_SYNTHETIC_DATA_ROOT",
    )
    cell["source"] = data


def test_notebook_has_clean_ordered_workflow_and_figure_cells() -> None:
    payload = _payload()
    cells = payload["cells"]

    assert payload["nbformat"] == 4
    assert [cell["id"] for cell in cells] == [
        "intro",
        "imports",
        "data",
        "config",
        "run",
        "run_many",
        "overnight_intro",
        "overnight_config",
        "overnight_run",
        "publication_figures",
        "figure_setup",
        "fig1_workflow",
        "fig2_detections",
        "figure_analysis",
        "fig3_conflict_graph",
        "fig4_neutral_atoms",
        "fig5_performance",
        "fig6_eligibility",
        "fig7_detection",
        "fig8_runtime",
    ]
    assert [
        index for index, cell in enumerate(cells) if cell["cell_type"] == "markdown"
    ] == [0, 6, 9]
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    assert all(cell.get("execution_count") is None for cell in code_cells)
    assert all(cell.get("outputs") == [] for cell in code_cells)
    for cell in code_cells:
        compile(_source(cell), f"user_notebook:{cell['id']}", "exec")


def test_notebook_uses_the_bounded_quantum_synthetic_data_by_default() -> None:
    intro, imports, data, config, run, run_many = map(
        _source, _payload()["cells"][:6]
    )

    assert "quantum-friendly synthetic sequence" in intro
    assert imports.index("sys.path.insert") < imports.index("from neutral_atom_mht")
    assert "from cell_data import DATASET_NAME" in imports
    assert "SyntheticDataGenerator" in imports
    assert "QUANTUM_DEMO_DATA_CONFIG" in imports
    assert "QuantumSolver" in imports
    assert 'project_root / "src"' in imports
    assert "USE_QUANTUM_DEMO_DATA = True" in data
    assert "or not raw_frame_path(real_dataset_root, frame).is_file()" in data
    assert "project_root / DEFAULT_SYNTHETIC_DATA_ROOT" in data
    assert "synthetic_data_config = QUANTUM_DEMO_DATA_CONFIG" in data
    assert "not synthetic_dataset.raw_frame_path(0).is_file()" in data
    assert ".generate(\n        synthetic_output_root" in data
    assert "frame = 0" in data
    assert "synthetic_dataset.raw_frame_path(frame)" in config
    assert "sequence = synthetic_dataset.config.sequence" in config
    assert "raw_frame_path(dataset_root, frame)" in config
    assert "HPC(config, sequence=sequence)" in config
    assert "# solver = ClassicalSolver(maximum_component_nodes=60)" in config
    assert "solver = QuantumSolver(maximum_component_nodes=8)" in config
    assert "load_tiff(frame_path)" in run
    assert "{dataset_label} sequence {sequence}" in run
    assert "RUN_MANY_FRAMES = True" in run_many
    assert "MANY_FRAME_COUNT = 3" in run_many
    assert "min(MANY_FRAME_COUNT, available_frames)" in run_many
    assert "sequence_controller.run_sequence(" in run_many


def test_notebook_figures_are_reproducible_and_compare_the_same_graph() -> None:
    cells = {cell["id"]: _source(cell) for cell in _payload()["cells"]}

    assert '"fig1_workflow.png"' in cells["fig1_workflow"]
    assert '"fig2_detection_overlays.png"' in cells["fig2_detections"]
    assert "Real sequence not installed" in cells["fig2_detections"]
    assert "load_tracking_labels(0)" in cells["fig2_detections"]
    assert "synthetic_truth" in cells["fig2_detections"]
    assert "SyntheticDataGenerator(QUANTUM_DEMO_DATA_CONFIG).iter_frames()" in cells["figure_analysis"]
    assert "example_quantum_solver.execute(" in cells["figure_analysis"]
    assert "example_exact_solver.solve(prepared_frame.solver_input())" in cells["figure_analysis"]
    assert "example_reference.advance(prepared_frame, exact_result)" in cells["figure_analysis"]
    assert "run.program is not None" in cells["figure_analysis"]
    assert '"fig3_conflict_graph.png"' in cells["fig3_conflict_graph"]
    assert "logical_layout(example_graph)" in cells["fig3_conflict_graph"]
    assert '"fig4_neutral_atom_embedding.png"' in cells["fig4_neutral_atoms"]
    assert "example_run.coordinates" in cells["fig4_neutral_atoms"]
    assert "program.omega" in cells["fig4_neutral_atoms"]
    assert "blockade_distance / 2.0" in cells["fig4_neutral_atoms"]
    assert "intended_edges & physical_edges" in cells["fig4_neutral_atoms"]
    assert '"fig5_quantum_reliability.png"' in cells["fig5_performance"]
    assert '"relative_objective"' in cells["fig5_performance"]
    assert '"quantum_attempted"' in cells["fig5_performance"]
    assert '"selection_agrees"' in cells["fig5_performance"]
    assert "failure_percentages" in cells["fig5_performance"]
    assert '"fig6_quantum_size_outcomes.png"' in cells["fig6_eligibility"]
    assert "maximum_nonclique_component_nodes" in cells["fig6_eligibility"]
    assert 'source_axis = "baseline"' in cells["fig6_eligibility"]
    assert "no_work" in cells["fig6_eligibility"]
    assert "supported" in cells["fig6_eligibility"]
    assert "oversized" in cells["fig6_eligibility"]
    assert '"fig7_detection_and_tracking_quality.png"' in cells["fig7_detection"]
    assert '"detection_recall"' in cells["fig7_detection"]
    assert '"detection_precision"' in cells["fig7_detection"]
    assert "scenario_quality_rows" in cells["fig7_detection"]
    assert '"cumulative_matched_track_gt_count"' in cells["fig7_detection"]
    assert '"cumulative_id_switch_count"' in cells["fig7_detection"]
    assert '"cumulative_fragmentation_count"' in cells["fig7_detection"]
    assert "per_100_matches" in cells["fig7_detection"]
    assert '"fig8_runtime_diagnostics.png"' in cells["fig8_runtime"]
    assert '"states_evaluated"' in cells["fig8_runtime"]
    assert '"quantum_runtime_seconds"' in cells["fig8_runtime"]
    assert 'row.get("exact_status") != "optimal"' in cells["fig8_runtime"]
    assert 'row.get("quantum_status") == "completed"' in cells["fig8_runtime"]
    assert "completion_rates" in cells["fig8_runtime"]


def test_notebook_defines_a_resumable_broad_overnight_campaign() -> None:
    cells = {cell["id"]: _source(cell) for cell in _payload()["cells"]}
    campaign = cells["overnight_config"]

    assert 'BENCHMARK_PROFILE = "overnight"' in campaign
    assert '("motion", "dropout", "clutter", "sensor_noise", "combined")' in campaign
    assert "BENCHMARK_SEVERITIES = (0.2, 0.4, 0.6, 0.8, 1.0)" in campaign
    assert "BENCHMARK_OBJECT_COUNTS = (4, 12, 30, 55)" in campaign
    assert "BENCHMARK_SEEDS = tuple(range(5))" in campaign
    assert "BENCHMARK_FRAME_COUNT = 40" in campaign
    assert "QUANTUM_QUOTA_PER_STRATUM = 5" in campaign
    assert "BENCHMARK_FORWARD_WORK_HOURS = 10.0" in campaign
    assert "forward_work_budget_seconds=BENCHMARK_FORWARD_WORK_HOURS" in campaign
    assert "quantum_quota_per_stratum=QUANTUM_QUOTA_PER_STRATUM" in campaign
    assert "build_synthetic_scenarios(" in campaign
    assert 'axes=("baseline", *BENCHMARK_AXES)' in campaign
    assert "exact_maximum_component_nodes=128" in campaign
    assert "quantum_max_nonclique_component_nodes=8" in campaign
    assert "store_detailed_records=False" in campaign
    assert "resume=True" in campaign
    assert "run_overnight_benchmark(" in cells["overnight_run"]
    assert "progress=lambda update: print(update, flush=True)" in cells["overnight_run"]
    assert "overnight_result.database_path" in cells["overnight_run"]
    assert "overnight_result.csv_path" in cells["overnight_run"]


@pytest.mark.filterwarnings(
    "ignore:Proactor event loop does not implement add_reader.*:RuntimeWarning"
)
def test_notebook_executes_end_to_end_without_real_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("PYTHONPATH", raising=False)
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    notebook.cells = notebook.cells[:5]
    _force_tiny_synthetic_run(notebook.cells[2], tmp_path, frame_count=1)
    notebook.cells[3].source = _source(notebook.cells[3]).replace(
        "solver = QuantumSolver(maximum_component_nodes=8)",
        "solver = ClassicalSolver(maximum_component_nodes=8)",
    )

    executed = NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()

    assert all(
        output.get("output_type") != "error"
        for cell in executed.cells
        for output in cell.get("outputs", [])
    )
    dataset_root = tmp_path / "data" / "synthetic" / "TINY-MHT"
    assert (dataset_root / "01" / "t000.tif").is_file()
    assert (dataset_root / "01_GT" / "TRA" / "man_track000.tif").is_file()
    assert (dataset_root / "01_GT" / "TRA" / "man_track.txt").is_file()
    run_cell = next(cell for cell in executed.cells if cell.id == "run")
    outputs = run_cell.get("outputs", [])
    assert any("image/png" in output.get("data", {}) for output in outputs)
    assert any(
        "detections" in output.get("data", {}).get("text/plain", "")
        for output in outputs
    )


@pytest.mark.filterwarnings(
    "ignore:Proactor event loop does not implement add_reader.*:RuntimeWarning"
)
def test_optional_many_frame_cell_runs_the_synthetic_sequence(
    tmp_path: Path,
) -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    notebook.cells = [
        notebook.cells[0],
        notebook.cells[1],
        notebook.cells[2],
        notebook.cells[3],
        notebook.cells[5],
    ]
    _force_tiny_synthetic_run(notebook.cells[2], tmp_path, frame_count=3)
    notebook.cells[3].source = _source(notebook.cells[3]).replace(
        "solver = QuantumSolver(maximum_component_nodes=8)",
        "solver = ClassicalSolver(maximum_component_nodes=8)",
    )

    executed = NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()

    outputs = executed.cells[-1].get("outputs", [])
    assert any("image/png" in output.get("data", {}) for output in outputs)
    assert any(
        "frames_processed" in output.get("data", {}).get("text/plain", "")
        and "3" in output.get("data", {}).get("text/plain", "")
        for output in outputs
    )
