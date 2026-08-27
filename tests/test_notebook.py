"""Keep the root notebook concise, reproducible, and internally consistent."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "user_notebook.ipynb"


def _payload() -> dict[str, object]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(cell: dict[str, object]) -> str:
    return "".join(cell["source"])


def _cells_by_id() -> dict[str, str]:
    return {cell["id"]: _source(cell) for cell in _payload()["cells"]}


def test_notebook_has_clean_reduced_cell_order() -> None:
    payload = _payload()
    cells = payload["cells"]

    assert payload["nbformat"] == 4
    assert [cell["id"] for cell in cells] == [
        "intro",
        "imports",
        "overnight_intro",
        "overnight_config",
        "overnight_run",
        "publication_figures",
        "figure_setup",
        "fig1_detections",
        "figure_analysis",
        "fig2_conflict_graph",
        "fig3_neutral_atoms",
        "fig4_performance",
        "fig5_eligibility",
        "fig6_detection",
        "fig7_runtime",
        "fig8_hardware_runtime",
    ]
    assert [
        index for index, cell in enumerate(cells)
        if cell["cell_type"] == "markdown"
    ] == [0, 2, 5]
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    assert all(cell.get("execution_count") is None for cell in code_cells)
    assert all(cell.get("outputs") == [] for cell in code_cells)
    for cell in code_cells:
        compile(_source(cell), f"user_notebook:{cell['id']}", "exec")


def test_notebook_uses_one_combined_noise_parameter_and_reuses_classical_data() -> None:
    cells = _cells_by_id()
    imports = cells["imports"]
    campaign = cells["overnight_config"]

    assert imports.index("sys.path.insert") < imports.index(
        "from neutral_atom_mht"
    )
    assert "SyntheticDataset" not in imports
    assert "DEFAULT_SYNTHETIC_DATA_ROOT" not in imports
    assert 'BENCHMARK_AXIS = "combined"' in campaign
    assert 'axes=("baseline", BENCHMARK_AXIS)' in campaign
    assert "BENCHMARK_SEVERITIES = (0.2, 0.4, 0.6, 0.8, 1.0)" in campaign
    assert "BENCHMARK_OBJECT_COUNTS = (4, 12, 30, 55)" in campaign
    assert "BENCHMARK_SEEDS = tuple(range(5))" in campaign
    assert "BENCHMARK_FRAME_COUNT = 40" in campaign
    assert "QUANTUM_QUOTA_PER_STRATUM = 3" in campaign
    assert "BENCHMARK_FORWARD_WORK_MINUTES = 25.0" in campaign
    assert "exact_checkpoint_source=classical_checkpoint" in campaign
    assert '"schema-2.1"' in campaign
    assert '"combined_quick"' in campaign
    assert "exact_maximum_component_nodes=128" in campaign
    assert "quantum_max_nonclique_component_nodes=8" in campaign
    assert "store_detailed_records=False" in campaign
    assert "resume=True" in campaign
    assert "BENCHMARK_AXES" not in campaign
    assert "BENCHMARK_PROFILE" not in campaign
    assert "run_overnight_benchmark(" in cells["overnight_run"]


def test_notebook_figures_are_renumbered_and_use_combined_noise() -> None:
    cells = _cells_by_id()

    assert "fig1_workflow" not in cells
    assert '"fig1_detection_overlays.png"' in cells["fig1_detections"]
    assert "SyntheticDataGenerator(QUANTUM_DEMO_DATA_CONFIG)" in cells[
        "fig1_detections"
    ]
    assert "Real sequence not installed" in cells["fig1_detections"]
    assert "synthetic_truth" in cells["fig1_detections"]

    assert "example_quantum_solver.execute(" in cells["figure_analysis"]
    assert "example_exact_solver.solve(prepared_frame.solver_input())" in cells[
        "figure_analysis"
    ]
    assert "example_reference.advance(prepared_frame, exact_result)" in cells[
        "figure_analysis"
    ]
    assert '"fig2_conflict_graph.png"' in cells["fig2_conflict_graph"]
    assert "logical_layout(example_graph)" in cells["fig2_conflict_graph"]
    assert '"fig3_neutral_atom_embedding.png"' in cells["fig3_neutral_atoms"]
    assert "example_run.coordinates" in cells["fig3_neutral_atoms"]
    assert "physical_edges != intended_edges" in cells["fig3_neutral_atoms"]

    assert '"fig4_quantum_reliability.png"' in cells["fig4_performance"]
    assert "combined-noise severity" in cells["fig4_performance"]
    assert '"selection_agrees"' in cells["fig4_performance"]
    assert 'row["graph_nodes"]' in cells["fig4_performance"]
    assert "total conflict-graph nodes" in cells["fig4_performance"]
    assert "maximum_nonclique_component_nodes" not in cells["fig4_performance"]

    assert '"fig5_quantum_size_outcomes.png"' in cells["fig5_eligibility"]
    assert "maximum_nonclique_component_nodes" in cells["fig5_eligibility"]
    assert 'else BENCHMARK_AXIS' in cells["fig5_eligibility"]
    assert all(
        label in cells["fig5_eligibility"]
        for label in ("no_work", "supported", "oversized")
    )

    assert '"fig6_detection_and_tracking_quality.png"' in cells["fig6_detection"]
    assert "condition_medians" in cells["fig6_detection"]
    assert "combined-noise severity" in cells["fig6_detection"]
    assert "BENCHMARK_AXES" not in cells["fig6_detection"]

    assert '"fig7_runtime_diagnostics.png"' in cells["fig7_runtime"]
    assert '"states_evaluated"' in cells["fig7_runtime"]
    assert '"quantum_runtime_seconds"' in cells["fig7_runtime"]
    assert 'row["graph_nodes"]' in cells["fig7_runtime"]

    assert (
        "sequence.get_duration(include_fall_time=True)"
        in cells["fig8_hardware_runtime"]
    )
    assert '"sample_count"' in cells["fig8_hardware_runtime"]
    assert '"qpu_sampled_seconds"' in cells["fig8_hardware_runtime"]
    assert 'row["graph_nodes"]' in cells["fig8_hardware_runtime"]
    assert "reference_cpu_count" in cells["fig8_hardware_runtime"]
    assert '"fig8_hardware_runtime_comparison.png"' in cells[
        "fig8_hardware_runtime"
    ]
    assert "atom loading, reset, readout" in cells["fig8_hardware_runtime"]


def test_notebook_contains_no_obsolete_workflow_or_demo_cells() -> None:
    cells = _cells_by_id()

    assert not {"data", "config", "run", "run_many", "fig1_workflow"} & cells.keys()
    all_source = "\n".join(cells.values())
    assert "fig1_workflow.png" not in all_source
    assert "BENCHMARK_PROFILE" not in all_source
    assert "BENCHMARK_AXES" not in all_source
