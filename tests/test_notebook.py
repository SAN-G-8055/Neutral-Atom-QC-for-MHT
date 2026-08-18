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


def test_notebook_is_one_markdown_and_five_clean_ordered_code_cells() -> None:
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
    ]
    assert cells[0]["cell_type"] == "markdown"
    assert all(cell["cell_type"] == "code" for cell in cells[1:])
    assert all(cell.get("execution_count") is None for cell in cells[1:])
    assert all(cell.get("outputs") == [] for cell in cells[1:])
    for cell in cells[1:]:
        compile(_source(cell), f"user_notebook:{cell['id']}", "exec")


def test_notebook_uses_the_bounded_quantum_synthetic_data_by_default() -> None:
    intro, imports, data, config, run, run_many = map(_source, _payload()["cells"])

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


@pytest.mark.filterwarnings(
    "ignore:Proactor event loop does not implement add_reader.*:RuntimeWarning"
)
def test_notebook_executes_end_to_end_without_real_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("PYTHONPATH", raising=False)
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    _force_tiny_synthetic_run(notebook.cells[2], tmp_path, frame_count=1)

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
