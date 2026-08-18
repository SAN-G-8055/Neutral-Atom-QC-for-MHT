"""Keep the root notebook minimal, clean, and executable from a checkout."""

from __future__ import annotations

import json
from pathlib import Path

from nbclient import NotebookClient
import nbformat
import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "user_notebook.ipynb"
EXAMPLE_FRAME = ROOT / "data" / "PhC-C2DL-PSC" / "01" / "t000.tif"


def _payload() -> dict[str, object]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(cell: dict[str, object]) -> str:
    return "".join(cell["source"])


def _enable_tiny_synthetic_generation(
    cell: dict[str, object],
    output_root: Path,
    *,
    frame_count: int,
) -> None:
    generation = _source(cell)
    generation = generation.replace(
        "GENERATE_SYNTHETIC_DATA = False",
        "GENERATE_SYNTHETIC_DATA = True",
    )
    generation = generation.replace(
        "frame_count=40",
        f"frame_count={frame_count}",
    )
    generation = generation.replace("object_count=55", "object_count=2")
    generation = generation.replace(
        "    seed=0,",
        "    seed=0,\n    image_shape=(64, 80),",
    )
    generation = generation.replace(
        "project_root / DEFAULT_SYNTHETIC_DATA_ROOT",
        f"Path({output_root.as_posix()!r}) / DEFAULT_SYNTHETIC_DATA_ROOT",
    )
    cell["source"] = generation


def test_notebook_is_five_clean_ordered_code_cells() -> None:
    payload = _payload()
    cells = payload["cells"]

    assert payload["nbformat"] == 4
    assert [cell["id"] for cell in cells] == [
        "imports",
        "generate",
        "config",
        "run",
        "run_many",
    ]
    assert all(cell["cell_type"] == "code" for cell in cells)
    assert all(cell.get("execution_count") is None for cell in cells)
    assert all(cell.get("outputs") == [] for cell in cells)
    for cell in cells:
        compile(_source(cell), f"user_notebook:{cell['id']}", "exec")


def test_notebook_can_select_real_or_generated_data() -> None:
    imports, generate, config, run, run_many = map(
        _source, _payload()["cells"]
    )

    assert imports.index("sys.path.insert") < imports.index("from neutral_atom_mht")
    assert "from cell_data import DATASET_NAME" in imports
    assert "SyntheticDataGenerator" in imports
    assert "SyntheticDataset" in imports
    assert "QuantumSolver" in imports
    assert 'project_root / "src"' in imports
    assert "GENERATE_SYNTHETIC_DATA = False" in generate
    assert "project_root / DEFAULT_SYNTHETIC_DATA_ROOT" in generate
    assert "SyntheticDataset(" in generate
    assert ".generate(\n        synthetic_output_root" in generate
    assert "USE_SYNTHETIC_DATA = False" in config
    assert "synthetic_dataset.raw_frame_path(frame)" in config
    assert "sequence = synthetic_dataset.config.sequence" in config
    assert 'project_root / "data" / DATASET_NAME' in config
    assert "frame = 0" in config
    assert "raw_frame_path(dataset_root, frame)" in config
    assert "HPC(config, sequence=sequence)" in config
    assert "solver = ClassicalSolver()" in config
    assert "# solver = QuantumSolver()" in config
    assert "load_tiff(frame_path)" in run
    assert "{dataset_label} sequence {sequence}" in run
    assert "RUN_MANY_FRAMES = False" in run_many
    assert "MANY_FRAME_COUNT = 40" in run_many
    assert "synthetic_dataset.load_frame(frame_index)" in run_many
    assert "load_tiff(raw_frame_path(dataset_root, frame_index))" in run_many
    assert "sequence_controller = HPC(config, sequence=sequence)" in run_many
    assert "sequence_controller.run_sequence(" in run_many
    assert "start_frame=0" in run_many
    assert all(
        legacy not in imports + generate + config + run + run_many
        for legacy in ("np.mgrid", "np.where")
    )


@pytest.mark.filterwarnings(
    "ignore:Proactor event loop does not implement add_reader.*:RuntimeWarning"
)
def test_notebook_executes_without_a_test_supplied_pythonpath(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYTHONPATH", raising=False)
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    has_data = EXAMPLE_FRAME.is_file()
    if not has_data:
        notebook.cells = notebook.cells[:2]

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
    if has_data:
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
def test_optional_generation_cell_writes_the_synthetic_layout(
    tmp_path: Path,
) -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    notebook.cells = notebook.cells[:2]
    _enable_tiny_synthetic_generation(
        notebook.cells[1],
        tmp_path,
        frame_count=1,
    )

    NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()

    dataset_root = tmp_path / "data" / "synthetic" / "SYN-MHT"
    assert (dataset_root / "01" / "t000.tif").is_file()
    assert (dataset_root / "01_GT" / "TRA" / "man_track000.tif").is_file()
    assert (dataset_root / "01_GT" / "TRA" / "man_track.txt").is_file()


@pytest.mark.filterwarnings(
    "ignore:Proactor event loop does not implement add_reader.*:RuntimeWarning"
)
def test_optional_many_frame_cell_runs_the_selected_synthetic_sequence(
    tmp_path: Path,
) -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    notebook.cells = [
        notebook.cells[0],
        notebook.cells[1],
        notebook.cells[2],
        notebook.cells[4],
    ]
    _enable_tiny_synthetic_generation(
        notebook.cells[1],
        tmp_path,
        frame_count=3,
    )
    notebook.cells[2].source = _source(notebook.cells[2]).replace(
        "USE_SYNTHETIC_DATA = False",
        "USE_SYNTHETIC_DATA = True",
    )
    notebook.cells[3].source = _source(notebook.cells[3]).replace(
        "RUN_MANY_FRAMES = False",
        "RUN_MANY_FRAMES = True",
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
