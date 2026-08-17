"""Check the three-cell notebook against one real sequence-01 image.

The import cell must make a source checkout usable without an editable install.
The remaining cells may configure and process one existing frame, but they must
not hide synthetic data generation or a multi-frame demonstration.
"""

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


def _cell_source(cell: dict[str, object]) -> str:
    return "".join(cell["source"])


def test_notebook_contains_only_import_config_and_run_cells() -> None:
    payload = _payload()
    cells = payload["cells"]

    assert payload["nbformat"] == 4
    assert len(cells) == 3
    assert all(cell["cell_type"] == "code" for cell in cells)
    assert [cell["id"] for cell in cells] == ["imports", "config", "run"]
    assert all(cell.get("execution_count") is None for cell in cells)
    assert all(cell.get("outputs") == [] for cell in cells)


def test_import_cell_fixes_the_src_layout_before_package_import() -> None:
    imports = _cell_source(_payload()["cells"][0])

    assert 'project_root / "src"' in imports
    assert "sys.path.insert" in imports
    assert "from neutral_atom_mht import" in imports
    assert imports.index("sys.path.insert") < imports.index("from neutral_atom_mht")
    assert "ClassicalSolver" in imports
    assert "HPC" in imports
    assert "HPCConfig" in imports
    assert "load_tiff" in imports
    assert "raw_frame_path" in imports


def test_config_and_run_use_exactly_one_existing_dataset_frame() -> None:
    cells = _payload()["cells"]
    config = _cell_source(cells[1])
    run = _cell_source(cells[2])
    all_code = "\n".join(_cell_source(cell) for cell in cells)

    assert 'project_root / "data" / DATASET_NAME' in config
    assert "frame = 0" in config
    assert "raw_frame_path(dataset_root, frame)" in config
    assert "HPCConfig()" in config
    assert 'HPC(config, sequence="01")' in config
    assert "ClassicalSolver()" in config
    assert "load_tiff(frame_path)" in run
    assert ".prepare_frame(" in run
    assert ".solve(" in run
    assert ".advance(" in run
    assert "axis.scatter(" in run

    forbidden = (
        "np.mgrid",
        "np.where",
        "np.zeros",
        "np.ones",
        "centres",
        "synthetic",
        "run_sequence",
        "QuantumSolver",
        "format_input",
        "def ",
        "class ",
    )
    assert all(token not in all_code for token in forbidden)


def test_import_cell_executes_without_a_test_supplied_pythonpath(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYTHONPATH", raising=False)
    source = nbformat.read(NOTEBOOK, as_version=4)
    import_only = nbformat.v4.new_notebook(
        cells=[source.cells[0]],
        metadata=source.metadata,
    )

    executed = NotebookClient(
        import_only,
        timeout=60,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()

    assert all(
        output.get("output_type") != "error"
        for output in executed.cells[0].get("outputs", [])
    )


@pytest.mark.skipif(
    not EXAMPLE_FRAME.is_file(),
    reason="the intentionally unversioned sequence-01 data are not installed",
)
def test_notebook_executes_one_real_frame_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYTHONPATH", raising=False)
    notebook = nbformat.read(NOTEBOOK, as_version=4)

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
    run_outputs = executed.cells[2].get("outputs", [])
    assert any("image/png" in output.get("data", {}) for output in run_outputs)
    assert any(
        "detections" in output.get("data", {}).get("text/plain", "")
        for output in run_outputs
    )
