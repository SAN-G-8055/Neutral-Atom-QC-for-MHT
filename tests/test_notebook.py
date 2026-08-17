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


def test_notebook_is_three_clean_ordered_code_cells() -> None:
    payload = _payload()
    cells = payload["cells"]

    assert payload["nbformat"] == 4
    assert [cell["id"] for cell in cells] == ["imports", "config", "run"]
    assert all(cell["cell_type"] == "code" for cell in cells)
    assert all(cell.get("execution_count") is None for cell in cells)
    assert all(cell.get("outputs") == [] for cell in cells)
    for cell in cells:
        compile(_source(cell), f"user_notebook:{cell['id']}", "exec")


def test_notebook_uses_the_checkout_and_one_real_frame() -> None:
    imports, config, run = map(_source, _payload()["cells"])

    assert imports.index("sys.path.insert") < imports.index("from neutral_atom_mht")
    assert 'project_root / "src"' in imports
    assert 'project_root / "data" / DATASET_NAME' in config
    assert "frame = 0" in config
    assert "raw_frame_path(dataset_root, frame)" in config
    assert "load_tiff(frame_path)" in run
    assert all(
        legacy not in imports + config + run
        for legacy in ("np.mgrid", "np.where", "synthetic", "run_sequence")
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
        notebook.cells = notebook.cells[:1]

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
        outputs = executed.cells[-1].get("outputs", [])
        assert any("image/png" in output.get("data", {}) for output in outputs)
        assert any(
            "detections" in output.get("data", {}).get("text/plain", "")
            for output in outputs
        )
