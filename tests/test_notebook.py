from __future__ import annotations

import json
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient
import pytest


NOTEBOOK = Path("notebooks/classical_vs_quantum.ipynb")


def test_notebook_is_a_clean_thin_package_interface() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    assert payload["nbformat"] == 4
    code = "\n".join(
        "".join(cell["source"])
        for cell in payload["cells"]
        if cell["cell_type"] == "code"
    )
    assert "from neutral_atom_mht" in code
    assert "def " not in code
    assert "class " not in code
    assert "compare_prepared" in code
    assert "advance(" in code
    assert all(
        cell.get("outputs", []) == []
        for cell in payload["cells"]
        if cell["cell_type"] == "code"
    )


def test_notebook_documents_explicit_backend_choice_and_identical_schema() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell["source"]) for cell in payload["cells"])

    assert "same immutable weighted conflict graph" in Path("README.md").read_text(
        encoding="utf-8"
    )
    assert "common result contract" in text
    assert "common_columns" in text
    assert "classical_exact" in text
    assert "neutral_atom_qutip" in text


@pytest.mark.quantum
def test_notebook_executes_end_to_end_without_persisting_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = str((Path.cwd() / "src").resolve())
    inherited = os.environ.get("PYTHONPATH")
    monkeypatch.setenv(
        "PYTHONPATH",
        source_path if not inherited else source_path + os.pathsep + inherited,
    )
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    graph_path = tmp_path / "notebook_conflict_graph.png"
    visualize = next(cell for cell in notebook.cells if cell.get("id") == "visualize")
    visualize.source = visualize.source.replace(
        'project_root / "outputs/tracking/notebook_conflict_graph.png"',
        f'Path({str(graph_path)!r})',
    )

    executed = NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(Path.cwd())}},
    ).execute()

    assert graph_path.is_file()
    assert all(
        output.get("output_type") != "error"
        for cell in executed.cells
        for output in cell.get("outputs", [])
    )
