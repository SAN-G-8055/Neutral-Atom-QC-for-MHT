"""Check that the root notebook is a readable, executable client of the package.

The notebook must demonstrate the public objects without hiding project logic
inside notebook-defined functions or classes.  Its default path uses the
implemented classical solver; the neutral-atom section may only format the
same frozen problem and report the explicit placeholder status.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from nbclient import NotebookClient
import nbformat


NOTEBOOK = Path("user_notebook.ipynb")


def _notebook_text() -> tuple[dict[str, object], str, str]:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in payload["cells"]
        if cell["cell_type"] == "code"
    )
    text = "\n".join("".join(cell["source"]) for cell in payload["cells"])
    return payload, code, text


def test_notebook_is_a_clean_root_level_object_oriented_interface() -> None:
    payload, code, text = _notebook_text()

    assert NOTEBOOK.parent == Path(".")
    assert payload["nbformat"] == 4
    assert payload["cells"][0]["cell_type"] == "markdown"
    assert "image" in "".join(payload["cells"][0]["source"]).lower()
    assert "from neutral_atom_mht import" in code
    assert "HPC(" in code
    assert "HPCConfig(" in code
    assert "ClassicalSolver(" in code
    assert "QuantumSolver(" in code
    assert "def " not in code
    assert "class " not in code
    assert ".observe(" in code
    assert ".prepare_frame(" in code
    assert ".solve(" in code
    assert ".advance(" in code
    assert ".run_sequence(" in code
    assert all(
        cell.get("outputs", []) == []
        for cell in payload["cells"]
        if cell["cell_type"] == "code"
    )
    assert "step by step" in text.lower()


def test_notebook_is_honest_about_the_quantum_placeholder() -> None:
    _, code, text = _notebook_text()
    lowered = text.lower()

    assert ".format_input(" in code
    assert "not_implemented" in lowered
    assert "manual" in lowered
    assert "qutip" not in lowered
    assert "neutralatombackend" not in lowered
    assert "neutral_atom_qutip" not in lowered


def test_notebook_executes_end_to_end_without_persisting_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_path = str((Path.cwd() / "src").resolve())
    inherited = os.environ.get("PYTHONPATH")
    monkeypatch.setenv(
        "PYTHONPATH",
        source_path if not inherited else source_path + os.pathsep + inherited,
    )
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    graph_path = tmp_path / "user_conflict_graph.png"
    visualize = next(cell for cell in notebook.cells if cell.get("id") == "visualize")
    visualize.source = visualize.source.replace(
        'project_root / "outputs/user_conflict_graph.png"',
        f"Path({str(graph_path)!r})",
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
