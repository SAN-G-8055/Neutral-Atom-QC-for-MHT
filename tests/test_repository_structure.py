"""Protect the intentionally flat and self-documenting repository layout."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "neutral_atom_mht"


def test_repository_has_one_readme_and_one_root_notebook() -> None:
    readmes = [
        path
        for path in ROOT.rglob("README*")
        if not any(part.startswith(".") for part in path.relative_to(ROOT).parts)
    ]
    assert readmes == [ROOT / "README.md"]
    assert (ROOT / "user_notebook.ipynb").is_file()


def test_source_package_contains_only_flat_python_modules() -> None:
    assert all(path.parent == PACKAGE for path in PACKAGE.rglob("*.py"))


def test_curated_outputs_are_directly_under_artifacts() -> None:
    expected = {
        "detections.csv",
        "detections_overview.png",
        "gold_events.csv",
        "matches.csv",
        "per_frame_metrics.csv",
        "performance_over_time.png",
        "summary.json",
    }

    assert {path.name for path in (ROOT / "artifacts").iterdir()} == expected


def test_every_python_file_starts_with_a_natural_language_description() -> None:
    code_files = sorted(PACKAGE.glob("*.py")) + sorted((ROOT / "tests").glob("*.py"))
    assert code_files
    for path in code_files:
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        description = ast.get_docstring(module, clean=True)
        assert description and len(description.split()) >= 8, path.name
