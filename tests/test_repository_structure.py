"""Protect the intentionally flat and self-documenting repository layout."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "neutral_atom_mht"


def test_repository_has_one_readme_and_one_root_notebook() -> None:
    readmes = []
    for path in ROOT.rglob("README*"):
        relative = path.relative_to(ROOT)
        if (
            not any(part.startswith(".") for part in relative.parts)
            and relative.parts[0] not in {"build", "data", "dist", "outputs"}
        ):
            readmes.append(relative)

    assert readmes == [Path("README.md")]
    assert (ROOT / "user_notebook.ipynb").is_file()
    assert not (ROOT / "docs").exists()
    assert not (ROOT / "notebooks").exists()


def test_source_package_is_flat_and_legacy_monoliths_are_absent() -> None:
    assert not (PACKAGE / "backends").exists()
    assert not (PACKAGE / "tracking").exists()
    assert not (ROOT / "poster").exists()
    assert not (ROOT / "report").exists()
    assert not (ROOT / "figures").exists()
    assert not (ROOT / "scripts" / "NielsBohrProject.py").exists()
    assert not (ROOT / "scripts" / "Route1_HypothesisDiscovery.py").exists()


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


def test_removed_global_and_qutip_implementations_were_not_reintroduced() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8").lower() for path in PACKAGE.glob("*.py")
    )

    assert "def enumerate_hypotheses" not in source
    assert "def hypothesis_probabilities" not in source
    assert "max_tracks_per_family" not in source
    assert "import qutip" not in source
