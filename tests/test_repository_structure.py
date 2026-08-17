from pathlib import Path


def test_legacy_report_poster_and_global_hypothesis_monoliths_are_absent() -> None:
    for path in (
        Path("poster"),
        Path("report"),
        Path("figures"),
        Path("scripts/NielsBohrProject.py"),
        Path("scripts/Route1_HypothesisDiscovery.py"),
    ):
        assert not path.exists(), f"legacy path unexpectedly remains: {path}"


def test_removed_global_population_functions_were_not_moved_into_package() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in Path("src/neutral_atom_mht").rglob("*.py")
    )

    assert "def enumerate_hypotheses" not in source
    assert "def hypothesis_probabilities" not in source
    assert "def n_scan_prune" not in source
    assert "max_tracks_per_family" not in source
