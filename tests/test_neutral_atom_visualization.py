"""Verify neutral-atom diagnostics save headlessly through structural records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pytest

from neutral_atom import NeutralAtomVisualizer
from neutral_atom_mht import NeutralAtomVisualizer as PublicNeutralAtomVisualizer


class FakeDrawable:
    def __init__(self, *, outputs: tuple[str, ...] = ("",), fail: bool = False):
        self.outputs = outputs
        self.fail = fail
        self.calls: list[dict[str, object]] = []

    def draw(self, **kwargs: object) -> None:
        self.calls.append(kwargs)
        figure, axis = plt.subplots()
        axis.plot((0, 1), (0, 1))
        if self.fail:
            raise RuntimeError("draw failed")

        requested = Path(str(kwargs["fig_name"]))
        savefig_options = dict(kwargs["kwargs_savefig"])  # type: ignore[arg-type]
        for suffix in self.outputs:
            path = requested.with_name(
                f"{requested.stem}{suffix}{requested.suffix}"
            )
            figure.savefig(path, **savefig_options)


class FakeDevice:
    def __init__(self) -> None:
        self.amplitudes: list[float] = []

    def rydberg_blockade_radius(self, amplitude: float) -> float:
        self.amplitudes.append(amplitude)
        return 4.25


class FakeSequence(FakeDrawable):
    def __init__(self) -> None:
        super().__init__(outputs=("_pulses", "_per_qubit", "_per_qubit_legend"))
        self.device = FakeDevice()


@dataclass(frozen=True, slots=True)
class FakeProgram:
    register: FakeDrawable
    detuning_map: FakeDrawable
    sequence: FakeSequence


@dataclass(frozen=True, slots=True)
class FakeRun:
    bitstring_counts: tuple[tuple[str, int], ...]
    selected_bitstring: str


def _program() -> FakeProgram:
    register = FakeDrawable()
    register.qubit_ids = ("q0", "q1")  # type: ignore[attr-defined]
    return FakeProgram(register, FakeDrawable(), FakeSequence())


def test_visualizer_is_exposed_from_the_merged_module_and_public_facade() -> None:
    assert NeutralAtomVisualizer.__module__ == "neutral_atom"
    assert PublicNeutralAtomVisualizer is NeutralAtomVisualizer


def test_visualizer_saves_all_program_views_without_showing_figures(tmp_path) -> None:
    program = _program()
    visualizer = NeutralAtomVisualizer(dpi=80)
    stale_sequence = tmp_path / "sequence" / "program_stale.png"
    stale_sequence.parent.mkdir(parents=True)
    stale_sequence.write_bytes(b"old")

    register_path = visualizer.save_register(
        program, tmp_path / "register" / "atoms.png"
    )
    detuning_path = visualizer.save_detuning_map(
        program, tmp_path / "detuning" / "weights.png"
    )
    sequence_figures = visualizer.save_sequence(
        program, tmp_path / "sequence" / "program.png"
    )

    assert register_path.is_file()
    assert detuning_path.is_file()
    assert len(sequence_figures.paths) == 3
    assert stale_sequence not in sequence_figures.paths
    assert all(path.is_file() and path.stat().st_size for path in sequence_figures.paths)
    assert program.sequence.device.amplitudes == [1.0]

    register_call = program.register.calls[0]
    assert register_call["show"] is False
    assert register_call["blockade_radius"] == 4.25
    assert register_call["draw_graph"] is True
    assert register_call["draw_half_radius"] is True

    detuning_call = program.detuning_map.calls[0]
    assert detuning_call["show"] is False
    assert detuning_call["labels"] == ("q0", "q1")

    sequence_call = program.sequence.calls[0]
    assert sequence_call["show"] is False
    assert sequence_call["draw_detuning_maps"] is True
    assert sequence_call["draw_qubit_det"] is True
    assert sequence_call["draw_qubit_amp"] is True
    assert plt.get_fignums() == []


def test_suffixless_outputs_are_normalized_to_png(tmp_path) -> None:
    output = NeutralAtomVisualizer(dpi=80).save_register(
        _program(),
        tmp_path / "register-without-suffix",
    )

    assert output == tmp_path / "register-without-suffix.png"
    assert output.is_file()


def test_distribution_is_deterministic_and_highlights_the_selected_sample(
    tmp_path, monkeypatch
) -> None:
    captured: dict[str, object] = {}
    original_bar = Axes.bar

    def capture_bar(self, labels, values, **kwargs):  # type: ignore[no-untyped-def]
        captured["labels"] = tuple(labels)
        captured["values"] = tuple(values)
        captured["colors"] = tuple(kwargs["color"])
        return original_bar(self, labels, values, **kwargs)

    monkeypatch.setattr(Axes, "bar", capture_bar)
    run = FakeRun(
        bitstring_counts=(("11", 2), ("10", 5), ("00", 5), ("01", 1)),
        selected_bitstring="10",
    )

    output = NeutralAtomVisualizer(dpi=80).save_distribution(
        run,
        tmp_path / "nested" / "distribution.png",
        max_bitstrings=3,
    )

    assert output.is_file() and output.stat().st_size > 0
    assert captured == {
        "labels": ("00", "10", "11"),
        "values": (5, 5, 2),
        "colors": ("#2A9D8F", "#D1495B", "#2A9D8F"),
    }
    assert plt.get_fignums() == []


def test_visualizer_closes_only_figures_created_by_a_failing_draw(tmp_path) -> None:
    retained = plt.figure()
    failing_register = FakeDrawable(fail=True)
    failing_register.qubit_ids = ("q0",)  # type: ignore[attr-defined]
    program = SimpleNamespace(
        register=failing_register,
        detuning_map=FakeDrawable(),
        sequence=FakeSequence(),
    )

    try:
        with pytest.raises(RuntimeError, match="draw failed"):
            NeutralAtomVisualizer().save_register(
                program, tmp_path / "failed-register.png"
            )
        assert plt.get_fignums() == [retained.number]
    finally:
        plt.close(retained)


@pytest.mark.parametrize("limit", [0, -1])
def test_distribution_rejects_nonpositive_limits(tmp_path, limit: int) -> None:
    run = FakeRun((("0", 1),), "0")
    with pytest.raises(ValueError, match="positive"):
        NeutralAtomVisualizer().save_distribution(
            run,
            tmp_path / "unused.png",
            max_bitstrings=limit,
        )
