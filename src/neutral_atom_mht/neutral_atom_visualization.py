"""Save neutral-atom program and sampling diagnostics without affecting solving.

The solver builds and executes Pulser objects; this module only renders those
already-built objects.  Keeping the boundary here means solver execution is
headless and callers opt into every figure explicitly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import matplotlib.pyplot as plt


class NeutralAtomProgramLike(Protocol):
    """Structural view of the immutable program record used for plotting."""

    register: Any
    detuning_map: Any
    sequence: Any


class NeutralAtomRunLike(Protocol):
    """Structural view of sampled bitstrings and the solver's chosen sample."""

    bitstring_counts: Sequence[tuple[str, int]]
    selected_bitstring: str


@dataclass(frozen=True, slots=True)
class NeutralAtomSequenceFigures:
    """Files produced by Pulser's potentially multi-figure sequence drawing."""

    output_prefix: Path
    paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class NeutralAtomVisualizer:
    """Explicit, reusable renderer for one neutral-atom program or run."""

    dpi: int = 150

    def __post_init__(self) -> None:
        if self.dpi < 1:
            raise ValueError("dpi must be positive")

    def save_register(
        self,
        program: NeutralAtomProgramLike,
        output: str | Path,
        *,
        interaction_amplitude: float = 1.0,
    ) -> Path:
        """Save the atom register with its blockade graph and half radii."""

        path = _prepare_output(output)
        blockade_radius = program.sequence.device.rydberg_blockade_radius(
            interaction_amplitude
        )
        self._draw(
            program.register.draw,
            blockade_radius=blockade_radius,
            draw_graph=True,
            draw_half_radius=True,
            fig_name=str(path),
        )
        _require_output(path, "register")
        return path

    def save_detuning_map(
        self,
        program: NeutralAtomProgramLike,
        output: str | Path,
    ) -> Path:
        """Save the local detuning weights using the register's qubit labels."""

        path = _prepare_output(output)
        self._draw(
            program.detuning_map.draw,
            labels=program.register.qubit_ids,
            fig_name=str(path),
        )
        _require_output(path, "detuning-map")
        return path

    def save_sequence(
        self,
        program: NeutralAtomProgramLike,
        output_prefix: str | Path,
    ) -> NeutralAtomSequenceFigures:
        """Save pulse, detuning-map, and per-qubit sequence figures."""

        prefix = _prepare_output(output_prefix)
        before = {
            path: (path.stat().st_mtime_ns, path.stat().st_size)
            for path in _sequence_outputs(prefix)
        }
        self._draw(
            program.sequence.draw,
            draw_detuning_maps=True,
            draw_qubit_det=True,
            draw_qubit_amp=True,
            fig_name=str(prefix),
        )
        paths = tuple(
            path
            for path in _sequence_outputs(prefix)
            if before.get(path) != (path.stat().st_mtime_ns, path.stat().st_size)
        )
        if not paths:
            raise RuntimeError("Pulser sequence drawing did not create any figure files")
        return NeutralAtomSequenceFigures(prefix, paths)

    def save_distribution(
        self,
        run: NeutralAtomRunLike,
        output: str | Path,
        *,
        max_bitstrings: int | None = None,
    ) -> Path:
        """Save sampled bitstrings ordered by count and highlight the selection."""

        if max_bitstrings is not None and max_bitstrings < 1:
            raise ValueError("max_bitstrings must be positive when supplied")

        counts = sorted(
            run.bitstring_counts,
            key=lambda item: (-item[1], item[0]),
        )
        if max_bitstrings is not None:
            counts = counts[:max_bitstrings]
        if not counts:
            raise ValueError("a distribution requires at least one sampled bitstring")

        path = _prepare_output(output)
        bitstrings = [bitstring for bitstring, _ in counts]
        values = [count for _, count in counts]
        colors = [
            "#D1495B" if bitstring == run.selected_bitstring else "#2A9D8F"
            for bitstring in bitstrings
        ]
        figure, axis = plt.subplots(
            figsize=(max(8.0, min(18.0, 0.55 * len(bitstrings))), 6.0)
        )
        try:
            axis.bar(bitstrings, values, width=0.5, color=colors)
            axis.set_xlabel("Bitstring")
            axis.set_ylabel("Count")
            axis.set_title("Neutral-atom sample distribution")
            axis.tick_params(axis="x", labelrotation=90)
            figure.tight_layout()
            figure.savefig(path, dpi=self.dpi, bbox_inches="tight")
        finally:
            plt.close(figure)
        return path

    def _draw(self, draw: Any, **kwargs: object) -> None:
        """Invoke a Pulser drawing method headlessly and close its figures."""

        existing_figures = set(plt.get_fignums())
        try:
            draw(
                **kwargs,
                kwargs_savefig={"dpi": self.dpi, "bbox_inches": "tight"},
                show=False,
            )
        finally:
            for figure_number in set(plt.get_fignums()) - existing_figures:
                plt.close(figure_number)


def _prepare_output(output: str | Path) -> Path:
    path = Path(output)
    if not path.suffix:
        path = path.with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _sequence_outputs(prefix: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for path in prefix.parent.iterdir()
            if path.is_file()
            and path.stem.startswith(prefix.stem)
            and path.suffix == prefix.suffix
        )
    )


def _require_output(path: Path, kind: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"Pulser {kind} drawing did not create {path}")


__all__ = [
    "NeutralAtomSequenceFigures",
    "NeutralAtomVisualizer",
]
