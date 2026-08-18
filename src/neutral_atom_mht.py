"""Expose the project's small object-oriented API from one flat facade.

The facade exposes the objects needed to configure the HPC, its solvers, and
synthetic datasets. Detection, evaluation, graph, and preprocessing details
remain in their focused modules instead of becoming compatibility aliases.
"""

from _version import __version__

from classical_solver import ClassicalSolver
from hpc import (
    HPC,
    HPCConfig,
    hpc,
)
from models import Observation, TrackState
from neutral_atom import (
    NeutralAtomComponent,
    NeutralAtomConfig,
    NeutralAtomDependencyError,
    NeutralAtomEmbeddingError,
    NeutralAtomExecution,
    NeutralAtomExecutionError,
    NeutralAtomProgram,
    NeutralAtomRun,
    NeutralAtomRunError,
    NeutralAtomRunner,
    NeutralAtomSampleError,
    PulserQutipRunner,
    QuantumSolver,
)
from neutral_atom_visualization import (
    NeutralAtomSequenceFigures,
    NeutralAtomVisualizer,
)
from solver import (
    Solver,
    SolverComparison,
    SolverInput,
    SolverResult,
    SolverSelection,
)
from synthetic_data import (
    DEFAULT_SYNTHETIC_DATA_ROOT,
    SyntheticDataConfig,
    SyntheticDataGenerator,
    SyntheticDataset,
)

__all__ = [
    "ClassicalSolver",
    "DEFAULT_SYNTHETIC_DATA_ROOT",
    "HPC",
    "HPCConfig",
    "NeutralAtomComponent",
    "NeutralAtomConfig",
    "NeutralAtomDependencyError",
    "NeutralAtomEmbeddingError",
    "NeutralAtomExecution",
    "NeutralAtomExecutionError",
    "NeutralAtomProgram",
    "NeutralAtomRun",
    "NeutralAtomRunError",
    "NeutralAtomRunner",
    "NeutralAtomSampleError",
    "NeutralAtomSequenceFigures",
    "NeutralAtomVisualizer",
    "Observation",
    "PulserQutipRunner",
    "QuantumSolver",
    "Solver",
    "SolverComparison",
    "SolverInput",
    "SolverResult",
    "SolverSelection",
    "SyntheticDataConfig",
    "SyntheticDataGenerator",
    "SyntheticDataset",
    "TrackState",
    "hpc",
]
