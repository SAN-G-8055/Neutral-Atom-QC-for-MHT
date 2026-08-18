"""Expose the project's small object-oriented API from one flat facade.

The facade exposes the objects needed to configure the HPC, its solvers, and
synthetic datasets. Detection, graph, and preprocessing details
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
    NeutralAtomError,
    NeutralAtomExecution,
    NeutralAtomProgram,
    NeutralAtomRun,
    NeutralAtomRunner,
    NeutralAtomSequenceFigures,
    NeutralAtomVisualizer,
    PulserQutipRunner,
    QuantumSolver,
)
from solver import (
    ComponentSolver,
    Solver,
    SolverComparison,
    SolverInput,
    SolverResult,
    SolverSelection,
)
from synthetic_data import (
    DEFAULT_SYNTHETIC_DATA_ROOT,
    QUANTUM_DEMO_DATA_CONFIG,
    SyntheticDataConfig,
    SyntheticDataGenerator,
    SyntheticDataset,
)

__all__ = [
    "ClassicalSolver",
    "ComponentSolver",
    "DEFAULT_SYNTHETIC_DATA_ROOT",
    "HPC",
    "HPCConfig",
    "NeutralAtomComponent",
    "NeutralAtomConfig",
    "NeutralAtomError",
    "NeutralAtomExecution",
    "NeutralAtomProgram",
    "NeutralAtomRun",
    "NeutralAtomRunner",
    "NeutralAtomSequenceFigures",
    "NeutralAtomVisualizer",
    "Observation",
    "PulserQutipRunner",
    "QUANTUM_DEMO_DATA_CONFIG",
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
