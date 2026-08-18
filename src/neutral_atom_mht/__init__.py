"""Expose the project's small object-oriented API from one import location.

The root package exposes only the objects needed to configure the HPC and its
solvers. Detection, evaluation, graph, and preprocessing details remain in
their focused modules instead of becoming permanent compatibility aliases.
"""

__version__ = "0.1.0"

from .classical_solver import ClassicalSolver
from .hpc import (
    HPC,
    HPCConfig,
    hpc,
)
from .models import Observation, TrackState
from .neutral_atom import (
    NeutralAtomInput,
    NeutralAtomOutput,
    QuantumSolver,
)
from .solver import (
    Solver,
    SolverComparison,
    SolverInput,
    SolverResult,
    SolverSelection,
)

__all__ = [
    "ClassicalSolver",
    "HPC",
    "HPCConfig",
    "NeutralAtomInput",
    "NeutralAtomOutput",
    "Observation",
    "QuantumSolver",
    "Solver",
    "SolverComparison",
    "SolverInput",
    "SolverResult",
    "SolverSelection",
    "TrackState",
    "hpc",
]
