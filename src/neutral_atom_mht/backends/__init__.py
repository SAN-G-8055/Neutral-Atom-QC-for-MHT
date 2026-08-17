"""Interchangeable optimization backends with one result schema."""

from .base import SolverBackend, SolverInput, SolverResult, validate_result
from .classical import ClassicalBackend
from .neutral_atom import AdiabaticPulse, NeutralAtomBackend, PasqalParameters

__all__ = [
    "ClassicalBackend",
    "AdiabaticPulse",
    "NeutralAtomBackend",
    "PasqalParameters",
    "SolverBackend",
    "SolverInput",
    "SolverResult",
    "validate_result",
]
