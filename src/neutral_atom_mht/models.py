"""Strict, solver-independent data models for local data association.

The tracker deliberately keeps one state per physical track.  Association
hypotheses are short-lived candidates for a single frame; they are never kept
as a family of global hypotheses.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, nextafter
from numbers import Integral, Real

import numpy as np


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _sigmoid_probability(log_odds: float) -> float:
    """Map finite log-odds to the nearest representable open probability."""

    if log_odds >= 0.0:
        probability = 1.0 / (1.0 + exp(-log_odds))
    else:
        exponential = exp(log_odds)
        probability = exponential / (1.0 + exponential)
    return min(nextafter(1.0, 0.0), max(nextafter(0.0, 1.0), probability))


def _vector(values: object, length: int, name: str) -> tuple[float, ...]:
    array = np.asarray(values, dtype=float)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {length} finite values")
    return tuple(float(value) for value in array)


def _positive_definite_matrix(
    values: object,
    size: int,
    name: str,
) -> tuple[tuple[float, ...], ...]:
    array = np.asarray(values, dtype=float)
    if array.shape != (size, size) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {size}x{size} matrix")
    if not np.allclose(array, array.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    try:
        np.linalg.cholesky(array)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite") from exc
    return tuple(tuple(float(value) for value in row) for row in array)


@dataclass(frozen=True, slots=True)
class Observation:
    """One unlabelled 2-D observation supplied to the tracker."""

    frame: int
    observation_id: int
    x: float
    y: float
    covariance: tuple[tuple[float, float], tuple[float, float]] = (
        (4.0, 0.0),
        (0.0, 4.0),
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame", _integer(self.frame, "frame"))
        object.__setattr__(
            self,
            "observation_id",
            _integer(self.observation_id, "observation_id", minimum=1),
        )
        object.__setattr__(self, "x", _finite(self.x, "x"))
        object.__setattr__(self, "y", _finite(self.y, "y"))
        object.__setattr__(
            self,
            "covariance",
            _positive_definite_matrix(self.covariance, 2, "covariance"),
        )

    @property
    def position(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True, slots=True)
class TrackState:
    """The single retained Bayesian/Kalman state for one object."""

    track_id: int
    frame: int
    state: tuple[float, float, float, float]
    covariance: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]
    log_odds: float
    hits: int = 1
    misses: int = 0
    observation_history: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "track_id", _integer(self.track_id, "track_id", minimum=1)
        )
        object.__setattr__(self, "frame", _integer(self.frame, "frame"))
        object.__setattr__(self, "state", _vector(self.state, 4, "state"))
        object.__setattr__(
            self,
            "covariance",
            _positive_definite_matrix(self.covariance, 4, "covariance"),
        )
        object.__setattr__(self, "log_odds", _finite(self.log_odds, "log_odds"))
        object.__setattr__(self, "hits", _integer(self.hits, "hits", minimum=1))
        object.__setattr__(self, "misses", _integer(self.misses, "misses"))
        history: list[tuple[int, int]] = []
        for frame, observation_id in self.observation_history:
            history_frame = _integer(frame, "history frame")
            if history_frame > self.frame:
                raise ValueError("history frames cannot follow the track frame")
            history.append(
                (history_frame, _integer(observation_id, "history observation_id", minimum=1))
            )
        if len(history) != len(set(history)):
            raise ValueError("observation_history cannot contain duplicate event keys")
        object.__setattr__(self, "observation_history", tuple(history))

    @property
    def position(self) -> tuple[float, float]:
        return (self.state[0], self.state[1])

    @property
    def posterior_probability(self) -> float:
        """Existence probability derived from the canonical log-odds state."""

        return _sigmoid_probability(self.log_odds)


@dataclass(frozen=True, slots=True)
class GatedAssociation:
    """A track/observation pair admitted by the declared validation gate."""

    frame: int
    track_id: int
    observation_id: int
    innovation: tuple[float, float]
    innovation_covariance: tuple[tuple[float, float], tuple[float, float]]
    mahalanobis_sq: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame", _integer(self.frame, "frame"))
        object.__setattr__(
            self, "track_id", _integer(self.track_id, "track_id", minimum=1)
        )
        object.__setattr__(
            self,
            "observation_id",
            _integer(self.observation_id, "observation_id", minimum=1),
        )
        object.__setattr__(
            self, "innovation", _vector(self.innovation, 2, "innovation")
        )
        object.__setattr__(
            self,
            "innovation_covariance",
            _positive_definite_matrix(
                self.innovation_covariance, 2, "innovation_covariance"
            ),
        )
        distance = _finite(self.mahalanobis_sq, "mahalanobis_sq")
        if distance < 0.0:
            raise ValueError("mahalanobis_sq cannot be negative")
        object.__setattr__(self, "mahalanobis_sq", distance)


@dataclass(frozen=True, slots=True)
class AssociationHypothesis:
    """One ephemeral, local association candidate for one frame."""

    hypothesis_id: int
    frame: int
    track_id: int
    observation_id: int
    log_likelihood_ratio: float
    posterior_log_odds: float
    weight: float
    gate: GatedAssociation

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "hypothesis_id",
            _integer(self.hypothesis_id, "hypothesis_id", minimum=1),
        )
        object.__setattr__(self, "frame", _integer(self.frame, "frame"))
        object.__setattr__(
            self, "track_id", _integer(self.track_id, "track_id", minimum=1)
        )
        object.__setattr__(
            self,
            "observation_id",
            _integer(self.observation_id, "observation_id", minimum=1),
        )
        if (
            self.gate.frame,
            self.gate.track_id,
            self.gate.observation_id,
        ) != (self.frame, self.track_id, self.observation_id):
            raise ValueError("gate scope must match the hypothesis scope")
        object.__setattr__(
            self,
            "log_likelihood_ratio",
            _finite(self.log_likelihood_ratio, "log_likelihood_ratio"),
        )
        object.__setattr__(
            self,
            "posterior_log_odds",
            _finite(self.posterior_log_odds, "posterior_log_odds"),
        )
        weight = _finite(self.weight, "weight")
        object.__setattr__(self, "weight", weight)

    @property
    def posterior_probability(self) -> float:
        """Association probability derived from its posterior log odds."""

        return _sigmoid_probability(self.posterior_log_odds)


def observations_from_detections(
    detections: object,
    *,
    variance_px2: float = 4.0,
) -> tuple[Observation, ...]:
    """Adapt strict detection events to tracking observations.

    The adapter is intentionally structural so the HPC can accept detector
    events without coupling its state logic to the segmentation algorithm.
    """

    variance = _finite(variance_px2, "variance_px2")
    if variance <= 0.0:
        raise ValueError("variance_px2 must be positive")
    observations = tuple(
        Observation(
            frame=event.frame,
            observation_id=event.detection_id,
            x=event.x_px,
            y=event.y_px,
            covariance=((variance, 0.0), (0.0, variance)),
        )
        for event in detections
    )
    keys = [(item.frame, item.observation_id) for item in observations]
    if len(keys) != len(set(keys)):
        raise ValueError("detection events must have unique frame-local IDs")
    return tuple(sorted(observations, key=lambda item: (item.frame, item.observation_id)))
