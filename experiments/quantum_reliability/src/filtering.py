"""Predict, correct, and retain object tracks with transparent Kalman math.

The functions here know nothing about graph solvers.  They move a track state
forward in time, apply one chosen observation, or enforce declared retention
limits.  :class:`HPC` exposes these operations as step-by-step methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from models import AssociationHypothesis, GatedAssociation, TrackState


MEASUREMENT_MATRIX = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float
)


@dataclass(frozen=True, slots=True)
class FilterConfig:
    acceleration_std: float = 1.0
    minimum_posterior: float = 0.01
    maximum_misses: int = 3
    maximum_tracks: int = 200

    def __post_init__(self) -> None:
        if not isfinite(float(self.acceleration_std)) or self.acceleration_std < 0.0:
            raise ValueError("acceleration_std must be finite and non-negative")
        if not 0.0 < float(self.minimum_posterior) < 1.0:
            raise ValueError("minimum_posterior must lie strictly between 0 and 1")
        if self.maximum_misses < 0:
            raise ValueError("maximum_misses cannot be negative")
        if self.maximum_tracks < 1:
            raise ValueError("maximum_tracks must be positive")
        object.__setattr__(self, "acceleration_std", float(self.acceleration_std))
        object.__setattr__(self, "minimum_posterior", float(self.minimum_posterior))


def constant_velocity_matrices(
    elapsed_seconds: float,
    acceleration_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the constant-velocity transition and process covariance."""

    dt = float(elapsed_seconds)
    q = float(acceleration_std)
    if not isfinite(dt) or dt <= 0.0:
        raise ValueError("elapsed_seconds must be finite and positive")
    if not isfinite(q) or q < 0.0:
        raise ValueError("acceleration_std must be finite and non-negative")
    transition = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    noise_map = np.array(
        [[0.5 * dt**2, 0.0], [0.0, 0.5 * dt**2], [dt, 0.0], [0.0, dt]]
    )
    process = noise_map @ (q**2 * np.eye(2)) @ noise_map.T
    return transition, process


def predict_tracks(
    tracks: tuple[TrackState, ...],
    *,
    frame: int,
    seconds_per_frame: float,
    config: FilterConfig,
) -> tuple[TrackState, ...]:
    """Predict each retained track to ``frame`` without using observations."""

    if frame < 0:
        raise ValueError("frame must be a non-negative integer")
    if not isfinite(float(seconds_per_frame)) or seconds_per_frame <= 0.0:
        raise ValueError("seconds_per_frame must be finite and positive")
    predicted: list[TrackState] = []
    for track in sorted(tracks, key=lambda item: item.track_id):
        frame_delta = frame - track.frame
        if frame_delta <= 0:
            raise ValueError("prediction frame must follow every track frame")
        transition, process = constant_velocity_matrices(
            frame_delta * float(seconds_per_frame), config.acceleration_std
        )
        state = transition @ np.asarray(track.state)
        covariance = transition @ np.asarray(track.covariance) @ transition.T + process
        predicted.append(
            TrackState(
                track_id=track.track_id,
                frame=frame,
                state=tuple(state),
                covariance=tuple(map(tuple, covariance)),
                log_odds=track.log_odds,
                hits=track.hits,
                misses=track.misses,
                observation_history=track.observation_history,
            )
        )
    return tuple(predicted)


def update_track_state(
    predicted: TrackState,
    gate: GatedAssociation,
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    """Apply the 2-D Kalman measurement update for one selected association."""

    if (predicted.frame, predicted.track_id) != (gate.frame, gate.track_id):
        raise ValueError("predicted track and gate scopes do not match")
    covariance = np.asarray(predicted.covariance)
    innovation_covariance = np.asarray(gate.innovation_covariance)
    gain = covariance @ MEASUREMENT_MATRIX.T @ np.linalg.inv(innovation_covariance)
    state = np.asarray(predicted.state) + gain @ np.asarray(gate.innovation)
    identity = np.eye(4)
    residual = identity - gain @ MEASUREMENT_MATRIX
    # Joseph form preserves positive semidefiniteness under floating-point error.
    measurement_covariance = (
        innovation_covariance
        - MEASUREMENT_MATRIX @ covariance @ MEASUREMENT_MATRIX.T
    )
    updated_covariance = (
        residual @ covariance @ residual.T
        + gain @ measurement_covariance @ gain.T
    )
    updated_covariance = 0.5 * (updated_covariance + updated_covariance.T)
    return tuple(state), tuple(map(tuple, updated_covariance))


def filter_association_hypotheses(
    hypotheses: tuple[AssociationHypothesis, ...],
    *,
    minimum_weight: float = 0.0,
) -> tuple[AssociationHypothesis, ...]:
    """Keep candidates at or above the declared MWIS benefit threshold."""

    threshold = float(minimum_weight)
    if not isfinite(threshold) or threshold < 0.0:
        raise ValueError("minimum_weight must be finite and non-negative")
    return tuple(
        sorted(
            (item for item in hypotheses if float(item.weight) >= threshold),
            key=lambda item: (item.track_id, item.observation_id, item.hypothesis_id),
        )
    )


def filter_tracks(
    tracks: tuple[TrackState, ...],
    config: FilterConfig,
) -> tuple[TrackState, ...]:
    """Apply declared probability/miss/cap rules after Bayesian updating."""

    eligible = [
        track
        for track in tracks
        if track.posterior_probability >= config.minimum_posterior
        and track.misses <= config.maximum_misses
    ]
    eligible.sort(key=lambda item: (-item.posterior_probability, item.track_id))
    retained = eligible[: config.maximum_tracks]
    return tuple(sorted(retained, key=lambda item: item.track_id))
