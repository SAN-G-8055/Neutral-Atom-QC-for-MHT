"""Decide which track-to-observation pairs are plausible before solving.

Each candidate must pass an inclusive Mahalanobis-distance check and, when
configured, a separate Euclidean innovation-distance check.  Gating only
removes impossible pairs; it neither updates tracks nor builds graph edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from filtering import MEASUREMENT_MATRIX
from models import GatedAssociation, Observation, TrackState


@dataclass(frozen=True, slots=True)
class GateConfig:
    mahalanobis_sq: float = 9.21
    maximum_innovation_distance: float | None = None

    def __post_init__(self) -> None:
        threshold = float(self.mahalanobis_sq)
        if not isfinite(threshold) or threshold <= 0.0:
            raise ValueError("mahalanobis_sq must be finite and positive")
        object.__setattr__(self, "mahalanobis_sq", threshold)
        if self.maximum_innovation_distance is not None:
            maximum_distance = float(self.maximum_innovation_distance)
            if not isfinite(maximum_distance) or maximum_distance <= 0.0:
                raise ValueError(
                    "maximum_innovation_distance must be finite and positive"
                )
            object.__setattr__(
                self, "maximum_innovation_distance", maximum_distance
            )


def gate_observations(
    predicted_tracks: tuple[TrackState, ...],
    observations: tuple[Observation, ...],
    config: GateConfig,
) -> tuple[GatedAssociation, ...]:
    """Return all track/observation pairs inside the inclusive gate."""

    if not predicted_tracks:
        return ()
    frame = predicted_tracks[0].frame
    if any(track.frame != frame for track in predicted_tracks):
        raise ValueError("predicted tracks must all have the same frame")
    if any(observation.frame != frame for observation in observations):
        raise ValueError("observations must share the predicted-track frame")
    admitted: list[GatedAssociation] = []
    for track in sorted(predicted_tracks, key=lambda item: item.track_id):
        state = np.asarray(track.state)
        covariance = np.asarray(track.covariance)
        predicted_position = MEASUREMENT_MATRIX @ state
        for observation in sorted(observations, key=lambda item: item.observation_id):
            measured_position = np.asarray(observation.position)
            if config.maximum_innovation_distance is not None:
                displacement = np.linalg.norm(measured_position - predicted_position)
                if displacement > config.maximum_innovation_distance:
                    continue
            innovation = measured_position - predicted_position
            innovation_covariance = (
                MEASUREMENT_MATRIX @ covariance @ MEASUREMENT_MATRIX.T
                + np.asarray(observation.covariance)
            )
            mahalanobis_sq = float(
                innovation @ np.linalg.solve(innovation_covariance, innovation)
            )
            if mahalanobis_sq <= config.mahalanobis_sq:
                admitted.append(
                    GatedAssociation(
                        frame=frame,
                        track_id=track.track_id,
                        observation_id=observation.observation_id,
                        innovation=tuple(innovation),
                        innovation_covariance=tuple(map(tuple, innovation_covariance)),
                        mahalanobis_sq=mahalanobis_sq,
                    )
                )
    return tuple(admitted)
