from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from neutral_atom_mht.tracking.models import (
    Observation,
    TrackState,
    observations_from_detections,
)


def make_track(**changes: object) -> TrackState:
    values: dict[str, object] = {
        "track_id": 1,
        "frame": 0,
        "state": (2.0, 3.0, 1.0, -1.0),
        "covariance": tuple(map(tuple, np.eye(4))),
        "log_odds": 0.0,
        "posterior_probability": 0.5,
        "observation_history": ((0, 3),),
    }
    values.update(changes)
    return TrackState(**values)


@pytest.mark.parametrize(
    "covariance",
    (
        ((1.0, 2.0), (0.0, 1.0)),
        ((1.0, 0.0), (0.0, 0.0)),
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ),
)
def test_observation_requires_a_symmetric_positive_definite_covariance(
    covariance: object,
) -> None:
    with pytest.raises(ValueError, match="covariance"):
        Observation(0, 1, 2.0, 3.0, covariance)  # type: ignore[arg-type]


def test_track_models_normalize_numpy_scalars_and_arrays() -> None:
    track = make_track(
        track_id=np.int64(2),
        state=np.array([1.0, 2.0, 3.0, 4.0]),
        covariance=np.eye(4),
    )

    assert track.track_id == 2
    assert track.state == (1.0, 2.0, 3.0, 4.0)
    assert isinstance(track.state[0], float)


def test_track_rejects_a_tail_probability_inconsistent_with_log_odds() -> None:
    with pytest.raises(ValueError, match="sigmoid"):
        make_track(log_odds=-100.0, posterior_probability=1e-13)


@dataclass(frozen=True)
class FakeDetection:
    frame: int
    detection_id: int
    x_px: float
    y_px: float


def test_detection_adapter_preserves_frame_ids_and_coordinate_convention() -> None:
    observations = observations_from_detections(
        [FakeDetection(2, 7, 11.5, 4.25)], variance_px2=9.0
    )

    assert observations == (
        Observation(2, 7, 11.5, 4.25, ((9.0, 0.0), (0.0, 9.0))),
    )


def test_detection_adapter_rejects_duplicate_event_keys() -> None:
    events = [FakeDetection(2, 7, 1.0, 2.0), FakeDetection(2, 7, 3.0, 4.0)]

    with pytest.raises(ValueError, match="unique"):
        observations_from_detections(events)
