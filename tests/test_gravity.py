import numpy as np
import pytest

from orbitforge import Body, SystemState, accelerations


def two_body_state():
    return SystemState.from_bodies(
        [Body("left", 2.0, np.array([-1.0, 0.0])),
         Body("right", 1.0, np.array([1.0, 0.0]))],
        np.zeros((2, 2)),
    )


def test_pair_accelerations_obey_newton_third_law():
    state = two_body_state()
    acc = accelerations(state, gravitational_constant=1.0)
    net_force = np.sum(state.masses[:, None] * acc, axis=0)
    np.testing.assert_allclose(net_force, 0.0, atol=1e-14)
    assert acc[0, 0] > 0
    assert acc[1, 0] < 0


def test_coincident_points_require_softening():
    state = SystemState.from_bodies(
        [Body("a", 1, np.array([0.0, 0.0])),
         Body("b", 1, np.array([0.0, 0.0]))],
        np.zeros((2, 2)),
    )
    with pytest.raises(ValueError, match="softening"):
        accelerations(state)
    assert np.all(np.isfinite(accelerations(state, softening=0.1)))
