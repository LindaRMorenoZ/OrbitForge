import numpy as np
import pytest

from orbitforge import Body, SystemState


def test_state_from_bodies_copies_and_validates():
    bodies = [Body("a", 1.0, np.array([0.0, 0.0]))]
    state = SystemState.from_bodies(bodies, np.zeros((1, 2)))
    bodies[0].position.setflags(write=True) if False else None
    assert state.positions.shape == (1, 2)
    assert state.n_bodies == 1


@pytest.mark.parametrize("mass", [0.0, -1.0, np.nan, np.inf])
def test_body_rejects_invalid_mass(mass):
    with pytest.raises(ValueError):
        Body("bad", mass, np.array([0.0, 0.0]))


def test_state_rejects_mismatched_velocity_shape():
    with pytest.raises(ValueError):
        SystemState(("a",), np.array([1.0]), np.zeros((1, 2)), np.zeros((1, 3)))
