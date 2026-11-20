
import numpy as np
import pytest

from orbitforge import Body, SystemState
from orbitforge.validation.convergence import run_convergence_test


def circular_binary():
    bodies = [
        Body("a", 1.0, np.array([-0.5, 0.0])),
        Body("b", 1.0, np.array([0.5, 0.0])),
    ]
    return SystemState.from_bodies(
        bodies, np.array([[0.0, -0.5], [0.0, 0.5]])
    )


def test_convergence_errors_decrease_with_smaller_time_steps():
    result = run_convergence_test(
        circular_binary(),
        duration=1.0,
        time_steps=[0.02, 0.01, 0.005],
        softening=0.01,
    )

    assert result.position_error.shape == (3,)
    assert result.energy_drift.shape == (3,)
    assert result.angular_momentum_drift.shape == (3,)

    assert np.all(np.diff(result.position_error) < 0)


def test_velocity_verlet_observed_order_is_second_order():
    result = run_convergence_test(
        circular_binary(),
        duration=1.0,
        time_steps=[0.02, 0.01, 0.005],
        softening=0.01,
    )

    assert result.observed_order[1] == pytest.approx(2.0, abs=0.2)
    assert result.observed_order[2] == pytest.approx(2.0, abs=0.2)


@pytest.mark.parametrize(
    "duration,time_steps",
    [
        (0.0, [0.01, 0.005]),
        (1.0, []),
        (1.0, [0.01, 0.01]),
        (1.0, [0.03, 0.01]),
    ],
)
def test_invalid_convergence_parameters_are_rejected(duration, time_steps):
    with pytest.raises(ValueError):
        run_convergence_test(
            circular_binary(),
            duration=duration,
            time_steps=time_steps,
            softening=0.01,
        )
