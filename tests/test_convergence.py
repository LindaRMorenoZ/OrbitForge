
import numpy as np
import pytest

from orbitforge import Body, SystemState
from orbitforge.validation.convergence import (
    ConvergenceResult,
    run_convergence_test,
)


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
@pytest.mark.parametrize(
    "reference_factor",
    [0, 1, -1, 1.5, True],
)
def test_invalid_reference_factor_is_rejected(reference_factor):
    with pytest.raises(ValueError, match="reference_factor"):
        run_convergence_test(
            circular_binary(),
            duration=1.0,
            time_steps=[0.02, 0.01],
            softening=0.01,
            reference_factor=reference_factor,
        )

def test_convergence_rejects_zero_initial_energy():
    state = SystemState.from_bodies(
        [
            Body("a", 1.0, np.array([-0.5, 0.0])),
            Body("b", 1.0, np.array([0.5, 0.0])),
        ],
        np.array([
            [0.0, -1.0],
            [0.0, 1.0],
        ]),
    )

    with pytest.raises(ValueError, match="initial energy"):
        run_convergence_test(
            state,
            duration=1.0,
            time_steps=[0.02, 0.01],
            gravitational_constant=1.0,
            softening=0.0,
        )
def test_convergence_does_not_modify_initial_state():
    state = circular_binary()

    original_positions = state.positions.copy()
    original_velocities = state.velocities.copy()
    original_masses = state.masses.copy()
    original_time = state.time

    run_convergence_test(
        state,
        duration=1.0,
        time_steps=[0.02, 0.01, 0.005],
        softening=0.01,
    )

    np.testing.assert_array_equal(state.positions, original_positions)
    np.testing.assert_array_equal(state.velocities, original_velocities)
    np.testing.assert_array_equal(state.masses, original_masses)
    assert state.time == original_time

def test_convergence_sorts_time_steps_from_largest_to_smallest():
    result = run_convergence_test(
        circular_binary(),
        duration=1.0,
        time_steps=[0.005, 0.02, 0.01],
        softening=0.01,
    )

    np.testing.assert_array_equal(
        result.dt,
        np.array([0.02, 0.01, 0.005]),
    )
def test_convergence_result_arrays_are_read_only_and_independent():
    dt = np.array([0.02, 0.01])
    errors = np.array([0.1, 0.01])
    orders = np.array([np.nan, 2.0])

    result = ConvergenceResult(
        dt=dt,
        position_error=errors,
        energy_drift=errors,
        angular_momentum_drift=errors,
        observed_order=orders,
    )

    # Modificar los datos originales no debe alterar el resultado.
    dt[0] = 99.0
    errors[0] = 99.0
    orders[1] = 99.0

    assert result.dt[0] == 0.02
    assert result.position_error[0] == 0.1
    assert result.observed_order[1] == 2.0

    # Los arrays del resultado no deben poder modificarse.
    with pytest.raises(ValueError):
        result.dt[0] = 0.03

    with pytest.raises(ValueError):
        result.position_error[0] = 0.5

@pytest.mark.parametrize(
    "changes",
    [
        {"dt": np.array([0.01, 0.02])},
        {"dt": np.array([0.02, 0.0])},
        {"position_error": np.array([0.1, -0.01])},
        {"energy_drift": np.array([0.1, np.inf])},
        {"angular_momentum_drift": np.array([0.1])},
        {"observed_order": np.array([np.nan, np.inf])},
    ],
)
def test_convergence_result_rejects_invalid_data(changes):
    data = {
        "dt": np.array([0.02, 0.01]),
        "position_error": np.array([0.1, 0.01]),
        "energy_drift": np.array([0.02, 0.01]),
        "angular_momentum_drift": np.array([0.03, 0.01]),
        "observed_order": np.array([np.nan, 2.0]),
    }
    data.update(changes)

    with pytest.raises(ValueError):
        ConvergenceResult(**data)
