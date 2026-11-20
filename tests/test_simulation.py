import numpy as np
import pytest

from orbitforge import Body, Simulation, SystemState, linear_momentum, total_energy
from orbitforge.config import SimulationConfig


def circular_binary():
    bodies = [
        Body("a", 1.0, np.array([-0.5, 0.0])),
        Body("b", 1.0, np.array([0.5, 0.0])),
    ]
    return SystemState.from_bodies(
        bodies, np.array([[0.0, -0.5], [0.0, 0.5]])
    )


def test_simulation_saves_initial_and_final_state():
    result = Simulation(circular_binary(), softening=0.01).run(steps=10, dt=0.001)
    assert result.times.shape == (11,)
    assert result.positions.shape == (11, 2, 2)
    assert result.final_state.time == pytest.approx(0.01)


def test_velocity_verlet_has_bounded_energy_drift():
    state = circular_binary()
    result = Simulation(state, softening=0.01).run(steps=1000, dt=0.001, save_every=10)
    relative = np.abs((result.energies - result.energies[0]) / result.energies[0])
    assert np.max(relative) < 2e-3


def test_linear_momentum_is_conserved():
    state = circular_binary()
    initial = linear_momentum(state)
    final = Simulation(state, softening=0.01).run(steps=500, dt=0.001).final_state
    np.testing.assert_allclose(linear_momentum(final), initial, atol=1e-12)


@pytest.mark.parametrize("steps,dt,save_every", [(-1, 0.1, 1), (1, 0, 1), (1, 0.1, 0)])
def test_invalid_run_parameters_are_rejected(steps, dt, save_every):
    with pytest.raises(ValueError):
        Simulation(circular_binary()).run(steps, dt, save_every)

def test_run_configured_matches_regular_run():
    state = SystemState(
        names=["A", "B"],
        masses=np.array([1.0, 0.001]),
        positions=np.array([
            [0.0, 0.0],
            [1.0, 0.0],
        ]),
        velocities=np.array([
            [0.0, 0.0],
            [0.0, 6.28],
        ]),
    )

    simulation = Simulation(state)

    config = SimulationConfig(
        total_time=1.0,
        dt=0.01,
        save_every=10,
    )

    configured_result = simulation.run_configured(config)
    regular_result = simulation.run(
        steps=100,
        dt=0.01,
        save_every=10,
    )

    assert np.allclose(
        configured_result.positions,
        regular_result.positions,
    )

    assert np.allclose(
        configured_result.energies,
        regular_result.energies,
    )


def test_run_configured_rejects_invalid_config():
    simulation = Simulation(
        SystemState(
            names=["A"],
            masses=np.array([1.0]),
            positions=np.array([[0.0, 0.0]]),
            velocities=np.array([[0.0, 0.0]]),
        )
    )

    with pytest.raises(TypeError):
        simulation.run_configured("invalid")


def test_simulation_result_arrays_are_read_only():
    result = Simulation(circular_binary()).run(steps=10, dt=0.001)

    with pytest.raises(ValueError):
        result.positions[0, 0, 0] = 100.0

    with pytest.raises(ValueError):
        result.energies[0] = 100.0


def test_simulation_result_rejects_inconsistent_snapshot_shapes():
    result = Simulation(circular_binary()).run(steps=10, dt=0.001)

    with pytest.raises(ValueError):
        type(result)(
            times=result.times,
            positions=result.positions[:-1],
            velocities=result.velocities,
            energies=result.energies,
            final_state=result.final_state,
            gravitational_constant=result.gravitational_constant,
            softening=result.softening,
        )


def test_simulation_result_rejects_non_finite_data():
    result = Simulation(circular_binary()).run(steps=10, dt=0.001)

    invalid_times = result.times.copy()
    invalid_times[0] = np.nan

    with pytest.raises(ValueError):
        type(result)(
            times=invalid_times,
            positions=result.positions,
            velocities=result.velocities,
            energies=result.energies,
            final_state=result.final_state,
            gravitational_constant=result.gravitational_constant,
            softening=result.softening,
        )


def test_simulation_result_rejects_invalid_physical_parameters():
    result = Simulation(circular_binary()).run(steps=10, dt=0.001)

    with pytest.raises(ValueError):
        type(result)(
            times=result.times,
            positions=result.positions,
            velocities=result.velocities,
            energies=result.energies,
            final_state=result.final_state,
            gravitational_constant=-1.0,
            softening=result.softening,
        )
def test_simulation_result_final_state_is_independent():
    simulation = Simulation(circular_binary())
    result = simulation.run(steps=10, dt=0.001)

    original_position = result.final_state.positions.copy()

    simulation.initial_state.positions[0, 0] = 100.0

    np.testing.assert_array_equal(
        result.final_state.positions,
        original_position,
    )


def test_simulation_result_final_state_arrays_are_read_only():
    result = Simulation(circular_binary()).run(steps=10, dt=0.001)

    with pytest.raises(ValueError):
        result.final_state.positions[0, 0] = 100.0

    with pytest.raises(ValueError):
        result.final_state.masses[0] = 100.0