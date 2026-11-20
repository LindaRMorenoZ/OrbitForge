import pytest

from orbitforge.config import SimulationConfig


def test_valid_configuration():
    config = SimulationConfig(
        total_time=100.0,
        dt=0.01,
        save_every=10,
    )

    assert config.total_time == 100.0
    assert config.dt == 0.01
    assert config.save_every == 10
    assert config.integrator == "velocity_verlet"


def test_calculate_number_of_steps():
    config = SimulationConfig(
        total_time=100.0,
        dt=0.01,
    )

    assert config.steps == 10000


def test_reject_zero_total_time():
    with pytest.raises(ValueError):
        SimulationConfig(total_time=0.0, dt=0.01)


def test_reject_negative_dt():
    with pytest.raises(ValueError):
        SimulationConfig(total_time=10.0, dt=-0.01)


def test_reject_invalid_save_every():
    with pytest.raises(ValueError):
        SimulationConfig(
            total_time=10.0,
            dt=0.01,
            save_every=0,
        )


def test_reject_unsupported_integrator():
    with pytest.raises(ValueError):
        SimulationConfig(
            total_time=10.0,
            dt=0.01,
            integrator="runge_kutta",
        )


def test_reject_incompatible_total_time_and_dt():
    with pytest.raises(ValueError):
        config = SimulationConfig(
            total_time=10.0,
            dt=3.0,
        )
        _ = config.steps


def test_configuration_is_immutable():
    config = SimulationConfig(total_time=10.0, dt=0.01)

    with pytest.raises(AttributeError):
        config.dt = 0.02