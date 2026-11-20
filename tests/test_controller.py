
import numpy as np
import pytest

from orbitforge.config import SimulationConfig
from orbitforge.controller.simulation_controller import (
    ControllerStatus,
    SimulationController,
)
from orbitforge.simulation import Simulation
from orbitforge.state import SystemState
from orbitforge.builder import SystemBuilder


def make_controller(save_every=1):
    state = SystemState(
        names=("Sun", "Planet"),
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
        total_time=0.1,
        dt=0.01,
        save_every=save_every,
    )

    return SimulationController(simulation, config)


def test_controller_initial_state():
    controller = make_controller()

    assert controller.status == ControllerStatus.READY
    assert controller.current_step == 0
    assert controller.progress == 0.0


def test_controller_start():
    controller = make_controller()

    controller.start()

    assert controller.status == ControllerStatus.RUNNING


def test_controller_cannot_step_before_start():
    controller = make_controller()

    with pytest.raises(RuntimeError):
        controller.step()


def test_controller_advances_one_step():
    controller = make_controller()
    controller.start()

    initial_time = controller.current_state.time
    new_state = controller.step()

    assert controller.current_step == 1
    assert new_state.time > initial_time
    assert controller.progress == pytest.approx(0.1)


def test_controller_pause_and_resume():
    controller = make_controller()
    controller.start()
    controller.step()

    controller.pause()
    assert controller.status == ControllerStatus.PAUSED

    with pytest.raises(RuntimeError):
        controller.step()

    controller.resume()
    assert controller.status == ControllerStatus.RUNNING

    controller.step()
    assert controller.current_step == 2


def test_controller_completes_simulation():
    controller = make_controller()
    controller.start()

    for _ in range(controller.config.steps):
        controller.step()

    assert controller.status == ControllerStatus.COMPLETED
    assert controller.progress == 1.0
    assert controller.current_step == controller.config.steps


def test_controller_cannot_start_twice():
    controller = make_controller()
    controller.start()

    with pytest.raises(RuntimeError):
        controller.start()


def test_controller_stop():
    controller = make_controller()
    controller.start()
    controller.step()

    controller.stop()

    assert controller.status == ControllerStatus.STOPPED


def test_controller_can_stop_while_paused():
    controller = make_controller()
    controller.start()
    controller.pause()

    controller.stop()

    assert controller.status == ControllerStatus.STOPPED


def test_controller_rejects_invalid_constructor_arguments():
    config = SimulationConfig(total_time=0.1, dt=0.01)

    with pytest.raises(TypeError):
        SimulationController("not a simulation", config)

    controller = make_controller()

    with pytest.raises(TypeError):
        SimulationController(controller.simulation, "not a config")
def test_controller_builds_result_after_completion():
    controller = make_controller()
    controller.start()

    for _ in range(controller.config.steps):
        controller.step()

    result = controller.result

    assert result is not None
    assert len(result.times) == controller.config.steps + 1
    assert result.positions.shape[0] == len(result.times)
    assert result.velocities.shape == result.positions.shape
    assert len(result.energies) == len(result.times)


def test_controller_result_matches_regular_simulation():
    controller = make_controller()
    controller.start()

    for _ in range(controller.config.steps):
        controller.step()

    controlled_result = controller.result
    regular_result = controller.simulation.run_configured(
        controller.config
    )

    assert controlled_result is not None

    np.testing.assert_allclose(
        controlled_result.positions,
        regular_result.positions,
    )
    np.testing.assert_allclose(
        controlled_result.velocities,
        regular_result.velocities,
    )
    np.testing.assert_allclose(
        controlled_result.energies,
        regular_result.energies,
    )

def test_controller_preserves_partial_result_when_stopped():
    controller = make_controller()
    controller.start()

    controller.step()
    controller.step()
    controller.stop()

    result = controller.result

    assert controller.status == ControllerStatus.STOPPED
    assert result is not None
    assert controller.current_step == 2
    assert result.final_state.time == pytest.approx(0.02)
    assert result.times[-1] == pytest.approx(0.02)   

def test_builder_creates_system_for_controller():
    builder = SystemBuilder(dimensions=2)

    builder.add_body(
        name="Sun",
        mass=1.0,
        position=[0.0, 0.0],
        velocity=[0.0, 0.0],
    )

    builder.add_body(
        name="Earth",
        mass=0.001,
        position=[1.0, 0.0],
        velocity=[0.0, 6.28],
    )

    simulation = builder.create_simulation()

    config = SimulationConfig(
        total_time=0.1,
        dt=0.01,
        save_every=2,
    )

    controller = SimulationController(simulation, config)

    assert controller.status == ControllerStatus.READY
    assert controller.current_state.n_bodies == 2

    controller.start()

    for _ in range(config.steps):
        controller.step()

    assert controller.status == ControllerStatus.COMPLETED
    assert controller.result is not None

def test_controller_respects_save_every():
    controller = make_controller(save_every=2)

    controller.start()

    for _ in range(controller.config.steps):
        controller.step()

    result = controller.result

    assert result is not None
    assert len(result.times) == 6

    np.testing.assert_allclose(
        result.times,
        [0.0, 0.02, 0.04, 0.06, 0.08, 0.1],
    )

def test_controller_saves_latest_state_when_stopped_between_snapshots():
    controller = make_controller(save_every=5)

    controller.start()

    controller.step()
    controller.step()
    controller.stop()

    result = controller.result

    assert result is not None
    assert len(result.times) == 2

    assert result.times[0] == pytest.approx(0.0)
    assert result.times[-1] == pytest.approx(0.02)

    assert result.final_state.time == pytest.approx(0.02)

    np.testing.assert_allclose(
        result.positions[-1],
        result.final_state.positions,
    )    

def test_controller_configuration_cannot_be_replaced():
    controller = make_controller()

    with pytest.raises(AttributeError):
        controller.config = SimulationConfig(
            total_time=0.2,
            dt=0.01,
        )

    with pytest.raises(AttributeError):
        controller.simulation = make_controller().simulation    


def test_controller_reset_after_completion():
    controller = make_controller()
    controller.start()

    for _ in range(controller.config.steps):
        controller.step()

    assert controller.status == ControllerStatus.COMPLETED

    controller.reset()

    assert controller.status == ControllerStatus.READY
    assert controller.current_step == 0
    assert controller.progress == 0.0
    assert controller.result is None

    np.testing.assert_allclose(
        controller.current_state.positions,
        controller.simulation.initial_state.positions,
    )


def test_controller_reset_after_stop():
    controller = make_controller()
    controller.start()
    controller.step()
    controller.stop()

    controller.reset()

    assert controller.status == ControllerStatus.READY
    assert controller.current_step == 0
    assert controller.result is None

    assert len(controller._times) == 1


def test_controller_cannot_reset_while_running():
    controller = make_controller()
    controller.start()

    with pytest.raises(RuntimeError):
        controller.reset()

    assert controller.status == ControllerStatus.RUNNING
