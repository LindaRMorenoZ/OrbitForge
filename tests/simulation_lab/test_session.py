import pytest

from orbitforge.builder import SystemBuilder
from orbitforge.config import SimulationConfig
from orbitforge.experiments import Experiment
from orbitforge.simulation_lab.session import SimulationSession


@pytest.fixture
def small_experiment():
    builder = SystemBuilder()
    builder.add_body(
        name="Body",
        mass=1.0,
        position=[0.0, 0.0],
        velocity=[1.0, 0.0],
    )

    config = SimulationConfig(total_time=0.1, dt=0.01)

    return Experiment(
        name="Test experiment",
        builder=builder,
        config=config,
    )


def test_session_requires_experiment():
    with pytest.raises(TypeError):
        SimulationSession()


def test_session_initializes_with_ready_controller(small_experiment):
    session = SimulationSession(small_experiment)

    assert session.status.name == "READY"
    assert session.result is None
    assert session.analysis is None
    assert session.metrics is None


def test_session_rejects_invalid_experiment():
    with pytest.raises(TypeError):
        SimulationSession("not an experiment")

def test_session_exposes_its_controller(small_experiment):
    session = SimulationSession(small_experiment)

    assert session.controller is not None
    assert session.controller.status.name == "READY"


def test_session_start_changes_status(small_experiment):
    session = SimulationSession(small_experiment)

    session.start()

    assert session.status.name == "RUNNING"


def test_session_pause_and_resume(small_experiment):
    session = SimulationSession(small_experiment)

    session.start()
    session.pause()

    assert session.status.name == "PAUSED"

    session.resume()

    assert session.status.name == "RUNNING"

def test_session_step_advances_simulation(small_experiment):
    session = SimulationSession(small_experiment)

    session.start()
    session.step()

    assert session.current_state.time > 0.0


def test_session_completes_and_stores_result(small_experiment):
    session = SimulationSession(small_experiment)

    session.start()

    while session.status.name == "RUNNING":
        session.step()

    assert session.status.name == "COMPLETED"
    assert session.result is not None


def test_session_analyze_requires_result(small_experiment):
    session = SimulationSession(small_experiment)

    with pytest.raises(RuntimeError):
        session.analyze()


def test_session_analyzes_completed_result(small_experiment):
    session = SimulationSession(small_experiment)

    session.start()

    while session.status.name == "RUNNING":
        session.step()

    analysis = session.analyze()

    assert analysis is not None
    assert session.analysis is analysis

def test_session_evaluate_requires_analysis(small_experiment):
    session = SimulationSession(small_experiment)

    with pytest.raises(RuntimeError):
        session.evaluate()


def test_session_evaluates_conservation_metrics(small_experiment):
    session = SimulationSession(small_experiment)
    session.start()

    while session.status.name == "RUNNING":
        session.step()

    session.analyze()
    metrics = session.evaluate()

    assert metrics is not None
    assert session.metrics is metrics


def test_session_reset_restores_initial_state(small_experiment):
    session = SimulationSession(small_experiment)
    initial_time = session.current_state.time

    session.start()
    session.step()
    session.reset()

    assert session.status.name == "READY"
    assert session.current_state.time == initial_time
    assert session.result is None
    assert session.analysis is None
    assert session.metrics is None

def test_session_exposes_simulation_progress(small_experiment):
    session = SimulationSession(small_experiment)

    assert session.progress == 0.0

    session.start()
    session.step()

    assert session.progress > 0.0


def test_session_reports_completion(small_experiment):
    session = SimulationSession(small_experiment)

    assert session.is_completed is False

    session.start()

    while session.status.name == "RUNNING":
        session.step()

    assert session.is_completed is True
    assert session.progress == 1.0
