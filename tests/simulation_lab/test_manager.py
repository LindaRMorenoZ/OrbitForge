import pytest

from orbitforge.builder import SystemBuilder
from orbitforge.config import SimulationConfig
from orbitforge.experiments import Experiment
from orbitforge.simulation_lab.manager import ExperimentManager
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


def test_manager_initializes_empty():
    manager = ExperimentManager()

    assert manager.sessions == {}


def test_manager_creates_session(small_experiment):
    manager = ExperimentManager()

    session = manager.create_session(small_experiment)

    assert isinstance(session, SimulationSession)
    assert manager.sessions["Test experiment"] is session


def test_manager_retrieves_session(small_experiment):
    manager = ExperimentManager()
    session = manager.create_session(small_experiment)

    retrieved = manager.get_session("Test experiment")

    assert retrieved is session

def test_manager_rejects_invalid_experiment():
    manager = ExperimentManager()

    with pytest.raises(TypeError):
        manager.create_session("not an experiment")


def test_manager_rejects_duplicate_names(small_experiment):
    manager = ExperimentManager()
    manager.create_session(small_experiment)

    with pytest.raises(ValueError):
        manager.create_session(small_experiment)


def test_manager_rejects_empty_experiment_name():
    manager = ExperimentManager()

    with pytest.raises(ValueError):
        manager.get_session("")


def test_manager_raises_for_unknown_session():
    manager = ExperimentManager()

    with pytest.raises(KeyError):
        manager.get_session("Unknown experiment")


def test_manager_removes_session(small_experiment):
    manager = ExperimentManager()
    session = manager.create_session(small_experiment)

    removed = manager.remove_session("Test experiment")

    assert removed is session
    assert manager.sessions == {}


def test_manager_raises_when_removing_unknown_session():
    manager = ExperimentManager()

    with pytest.raises(KeyError):
        manager.remove_session("Unknown experiment")

def test_manager_keeps_multiple_sessions_independent():
    manager = ExperimentManager()

    builder_a = SystemBuilder()
    builder_a.add_body(
        name="Body A",
        mass=1.0,
        position=[0.0, 0.0],
        velocity=[1.0, 0.0],
    )

    builder_b = SystemBuilder()
    builder_b.add_body(
        name="Body B",
        mass=2.0,
        position=[0.0, 0.0],
        velocity=[0.0, 1.0],
    )

    config = SimulationConfig(total_time=0.1, dt=0.01)

    experiment_a = Experiment("Experiment A", builder_a, config)
    experiment_b = Experiment("Experiment B", builder_b, config)

    session_a = manager.create_session(experiment_a)
    session_b = manager.create_session(experiment_b)

    session_a.start()
    session_a.step()

    assert session_a.current_state.time > 0.0
    assert session_b.current_state.time == 0.0
    assert session_b.status.name == "READY"
