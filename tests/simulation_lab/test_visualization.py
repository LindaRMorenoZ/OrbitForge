"""Tests for interactive simulation visualization."""

import matplotlib

matplotlib.use("Agg")

import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orbitforge.builder import SystemBuilder
from orbitforge.config import SimulationConfig
from orbitforge.experiments import Experiment
from orbitforge.simulation_lab.session import SimulationSession
from orbitforge.simulation_lab.visualization import (
    InteractiveSimulationViewer,
    _create_axes,
    _plot_trajectories,
    _validate_session,
    visualize_result,
)


@pytest.mark.parametrize("dimensions", [2, 3])
def test_create_axes_supports_dimensions(dimensions):
    figure, axes = _create_axes(dimensions)

    try:
        if dimensions == 2:
            assert axes.name == "rectilinear"
        else:
            assert axes.name == "3d"
    finally:
        plt.close(figure)


def test_create_axes_rejects_unsupported_dimensions():
    with pytest.raises(ValueError, match="Only 2D and 3D"):
        _create_axes(4)


def test_validate_session_rejects_invalid_object():
    with pytest.raises(TypeError, match="SimulationSession"):
        _validate_session(object())


@pytest.mark.parametrize("dimensions", [2, 3])
def test_plot_trajectories_supports_dimensions(dimensions):
    figure, axes = _create_axes(dimensions)

    positions = np.zeros((5, 2, dimensions))

    try:
        lines = _plot_trajectories(axes, positions, dimensions)

        assert len(lines) == 2
    finally:
        plt.close(figure)


def test_plot_trajectories_rejects_invalid_shape():
    figure, axes = _create_axes(2)

    positions = np.zeros((5, 2, 3))

    try:
        with pytest.raises(ValueError, match="positions must have shape"):
            _plot_trajectories(axes, positions, 2)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("dimensions", [2, 3])
def test_visualize_result_supports_dimensions(dimensions, monkeypatch):
    positions = np.zeros((5, 2, dimensions))
    result = SimpleNamespace(positions=positions)

    monkeypatch.setattr(plt, "show", lambda: None)

    try:
        visualize_result(result, ("Body A", "Body B"))
    finally:
        plt.close("all")


def test_visualize_result_rejects_invalid_names(monkeypatch):
    positions = np.zeros((5, 2, 2))
    result = SimpleNamespace(positions=positions)

    monkeypatch.setattr(plt, "show", lambda: None)

    with pytest.raises(ValueError, match="one name per body"):
        visualize_result(result, ("Body A",))

    plt.close("all")


@pytest.fixture
def small_session():
    builder = SystemBuilder()
    builder.add_body(
        name="Body",
        mass=1.0,
        position=[0.0, 0.0],
        velocity=[1.0, 0.0],
    )

    config = SimulationConfig(total_time=0.1, dt=0.01)

    experiment = Experiment(
        name="Interactive test",
        builder=builder,
        config=config,
    )

    return SimulationSession(experiment)


def test_interactive_viewer_initializes_with_session(small_session):
    viewer = InteractiveSimulationViewer(small_session)

    assert viewer.session is small_session
    assert viewer.interval == 30
    assert viewer.steps_per_frame == 10
    assert viewer.figure is None
    assert viewer.axes is None
    assert viewer.animation is None


def test_interactive_viewer_rejects_invalid_session():
    with pytest.raises(TypeError):
        InteractiveSimulationViewer("not a session")


@pytest.mark.parametrize(
    ("interval", "steps_per_frame"),
    [
        (0, 10),
        (-1, 10),
        (30, 0),
        (30, -1),
        (True, 10),
        (30, False),
    ],
)
def test_interactive_viewer_rejects_invalid_parameters(
    small_session,
    interval,
    steps_per_frame,
):
    with pytest.raises(ValueError):
        InteractiveSimulationViewer(
            small_session,
            interval=interval,
            steps_per_frame=steps_per_frame,
        )


@pytest.mark.parametrize("dimensions", [2, 3])
def test_interactive_viewer_initializes_plot(
    small_session,
    dimensions,
):
    if dimensions == 3:
        builder = SystemBuilder(dimensions=3)
        builder.add_body(
            name="Body",
            mass=1.0,
            position=[0.0, 0.0, 0.0],
            velocity=[1.0, 0.0, 0.0],
        )
        experiment = Experiment(
            name="3D test",
            builder=builder,
            config=SimulationConfig(total_time=0.1, dt=0.01),
        )
        session = SimulationSession(experiment)
    else:
        session = small_session

    viewer = InteractiveSimulationViewer(session)
    viewer._initialize_plot()

    assert viewer.figure is not None
    assert viewer.axes is not None
    assert viewer._points is not None
    assert len(viewer._trajectories) == 1
    assert len(viewer._history) == 1

    plt.close(viewer.figure)


def test_interactive_viewer_updates_history_only_when_simulation_advances(
    small_session,
):
    viewer = InteractiveSimulationViewer(
        small_session,
        steps_per_frame=2,
    )
    viewer._initialize_plot()

    initial_history_length = len(viewer._history)

    # No advancement while the session is READY.
    viewer._update_frame(0)

    assert len(viewer._history) == initial_history_length

    # Advance the simulation.
    small_session.start()
    viewer._update_frame(1)

    assert small_session.controller.current_step == 2
    assert len(viewer._history) == initial_history_length + 1

    # Pause and verify that the history does not grow.
    small_session.pause()
    history_length = len(viewer._history)

    viewer._update_frame(2)

    assert len(viewer._history) == history_length

    plt.close(viewer.figure)


def test_interactive_viewer_limits_history(small_session):
    viewer = InteractiveSimulationViewer(small_session)
    viewer._initialize_plot()

    viewer._max_history = 3
    viewer._history = [small_session.current_state.positions.copy() for _ in range(3)]

    small_session.start()
    viewer.steps_per_frame = 1

    viewer._update_frame(0)

    assert len(viewer._history) == 3

    plt.close(viewer.figure)


def test_interactive_viewer_execution_controls(small_session):
    viewer = InteractiveSimulationViewer(small_session)

    viewer._start_or_resume(None)
    assert small_session.status.name == "RUNNING"

    viewer._pause(None)
    assert small_session.status.name == "PAUSED"

    viewer._start_or_resume(None)
    assert small_session.status.name == "RUNNING"

    viewer._stop(None)
    assert small_session.status.name == "STOPPED"


def test_interactive_viewer_reset(small_session):
    viewer = InteractiveSimulationViewer(small_session)
    viewer._initialize_plot()

    initial_positions = small_session.current_state.positions.copy()

    small_session.start()
    viewer._update_frame(0)

    assert small_session.controller.current_step > 0

    viewer._reset(None)

    assert small_session.status.name == "READY"
    assert small_session.controller.current_step == 0
    assert len(viewer._history) == 1
    assert (small_session.current_state.positions == initial_positions).all()

    plt.close(viewer.figure)


def test_interactive_viewer_initializes_controls(small_session):
    viewer = InteractiveSimulationViewer(small_session)
    viewer._initialize_plot()
    viewer._initialize_controls()

    assert len(viewer._buttons) == 4

    plt.close(viewer.figure)


def test_interactive_viewer_show_rejects_second_initialization(
    small_session,
    monkeypatch,
):
    viewer = InteractiveSimulationViewer(small_session)

    monkeypatch.setattr(plt, "show", lambda: None)
    monkeypatch.setattr(
        "orbitforge.simulation_lab.visualization.FuncAnimation",
        lambda *args, **kwargs: object(),
    )

    viewer.show()

    assert viewer.figure is not None
    assert viewer.animation is not None

    with pytest.raises(RuntimeError):
        viewer.show()

    plt.close(viewer.figure)
