"""Interactive 2D and 3D visualization for simulation sessions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

from orbitforge.controller.simulation_controller import ControllerStatus
from orbitforge.simulation_lab.session import SimulationSession


def _validate_session(session: SimulationSession) -> None:
    """Validate the simulation session."""
    if not isinstance(session, SimulationSession):
        raise TypeError("session must be a SimulationSession instance")


class InteractiveSimulationViewer:
    """Interactive 2D and 3D viewer for an N-body simulation session."""

    def __init__(
        self,
        session: SimulationSession,
        *,
        interval: int = 30,
        steps_per_frame: int = 10,
    ) -> None:
        _validate_session(session)

        if not isinstance(interval, int) or isinstance(interval, bool) or interval <= 0:
            raise ValueError("interval must be a positive integer")

        if (
            not isinstance(steps_per_frame, int)
            or isinstance(steps_per_frame, bool)
            or steps_per_frame <= 0
        ):
            raise ValueError("steps_per_frame must be a positive integer")

        self.session = session
        self.interval = interval
        self.steps_per_frame = steps_per_frame
        self._max_history = 2000

        self.figure = None
        self.axes = None
        self.animation = None

        self._points = None
        self._trajectories = []
        self._history = []
        self._buttons = []

    def _initialize_plot(self) -> None:
        """Initialize the figure, body markers, and trajectory lines."""
        state = self.session.current_state
        dimensions = state.dimensions
        positions = state.positions

        self.figure, self.axes = _create_axes(dimensions)

        if dimensions == 2:
            self._points = self.axes.scatter(
                positions[:, 0],
                positions[:, 1],
            )
        else:
            self._points = self.axes.scatter(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
            )

        for _ in range(state.n_bodies):
            if dimensions == 2:
                (line,) = self.axes.plot([], [])
            else:
                (line,) = self.axes.plot([], [], [])

            self._trajectories.append(line)

        self._history = [positions.copy()]

        self.figure.tight_layout()

    def _update_frame(self, _frame):
        """Advance the simulation and update the graphical elements."""
        controller = self.session.controller
        previous_step = controller.current_step

        if controller.status == ControllerStatus.RUNNING:
            for _ in range(self.steps_per_frame):
                if controller.status != ControllerStatus.RUNNING:
                    break
                self.session.step()

        state = self.session.current_state
        positions = state.positions
        dimensions = state.dimensions

        if controller.current_step != previous_step:
            self._history.append(positions.copy())

        if len(self._history) > self._max_history:
            self._history.pop(0)

        history = np.asarray(self._history)

        if dimensions == 2:
            self._points.set_offsets(positions[:, :2])
        else:
            self._points._offsets3d = (
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
            )

        for body_index, line in enumerate(self._trajectories):
            trajectory = history[:, body_index, :]

            if dimensions == 2:
                line.set_data(
                    trajectory[:, 0],
                    trajectory[:, 1],
                )
            else:
                line.set_data(
                    trajectory[:, 0],
                    trajectory[:, 1],
                )
                line.set_3d_properties(trajectory[:, 2])

        self.axes.set_title(
            f"OrbitForge — {dimensions}D Simulation | "
            f"t = {state.time:.4f} | {controller.status.name}"
        )

        return [self._points, *self._trajectories]

    def _start_or_resume(self, _event) -> None:
        """Start or resume the simulation."""
        status = self.session.status

        if status == ControllerStatus.READY:
            self.session.start()
        elif status == ControllerStatus.PAUSED:
            self.session.resume()

    def _pause(self, _event) -> None:
        """Pause the simulation."""
        if self.session.status == ControllerStatus.RUNNING:
            self.session.pause()

    def _stop(self, _event) -> None:
        """Stop the simulation."""
        if self.session.status in (
            ControllerStatus.RUNNING,
            ControllerStatus.PAUSED,
        ):
            self.session.stop()

    def _reset(self, _event) -> None:
        """Reset the simulation and clear the visual history."""
        self.session.reset()

        positions = self.session.current_state.positions
        self._history = [positions.copy()]

        for line in self._trajectories:
            if self.session.current_state.dimensions == 2:
                line.set_data([], [])
            else:
                line.set_data([], [])
                line.set_3d_properties([])

        self._update_frame(None)

    def _initialize_controls(self) -> None:
        """Create interactive simulation control buttons."""
        self.figure.subplots_adjust(bottom=0.22)

        button_positions = [
            [0.08, 0.06, 0.18, 0.08],
            [0.29, 0.06, 0.18, 0.08],
            [0.50, 0.06, 0.18, 0.08],
            [0.71, 0.06, 0.18, 0.08],
        ]

        callbacks = [
            ("Start / Resume", self._start_or_resume),
            ("Pause", self._pause),
            ("Stop", self._stop),
            ("Reset", self._reset),
        ]

        for position, (label, callback) in zip(
            button_positions,
            callbacks,
            strict=True,
        ):
            button_axes = self.figure.add_axes(position)
            button = Button(button_axes, label)
            button.on_clicked(callback)
            self._buttons.append(button)

    def show(self) -> None:
        """Display the interactive simulation viewer."""
        if self.figure is not None:
            raise RuntimeError("viewer has already been initialized")

        self._initialize_plot()
        self._initialize_controls()

        self.animation = FuncAnimation(
            self.figure,
            self._update_frame,
            interval=self.interval,
            blit=False,
            cache_frame_data=False,
        )

        plt.show()


def _create_axes(dimensions: int):
    """Create a Matplotlib figure with dimension-appropriate axes."""
    if dimensions not in (2, 3):
        raise ValueError("Only 2D and 3D systems are supported")

    figure = plt.figure(figsize=(9, 7))

    if dimensions == 2:
        axes = figure.add_subplot(111)
        axes.set_xlabel("X")
        axes.set_ylabel("Y")
        axes.set_aspect("equal", adjustable="datalim")
    else:
        axes = figure.add_subplot(111, projection="3d")
        axes.set_xlabel("X")
        axes.set_ylabel("Y")
        axes.set_zlabel("Z")

    axes.set_title(f"OrbitForge — {dimensions}D Simulation")

    return figure, axes


def _plot_trajectories(axes, positions, dimensions: int):
    """Plot the trajectories of all bodies in a 2D or 3D system."""
    if dimensions not in (2, 3):
        raise ValueError("Only 2D and 3D systems are supported")

    if positions.ndim != 3 or positions.shape[2] != dimensions:
        raise ValueError("positions must have shape (n_steps, n_bodies, dimensions)")

    lines = []

    for body_index in range(positions.shape[1]):
        trajectory = positions[:, body_index, :]

        if dimensions == 2:
            (line,) = axes.plot(
                trajectory[:, 0],
                trajectory[:, 1],
            )
        else:
            (line,) = axes.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                trajectory[:, 2],
            )

        lines.append(line)

    return lines


def visualize_session(
    session: SimulationSession,
) -> None:
    """Display the current state of a simulation session."""
    _validate_session(session)

    state = session.current_state
    dimensions = state.dimensions

    figure, axes = _create_axes(dimensions)

    positions = state.positions

    if dimensions == 2:
        axes.scatter(positions[:, 0], positions[:, 1])
    else:
        axes.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
        )

    figure.tight_layout()
    plt.show()


def visualize_result(result, names: tuple[str, ...]) -> None:
    """Display the complete trajectories of a simulation result."""
    if not isinstance(names, tuple) or not all(
        isinstance(name, str) and name.strip() for name in names
    ):
        raise TypeError("names must be a tuple of non-empty strings")

    positions = np.asarray(result.positions)

    if positions.ndim != 3:
        raise ValueError("result.positions must have shape (n_steps, n_bodies, dimensions)")

    if positions.shape[0] == 0 or positions.shape[1] == 0:
        raise ValueError("result.positions must contain at least one step and body")

    dimensions = positions.shape[2]

    if dimensions not in (2, 3):
        raise ValueError("Only 2D and 3D results are supported")

    if len(names) != positions.shape[1]:
        raise ValueError("names must contain one name per body")

    if not np.all(np.isfinite(positions)):
        raise ValueError("result.positions must contain only finite values")

    figure, axes = _create_axes(dimensions)
    lines = _plot_trajectories(axes, positions, dimensions)

    final_positions = positions[-1]

    for index, line in enumerate(lines):
        line.set_label(names[index])

    if dimensions == 2:
        axes.scatter(
            final_positions[:, 0],
            final_positions[:, 1],
            marker="o",
        )
    else:
        axes.scatter(
            final_positions[:, 0],
            final_positions[:, 1],
            final_positions[:, 2],
            marker="o",
        )

    axes.legend()
    figure.tight_layout()
    plt.show()


def visualize_interactive_session(
    session: SimulationSession,
    *,
    interval: int = 30,
    steps_per_frame: int = 10,
) -> InteractiveSimulationViewer:
    """Open an interactive viewer for a simulation session."""
    viewer = InteractiveSimulationViewer(
        session,
        interval=interval,
        steps_per_frame=steps_per_frame,
    )

    viewer.show()
    return viewer
