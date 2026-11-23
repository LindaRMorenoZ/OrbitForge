"""Interactive 2D and 3D visualization for simulation sessions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

from orbitforge.controller.simulation_controller import ControllerStatus
from orbitforge.diagnostics import (
    kinetic_energy,
    potential_energy,
)
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
        self._body_colors = []
        self._history = []
        self._buttons = []
        self._energy_reference = None
        self._diagnostics_text = None
        self._diagnostics_figure = None
        self._diagnostics_scroll_offset = 0
        self._diagnostics_visible_lines = 30
        self._diagnostics_lines = []
        self._diagnostics_line_count = 0
        self._diagnostics_scrollbar = None                

    def _initialize_plot(self) -> None:
        """Initialize the figure, body markers, and trajectory lines."""
        state = self.session.current_state
        dimensions = state.dimensions
        positions = state.positions
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self._body_colors = [
            color_cycle[i % len(color_cycle)]
            for i in range(state.n_bodies)
        ]


        self.figure, self.axes = _create_axes(dimensions)

        if dimensions == 2:
            self._points = self.axes.scatter(
                positions[:, 0],
                positions[:, 1],
                c=self._body_colors,
            )
        else:
            self._points = self.axes.scatter(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                c=self._body_colors,
            )

        for body_index in range(state.n_bodies):
            if dimensions == 2:
                (line,) = self.axes.plot([], [], color=self._body_colors[body_index])
            else:
                (line,) = self.axes.plot([], [], color=self._body_colors[body_index])

            self._trajectories.append(line)

        legend_handles = []

        for index, name in enumerate(state.names):
            handle = plt.Line2D(
                [],
                [],
                color=self._body_colors[index],
                marker="o",
                linestyle="-",
                label=name,
            )
            legend_handles.append(handle)

        self.axes.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.05, 1.0),
            borderaxespad=0,
            title="Bodies",
            fontsize=9,
            title_fontsize=10,
        )            

        
        self._history = [positions.copy()]

        experiment = self.session._experiment
        gravitational_constant = experiment._gravitational_constant
        softening = experiment._softening

        initial_state = experiment.initial_state

        self._energy_reference = (
            kinetic_energy(initial_state)
            + potential_energy(
                initial_state,
                gravitational_constant,
                softening,
            )
        )

        self._diagnostics_figure = plt.figure(
            figsize=(6, 8),
            num="OrbitForge — Physical Diagnostics",
        )

        self._diagnostics_text = self._diagnostics_figure.add_axes(
            [0.08, 0.05, 0.82, 0.9]
        )
        self._diagnostics_text.axis("off")

        scrollbar_axes = self._diagnostics_figure.add_axes(
            [0.94, 0.05, 0.025, 0.9]
        )

        self._diagnostics_scrollbar = Slider(
            scrollbar_axes,
            "",
            valmin=-1,
            valmax=0,
            valinit=0,
            orientation="vertical",
        )

        self._diagnostics_scrollbar.valtext.set_visible(False)
        self._diagnostics_scrollbar.on_changed(
            self._on_diagnostics_scrollbar
        )        

        self._diagnostics_figure.canvas.mpl_connect(
            "scroll_event",
            self._scroll_diagnostics,
        )        

        self._adjust_diagnostics_font()

        self.figure.canvas.mpl_connect(
            "resize_event",
            self._adjust_diagnostics_font,
        )

        self._update_diagnostics(state)

        self.figure.tight_layout()

    def _update_diagnostics(self, state):
        """Update global and per-body physical diagnostics."""

        experiment = self.session._experiment

        kinetic = kinetic_energy(state)
        potential = potential_energy(
            state,
            experiment._gravitational_constant,
            experiment._softening,
        )
        total = kinetic + potential

        relative_drift = (
            (total - self._energy_reference)
            / abs(self._energy_reference)
        )

        lines = []

        def add_line(text, color="black", weight="normal"):
            lines.append((text, color, weight))

        add_line("SYSTEM DIAGNOSTICS", weight="bold")
        add_line(f"Kinetic:   {kinetic:.3e}")
        add_line(f"Potential: {potential:.3e}")
        add_line(f"Total:     {total:.3e}")
        add_line(f"Drift:     {relative_drift:.3e}")

        add_line("")
        add_line("BODY DIAGNOSTICS", weight="bold")

        for index in range(state.n_bodies):
            name = state.names[index]
            color = self._body_colors[index]

            mass = state.masses[index]
            position = state.positions[index]
            velocity = state.velocities[index]

            speed = float(np.linalg.norm(velocity))
            body_kinetic = 0.5 * mass * speed**2

            add_line(f"● {name}", color=color, weight="bold")
            add_line(f"  Mass:  {mass:.3e}")
            add_line(f"  Speed: {speed:.3e}")
            add_line(f"  K:     {body_kinetic:.3e}")

            position_text = ", ".join(
                f"{value:.2e}" for value in position
            )
            velocity_text = ", ".join(
                f"{value:.2e}" for value in velocity
            )

            add_line(f"  Pos: ({position_text})")
            add_line(f"  Vel: ({velocity_text})")
            add_line("")

        self._diagnostics_lines = lines
        self._diagnostics_line_count = len(lines)

        max_offset = max(
            0,
            self._diagnostics_line_count - self._diagnostics_visible_lines,
        )
        self._diagnostics_scroll_offset = min(
            self._diagnostics_scroll_offset,
            max_offset,
        )

        self._render_diagnostics()

    def _adjust_diagnostics_font(self, _event=None):
        """Adjust diagnostic text size to the figure dimensions."""

        if self._diagnostics_text is None:
            return

        width, height = self.figure.get_size_inches()

        available_height = height * 0.78
        lines_per_body = 7
        total_lines = 7 + self.session.current_state.n_bodies * lines_per_body

        font_size = np.clip(
            available_height * 72 / (total_lines * 1.8),
            5,
            8,
        )

        for text in self._diagnostics_text.texts:
            if text.get_fontweight() == "bold":
                text.set_fontsize(font_size * 1.05)
            else:
                text.set_fontsize(font_size)

    def _update_axis_limits(self, positions):
        """Expand 3D axis limits when bodies approach the boundaries."""
        if positions.shape[1] != 3:
            return

        current_limits = [
            self.axes.get_xlim3d(),
            self.axes.get_ylim3d(),
            self.axes.get_zlim3d(),
        ]

        current_radius = max(
            max(abs(lower), abs(upper))
            for lower, upper in current_limits
        )

        required_radius = float(np.max(np.abs(positions))) * 1.2

        if required_radius > current_radius:
            new_radius = max(required_radius, current_radius * 1.5)

            self.axes.set_xlim3d(-new_radius, new_radius)
            self.axes.set_ylim3d(-new_radius, new_radius)
            self.axes.set_zlim3d(-new_radius, new_radius)
            self.axes.set_box_aspect((1, 1, 1))

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
        self._update_diagnostics(state)


        if dimensions == 3:
            self._update_axis_limits(positions)        

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

    def _scroll_diagnostics(self, event):
        """Scroll through diagnostics using the mouse wheel."""

        if self._diagnostics_line_count <= self._diagnostics_visible_lines:
            return

        if event.button == "down":
            self._diagnostics_scroll_offset += 3
        elif event.button == "up":
            self._diagnostics_scroll_offset -= 3

        max_offset = max(
            0,
            self._diagnostics_line_count - self._diagnostics_visible_lines,
        )

        self._diagnostics_scroll_offset = min(
            max(self._diagnostics_scroll_offset, 0),
            max_offset,
        )

        if self._diagnostics_scrollbar is not None:
            max_offset = max(
                0,
                self._diagnostics_line_count
                - self._diagnostics_visible_lines,
            )

            self._diagnostics_scrollbar.set_val(
                -self._diagnostics_scroll_offset / max_offset
                if max_offset > 0
                else 0
            )

        self._render_diagnostics()

    def _on_diagnostics_scrollbar(self, value):
        """Update diagnostics position from the scrollbar."""

        max_offset = max(
            0,
            self._diagnostics_line_count - self._diagnostics_visible_lines,
        )

        self._diagnostics_scroll_offset = int(
            round(-value * max_offset)
        )

        self._render_diagnostics()

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

        experiment = self.session._experiment
        initial_state = experiment.initial_state

        self._energy_reference = (
            kinetic_energy(initial_state)
            + potential_energy(
                initial_state,
                experiment._gravitational_constant,
                experiment._softening,
            )
        )

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
        self.figure.subplots_adjust(
            left=0.08,
            right=0.92,            
            bottom=0.22,
        )

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

    def _render_diagnostics(self):
        """Render the currently visible diagnostics lines."""

        ax = self._diagnostics_text
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(
            self._diagnostics_scroll_offset
            + self._diagnostics_visible_lines,
            self._diagnostics_scroll_offset,
        )
        ax.axis("off")

        start = self._diagnostics_scroll_offset
        end = min(
            start + self._diagnostics_visible_lines,
            self._diagnostics_line_count,
        )

        for index in range(start, end):
            text, color, weight = self._diagnostics_lines[index]

            ax.text(
                0.02,
                index + 0.5,
                text,
                color=color,
                fontsize=9,
                fontweight=weight,
                verticalalignment="center",
            )

        self._diagnostics_figure.canvas.draw_idle()


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
