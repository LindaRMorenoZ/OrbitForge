"""High-level deterministic simulation runner."""
from dataclasses import dataclass

import numpy as np

from .config import SimulationConfig
from .diagnostics import total_energy
from .gravity import accelerations
from .integrators import velocity_verlet
from .state import SystemState


@dataclass
class SimulationResult:
    """Saved snapshots from a completed simulation."""

    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    energies: np.ndarray
    final_state: SystemState
    gravitational_constant: float
    softening: float

    def __post_init__(self) -> None:
        """Validate and protect saved simulation data."""

        if not isinstance(self.final_state, SystemState):
            raise TypeError("final_state must be a SystemState instance")

        final_state = self.final_state.copy()

        final_state.masses.setflags(write=False)
        final_state.positions.setflags(write=False)
        final_state.velocities.setflags(write=False)

        object.__setattr__(self, "final_state", final_state)

        times = np.array(self.times, dtype=float, copy=True)
        positions = np.array(self.positions, dtype=float, copy=True)
        velocities = np.array(self.velocities, dtype=float, copy=True)
        energies = np.array(self.energies, dtype=float, copy=True)

        if times.ndim != 1 or len(times) == 0:
            raise ValueError("times must be a non-empty one-dimensional array")

        if energies.ndim != 1 or energies.shape != times.shape:
            raise ValueError("energies must have the same shape as times")

        if positions.ndim != 3:
            raise ValueError(
                "positions must have shape (snapshots, bodies, dimensions)"
            )

        if velocities.shape != positions.shape:
            raise ValueError("velocities must have the same shape as positions")

        expected_shape = (
            len(times),
            self.final_state.n_bodies,
            self.final_state.dimensions,
        )

        if positions.shape != expected_shape:
            raise ValueError(
                "snapshot dimensions must match times and final_state"
            )

        for name, array in (
            ("times", times),
            ("positions", positions),
            ("velocities", velocities),
            ("energies", energies),
        ):
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain only finite values")

            array.setflags(write=False)
            object.__setattr__(self, name, array)

        if (
            not np.isfinite(self.gravitational_constant)
            or self.gravitational_constant <= 0
        ):
            raise ValueError(
                "gravitational_constant must be finite and positive"
            )

        if not np.isfinite(self.softening) or self.softening < 0:
            raise ValueError("softening must be finite and non-negative")

        object.__setattr__(
            self, "gravitational_constant", float(self.gravitational_constant)
        )
        object.__setattr__(self, "softening", float(self.softening))


class Simulation:
    def __init__(
        self, initial_state: SystemState, gravitational_constant: float = 1.0,
        softening: float = 0.0,
    ) -> None:
        if not np.isfinite(gravitational_constant) or gravitational_constant <= 0:
            raise ValueError("gravitational_constant must be finite and positive")
        if not np.isfinite(softening) or softening < 0:
            raise ValueError("softening must be finite and non-negative")
        self.initial_state = initial_state.copy()
        self.gravitational_constant = float(gravitational_constant)
        self.softening = float(softening)

    def _acceleration(self, state: SystemState) -> np.ndarray:
        return accelerations(state, self.gravitational_constant, self.softening)

    def _advance_one_step(
        self,
        state: SystemState,
        dt: float,
    ) -> SystemState:
        """Advance a system state by one integration step."""

        return velocity_verlet(
            state,
            dt,
            self._acceleration,
        )

    def run(self, steps: int, dt: float, save_every: int = 1) -> SimulationResult:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        if isinstance(save_every, bool) or not isinstance(save_every, int) or save_every < 1:
            raise ValueError("save_every must be a positive integer")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")

        state = self.initial_state.copy()
        times, positions, velocities, energies = [], [], [], []

        def save() -> None:
            times.append(state.time)
            positions.append(state.positions.copy())
            velocities.append(state.velocities.copy())
            energies.append(total_energy(
                state, self.gravitational_constant, self.softening
            ))

        save()
        for index in range(1, steps + 1):
            state = self._advance_one_step(state, dt)
            if index % save_every == 0 or index == steps:
                save()

        return SimulationResult(
            times=np.asarray(times),
            positions=np.asarray(positions),
            velocities=np.asarray(velocities),
            energies=np.asarray(energies),
            final_state=state,
            gravitational_constant=self.gravitational_constant,
            softening=self.softening,
        )
    def run_configured(
        self, config: SimulationConfig
    ) -> SimulationResult:
        """Run the simulation using a predefined configuration."""

        if not isinstance(config, SimulationConfig):
            raise TypeError("config must be a SimulationConfig instance")

        return self.run(
            steps=config.steps,
            dt=config.dt,
            save_every=config.save_every,
        )