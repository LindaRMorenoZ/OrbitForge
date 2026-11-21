"""Controller for interactive N-body simulations."""
from enum import Enum, auto

import numpy as np

from ..config import SimulationConfig
from ..diagnostics import total_energy
from ..simulation import Simulation, SimulationResult
from ..state import SystemState


class ControllerStatus(Enum):
    """Possible states of a simulation controller."""

    READY = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    STOPPED = auto()


class SimulationController:
    """Controls the step-by-step execution of a simulation."""

    def __init__(
        self,
        simulation: Simulation,
        config: SimulationConfig,
    ) -> None:
        if not isinstance(simulation, Simulation):
            raise TypeError("simulation must be a Simulation instance")

        if not isinstance(config, SimulationConfig):
            raise TypeError("config must be a SimulationConfig instance")

        self._simulation = simulation
        self._config = config
        self.current_state: SystemState = simulation.initial_state.copy()

        self.status = ControllerStatus.READY
        self.current_step = 0
        self._result: SimulationResult | None = None
        self._times = [self.current_state.time]
        self._positions = [self.current_state.positions.copy()]
        self._velocities = [self.current_state.velocities.copy()]
        self._energies = [
            total_energy(
                self.current_state,
                simulation.gravitational_constant,
                simulation.softening,
            )
        ]
    @property
    def simulation(self) -> Simulation:
        """Return the simulation engine."""

        return self._simulation

    @property
    def config(self) -> SimulationConfig:
        """Return the immutable simulation configuration."""

        return self._config
    @property
    def progress(self) -> float:
        """Return simulation progress between 0 and 1."""

        if self.config.steps == 0:
            return 1.0

        return self.current_step / self.config.steps

    @property
    def result(self) -> SimulationResult | None:
        """Return the result when the simulation is completed."""

        return self._result

    def start(self) -> None:
        """Start the simulation."""

        if self.status != ControllerStatus.READY:
            raise RuntimeError("simulation can only start from READY")

        self.status = ControllerStatus.RUNNING

    def step(self) -> SystemState:
        """Advance the simulation by one time step."""

        if self.status != ControllerStatus.RUNNING:
            raise RuntimeError("simulation must be RUNNING to advance")

        if self.current_step >= self.config.steps:
            self.status = ControllerStatus.COMPLETED
            return self.current_state.copy()

        self.current_state = self.simulation._advance_one_step(
            self.current_state,
            self.config.dt,
        )

        self.current_step += 1

        if (
            self.current_step % self.config.save_every == 0
            or self.current_step == self.config.steps
        ):
            self._save_snapshot()

        if self.current_step >= self.config.steps:
            self.status = ControllerStatus.COMPLETED
            self._build_result()

        return self.current_state.copy()

    def pause(self) -> None:
        """Pause a running simulation."""

        if self.status != ControllerStatus.RUNNING:
            raise RuntimeError("only a RUNNING simulation can be paused")

        self.status = ControllerStatus.PAUSED

    def resume(self) -> None:
        """Resume a paused simulation."""

        if self.status != ControllerStatus.PAUSED:
            raise RuntimeError("only a PAUSED simulation can be resumed")

        self.status = ControllerStatus.RUNNING

    def stop(self) -> None:
        """Stop the simulation and preserve its partial results."""

        if self.status not in (
            ControllerStatus.RUNNING,
            ControllerStatus.PAUSED,
        ):
            raise RuntimeError("only a RUNNING or PAUSED simulation can stop")

        if self._times[-1] != self.current_state.time:
            self._save_snapshot()
        
        self.status = ControllerStatus.STOPPED
        self._build_result()

    def reset(self) -> None:
        """Reset the controller to its initial simulation state."""

        if self.status == ControllerStatus.RUNNING:
            raise RuntimeError("cannot reset a RUNNING simulation")

        self.current_state = self.simulation.initial_state.copy()
        self.current_step = 0
        self.status = ControllerStatus.READY
        self._result = None

        self._times = [self.current_state.time]
        self._positions = [self.current_state.positions.copy()]
        self._velocities = [self.current_state.velocities.copy()]

        self._energies = [
            total_energy(
                self.current_state,
                self.simulation.gravitational_constant,
                self.simulation.softening,
            )
        ]

    def _save_snapshot(self) -> None:
        """Save the current simulation state."""

        self._times.append(self.current_state.time)
        self._positions.append(self.current_state.positions.copy())
        self._velocities.append(self.current_state.velocities.copy())

        self._energies.append(
            total_energy(
                self.current_state,
                self.simulation.gravitational_constant,
                self.simulation.softening,
            )
        )

    def _build_result(self) -> None:
        """Build the final simulation result."""

        self._result = SimulationResult(
            times=np.asarray(self._times),
            positions=np.asarray(self._positions),
            velocities=np.asarray(self._velocities),
            energies=np.asarray(self._energies),
            final_state=self.current_state.copy(),
            gravitational_constant=self.simulation.gravitational_constant,
            softening=self.simulation.softening,
        )
