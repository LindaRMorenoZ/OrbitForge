"""Reusable configuration for computational N-body experiments."""

import json
import math
from pathlib import Path

from orbitforge.builder import SystemBuilder
from orbitforge.config import SimulationConfig
from orbitforge.state import SystemState
from orbitforge.units import UnitSystem

from .experiment import Experiment


class ExperimentConfiguration:
    """A reusable and independent definition of an N-body experiment."""

    def __init__(
        self,
        name: str,
        initial_state: SystemState,
        config: SimulationConfig,
        gravitational_constant: float,
        softening: float = 0.0,
        description: str = "",
        unit_system: UnitSystem | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Configuration name must be a non-empty string.")

        if not isinstance(initial_state, SystemState):
            raise TypeError("initial_state must be a SystemState.")

        if not isinstance(config, SimulationConfig):
            raise TypeError("config must be a SimulationConfig.")

        if not math.isfinite(gravitational_constant) or gravitational_constant <= 0:
            raise ValueError("gravitational_constant must be finite and positive.")

        if not math.isfinite(softening) or softening < 0:
            raise ValueError("softening must be finite and non-negative.")

        if unit_system is not None and not isinstance(unit_system, UnitSystem):
            raise TypeError("unit_system must be a UnitSystem or None.")

        if not isinstance(description, str):
            raise TypeError("description must be a string.")

        self._name = name.strip()
        self._description = description.strip()
        self._initial_state = initial_state.copy()
        self._config = config
        self._gravitational_constant = float(gravitational_constant)
        self._softening = float(softening)
        self._unit_system = unit_system

    @classmethod
    def from_builder(
        cls,
        name: str,
        builder: SystemBuilder,
        config: SimulationConfig,
        softening: float = 0.0,
        description: str = "",
    ) -> "ExperimentConfiguration":
        """Create a configuration from a system builder."""

        if not isinstance(builder, SystemBuilder):
            raise TypeError("builder must be a SystemBuilder.")

        return cls(
            name=name,
            initial_state=builder.build(),
            config=config,
            gravitational_constant=builder.gravitational_constant,
            softening=softening,
            description=description,
            unit_system=builder.unit_system,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def initial_state(self) -> SystemState:
        return self._initial_state.copy()

    @property
    def config(self) -> SimulationConfig:
        return self._config

    @property
    def gravitational_constant(self) -> float:
        return self._gravitational_constant

    @property
    def softening(self) -> float:
        return self._softening

    @property
    def unit_system(self) -> UnitSystem | None:
        return self._unit_system

    def create_experiment(self, name: str | None = None) -> Experiment:
        """Create an independent experiment from this configuration."""

        return Experiment.from_configuration(
            self,
            name=name,
        )

    def with_updates(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        initial_state: SystemState | None = None,
        total_time: float | None = None,
        dt: float | None = None,
        save_every: int | None = None,
        integrator: str | None = None,
        gravitational_constant: float | None = None,
        softening: float | None = None,
        unit_system: UnitSystem | None = None,
    ) -> "ExperimentConfiguration":
        """Create a new configuration with selected parameters updated."""

        updated_config = SimulationConfig(
            total_time=(self.config.total_time if total_time is None else total_time),
            dt=self.config.dt if dt is None else dt,
            save_every=(self.config.save_every if save_every is None else save_every),
            integrator=(self.config.integrator if integrator is None else integrator),
        )

        return ExperimentConfiguration(
            name=self.name if name is None else name,
            description=(self.description if description is None else description),
            initial_state=(self.initial_state if initial_state is None else initial_state),
            config=updated_config,
            gravitational_constant=(
                self.gravitational_constant
                if gravitational_constant is None
                else gravitational_constant
            ),
            softening=(self.softening if softening is None else softening),
            unit_system=(self.unit_system if unit_system is None else unit_system),
        )

    def to_dict(self) -> dict:
        """Convert the configuration into a JSON-compatible dictionary."""

        unit_system = None

        if self.unit_system is not None:
            unit_system = {
                "name": self.unit_system.name,
                "length_in_meters": self.unit_system.length_in_meters,
                "mass_in_kg": self.unit_system.mass_in_kg,
                "time_in_seconds": self.unit_system.time_in_seconds,
                "length_symbol": self.unit_system.length_symbol,
                "mass_symbol": self.unit_system.mass_symbol,
                "time_symbol": self.unit_system.time_symbol,
            }

        return {
            "format_version": 1,
            "name": self.name,
            "description": self.description,
            "initial_state": {
                "names": list(self._initial_state.names),
                "masses": self._initial_state.masses.tolist(),
                "positions": self._initial_state.positions.tolist(),
                "velocities": self._initial_state.velocities.tolist(),
                "time": self._initial_state.time,
            },
            "simulation": {
                "total_time": self.config.total_time,
                "dt": self.config.dt,
                "save_every": self.config.save_every,
                "integrator": self.config.integrator,
            },
            "gravitational_constant": self.gravitational_constant,
            "softening": self.softening,
            "unit_system": unit_system,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfiguration":
        """Create a configuration from a validated dictionary."""

        if not isinstance(data, dict):
            raise TypeError("Configuration data must be a dictionary.")

        version = data.get("format_version")

        if isinstance(version, bool) or version != 1:
            raise ValueError("Unsupported configuration format version.")

        required_fields = {
            "name",
            "description",
            "initial_state",
            "simulation",
            "gravitational_constant",
            "softening",
            "unit_system",
        }

        missing_fields = required_fields - data.keys()

        if missing_fields:
            raise ValueError(f"Missing configuration fields: {sorted(missing_fields)}")

        state_data = data["initial_state"]
        simulation_data = data["simulation"]
        unit_data = data["unit_system"]

        if not isinstance(state_data, dict):
            raise TypeError("initial_state must be a dictionary.")

        if not isinstance(simulation_data, dict):
            raise TypeError("simulation must be a dictionary.")

        state_fields = {
            "names",
            "masses",
            "positions",
            "velocities",
            "time",
        }

        simulation_fields = {
            "total_time",
            "dt",
            "save_every",
            "integrator",
        }

        missing_state = state_fields - state_data.keys()
        missing_simulation = simulation_fields - simulation_data.keys()

        if missing_state:
            raise ValueError(f"Missing initial_state fields: {sorted(missing_state)}")

        if missing_simulation:
            raise ValueError(f"Missing simulation fields: {sorted(missing_simulation)}")

        if unit_data is not None and not isinstance(unit_data, dict):
            raise TypeError("unit_system must be a dictionary or None.")

        initial_state = SystemState(
            names=tuple(state_data["names"]),
            masses=state_data["masses"],
            positions=state_data["positions"],
            velocities=state_data["velocities"],
            time=state_data["time"],
        )

        config = SimulationConfig(
            total_time=simulation_data["total_time"],
            dt=simulation_data["dt"],
            save_every=simulation_data["save_every"],
            integrator=simulation_data["integrator"],
        )

        unit_system = UnitSystem(**unit_data) if unit_data is not None else None

        return cls(
            name=data["name"],
            description=data["description"],
            initial_state=initial_state,
            config=config,
            gravitational_constant=data["gravitational_constant"],
            softening=data["softening"],
            unit_system=unit_system,
        )

    def save(self, path: str | Path) -> None:
        """Save the configuration to a JSON file."""

        file_path = Path(path)

        with file_path.open("w", encoding="utf-8") as file:
            json.dump(
                self.to_dict(),
                file,
                indent=4,
                allow_nan=False,
            )

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfiguration":
        """Load a configuration from a JSON file."""

        file_path = Path(path)

        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        return cls.from_dict(data)
