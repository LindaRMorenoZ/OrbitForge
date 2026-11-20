
"""Tools for constructing custom N-body systems."""

from __future__ import annotations

import numpy as np

from ..state import Body, SystemState
from ..units import UnitSystem
from ..simulation import Simulation


class SystemBuilder:
    """Build a custom N-body system from individual bodies."""

    def __init__(self, dimensions: int = 2, unit_system: UnitSystem | None = None,
    ) -> None:           
        if dimensions not in (2, 3):
            raise ValueError("dimensions must be 2 or 3")
        if unit_system is not None and not isinstance(unit_system, UnitSystem):
            raise TypeError("unit_system must be a UnitSystem instance")

        self.dimensions = dimensions
        self.unit_system = unit_system
        self._bodies: list[Body] = []
        self._velocities: list[np.ndarray] = []

    @property
    def n_bodies(self) -> int:
        return len(self._bodies)

    @property
    def gravitational_constant(self) -> float:
        """Return G in the selected unit system, or 1.0 for arbitrary units."""
        if self.unit_system is None:
            return 1.0

        return self.unit_system.gravitational_constant

    @property
    def bodies(self) -> tuple[Body, ...]:
        return tuple(self._bodies)

    def add_body(
        self,
        name: str,
        mass: float,
        position: list[float] | tuple[float, ...],
        velocity: list[float] | tuple[float, ...],
    ) -> None:
        """Add a body with its initial position and velocity."""

        if any(body.name == name for body in self._bodies):
            raise ValueError(f"body name already exists: {name}")

        position_array = np.asarray(position, dtype=float)
        velocity_array = np.asarray(velocity, dtype=float)

        if position_array.shape != (self.dimensions,):
            raise ValueError(
                f"position must contain {self.dimensions} coordinates"
            )

        if velocity_array.shape != (self.dimensions,):
            raise ValueError(
                f"velocity must contain {self.dimensions} components"
            )

        body = Body(
            name=name,
            mass=mass,
            position=position_array,
        )

        if not np.all(np.isfinite(velocity_array)):
            raise ValueError("velocity must contain only finite values")

        self._bodies.append(body)
        self._velocities.append(velocity_array.copy())


    def update_body(
        self,
        name: str,
        new_name: str | None = None,
        mass: float | None = None,
        position: list[float] | tuple[float, ...] | None = None,
        velocity: list[float] | tuple[float, ...] | None = None,
    ) -> None:
        """Update the properties of an existing body."""

        for index, body in enumerate(self._bodies):
            if body.name == name:
                updated_name = body.name if new_name is None else new_name

                if any(
                    other.name == updated_name
                    for i, other in enumerate(self._bodies)
                    if i != index
                ):
                    raise ValueError(
                        f"body name already exists: {updated_name}"
                    )

                updated_mass = body.mass if mass is None else mass

                updated_position = (
                    body.position.copy()
                    if position is None
                    else np.asarray(position, dtype=float)
                )

                updated_velocity = (
                    self._velocities[index].copy()
                    if velocity is None
                    else np.asarray(velocity, dtype=float)
                )

                if updated_position.shape != (self.dimensions,):
                    raise ValueError(
                        f"position must contain {self.dimensions} coordinates"
                    )

                if updated_velocity.shape != (self.dimensions,):
                    raise ValueError(
                        f"velocity must contain {self.dimensions} components"
                    )

                if not np.all(np.isfinite(updated_velocity)):
                    raise ValueError(
                        "velocity must contain only finite values"
                    )

                updated_body = Body(
                    name=updated_name,
                    mass=updated_mass,
                    position=updated_position,
                )

                self._bodies[index] = updated_body
                self._velocities[index] = updated_velocity.copy()
                return

        raise ValueError(f"body not found: {name}")


    def remove_body(self, name: str) -> None:
        """Remove a body by name."""

        for index, body in enumerate(self._bodies):
            if body.name == name:
                del self._bodies[index]
                del self._velocities[index]
                return

        raise ValueError(f"body not found: {name}")

    def build(self, time: float = 0.0) -> SystemState:
        """Create a validated SystemState."""

        if not self._bodies:
            raise ValueError("cannot build an empty system")

        velocities = np.stack(self._velocities)

        return SystemState.from_bodies(
            bodies=self._bodies,
            velocities=velocities,
            time=time,
        )

    def create_simulation(
        self,
        time: float = 0.0,
        softening: float = 0.0,
    ) -> Simulation:
        """Create a simulation using the builder's units and initial conditions."""

        initial_state = self.build(time=time)

        return Simulation(
            initial_state=initial_state,
            gravitational_constant=self.gravitational_constant,
            softening=softening,
     )