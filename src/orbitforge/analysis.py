
"""High-level physical analysis of N-body simulation results."""

from dataclasses import dataclass

import numpy as np

from .diagnostics import (
    angular_momentum,
    kinetic_energy,
    linear_momentum,
    potential_energy,
)
from .simulation import SimulationResult
from .state import SystemState


@dataclass(frozen=True)
class SimulationAnalysis:
    """Physical diagnostics evaluated at saved simulation snapshots."""

    times: np.ndarray
    kinetic_energies: np.ndarray
    potential_energies: np.ndarray
    total_energies: np.ndarray
    linear_momenta: np.ndarray
    angular_momenta: np.ndarray
    relative_energy_drift: np.ndarray


    def __post_init__(self) -> None:
        """Validate, copy and protect diagnostic arrays."""

        array_names = (
            "times",
            "kinetic_energies",
            "potential_energies",
            "total_energies",
            "linear_momenta",
            "angular_momenta",
            "relative_energy_drift",
        )

        for name in array_names:
            array = np.array(getattr(self, name), dtype=float, copy=True)

            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain only finite values")

            array.setflags(write=False)
            object.__setattr__(self, name, array)

        if self.times.ndim != 1 or len(self.times) == 0:
            raise ValueError("times must be a non-empty one-dimensional array")
        if np.any(np.diff(self.times) <= 0):
            raise ValueError("times must be strictly increasing")

        n_snapshots = len(self.times)

        if self.total_energies[0] == 0.0:
            raise ValueError(
                "relative energy drift is undefined when initial energy is zero"
            )

        scalar_series = (
            "kinetic_energies",
            "potential_energies",
            "total_energies",
            "relative_energy_drift",
        )

        for name in scalar_series:
            if getattr(self, name).shape != (n_snapshots,):
                raise ValueError(
                    f"{name} must have shape ({n_snapshots},)"
                )

        expected_total_energy = (
            self.kinetic_energies + self.potential_energies
        )

        if not np.allclose(
            self.total_energies,
            expected_total_energy,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(
                "total_energies must equal kinetic_energies "
                "+ potential_energies"
            )

        expected_energy_drift = (
            self.total_energies - self.total_energies[0]
        ) / abs(self.total_energies[0])

        if not np.allclose(
            self.relative_energy_drift,
            expected_energy_drift,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(
                "relative_energy_drift must match the drift "
                "calculated from total_energies"
            )

        if self.linear_momenta.ndim != 2:
            raise ValueError("linear_momenta must be a two-dimensional array")

        if self.linear_momenta.shape[0] != n_snapshots:
            raise ValueError(
                "linear_momenta must have one row per snapshot"
            )

        dimensions = self.linear_momenta.shape[1]

        if dimensions not in (2, 3):
            raise ValueError("spatial dimension must be 2 or 3")

        expected_angular_dimensions = 1 if dimensions == 2 else 3

        if self.angular_momenta.shape != (
            n_snapshots,
            expected_angular_dimensions,
        ):
            raise ValueError(
                "angular_momenta has incompatible dimensions"
            )


    @classmethod
    def from_result(
        cls,
        result: SimulationResult,
        gravitational_constant: float | None = None,
        softening: float | None = None,
    ) -> "SimulationAnalysis":
        """Calculate physical diagnostics from a simulation result."""

        if not isinstance(result, SimulationResult):
            raise TypeError("result must be a SimulationResult instance")
        if len(result.times) == 0:
            raise ValueError("result must contain at least one saved snapshot")


        if gravitational_constant is None:
            gravitational_constant = result.gravitational_constant

        if softening is None:
            softening = result.softening

        if not np.isfinite(gravitational_constant) or gravitational_constant <= 0:
            raise ValueError(
                "gravitational_constant must be finite and positive"
            )

        if not np.isfinite(softening) or softening < 0:
            raise ValueError("softening must be finite and non-negative")

        final_state = result.final_state
        masses = final_state.masses
        names = final_state.names

        kinetic = []
        potential = []
        momenta = []
        angular = []

        for index, time in enumerate(result.times):
            state = SystemState(
                names=names,
                masses=masses,
                positions=result.positions[index],
                velocities=result.velocities[index],
                time=float(time),
            )

            kinetic.append(kinetic_energy(state))
            potential.append(
                potential_energy(state, gravitational_constant, softening)
            )
            momenta.append(linear_momentum(state))
            angular.append(angular_momentum(state))

        kinetic = np.asarray(kinetic)
        potential = np.asarray(potential)
        total = kinetic + potential

        initial_energy = total[0]

        if initial_energy == 0.0:
            raise ValueError(
                "relative energy drift is undefined when initial energy is zero"
            )

        energy_drift = (total - initial_energy) / abs(initial_energy)

        return cls(
            times=result.times.copy(),
            kinetic_energies=kinetic,
            potential_energies=potential,
            total_energies=total,
            linear_momenta=np.asarray(momenta),
            angular_momenta=np.asarray(angular),
            relative_energy_drift=energy_drift,
        )
