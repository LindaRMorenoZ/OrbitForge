
"""Quantitative conservation metrics for N-body simulations."""

from dataclasses import dataclass

import numpy as np

from .analysis import SimulationAnalysis


@dataclass(frozen=True)
class ConservationMetrics:
    """Maximum conservation deviations measured during a simulation."""

    max_absolute_energy_drift: float
    max_relative_energy_drift: float
    max_absolute_linear_momentum_drift: float
    max_relative_linear_momentum_drift: float | None
    max_absolute_angular_momentum_drift: float
    max_relative_angular_momentum_drift: float | None

    @classmethod
    def from_analysis(
        cls,
        analysis: SimulationAnalysis,
    ) -> "ConservationMetrics":
        """Calculate conservation metrics from simulation diagnostics."""

        if not isinstance(analysis, SimulationAnalysis):
            raise TypeError(
                "analysis must be a SimulationAnalysis instance"
            )

        initial_energy = analysis.total_energies[0]

        if initial_energy == 0.0:
            raise ValueError(
                "relative energy drift is undefined when initial energy is zero"
            )

        energy_difference = (
            analysis.total_energies - initial_energy
        )

        max_absolute_energy_drift = float(
            np.max(np.abs(energy_difference))
        )

        max_relative_energy_drift = float(
            max_absolute_energy_drift / abs(initial_energy)
        )

        initial_linear = analysis.linear_momenta[0]
        linear_difference = analysis.linear_momenta - initial_linear

        max_absolute_linear = float(
            np.max(np.linalg.norm(linear_difference, axis=1))
        )

        initial_linear_norm = float(np.linalg.norm(initial_linear))

        max_relative_linear = (
            max_absolute_linear / initial_linear_norm
            if initial_linear_norm > 0.0
            else None
        )

        initial_angular = analysis.angular_momenta[0]
        angular_difference = analysis.angular_momenta - initial_angular

        max_absolute_angular = float(
            np.max(np.linalg.norm(angular_difference, axis=1))
        )

        initial_angular_norm = float(np.linalg.norm(initial_angular))

        max_relative_angular = (
            max_absolute_angular / initial_angular_norm
            if initial_angular_norm > 0.0
            else None
        )

        return cls(
            max_absolute_energy_drift=max_absolute_energy_drift,
            max_relative_energy_drift=max_relative_energy_drift,
            max_absolute_linear_momentum_drift=max_absolute_linear,
            max_relative_linear_momentum_drift=max_relative_linear,
            max_absolute_angular_momentum_drift=max_absolute_angular,
            max_relative_angular_momentum_drift=max_relative_angular,
        )
