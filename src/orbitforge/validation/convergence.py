
"""Numerical convergence and conservation diagnostics."""

from dataclasses import dataclass

import numpy as np

from ..diagnostics import angular_momentum, total_energy
from ..simulation import Simulation
from ..state import SystemState


@dataclass
class ConvergenceResult:
    """Results from simulations performed with different time steps."""

    dt: np.ndarray
    position_error: np.ndarray
    energy_drift: np.ndarray
    angular_momentum_drift: np.ndarray
    observed_order: np.ndarray


def _relative_vector_error(final: np.ndarray, reference: np.ndarray) -> float:
    """Return the relative Euclidean error between two arrays."""
    denominator = np.linalg.norm(reference)

    if denominator == 0:
        return float(np.linalg.norm(final - reference))

    return float(np.linalg.norm(final - reference) / denominator)


def run_convergence_test(
    initial_state: SystemState,
    duration: float,
    time_steps: list[float],
    gravitational_constant: float = 1.0,
    softening: float = 0.0,
    reference_factor: int = 16,
) -> ConvergenceResult:
    """Compare numerical solutions using progressively smaller time steps.

    All simulations cover the same duration. The reference solution uses
    a time step smaller than the smallest tested step.
    """
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")

    if not time_steps:
        raise ValueError("time_steps cannot be empty")

    if any(not np.isfinite(dt) or dt <= 0 for dt in time_steps):
        raise ValueError("all time steps must be finite and positive")

    if len(set(time_steps)) != len(time_steps):
        raise ValueError("time_steps must be unique")

    if isinstance(reference_factor, bool) or not isinstance(reference_factor, int):
        raise ValueError("reference_factor must be an integer")

    if reference_factor < 2:
        raise ValueError("reference_factor must be at least 2")

    ordered_steps = np.array(sorted(time_steps, reverse=True), dtype=float)

    for dt in ordered_steps:
        steps = round(duration / dt)
        if steps < 1 or not np.isclose(steps * dt, duration):
            raise ValueError(
                f"duration ({duration}) must be an integer multiple of dt ({dt})"
            )

    reference_dt = ordered_steps[-1] / reference_factor
    reference_steps = round(duration / reference_dt)

    reference = Simulation(
        initial_state,
        gravitational_constant=gravitational_constant,
        softening=softening,
    ).run(reference_steps, reference_dt)

    reference_positions = reference.final_state.positions

    initial_energy = total_energy(
        initial_state, gravitational_constant, softening
    )
    initial_angular_momentum = angular_momentum(initial_state)

    position_errors = []
    energy_drifts = []
    angular_momentum_drifts = []

    for dt in ordered_steps:
        steps = round(duration / dt)

        result = Simulation(
            initial_state,
            gravitational_constant=gravitational_constant,
            softening=softening,
        ).run(steps, dt)

        position_errors.append(
            _relative_vector_error(
                result.final_state.positions,
                reference_positions,
            )
        )

        energy_drifts.append(
            float(
                np.max(
                    np.abs(result.energies - initial_energy)
                ) / abs(initial_energy)
            )
        )

        final_angular_momentum = angular_momentum(result.final_state)

        angular_momentum_drifts.append(
            _relative_vector_error(
                final_angular_momentum,
                initial_angular_momentum,
            )
        )

    position_errors = np.asarray(position_errors)
    energy_drifts = np.asarray(energy_drifts)
    angular_momentum_drifts = np.asarray(angular_momentum_drifts)

    observed_order = np.full(len(ordered_steps), np.nan)

    for index in range(1, len(ordered_steps)):
        coarse_error = position_errors[index - 1]
        fine_error = position_errors[index]

        if coarse_error > 0 and fine_error > 0 and coarse_error != fine_error:
            observed_order[index] = (
                np.log(coarse_error / fine_error)
                / np.log(ordered_steps[index - 1] / ordered_steps[index])
            )

    return ConvergenceResult(
        dt=ordered_steps,
        position_error=position_errors,
        energy_drift=energy_drifts,
        angular_momentum_drift=angular_momentum_drifts,
        observed_order=observed_order,
    )
