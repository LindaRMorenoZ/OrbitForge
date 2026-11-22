"""Scientific recommendations for N-body simulation parameters."""

import math
from dataclasses import dataclass

import numpy as np

from orbitforge.state import SystemState


@dataclass(frozen=True)
class ParameterRecommendation:
    """A suggested simulation parameter with its scientific justification."""

    value: float | int | str
    explanation: str


@dataclass(frozen=True)
class SimulationRecommendations:
    """Collection of recommended parameters and diagnostic warnings."""

    dt: ParameterRecommendation
    total_time: ParameterRecommendation
    save_every: ParameterRecommendation
    integrator: ParameterRecommendation
    warnings: tuple[str, ...] = ()


def _pairwise_dynamical_timescales(
    state: SystemState,
    gravitational_constant: float,
) -> list[float]:
    """Estimate characteristic gravitational timescales for all body pairs."""

    timescales = []

    for i in range(state.n_bodies - 1):
        for j in range(i + 1, state.n_bodies):
            displacement = state.positions[j] - state.positions[i]
            relative_velocity = state.velocities[j] - state.velocities[i]

            distance = float(np.hypot.reduce(displacement))
            total_mass = float(state.masses[i]) + float(state.masses[j])

            if not math.isfinite(total_mass):
                raise ValueError("Total pair mass must be finite.")

            if distance == 0.0:
                raise ValueError("Coincident bodies prevent a reliable timestep recommendation.")

            gravitational_timescale = (
                math.sqrt(distance / (gravitational_constant * total_mass)) * distance
            )

            if not math.isfinite(gravitational_timescale) or gravitational_timescale <= 0.0:
                raise ValueError("Could not calculate a finite positive gravitational timescale.")

            relative_speed = float(np.hypot.reduce(relative_velocity))

            if not math.isfinite(relative_speed):
                raise ValueError("Relative speed must be finite.")

            if relative_speed > 0.0:
                encounter_timescale = distance / relative_speed
                if not math.isfinite(encounter_timescale) or encounter_timescale <= 0.0:
                    raise ValueError("Encounter timescale must be finite and positive.")

                timescales.append(min(gravitational_timescale, encounter_timescale))
            else:
                timescales.append(gravitational_timescale)

    return timescales


class SimulationRecommender:
    """Recommend initial numerical parameters for N-body simulations."""

    def __init__(
        self,
        safety_factor: float = 0.02,
        duration_factor: float = 10.0,
        target_saved_states: int = 1000,
    ) -> None:

        if (
            isinstance(safety_factor, bool)
            or not isinstance(safety_factor, (int, float))
            or not math.isfinite(safety_factor)
            or not 0.0 < safety_factor < 1.0
        ):
            raise ValueError("safety_factor must be finite and between 0 and 1.")

        if (
            isinstance(duration_factor, bool)
            or not isinstance(duration_factor, (int, float))
            or not math.isfinite(duration_factor)
            or duration_factor <= 0.0
        ):
            raise ValueError("duration_factor must be finite and positive.")

        if (
            isinstance(target_saved_states, bool)
            or not isinstance(target_saved_states, int)
            or target_saved_states < 2
        ):
            raise ValueError("target_saved_states must be an integer greater than 1.")

        self.safety_factor = float(safety_factor)
        self.duration_factor = float(duration_factor)
        self.target_saved_states = target_saved_states

    def recommend_dt(
        self,
        state: SystemState,
        gravitational_constant: float,
    ) -> ParameterRecommendation:
        """Recommend an initial timestep from pairwise timescales."""

        if not isinstance(state, SystemState):
            raise TypeError("state must be a SystemState.")
        state.validate()

        if (
            isinstance(gravitational_constant, bool)
            or not isinstance(gravitational_constant, (int, float))
            or not math.isfinite(gravitational_constant)
            or gravitational_constant <= 0.0
        ):
            raise ValueError("gravitational_constant must be finite and positive.")

        timescales = _pairwise_dynamical_timescales(
            state,
            float(gravitational_constant),
        )

        if not timescales:
            raise ValueError("A timestep cannot be recommended without a valid body pair.")

        minimum_timescale = min(timescales)
        recommended_dt = self.safety_factor * minimum_timescale

        return ParameterRecommendation(
            value=recommended_dt,
            explanation=(
                f"Based on the minimum estimated pairwise timescale "
                f"({minimum_timescale:.6g}), using a safety factor of "
                f"{self.safety_factor:g}. This is an initial estimate; "
                "verify accuracy through timestep convergence tests."
            ),
        )

    def recommend_total_time(
        self,
        state: SystemState,
        gravitational_constant: float,
    ) -> ParameterRecommendation:
        """Recommend an initial simulation duration from pairwise timescales."""

        if not isinstance(state, SystemState):
            raise TypeError("state must be a SystemState.")

        state.validate()

        if (
            isinstance(gravitational_constant, bool)
            or not isinstance(gravitational_constant, (int, float))
            or not math.isfinite(gravitational_constant)
            or gravitational_constant <= 0.0
        ):
            raise ValueError("gravitational_constant must be finite and positive.")

        timescales = _pairwise_dynamical_timescales(
            state,
            float(gravitational_constant),
        )

        if not timescales:
            raise ValueError(
                "A simulation duration cannot be recommended without a valid body pair."
            )

        maximum_timescale = max(timescales)
        raw_total_time = self.duration_factor * maximum_timescale

        recommended_dt = self.recommend_dt(
            state,
            gravitational_constant,
        ).value

        number_of_steps = math.ceil(raw_total_time / recommended_dt)
        recommended_total_time = number_of_steps * recommended_dt

        return ParameterRecommendation(
            value=recommended_total_time,
            explanation=(
                f"Based on the maximum estimated pairwise timescale "
                f"({maximum_timescale:.6g}), using a duration factor of "
                f"{self.duration_factor:g}. This is an initial heuristic; "
                "adjust the duration to the physical phenomenon being studied."
            ),
        )

    def recommend_save_every(self, total_steps: int) -> ParameterRecommendation:
        """Recommend a storage interval targeting a bounded number of saved states."""

        if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps < 1:
            raise ValueError("total_steps must be a positive integer.")

        save_every = max(
            1,
            math.ceil(total_steps / (self.target_saved_states - 1)),
        )

        estimated_saved_states = math.ceil(total_steps / save_every) + 1

        return ParameterRecommendation(
            value=save_every,
            explanation=(
                f"Stores approximately {estimated_saved_states} states "
                f"for {total_steps} integration steps, targeting "
                f"{self.target_saved_states} saved states. "
                "This controls storage frequency, not integration accuracy."
            ),
        )

    def recommend_integrator(self) -> ParameterRecommendation:
        """Recommend the available numerical integration method."""

        return ParameterRecommendation(
            value="velocity_verlet",
            explanation=(
                "Velocity-Verlet is the currently implemented integrator. "
                "It is a second-order symplectic method suitable for "
                "many gravitational N-body simulations. Its accuracy "
                "must still be verified through convergence tests."
            ),
        )

    def recommend(
        self,
        state: SystemState,
        gravitational_constant: float,
    ) -> SimulationRecommendations:
        """Generate a complete set of initial simulation recommendations."""

        dt = self.recommend_dt(state, gravitational_constant)
        total_time = self.recommend_total_time(state, gravitational_constant)

        total_steps = round(total_time.value / dt.value)
        save_every = self.recommend_save_every(total_steps)
        integrator = self.recommend_integrator()

        warnings = (
            "All recommendations are initial heuristics, not guarantees of numerical accuracy.",
            "Validate the timestep through convergence tests.",
            "Review the simulation duration according to the physical phenomenon being studied.",
        )

        return SimulationRecommendations(
            dt=dt,
            total_time=total_time,
            save_every=save_every,
            integrator=integrator,
            warnings=warnings,
        )
