"""Configuration settings for N-body simulations."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SimulationConfig:
    """Defines the numerical parameters of a simulation."""

    total_time: float
    dt: float
    save_every: int = 1
    integrator: str = "velocity_verlet"

    def __post_init__(self) -> None:
        """Validate simulation parameters."""

        if not math.isfinite(self.total_time) or self.total_time <= 0:
            raise ValueError("total_time must be finite and positive")

        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be finite and positive")

        if (
            isinstance(self.save_every, bool)
            or not isinstance(self.save_every, int)
            or self.save_every < 1
        ):
            raise ValueError("save_every must be a positive integer")

        if self.integrator != "velocity_verlet":
            raise ValueError("Unsupported integrator")

    @property
    def steps(self) -> int:
        """Return the number of simulation steps."""

        ratio = self.total_time / self.dt

        if not math.isclose(ratio, round(ratio), rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(
                "total_time must be an integer multiple of dt"
            )

        return round(ratio)