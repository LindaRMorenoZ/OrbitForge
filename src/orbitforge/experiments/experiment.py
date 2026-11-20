"""Computational experiment definition and management."""

import math

from orbitforge.analysis import SimulationAnalysis
from orbitforge.builder import SystemBuilder
from orbitforge.config import SimulationConfig
from orbitforge.controller import SimulationController
from orbitforge.metrics import ConservationMetrics
from orbitforge.simulation import Simulation, SimulationResult

class Experiment:
    """A reproducible computational experiment."""

    def __init__(
        self,
        name: str,
        builder: SystemBuilder,
        config: SimulationConfig,
        softening: float = 0.0,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Experiment name must be a non-empty string.")

        if not isinstance(builder, SystemBuilder):
            raise TypeError("builder must be a SystemBuilder.")

        if not isinstance(config, SimulationConfig):
            raise TypeError("config must be a SimulationConfig.")

        self._name = name.strip()
        self._config = config
        self._gravitational_constant = builder.gravitational_constant

        self._softening = float(softening)
        if not math.isfinite(self._softening) or self._softening < 0:
            raise ValueError("softening must be finite and non-negative")

        self._initial_state = builder.build().copy()
        self._result: SimulationResult | None = None
        self._analysis: SimulationAnalysis | None = None
        self._metrics: ConservationMetrics | None = None

    @property
    def analysis(self) -> SimulationAnalysis | None:
        return self._analysis

    @property
    def metrics(self) -> ConservationMetrics | None:
        return self._metrics

    @property
    def name(self) -> str:
        return self._name

    @property
    def config(self) -> SimulationConfig:
        return self._config

    @property
    def initial_state(self):
        return self._initial_state.copy()

    @property
    def result(self) -> SimulationResult | None:
        return self._result

    def run(self) -> SimulationResult:
        """Run the experiment and store its simulation result."""
        self._analysis = None
        self._metrics = None

        simulation = Simulation(
            self._initial_state,
            gravitational_constant=self._gravitational_constant,
            softening=self._softening,
        )
        self._result = simulation.run_configured(self._config)

        return self._result

    def analyze(self) -> SimulationAnalysis:
        """Analyze the results of the experiment."""
        if self._result is None:
            raise RuntimeError(
                "The experiment must be run before it can be analyzed."
            )
        self._analysis = SimulationAnalysis.from_result(
            self._result,
        )

        return self._analysis

    def create_controller(self) -> SimulationController:
        """Create an interactive controller for this experiment."""
        simulation = Simulation(
            self._initial_state,
            gravitational_constant=self._gravitational_constant,
            softening=self._softening,
        )

        return SimulationController(simulation, self._config)
    def evaluate(self) -> ConservationMetrics:
        """Evaluate the conservation properties of the experiment."""
        if self._analysis is None:
            raise RuntimeError(
                "The experiment must be analyzed before it can be evaluated."
            )
        self._metrics = ConservationMetrics.from_analysis(
            self._analysis,
        )

        return self._metrics
