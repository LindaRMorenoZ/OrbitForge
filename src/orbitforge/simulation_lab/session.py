"""Interactive simulation session management."""

from orbitforge.analysis import SimulationAnalysis
from orbitforge.experiments import Experiment
from orbitforge.metrics import ConservationMetrics


class SimulationSession:
    """Manage the lifecycle and analysis of an interactive experiment."""

    def __init__(self, experiment: Experiment):
        if not isinstance(experiment, Experiment):
            raise TypeError("experiment must be an Experiment instance")

        self._experiment = experiment
        self._controller = experiment.create_controller()
        self._analysis = None
        self._metrics = None

    @property
    def controller(self):
        return self._controller

    @property
    def status(self):
        return self._controller.status

    @property
    def progress(self) -> float:
        """Return simulation progress between 0 and 1."""

        return self._controller.progress

    @property
    def is_completed(self) -> bool:
        """Return whether the simulation has completed."""

        return self.status.name == "COMPLETED"

    @property
    def current_state(self):
        return self._controller.current_state

    @property
    def result(self):
        return self._controller.result

    @property
    def analysis(self):
        return self._analysis

    @property
    def metrics(self):
        return self._metrics

    def start(self):
        self._controller.start()

    def step(self):
        self._controller.step()

    def pause(self):
        self._controller.pause()

    def resume(self):
        self._controller.resume()

    def stop(self):
        self._controller.stop()

    def reset(self):
        if self.status.name in ("RUNNING", "PAUSED"):
            self.stop()

        self._controller.reset()
        self._analysis = None
        self._metrics = None

    def analyze(self):
        if self.result is None:
            raise RuntimeError("A simulation result is required before analysis")

        self._analysis = SimulationAnalysis.from_result(self.result)
        self._metrics = None
        return self._analysis

    def evaluate(self):
        if self._analysis is None:
            raise RuntimeError("Analysis is required before evaluation")

        self._metrics = ConservationMetrics.from_analysis(self._analysis)
        return self._metrics
