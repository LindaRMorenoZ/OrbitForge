"""Management of interactive experiment sessions."""

from orbitforge.experiments import Experiment
from orbitforge.simulation_lab.session import SimulationSession


class ExperimentManager:
    """Create and manage multiple simulation sessions."""

    def __init__(self):
        self._sessions = {}

    @property
    def sessions(self):
        """Return a copy of the registered sessions."""

        return self._sessions.copy()

    def create_session(self, experiment: Experiment) -> SimulationSession:
        """Create and register a session for an experiment."""

        if not isinstance(experiment, Experiment):
            raise TypeError("experiment must be an Experiment instance")

        name = experiment.name

        if name in self._sessions:
            raise ValueError(f"Session '{name}' already exists")

        session = SimulationSession(experiment)
        self._sessions[name] = session

        return session

    def get_session(self, name: str) -> SimulationSession:
        """Retrieve a session by experiment name."""

        if not isinstance(name, str) or not name.strip():
            raise ValueError("session name must be a non-empty string")

        if name not in self._sessions:
            raise KeyError(f"Session '{name}' does not exist")

        return self._sessions[name]

    def remove_session(self, name: str) -> SimulationSession:
        """Remove and return a registered session."""

        if not isinstance(name, str) or not name.strip():
            raise ValueError("session name must be a non-empty string")

        if name not in self._sessions:
            raise KeyError(f"Session '{name}' does not exist")

        return self._sessions.pop(name)