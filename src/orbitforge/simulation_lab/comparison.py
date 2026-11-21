"""Pairwise comparison of conservation metrics across experiments."""

from itertools import combinations

from orbitforge.simulation_lab.manager import ExperimentManager


class ExperimentComparison:
    """Compare dimensionless conservation metrics between experiments."""

    def __init__(self, manager: ExperimentManager):
        if not isinstance(manager, ExperimentManager):
            raise TypeError("manager must be an ExperimentManager instance")

        self._manager = manager

    def compare(self) -> dict:
        """Return pairwise differences in relative conservation metrics."""

        sessions = self._manager.sessions

        if len(sessions) < 2:
            raise ValueError("At least two sessions are required for comparison")

        for name, session in sessions.items():
            if session.metrics is None:
                raise ValueError(
                    f"Session '{name}' must be evaluated before comparison"
                )

        comparisons = {}

        metric_names = (
            "max_relative_energy_drift",
            "max_relative_linear_momentum_drift",
            "max_relative_angular_momentum_drift",
        )

        for (name_a, session_a), (name_b, session_b) in combinations(
            sessions.items(), 2
        ):
            metrics_a = session_a.metrics
            metrics_b = session_b.metrics

            differences = {}

            for metric_name in metric_names:
                value_a = getattr(metrics_a, metric_name)
                value_b = getattr(metrics_b, metric_name)

                differences[metric_name] = (
                    None
                    if value_a is None or value_b is None
                    else value_a - value_b
                )

            comparisons[(name_a, name_b)] = differences

        return comparisons
        return comparisons
