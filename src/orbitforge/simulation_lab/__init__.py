"""Interactive simulation lab."""

from orbitforge.recommendations import (
    ParameterRecommendation,
    SimulationRecommendations,
    SimulationRecommender,
)

from .comparison import ExperimentComparison
from .configuration import apply_recommendations
from .manager import ExperimentManager
from .session import SimulationSession

__all__ = [
    "ExperimentComparison",
    "ExperimentManager",
    "ParameterRecommendation",
    "SimulationRecommendations",
    "SimulationRecommender",
    "SimulationSession",
    "apply_recommendations",
    "visualize_interactive_session",
]


def visualize_interactive_session(
    session,
    *,
    interval: int = 30,
    steps_per_frame: int = 10,
):
    """Open an interactive viewer for an existing simulation session."""
    from .visualization import visualize_interactive_session as visualize

    return visualize(
        session,
        interval=interval,
        steps_per_frame=steps_per_frame,
    )
