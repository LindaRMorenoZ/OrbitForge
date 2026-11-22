
"""Configuration utilities for the interactive simulation lab."""

from orbitforge.experiments import ExperimentConfiguration
from orbitforge.recommendations import SimulationRecommendations


def apply_recommendations(
    configuration: ExperimentConfiguration,
    recommendations: SimulationRecommendations,
    *,
    parameters: tuple[str, ...] | None = None,
) -> ExperimentConfiguration:
    """Return a new configuration with selected recommendations applied.

    By default, all recommended simulation parameters are applied.
    The original configuration remains unchanged.
    """

    if not isinstance(configuration, ExperimentConfiguration):
        raise TypeError("configuration must be an ExperimentConfiguration instance")

    if not isinstance(recommendations, SimulationRecommendations):
        raise TypeError("recommendations must be a SimulationRecommendations instance")

    available_parameters = ("dt", "total_time", "save_every", "integrator")

    if parameters is None:
        parameters = available_parameters

    if not isinstance(parameters, tuple):
        raise TypeError("parameters must be a tuple of parameter names")

    if any(parameter not in available_parameters for parameter in parameters):
        raise ValueError("parameters contains an unsupported recommendation")

    if len(parameters) != len(set(parameters)):
        raise ValueError("parameters must not contain duplicates")

    updates = {
        parameter: getattr(recommendations, parameter).value
        for parameter in parameters
    }

    return configuration.with_updates(**updates)
