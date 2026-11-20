
"""Fixed Solar System reference scenario.

Initial conditions are based on approximate JPL Keplerian elements
at J2000.0. Units: AU, years, and solar masses.

This is a Newtonian educational model, not a precision ephemeris.
"""

from __future__ import annotations

import numpy as np

from ..state import SystemState


G_AU = 4.0 * np.pi**2
SOLAR_MASS_KG = 1.98847e30

# name, mass (10^24 kg), a (AU), e, inclination (deg),
# mean longitude (deg), longitude of perihelion (deg),
# ascending node (deg), J2000 mean-anomaly correction (deg)
#
# Orbital elements: JPL Table 2a and 2b.
# Earth uses the Earth-Moon barycenter approximation.

_PLANETS = (
    ("Mercury", 0.330103, 0.38709843, 0.20563661, 7.00559432,
     252.25166724, 77.45771895, 48.33961819, 0.0),

    ("Venus", 4.86731, 0.72332102, 0.00676399, 3.39777545,
     181.97970850, 131.76755713, 76.67261496, 0.0),

    ("Earth", 5.97217, 1.00000018, 0.01673163, -0.00054346,
     100.46691572, 102.93005885, -5.11260389, 0.0),

    ("Mars", 0.641691, 1.52371243, 0.09336511, 1.85181869,
     -4.56813164, -23.91744784, 49.71320984, 0.0),

    ("Jupiter", 1898.125, 5.20248019, 0.04853590, 1.29861416,
     34.33479152, 14.27495244, 100.29282654, 0.06064060),

    ("Saturn", 568.317, 9.54149883, 0.05550825, 2.49424102,
     50.07571329, 92.86136063, 113.63998702, -0.13434469),

    ("Uranus", 86.8099, 19.18797948, 0.04685740, 0.77298127,
     314.20276625, 172.43404441, 73.96250215, -0.97731848),

    ("Neptune", 102.4092, 30.06952752, 0.00895439, 1.77005520,
     304.22289287, 46.68158724, 131.78635853, 0.68346318),
)


def _kepler_state(
    semi_major_axis: float,
    eccentricity: float,
    inclination_deg: float,
    mean_anomaly_deg: float,
    argument_perihelion_deg: float,
    ascending_node_deg: float,
    gravitational_parameter: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate position and velocity from Keplerian elements."""

    a = semi_major_axis
    e = eccentricity

    inclination = np.radians(inclination_deg)
    omega = np.radians(argument_perihelion_deg)
    node = np.radians(ascending_node_deg)
    mean_anomaly = np.radians(mean_anomaly_deg)

    # Solve Kepler's equation using Newton-Raphson iteration.
    eccentric_anomaly = mean_anomaly

    for _ in range(50):
        residual = (
            eccentric_anomaly
            - e * np.sin(eccentric_anomaly)
            - mean_anomaly
        )
        correction = residual / (
            1.0 - e * np.cos(eccentric_anomaly)
        )
        eccentric_anomaly -= correction

        if abs(correction) < 1e-14:
            break

    E = eccentric_anomaly

    # Position in the orbital plane.
    x = a * (np.cos(E) - e)
    y = a * np.sqrt(1.0 - e**2) * np.sin(E)

    # Velocity in the orbital plane.
    mean_motion = np.sqrt(gravitational_parameter / a**3)
    eccentric_rate = mean_motion / (1.0 - e * np.cos(E))

    vx = -a * np.sin(E) * eccentric_rate
    vy = a * np.sqrt(1.0 - e**2) * np.cos(E) * eccentric_rate

    # Rotation from orbital plane to J2000 ecliptic coordinates.
    cw, sw = np.cos(omega), np.sin(omega)
    cn, sn = np.cos(node), np.sin(node)
    ci, si = np.cos(inclination), np.sin(inclination)

    rotation = np.array([
        [cw * cn - sw * sn * ci, -sw * cn - cw * sn * ci],
        [cw * sn + sw * cn * ci, -sw * sn + cw * cn * ci],
        [sw * si, cw * si],
    ])

    position = rotation @ np.array([x, y])
    velocity = rotation @ np.array([vx, vy])

    return position, velocity


def build_solar_system() -> SystemState:
    """Build the fixed Sun + eight planets J2000 reference system.

    Returns a barycentric SystemState in AU, years, and solar masses.
    """

    names = ["Sun"]
    masses = [1.0]
    positions = [np.zeros(3)]
    velocities = [np.zeros(3)]

    for (
        name,
        mass_1e24_kg,
        a,
        eccentricity,
        inclination,
        mean_longitude,
        longitude_perihelion,
        ascending_node,
        anomaly_correction,
    ) in _PLANETS:

        mass = mass_1e24_kg * 1e24 / SOLAR_MASS_KG

        argument_perihelion = (
            longitude_perihelion - ascending_node
        )

        mean_anomaly = (
            mean_longitude
            - longitude_perihelion
            + anomaly_correction
        )

        gravitational_parameter = G_AU * (1.0 + mass)

        position, velocity = _kepler_state(
            a,
            eccentricity,
            inclination,
            mean_anomaly,
            argument_perihelion,
            ascending_node,
            gravitational_parameter,
        )

        names.append(name)
        masses.append(mass)
        positions.append(position)
        velocities.append(velocity)

    masses_array = np.asarray(masses, dtype=float)
    positions_array = np.asarray(positions, dtype=float)
    velocities_array = np.asarray(velocities, dtype=float)

    # Shift to the barycentric frame.
    total_mass = np.sum(masses_array)

    center_of_mass = np.sum(
        masses_array[:, None] * positions_array,
        axis=0,
    ) / total_mass

    center_of_mass_velocity = np.sum(
        masses_array[:, None] * velocities_array,
        axis=0,
    ) / total_mass

    positions_array -= center_of_mass
    velocities_array -= center_of_mass_velocity

    return SystemState(
        names=tuple(names),
        masses=masses_array,
        positions=positions_array,
        velocities=velocities_array,
    )
