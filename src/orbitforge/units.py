"""Physical unit systems for N-body simulations."""

from dataclasses import dataclass
from math import isfinite


# Gravitational constant in SI units:
# m^3 kg^-1 s^-2
G_SI = 6.67430e-11

# Astronomical unit in meters
AU_IN_METERS = 149597870700.0

# Solar mass in kilograms
SOLAR_MASS_IN_KG = 1.98847e30

# Julian year in seconds
YEAR_IN_SECONDS = 365.25 * 24.0 * 3600.0


@dataclass(frozen=True)
class UnitSystem:
    """Defines the fundamental units of a physical system."""

    name: str
    length_in_meters: float
    mass_in_kg: float
    time_in_seconds: float
    length_symbol: str
    mass_symbol: str
    time_symbol: str

    

    def __post_init__(self) -> None:
        """Validate the physical scales and unit labels."""

        scales = {
            "length_in_meters": self.length_in_meters,
            "mass_in_kg": self.mass_in_kg,
            "time_in_seconds": self.time_in_seconds,
        }

        for name, value in scales.items():
            if not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
                raise ValueError(
                    f"{name} must be a finite positive number."
                )

        labels = {
            "name": self.name,
            "length_symbol": self.length_symbol,
            "mass_symbol": self.mass_symbol,
            "time_symbol": self.time_symbol,
        }

        for name, value in labels.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
    
    @property
    def gravitational_constant(self) -> float:
        """Return G expressed in this unit system."""
        return (
            G_SI
            * self.mass_in_kg
            * self.time_in_seconds**2
            / self.length_in_meters**3
        )
    def length_to_si(self, value: float) -> float:
        """Convert a length to meters."""
        return value * self.length_in_meters

    def length_from_si(self, value: float) -> float:
        """Convert meters to the selected length unit."""
        return value / self.length_in_meters

    def mass_to_si(self, value: float) -> float:
        """Convert a mass to kilograms."""
        return value * self.mass_in_kg

    def mass_from_si(self, value: float) -> float:
        """Convert kilograms to the selected mass unit."""
        return value / self.mass_in_kg

    def time_to_si(self, value: float) -> float:
        """Convert time to seconds."""
        return value * self.time_in_seconds

    def time_from_si(self, value: float) -> float:
        """Convert seconds to the selected time unit."""
        return value / self.time_in_seconds

    def velocity_to_si(self, value: float) -> float:
        """Convert velocity to meters per second."""
        return value * self.length_in_meters / self.time_in_seconds

    def velocity_from_si(self, value: float) -> float:
        """Convert meters per second to the selected velocity unit."""
        return value * self.time_in_seconds / self.length_in_meters

SI_UNITS = UnitSystem(
    name="SI",
    length_in_meters=1.0,
    mass_in_kg=1.0,
    time_in_seconds=1.0,
    length_symbol="m",
    mass_symbol="kg",
    time_symbol="s",
)

ASTRONOMICAL_UNITS = UnitSystem(
    name="Astronomical",
    length_in_meters=AU_IN_METERS,
    mass_in_kg=SOLAR_MASS_IN_KG,
    time_in_seconds=YEAR_IN_SECONDS,
    length_symbol="AU",
    mass_symbol="M_sun",
    time_symbol="yr",
)
    