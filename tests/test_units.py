import numpy as np
import pytest

from orbitforge.units import (
    ASTRONOMICAL_UNITS,
    SI_UNITS,
    UnitSystem,
)


def test_si_gravitational_constant():
    assert SI_UNITS.gravitational_constant == pytest.approx(6.67430e-11)


def test_astronomical_gravitational_constant():
    expected = 4.0 * np.pi**2

    assert ASTRONOMICAL_UNITS.gravitational_constant == pytest.approx(
        expected, rel=1e-4
    )


def test_custom_unit_system():
    custom = UnitSystem(
        name="Custom",
        length_in_meters=1000.0,
        mass_in_kg=1.0,
        time_in_seconds=1.0,
        length_symbol="km",
        mass_symbol="kg",
        time_symbol="s",
    )

    expected = 6.67430e-20

    assert custom.gravitational_constant == pytest.approx(expected)

def test_astronomical_length_conversion():
    assert ASTRONOMICAL_UNITS.length_to_si(1.0) == pytest.approx(
        149597870700.0
    )


def test_astronomical_mass_conversion():
    assert ASTRONOMICAL_UNITS.mass_to_si(1.0) == pytest.approx(
        1.98847e30
    )


def test_astronomical_time_conversion():
    assert ASTRONOMICAL_UNITS.time_to_si(1.0) == pytest.approx(
        365.25 * 24 * 3600
    )


def test_astronomical_velocity_conversion():
    expected = 149597870700.0 / (365.25 * 24 * 3600)

    assert ASTRONOMICAL_UNITS.velocity_to_si(1.0) == pytest.approx(
        expected
    )


def test_length_round_trip():
    original = 3.5
    converted = ASTRONOMICAL_UNITS.length_to_si(original)

    assert ASTRONOMICAL_UNITS.length_from_si(converted) == pytest.approx(
        original
    )


def test_mass_round_trip():
    original = 2.5
    converted = ASTRONOMICAL_UNITS.mass_to_si(original)

    assert ASTRONOMICAL_UNITS.mass_from_si(converted) == pytest.approx(
        original
    )


def test_time_round_trip():
    original = 4.2
    converted = ASTRONOMICAL_UNITS.time_to_si(original)

    assert ASTRONOMICAL_UNITS.time_from_si(converted) == pytest.approx(
        original
    )


def test_velocity_round_trip():
    original = 12.5
    converted = ASTRONOMICAL_UNITS.velocity_to_si(original)

    assert ASTRONOMICAL_UNITS.velocity_from_si(converted) == pytest.approx(
        original
    )

def test_reject_zero_length_scale():
    with pytest.raises(ValueError):
        UnitSystem(
            name="Invalid",
            length_in_meters=0.0,
            mass_in_kg=1.0,
            time_in_seconds=1.0,
            length_symbol="m",
            mass_symbol="kg",
            time_symbol="s",
        )


def test_reject_negative_mass_scale():
    with pytest.raises(ValueError):
        UnitSystem(
            name="Invalid",
            length_in_meters=1.0,
            mass_in_kg=-1.0,
            time_in_seconds=1.0,
            length_symbol="m",
            mass_symbol="kg",
            time_symbol="s",
        )


def test_reject_infinite_time_scale():
    with pytest.raises(ValueError):
        UnitSystem(
            name="Invalid",
            length_in_meters=1.0,
            mass_in_kg=1.0,
            time_in_seconds=float("inf"),
            length_symbol="m",
            mass_symbol="kg",
            time_symbol="s",
        )


def test_reject_empty_unit_symbol():
    with pytest.raises(ValueError):
        UnitSystem(
            name="Invalid",
            length_in_meters=1.0,
            mass_in_kg=1.0,
            time_in_seconds=1.0,
            length_symbol="",
            mass_symbol="kg",
            time_symbol="s",
        )