"""Tests for the Quan & Fry 1995 seawater refractive index.

Validates against textbook values for the sodium D line (589 nm):
    - Pure water at 20 C: n ~ 1.333 (Tilton & Taylor 1938)
    - Seawater (35 PSU) at 20 C: n ~ 1.339 (Mehu & Johannin-Gilles 1968)

References
----------
.. [1] Quan & Fry (1995). Applied Optics, 34(18), 3477-3480.
"""

import pytest
from refraction.ocean.quan_fry import refractive_index


def test_pure_water_20C_589nm():
    """Tilton & Taylor (1938): n(20 C, S=0, 589 nm) ~ 1.333."""
    assert refractive_index(20.0, 0.0, 589.0) == pytest.approx(1.333, abs=0.001)


def test_seawater_20C_35psu_589nm():
    """Mehu & Johannin-Gilles (1968): n(20 C, S=35, 589 nm) ~ 1.339."""
    assert refractive_index(20.0, 35.0, 589.0) == pytest.approx(1.339, abs=0.001)


def test_salinity_increases_n():
    """Dissolved salts increase polarizability and thus refractive index."""
    assert refractive_index(20.0, 35.0, 589.0) > refractive_index(20.0, 0.0, 589.0)


def test_temperature_decreases_n():
    """Above ~4 C, thermal expansion decreases density and thus n."""
    assert refractive_index(25.0, 35.0, 589.0) < refractive_index(10.0, 35.0, 589.0)


def test_normal_dispersion():
    """Shorter wavelength -> higher n in the visible (Cauchy/Sellmeier)."""
    assert refractive_index(20.0, 35.0, 450.0) > refractive_index(20.0, 35.0, 650.0)
