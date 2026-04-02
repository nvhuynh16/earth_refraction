"""Tests for water_vapor module."""

import math
import pytest
from refraction.water_vapor import (
    svp_giacomo, enhancement_factor, water_vapor_mole_fraction,
    water_vapor_profile,
)


def test_svp_at_20C():
    svp = svp_giacomo(293.15)
    assert svp == pytest.approx(2339.0, abs=5.0)


def test_svp_at_50C():
    svp = svp_giacomo(323.15)
    assert svp == pytest.approx(12351.0, abs=50.0)


def test_svp_at_100C():
    svp = svp_giacomo(373.15)
    assert svp == pytest.approx(101418.0, abs=500.0)


def test_enhancement_factor_standard():
    f = enhancement_factor(20.0, 101325.0)
    assert 1.003 < f < 1.005


def test_mole_fraction_20C_100RH():
    x_w = water_vapor_mole_fraction(20.0, 101325.0, 1.0)
    assert x_w == pytest.approx(0.0232, abs=0.001)


def test_mole_fraction_dry():
    x_w = water_vapor_mole_fraction(20.0, 101325.0, 0.0)
    assert x_w == pytest.approx(0.0, abs=1e-15)


def test_profile_decay():
    n0 = 1e23
    n_5km = water_vapor_profile(n0, 5.0, 0.0, 2.0)
    expected = n0 * math.exp(-5.0 / 2.0)
    assert n_5km == pytest.approx(expected, rel=1e-10)


def test_profile_negligible_above_15km():
    n0 = 1e23
    n_15km = water_vapor_profile(n0, 15.0, 0.0, 2.0)
    assert n_15km < n0 * 1e-3


# ---- New tests ----

import numpy as np
from refraction.water_vapor import water_vapor_number_density


def test_water_vapor_number_density_ideal_gas():
    """Cross-check: n = x_w * P / (kB * T)."""
    KB = 1.380649e-23
    T_K = 293.15
    P_Pa = 101325.0
    RH = 0.5
    n = water_vapor_number_density(T_K, P_Pa, RH)
    x_w = water_vapor_mole_fraction(20.0, P_Pa, RH)
    expected = x_w * P_Pa / (KB * T_K)
    assert n == pytest.approx(expected, rel=1e-10)


def test_profile_array_input():
    n0 = 1e23
    h_arr = np.array([0.0, 2.0, 5.0, 10.0])
    n_arr = water_vapor_profile(n0, h_arr, 0.0, 2.0)
    assert n_arr.shape == (4,)
    for i, h in enumerate(h_arr):
        n_scalar = float(water_vapor_profile(n0, h, 0.0, 2.0))
        assert n_arr[i] == pytest.approx(n_scalar, rel=1e-14)


def test_profile_below_surface():
    n0 = 1e23
    n_below = water_vapor_profile(n0, 0.0, 1.0, 2.0)
    assert float(n_below) == pytest.approx(n0, rel=1e-14)


def test_profile_at_surface():
    n0 = 1e23
    n_at = water_vapor_profile(n0, 1.0, 1.0, 2.0)
    assert float(n_at) == pytest.approx(n0, rel=1e-14)
