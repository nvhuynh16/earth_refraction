"""Tests for water_vapor module."""

import math
import pytest
from refraction.water_vapor import (
    svp_giacomo, enhancement_factor, water_vapor_mole_fraction,
    water_vapor_profile, water_vapor_profile_derivative,
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


def test_profile_below_surface_bounded():
    """Below the surface the gradient tapers off, so n levels off at a
    bounded multiple of n0 instead of growing without limit."""
    n0, H_w, L = 1e23, 2.0, 0.3
    c = L / (H_w * math.sqrt(2.0))
    limit = 1.0 + (L / H_w) * math.sqrt(math.pi / 2.0) * math.exp(c * c) * (1.0 + math.erf(c))
    vals = [float(water_vapor_profile(n0, h, 1.0, H_w)) / n0 for h in (0.9, 0.5, 0.0, -5.0)]
    assert all(b > a for a, b in zip(vals, vals[1:]))  # increases downward
    assert vals[-1] == pytest.approx(limit, rel=1e-12)
    assert limit == pytest.approx(1.2128, abs=1e-4)


@pytest.mark.parametrize("h_s", [0.0, 1.0, 3.0])
def test_profile_smooth_at_surface(h_s):
    """n, dn/dh and d2n/dh2 are continuous at the surface altitude."""
    n0, H_w = 1e23, 2.0
    eps = 1e-9
    n = lambda h: float(water_vapor_profile(n0, h, h_s, H_w)) / n0
    d = lambda h: float(water_vapor_profile_derivative(n0, h, h_s, H_w)) / n0
    assert n(h_s - eps) == pytest.approx(n(h_s + eps), abs=1e-8)
    assert d(h_s - eps) == pytest.approx(d(h_s + eps), abs=1e-8)
    assert d(h_s) == pytest.approx(-1.0 / H_w, rel=1e-14)
    # Curvature n'' = 1/H_w^2 on both sides (one-sided differences, O(step) error)
    step = 1e-5
    assert (d(h_s) - d(h_s - step)) / step == pytest.approx(1.0 / H_w**2, rel=1e-3)
    assert (d(h_s + step) - d(h_s)) / step == pytest.approx(1.0 / H_w**2, rel=1e-3)


def test_profile_derivative_matches_fd():
    n0, h_s = 1e23, 1.0
    h = np.array([-1.0, 0.2, 0.7, 0.95, 1.05, 3.0, 8.0])
    e = 1e-5
    fd = (water_vapor_profile(n0, h + e, h_s) - water_vapor_profile(n0, h - e, h_s)) / (2 * e)
    assert np.allclose(water_vapor_profile_derivative(n0, h, h_s), fd, rtol=1e-7, atol=1e12)


def test_profile_at_surface():
    n0 = 1e23
    n_at = water_vapor_profile(n0, 1.0, 1.0, 2.0)
    assert float(n_at) == pytest.approx(n0, rel=1e-14)
