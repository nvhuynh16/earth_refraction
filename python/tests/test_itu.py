"""Tests for ITU-R P.453 model."""

import pytest
from refraction.itu_r_p453 import itu_N, itu_N_from_surface, k2, k3


def test_dry_air_standard():
    N = itu_N(288.15, 1013.25, 0.0)
    assert N == pytest.approx(273.0, abs=0.5)


def test_humid_air_20C():
    N = itu_N_from_surface(20.0, 101.325, 0.5)
    assert 290.0 < N < 340.0


def test_itu_exponential_reference_gradient():
    N_surface = 315.0
    H_scale = 7.35
    gradient = -N_surface / H_scale
    assert gradient == pytest.approx(-42.86, abs=0.01)


def test_dry_vs_humid_difference():
    N_dry = itu_N_from_surface(20.0, 101.325, 0.0)
    N_humid = itu_N_from_surface(20.0, 101.325, 1.0)
    assert N_humid > N_dry
    assert N_humid - N_dry > 30.0


def test_k3_term_dominance():
    T_K = 293.15
    e_hPa = 23.39
    k2_term = k2 * e_hPa / T_K
    k3_term = k3 * e_hPa / (T_K * T_K)
    assert k3_term > 5.0 * k2_term
