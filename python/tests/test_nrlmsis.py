"""Tests for NRLMSIS 2.1 model."""

import pytest


def test_sea_level_temperature(msis, standard_input):
    standard_input.alt = 0.0
    out = msis.msiscalc(standard_input)
    assert 200.0 < out.tn < 330.0


def test_sea_level_species_dominance(msis, standard_input):
    standard_input.alt = 0.0
    out = msis.msiscalc(standard_input)
    n_total = out.dn[2] + out.dn[3]
    n_minor = out.dn[5] + out.dn[7]
    assert n_total > 100.0 * n_minor
    assert out.dn[2] > out.dn[3]


def test_altitude_50km(msis, standard_input):
    standard_input.alt = 50.0
    out = msis.msiscalc(standard_input)
    assert 200.0 < out.tn < 350.0
    assert out.dn[2] > 0.0


def test_altitude_100km(msis, standard_input):
    standard_input.alt = 100.0
    out = msis.msiscalc(standard_input)
    assert 150.0 < out.tn < 400.0


def test_altitude_120km_O_dominates(msis, standard_input):
    standard_input.alt = 300.0
    out = msis.msiscalc(standard_input)
    assert out.dn[4] > out.dn[2]
    assert out.dn[4] > out.dn[3]


def test_exospheric_temperature(msis, standard_input):
    standard_input.alt = 500.0
    out = msis.msiscalc(standard_input)
    assert 600.0 < out.tex < 2000.0
    assert out.tn == pytest.approx(out.tex, rel=0.05)


def test_dT_dz_troposphere_negative(msis, standard_input):
    standard_input.alt = 5.0
    full = msis.msiscalc_with_derivative(standard_input)
    assert full.dT_dz < 0.0
    assert -15.0 < full.dT_dz < -1.0


def test_dT_dz_stratosphere_positive(msis, standard_input):
    standard_input.alt = 50.0
    full = msis.msiscalc_with_derivative(standard_input)
    assert full.dT_dz > -3.0


def test_dT_dz_finite_difference_crosscheck(msis, standard_input):
    alts = [10.0, 30.0, 60.0, 90.0, 110.0, 130.0, 200.0]
    for alt in alts:
        standard_input.alt = alt
        full = msis.msiscalc_with_derivative(standard_input)

        dh = 0.01
        standard_input.alt = alt + dh
        out_plus = msis.msiscalc(standard_input)
        standard_input.alt = alt - dh
        out_minus = msis.msiscalc(standard_input)
        standard_input.alt = alt

        dT_dz_fd = (out_plus.tn - out_minus.tn) / (2.0 * dh)
        tol = max(abs(dT_dz_fd) * 0.08, 0.05)
        assert abs(full.dT_dz - dT_dz_fd) < tol, (
            f"alt={alt}: analytical={full.dT_dz:.4f}, FD={dT_dz_fd:.4f}"
        )


# ---- msiscalc_profile tests ----

import numpy as np


def test_msiscalc_profile_shapes(msis, standard_input):
    alts = np.array([0.0, 10.0, 50.0, 100.0, 200.0])
    out = msis.msiscalc_profile(standard_input, alts)
    assert out.tn.shape == (5,)
    assert out.dT_dz.shape == (5,)
    assert out.dn.shape == (5, 11)


def test_msiscalc_profile_matches_scalar(msis, standard_input):
    alts = np.array([0.0, 5.0, 30.0, 80.0, 120.0])
    profile_out = msis.msiscalc_profile(standard_input, alts)

    for i, alt in enumerate(alts):
        standard_input.alt = alt
        scalar_out = msis.msiscalc(standard_input)
        full = msis.msiscalc_with_derivative(standard_input)

        assert profile_out.tn[i] == pytest.approx(scalar_out.tn, rel=1e-12)
        for j in range(11):
            assert profile_out.dn[i, j] == pytest.approx(scalar_out.dn[j], rel=1e-12), \
                f"dn[{j}] mismatch at alt={alt}"
        assert profile_out.dT_dz[i] == pytest.approx(full.dT_dz, rel=1e-12)


def test_msiscalc_profile_single_altitude(msis, standard_input):
    alts = np.array([50.0])
    out = msis.msiscalc_profile(standard_input, alts)
    assert out.tn.shape == (1,)
    assert out.dn.shape == (1, 11)

    standard_input.alt = 50.0
    scalar = msis.msiscalc(standard_input)
    assert out.tn[0] == pytest.approx(scalar.tn, rel=1e-12)


def test_msiscalc_boundary_122km(msis, standard_input):
    """Test near ZETA_B boundary (122.5 km geopotential) — both branches exercised."""
    alts = np.array([120.0, 122.0, 123.0, 125.0])
    out = msis.msiscalc_profile(standard_input, alts)

    for i in range(len(alts)):
        assert 200.0 < out.tn[i] < 2000.0
        assert out.dn[i, 2] > 0.0  # N2 still present
        assert np.isfinite(out.dT_dz[i])
