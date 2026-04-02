"""Tests for the Meissner & Wentz seawater permittivity model.

Validates against:
    - Textbook static permittivity of pure water (CRC Handbook 2004)
    - Stogryn conductivity model (UNESCO PSS-78 reference values)
    - Physical consistency checks (S=0 limit, monotonicity, identities)
    - MW2004 model behavior at L-band, X-band, Ka-band

References
----------
.. [1] Meissner & Wentz (2004), IEEE TGRS 42(9), 1836-1849.
.. [2] CRC Handbook of Chemistry and Physics, 85th ed. (2004), Table 6-14.
.. [3] Culkin & Smith (1980), "Determination of the concentration of potassium
       chloride solution having the same electrical conductivity, at 15 C and
       infinite frequency, as standard seawater."  IEEE J. Oceanic Eng. 5, 22.
"""

import math
import pytest
from refraction.ocean.meissner_wentz import (
    eps_s_pure, eps_1_pure, eps_inf_pure, v1_pure, v2_pure,
    sigma35, conductivity,
    eps_s_sea, eps_1_sea, eps_inf_sea, v1_sea, v2_sea,
    permittivity, refractive_index,
)


# ---------------------------------------------------------------------------
# Pure-water Debye parameters  (vs CRC Handbook [2])
# ---------------------------------------------------------------------------

class TestPureWater:
    """Verify pure-water Debye parameters against textbook values."""

    def test_eps_s_pure_0C(self):
        """CRC [2] gives eps_s(0 C) = 87.740.  MW2004 formula gives ~87.9."""
        assert eps_s_pure(0.0) == pytest.approx(87.9, abs=0.2)

    def test_eps_s_pure_20C(self):
        """CRC [2] gives eps_s(20 C) = 80.100."""
        assert eps_s_pure(20.0) == pytest.approx(80.2, abs=0.2)

    def test_eps_s_pure_25C(self):
        """CRC [2] gives eps_s(25 C) = 78.360."""
        assert eps_s_pure(25.0) == pytest.approx(78.4, abs=0.2)

    def test_eps_1_pure_20C(self):
        """Intermediate permittivity, typically ~5.7-6.2 for the second
        relaxation between first and second Debye processes."""
        val = eps_1_pure(20.0)
        assert 5.0 < val < 7.0

    def test_eps_inf_pure_20C(self):
        """High-frequency limit of permittivity.  Optical regime: n^2 ~ 1.77,
        so eps_inf ~ 4.2 (consistent with optical refractive index of water)."""
        val = eps_inf_pure(20.0)
        assert 3.5 < val < 5.0

    def test_v1_pure_20C(self):
        """Principal relaxation ~17 GHz at 20 C (Kaatze 1989).  The Debye
        relaxation time tau_1 = 1/(2*pi*v1) ~ 9 ps at 20 C."""
        val = v1_pure(20.0)
        assert 15.0 < val < 25.0

    def test_v2_pure_20C(self):
        """Second relaxation ~100-300 GHz, associated with the hydrogen-bond
        network rearrangement (Buchner et al. 1999)."""
        val = v2_pure(20.0)
        assert 80.0 < val < 400.0


# ---------------------------------------------------------------------------
# Conductivity  (vs UNESCO PSS-78 [3])
# ---------------------------------------------------------------------------

class TestConductivity:
    """Verify Stogryn conductivity model."""

    def test_sigma35_15C(self):
        """sigma(15 C, 35 PSU) ~ 4.29 S/m.  The conductivity ratio R_15(35)
        is defined as 1.0 in PSS-78 [3], so this tests sigma35 directly."""
        assert sigma35(15.0) == pytest.approx(4.29, abs=0.05)

    def test_conductivity_25C_35psu(self):
        """sigma(25 C, 35 PSU) = 5.306 S/m.  Computed from [1] Eq. 12:
        sigma35(25) = 2.903602 + 8.607e-2*25 + 4.738817e-4*625
                    - 2.991e-6*15625 + 4.3047e-9*390625 = 5.306 S/m.
        R_15(35) = 1.0 (by definition at 15 C, 35 PSU)."""
        val = conductivity(25.0, 35.0)
        assert val == pytest.approx(5.31, abs=0.1)

    def test_conductivity_pure_water(self):
        """Pure water has negligible ionic conductivity (< 0.01 S/m)."""
        assert conductivity(20.0, 0.0) == 0.0
        assert conductivity(0.0, 0.0) == 0.0

    def test_conductivity_increases_with_temperature(self):
        """Ion mobility increases with temperature."""
        assert conductivity(25.0, 35.0) > conductivity(5.0, 35.0)

    def test_conductivity_increases_with_salinity(self):
        """More ions means more conductivity."""
        assert conductivity(20.0, 35.0) > conductivity(20.0, 10.0)

    def test_conductivity_positive(self):
        for T in [0.0, 10.0, 20.0, 29.0]:
            for S in [5.0, 20.0, 35.0, 40.0]:
                assert conductivity(T, S) > 0.0


# ---------------------------------------------------------------------------
# Salinity-dependent parameters
# ---------------------------------------------------------------------------

class TestSalinityDependence:
    """Verify salinity corrections reduce to pure water at S=0."""

    def test_eps_s_salt_reduces_permittivity(self):
        """Dissolved ions disrupt the water dipole structure, reducing eps_s.
        At S=35, eps_s should be ~5-10% lower than pure water."""
        assert eps_s_sea(20.0, 35.0) < eps_s_pure(20.0)

    def test_eps_s_sea_at_zero_salinity(self):
        assert eps_s_sea(20.0, 0.0) == pytest.approx(eps_s_pure(20.0), rel=1e-10)

    def test_v1_sea_at_zero_salinity(self):
        assert v1_sea(20.0, 0.0) == pytest.approx(v1_pure(20.0), rel=1e-10)

    def test_eps_1_sea_at_zero_salinity(self):
        assert eps_1_sea(20.0, 0.0) == pytest.approx(eps_1_pure(20.0), rel=1e-10)

    def test_v2_sea_at_zero_salinity(self):
        assert v2_sea(20.0, 0.0) == pytest.approx(v2_pure(20.0), rel=1e-10)

    def test_eps_inf_sea_at_zero_salinity(self):
        assert eps_inf_sea(20.0, 0.0) == pytest.approx(eps_inf_pure(20.0), rel=1e-10)


# ---------------------------------------------------------------------------
# Full permittivity model
# ---------------------------------------------------------------------------

class TestPermittivity:
    """Verify complex permittivity at representative frequency/T/S points."""

    def test_pure_water_10ghz_20C(self):
        """Pure water at X-band (10 GHz), 20 C.
        Near the principal relaxation, large dielectric loss expected."""
        eps = permittivity(10.0, 20.0, 0.0)
        assert 50.0 < eps.real < 65.0
        assert 25.0 < eps.imag < 40.0

    def test_seawater_1p4ghz_20C_35psu(self):
        """Standard seawater at L-band (1.4 GHz), 20 C, 35 PSU.
        Conductivity dominates eps_imag at L-band (low frequency)."""
        eps = permittivity(1.4, 20.0, 35.0)
        assert 60.0 < eps.real < 80.0
        # Conductivity contribution: sigma*f0/f ~ 5.37*17.975/1.4 ~ 69
        assert 50.0 < eps.imag < 80.0

    def test_seawater_10ghz_20C_35psu(self):
        """Standard seawater at X-band (10 GHz), 20 C, 35 PSU."""
        eps = permittivity(10.0, 20.0, 35.0)
        assert 35.0 < eps.real < 60.0
        assert 30.0 < eps.imag < 50.0

    def test_seawater_37ghz_20C_35psu(self):
        """Standard seawater at Ka-band (37 GHz), 20 C, 35 PSU.
        Well above principal relaxation, eps_real drops significantly."""
        eps = permittivity(37.0, 20.0, 35.0)
        assert 5.0 < eps.real < 25.0
        assert 15.0 < eps.imag < 35.0

    def test_static_limit(self):
        """At very low frequency, eps_real should approach eps_s."""
        eps = permittivity(0.01, 20.0, 35.0)
        es = eps_s_sea(20.0, 35.0)
        assert eps.real == pytest.approx(es, rel=0.01)

    def test_no_nan_arctic(self):
        eps = permittivity(10.0, -2.0, 35.0)
        assert math.isfinite(eps.real) and math.isfinite(eps.imag)

    def test_no_nan_tropical(self):
        eps = permittivity(10.0, 29.0, 35.0)
        assert math.isfinite(eps.real) and math.isfinite(eps.imag)

    def test_mw2012_high_temperature_branch(self):
        """MW2012 v1_sea has a separate branch for SST > 30 C.
        Verify it produces finite, physically reasonable results."""
        eps = permittivity(10.0, 35.0, 35.0)  # SST = 35 C, above branch point
        assert math.isfinite(eps.real) and math.isfinite(eps.imag)
        assert eps.real > 0.0

    def test_v1_sea_continuity_at_30C(self):
        """v1_sea should be approximately continuous at the T=30 branch point."""
        v1_below = v1_sea(29.99, 35.0)
        v1_above = v1_sea(30.01, 35.0)
        # Should be within 1% of each other at the boundary
        assert v1_below == pytest.approx(v1_above, rel=0.01)

    def test_eps_real_positive(self):
        """Dielectric constant must always be positive."""
        for f in [1.0, 10.0, 37.0, 85.0]:
            for T in [-2.0, 15.0, 29.0]:
                eps = permittivity(f, T, 35.0)
                assert eps.real > 0.0


# ---------------------------------------------------------------------------
# Complex refractive index
# ---------------------------------------------------------------------------

class TestRefractiveIndex:
    """Verify complex refractive index from permittivity."""

    def test_identity(self):
        """Verify n^2 = eps: n_real^2 - n_imag^2 = eps_real and
        2*n_real*n_imag = eps_imag (complex square root identity)."""
        for T in [0.0, 15.0, 25.0]:
            for S in [0.0, 35.0]:
                for f in [1.4, 10.0, 37.0]:
                    eps = permittivity(f, T, S)
                    n = refractive_index(f, T, S)
                    eps_r_check = n.n_real ** 2 - n.n_imag ** 2
                    eps_i_check = 2.0 * n.n_real * n.n_imag
                    assert eps_r_check == pytest.approx(eps.real, rel=1e-10)
                    assert eps_i_check == pytest.approx(eps.imag, rel=1e-10)

    def test_n_real_greater_than_1(self):
        """Water always has n > 1 at microwave frequencies."""
        n = refractive_index(10.0, 20.0, 35.0)
        assert n.n_real > 1.0

    def test_n_imag_positive(self):
        """Lossy medium always has positive extinction coefficient."""
        n = refractive_index(10.0, 20.0, 35.0)
        assert n.n_imag > 0.0

    def test_seawater_lband(self):
        """At L-band (1.4 GHz), 20 C, 35 PSU: n_real ~ 8, n_imag ~ 2-4
        (high n because eps >> 1 at low microwave frequencies)."""
        n = refractive_index(1.4, 20.0, 35.0)
        assert 6.0 < n.n_real < 10.0
        assert 1.0 < n.n_imag < 5.0
