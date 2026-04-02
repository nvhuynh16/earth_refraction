"""Tests for OceanRefractionProfile integration class."""

import math
import numpy as np
import pytest

from refraction.ocean.ocean_refraction import (
    OceanConditions, OceanMode, OceanRefractionProfile,
    OceanRefractivityResult, VectorOceanRefractivityResult,
)
from refraction.ocean.ocean_profile import FLORIDA_GULF_SUMMER, CALIFORNIA_SUMMER
from refraction.ocean import meissner_wentz as mw
from refraction.ocean import millard_seaver as ms
from refraction.ocean import quan_fry as qf


@pytest.fixture
def standard_cond(ocean_cond_radio):
    return ocean_cond_radio


@pytest.fixture
def standard_profile(standard_cond):
    return OceanRefractionProfile(standard_cond)


class TestCompute:
    """Single-depth computation."""

    def test_returns_result(self, standard_profile):
        r = standard_profile.compute(0.0, OceanMode.MeissnerWentz)
        assert isinstance(r, OceanRefractivityResult)

    def test_radio_surface_matches_standalone(self, standard_cond, standard_profile):
        r = standard_profile.compute(0.0, OceanMode.MeissnerWentz)
        n_standalone = mw.refractive_index(standard_cond.freq_ghz,
                                           standard_cond.sst_C,
                                           standard_cond.sss_psu)
        assert r.n_real == pytest.approx(n_standalone.n_real, rel=0.01)

    def test_optical_surface_matches_standalone(self, standard_cond, standard_profile):
        r = standard_profile.compute(0.0, OceanMode.QuanFry)
        n_standalone = qf.refractive_index(standard_cond.sst_C,
                                           standard_cond.sss_psu,
                                           standard_cond.wavelength_nm)
        assert r.n_real == pytest.approx(n_standalone, rel=0.01)

    def test_millard_seaver_surface(self, standard_cond, standard_profile):
        r = standard_profile.compute(0.0, OceanMode.MillardSeaver)
        n_standalone = ms.refractive_index(standard_cond.sst_C,
                                           standard_cond.sss_psu,
                                           0.0, standard_cond.wavelength_nm)
        assert r.n_real == pytest.approx(n_standalone, rel=0.01)

    def test_radio_has_permittivity(self, standard_profile):
        r = standard_profile.compute(50.0, OceanMode.MeissnerWentz)
        assert r.eps_real > 0.0
        assert r.eps_imag > 0.0

    def test_optical_no_imaginary(self, standard_profile):
        r = standard_profile.compute(50.0, OceanMode.QuanFry)
        assert r.n_imag == 0.0

    def test_dn_dz_computed(self, standard_profile):
        r = standard_profile.compute(50.0, OceanMode.MeissnerWentz)
        assert math.isfinite(r.dn_dz)


class TestProfile:
    """Depth profile computation."""

    def test_profile_length(self, standard_profile):
        results = standard_profile.profile(0.0, 100.0, 10.0, OceanMode.MeissnerWentz)
        assert len(results) == 11  # 0, 10, 20, ..., 100

    def test_profile_smooth(self, standard_profile):
        results = standard_profile.profile(0.0, 200.0, 5.0, OceanMode.MeissnerWentz)
        n_values = [r.n_real for r in results]
        # No NaN or Inf
        assert all(math.isfinite(n) for n in n_values)

    def test_millard_seaver_profile(self, standard_profile):
        results = standard_profile.profile(0.0, 200.0, 50.0, OceanMode.MillardSeaver)
        assert len(results) == 5
        # All values should be finite and physically reasonable (n > 1.3)
        for r in results:
            assert math.isfinite(r.n_real)
            assert r.n_real > 1.3


class TestCallable:
    """Scalar/array dispatch via __call__."""

    def test_scalar_call(self, standard_profile):
        r = standard_profile(50.0, OceanMode.MeissnerWentz)
        assert isinstance(r, OceanRefractivityResult)

    def test_array_call(self, standard_profile):
        r = standard_profile([0.0, 50.0, 100.0], OceanMode.MeissnerWentz)
        assert isinstance(r, VectorOceanRefractivityResult)
        assert len(r.depth_m) == 3


class TestIAPWSMode:
    """IAPWS mode through the profile class."""

    def test_iapws_surface(self, standard_profile):
        """IAPWS mode should return pure water RI (ignores salinity)."""
        r = standard_profile.compute(0.0, OceanMode.IAPWS)
        assert r.n_real > 1.3
        assert r.n_real < 1.4
        assert r.n_imag == 0.0

    def test_iapws_at_depth(self):
        """IAPWS with pressure: n should increase with depth."""
        cond = OceanConditions(sst_C=10.0, sss_psu=0.0, wavelength_nm=589.3)
        prof = OceanRefractionProfile(cond)
        r0 = prof.compute(0.0, OceanMode.IAPWS)
        r1000 = prof.compute(1000.0, OceanMode.IAPWS)
        assert r1000.n_real > r0.n_real


class TestPreset:
    """Preset ocean profiles."""

    def test_florida_preset(self):
        prof = OceanRefractionProfile.from_preset(FLORIDA_GULF_SUMMER)
        r = prof.compute(0.0, OceanMode.MeissnerWentz)
        assert r.temperature_C > 22.0  # tanh model doesn't reach exact SST

    def test_california_preset(self):
        prof = OceanRefractionProfile.from_preset(CALIFORNIA_SUMMER)
        r = prof.compute(0.0, OceanMode.MeissnerWentz)
        assert r.temperature_C > 14.0


class TestUserArrays:
    """User-supplied T(z), S(z) profiles."""

    def test_user_arrays(self):
        depths = [0.0, 50.0, 100.0, 200.0, 500.0]
        temps = [25.0, 20.0, 15.0, 8.0, 4.0]
        sals = [35.0, 35.0, 35.0, 34.5, 34.5]
        prof = OceanRefractionProfile.from_arrays(depths, temps, sals)
        r = prof.compute(0.0, OceanMode.MeissnerWentz)
        assert r.temperature_C == pytest.approx(25.0, abs=0.1)

    def test_interpolated_depth(self):
        depths = [0.0, 100.0]
        temps = [25.0, 5.0]
        sals = [35.0, 35.0]
        prof = OceanRefractionProfile.from_arrays(depths, temps, sals)
        r = prof.compute(50.0, OceanMode.MeissnerWentz)
        assert r.temperature_C == pytest.approx(15.0, abs=0.1)
