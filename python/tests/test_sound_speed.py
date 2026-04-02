"""Tests for SoundSpeedProfile integration class."""

import numpy as np
import pytest

from refraction.ocean import (
    SoundMode,
    SoundSpeedResult,
    VectorSoundSpeedResult,
    SoundSpeedProfile,
    FLORIDA_GULF_SUMMER,
    CALIFORNIA_SUMMER,
)
from refraction.ocean import OceanConditions, del_grosso, chen_millero, mackenzie


@pytest.fixture
def standard_cond(ocean_cond):
    return ocean_cond


@pytest.fixture
def standard_profile(standard_cond):
    return SoundSpeedProfile(standard_cond)


class TestCompute:
    """Single-depth computation."""

    def test_returns_result(self, standard_profile):
        r = standard_profile.compute(0.0, SoundMode.DelGrosso)
        assert isinstance(r, SoundSpeedResult)

    def test_del_grosso_surface_matches_standalone(self, standard_profile):
        r = standard_profile.compute(0.0, SoundMode.DelGrosso)
        # Tanh thermocline shifts T(0) slightly from SST; use 1% rel tolerance
        c_expected = del_grosso.sound_speed(r.temperature_C, r.salinity_psu, 0.0)
        assert r.sound_speed_m_s == pytest.approx(c_expected, abs=0.01)

    def test_chen_millero_surface_matches_standalone(self, standard_profile):
        r = standard_profile.compute(0.0, SoundMode.ChenMillero)
        c_expected = chen_millero.sound_speed(r.temperature_C, r.salinity_psu, 0.0)
        assert r.sound_speed_m_s == pytest.approx(c_expected, abs=0.01)

    def test_mackenzie_surface_matches_standalone(self, standard_profile):
        r = standard_profile.compute(0.0, SoundMode.Mackenzie)
        c_expected = mackenzie.sound_speed(r.temperature_C, r.salinity_psu, 0.0)
        assert r.sound_speed_m_s == pytest.approx(c_expected, abs=0.01)

    def test_physical_range(self, standard_profile):
        r = standard_profile.compute(0.0, SoundMode.DelGrosso)
        assert 1400.0 < r.sound_speed_m_s < 1600.0

    def test_dc_dz_is_finite(self, standard_profile):
        r = standard_profile.compute(50.0, SoundMode.DelGrosso)
        assert np.isfinite(r.dc_dz)
        assert r.dc_dz != 0.0

    def test_dc_dz_negative_in_thermocline(self):
        """In the thermocline, temperature drops rapidly -> dc/dz < 0."""
        cond = OceanConditions(
            sst_C=25.0, sss_psu=35.0,
            mld_m=50.0, d_thermo_m=50.0, t_deep_C=4.0,
        )
        prof = SoundSpeedProfile(cond)
        r = prof.compute(75.0, SoundMode.DelGrosso)  # center of thermocline
        assert r.dc_dz < 0.0


class TestProfile:
    """Depth-range sweep."""

    def test_profile_length(self, standard_profile):
        results = standard_profile.profile(0, 100, 10, SoundMode.DelGrosso)
        assert len(results) == 11  # 0, 10, ..., 100

    def test_profile_no_nan(self, standard_profile):
        results = standard_profile.profile(0, 200, 5, SoundMode.DelGrosso)
        for r in results:
            assert np.isfinite(r.sound_speed_m_s)
            assert np.isfinite(r.dc_dz)


class TestCallable:
    """Scalar and array dispatch via __call__."""

    def test_scalar_call(self, standard_profile):
        r = standard_profile(0.0, SoundMode.DelGrosso)
        assert isinstance(r, SoundSpeedResult)

    def test_array_call(self, standard_profile):
        r = standard_profile([0.0, 50.0, 100.0], SoundMode.DelGrosso)
        assert isinstance(r, VectorSoundSpeedResult)
        assert len(r.depth_m) == 3


class TestPreset:
    """Preset profiles."""

    def test_florida_preset(self):
        prof = SoundSpeedProfile.from_preset(FLORIDA_GULF_SUMMER)
        r = prof.compute(0.0, SoundMode.DelGrosso)
        # Florida summer SST=30 (tanh-adjusted T(0) ~ 23-28)
        assert r.sound_speed_m_s > 1520.0

    def test_california_preset(self):
        prof = SoundSpeedProfile.from_preset(CALIFORNIA_SUMMER)
        r = prof.compute(0.0, SoundMode.DelGrosso)
        # California summer SST=19 -> moderate
        assert 1500.0 < r.sound_speed_m_s < 1540.0


class TestUserArrays:
    """User-supplied T(z), S(z) arrays."""

    def test_user_arrays(self):
        prof = SoundSpeedProfile.from_arrays(
            [0, 100], [20, 10], [35, 35],
        )
        r = prof.compute(0.0, SoundMode.DelGrosso)
        c_expected = del_grosso.sound_speed(20.0, 35.0, 0.0)
        assert r.sound_speed_m_s == pytest.approx(c_expected, abs=0.01)

    def test_interpolated_depth(self):
        prof = SoundSpeedProfile.from_arrays(
            [0, 100], [20, 10], [35, 35],
        )
        r = prof.compute(50.0, SoundMode.DelGrosso)
        # Interpolated T=15 at z=50
        assert r.temperature_C == pytest.approx(15.0, abs=0.1)


class TestCrossEquation:
    """All three equations agree at surface within 0.5 m/s."""

    def test_surface_agreement(self):
        cond = OceanConditions(sst_C=15.0, sss_psu=35.0)
        prof = SoundSpeedProfile(cond)
        c_dg = prof.compute(0.0, SoundMode.DelGrosso).sound_speed_m_s
        c_cm = prof.compute(0.0, SoundMode.ChenMillero).sound_speed_m_s
        c_mk = prof.compute(0.0, SoundMode.Mackenzie).sound_speed_m_s
        assert c_dg == pytest.approx(c_cm, abs=0.5)
        assert c_dg == pytest.approx(c_mk, abs=0.5)
