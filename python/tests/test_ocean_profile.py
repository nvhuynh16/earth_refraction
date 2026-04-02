"""Tests for ocean depth-pressure conversion and parametric profiles.

Validates depth-pressure against:
    - Saunders (1981) / UNESCO (1983) check value at 10000 dbar, 30 deg lat
    - Round-trip consistency across latitude and depth range

References
----------
.. [1] Saunders (1981). J. Phys. Oceanogr., 11, 573-574.
.. [2] Fofonoff & Millard (1983). UNESCO Tech. Papers No. 44.
"""

import pytest
from refraction.ocean.ocean_profile import (
    depth_to_pressure, pressure_to_depth,
    temperature_at_depth, salinity_at_depth,
    FLORIDA_GULF_SUMMER, CALIFORNIA_SUMMER,
)


class TestDepthPressure:
    """Verify depth <-> pressure conversion."""

    def test_surface(self):
        """Zero depth gives zero pressure."""
        assert depth_to_pressure(0.0, 45.0) == pytest.approx(0.0, abs=0.01)

    def test_1000m(self):
        """At 1000 m, ~45 deg lat: p ~ 1008 dbar (slightly > 1 dbar/m due
        to compression).  [2] standard ocean gives ~1008.6 dbar."""
        p = depth_to_pressure(1000.0, 45.0)
        assert 1005.0 < p < 1015.0

    def test_round_trip(self):
        """Round-trip z -> p -> z should recover original depth within 0.5 m
        across all latitudes and depths to 5000 m."""
        for z in [100.0, 500.0, 2000.0, 5000.0]:
            for lat in [0.0, 30.0, 45.0, 60.0, 90.0]:
                p = depth_to_pressure(z, lat)
                z2 = pressure_to_depth(p, lat)
                assert z2 == pytest.approx(z, abs=0.5), (
                    f"Round-trip failed: z={z}, lat={lat}, p={p:.1f}, z2={z2:.1f}"
                )

    def test_unesco_check_value(self):
        """UNESCO [2] check value: p = 10000 dbar at lat = 30 deg gives
        depth = 9712.653 m.  Saunders [1] formula is accurate to ~1 m
        at full ocean depth."""
        z = pressure_to_depth(10000.0, 30.0)
        assert z == pytest.approx(9712.65, abs=10.0)

    def test_monotonic(self):
        """Deeper water always has higher pressure."""
        p1 = depth_to_pressure(100.0, 45.0)
        p2 = depth_to_pressure(200.0, 45.0)
        assert p2 > p1

    def test_latitude_effect(self):
        """Higher latitude has stronger gravity, so same depth gives slightly
        higher pressure.  Effect is ~0.5% between equator and pole."""
        p_eq = depth_to_pressure(1000.0, 0.0)
        p_pole = depth_to_pressure(1000.0, 90.0)
        assert p_pole > p_eq


class TestParametricProfiles:
    """Verify tanh thermocline/halocline model."""

    def test_surface_temperature(self):
        """With MLD >> D_thermo, tanh(-MLD/D) -> -1 and T(0) -> SST."""
        T = temperature_at_depth(0.0, 30.0, 4.0, 200.0, 50.0)
        assert T == pytest.approx(30.0, abs=0.5)

    def test_deep_temperature(self):
        """At z >> MLD + D_thermo, tanh -> +1 and T -> T_deep."""
        T = temperature_at_depth(2000.0, 30.0, 4.0, 50.0, 50.0)
        assert T == pytest.approx(4.0, abs=0.5)

    def test_thermocline_center(self):
        """At z = MLD, tanh(0) = 0, so T = (SST + T_deep) / 2 exactly."""
        T = temperature_at_depth(50.0, 30.0, 4.0, 50.0, 50.0)
        assert T == pytest.approx(17.0, abs=0.1)

    def test_surface_salinity(self):
        S = salinity_at_depth(0.0, 36.0, 35.0, 200.0, 50.0)
        assert S == pytest.approx(36.0, abs=0.1)

    def test_deep_salinity(self):
        S = salinity_at_depth(2000.0, 36.0, 35.0, 50.0, 50.0)
        assert S == pytest.approx(35.0, abs=0.1)

    def test_monotonic_cooling_with_depth(self):
        """When SST > T_deep, temperature must decrease monotonically."""
        temps = [temperature_at_depth(z, 30.0, 4.0, 50.0, 50.0)
                 for z in range(0, 500, 10)]
        for i in range(1, len(temps)):
            assert temps[i] <= temps[i - 1]

    def test_florida_summer_preset(self):
        """Florida Gulf summer: SST ~ 30 C, shallow MLD (25 m).
        Surface temperature should be warm even with tanh smoothing."""
        p = FLORIDA_GULF_SUMMER
        T = temperature_at_depth(0.0, p.sst_C, p.t_deep_C, p.mld_m, p.d_thermo_m)
        assert T > 22.0

    def test_california_summer_preset(self):
        """California summer: SST ~ 19 C (cooler due to upwelling [CalCOFI])."""
        p = CALIFORNIA_SUMMER
        T = temperature_at_depth(0.0, p.sst_C, p.t_deep_C, p.mld_m, p.d_thermo_m)
        assert T > 14.0
