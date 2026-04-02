"""Tests for Chen-Millero / UNESCO (1977) sound speed equation.

Reference data:
    - Pond & Pickard (1986) [A] check values at P=0 for S=25 and S=35.
      These use IPTS-68 temperature scale; our ITS-90 (Wong & Zhu 1995)
      coefficients give slightly different results (max ~0.03 m/s at T=40).
    - UNESCO Technical Paper 44 [B] check value:
      S=40, T_68=40, P=10000 dbar => c=1731.995 m/s (IPTS-68).
    - SONAR.m computational tables [C] at S=30 for cross-validation.
    - Self-consistency: C00 = 1402.388 m/s at T=0, S=0, P=0.
"""

import pytest

from refraction.ocean import del_grosso, chen_millero
from refraction.ocean.ocean_profile import depth_to_pressure


# -- Pond & Pickard (1986) [A] reference values, P=0 dbar --------------------
# (T_C, S_ppt, c_ref_m_s)  — IPTS-68 scale; tolerance 0.03 for ITS-90 drift.
POND_PICKARD_P0 = [
    # S=25
    ( 0, 25, 1435.789875),
    (10, 25, 1477.681131),
    (20, 25, 1510.316386),
    (30, 25, 1535.214379),
    (40, 25, 1553.440561),
    # S=35
    ( 0, 35, 1449.138828),
    (10, 35, 1489.830942),
    (20, 35, 1521.475257),
    (30, 35, 1545.609796),
    (40, 35, 1563.223222),
]


class TestPondPickard:
    """Validate against Pond & Pickard (1986) [A] at P=0."""

    @pytest.mark.parametrize(
        "T, S, c_ref",
        POND_PICKARD_P0,
        ids=[f"T{t}S{s}" for t, s, _ in POND_PICKARD_P0],
    )
    def test_p0(self, T, S, c_ref):
        """ITS-90 coefficients match IPTS-68 values within 0.03 m/s."""
        c = chen_millero.sound_speed(T, S, 0)
        assert c == pytest.approx(c_ref, abs=0.03)


class TestUNESCO:
    """UNESCO Technical Paper 44 [B] check value."""

    def test_s40_t40_p10000(self):
        """S=40, T_68=40 C, P=10000 dbar => c=1731.995 m/s (IPTS-68).

        Our ITS-90 coefficients give ~1732.02 m/s; allow 0.05 m/s for the
        temperature scale difference.
        """
        c = chen_millero.sound_speed(40, 40, 10000)
        assert c == pytest.approx(1731.995, abs=0.05)


class TestBaseConstant:
    """The C00 base constant is recovered at T=0, S=0, P=0."""

    def test_c00(self):
        assert chen_millero.sound_speed(0, 0, 0) == pytest.approx(1402.388, abs=0.001)


class TestSONAR:
    """Cross-validate against SONAR.m tables [C] at S=30, Z=10 m."""

    @pytest.mark.parametrize("T, c_ref", [
        (0, 1442.62), (10, 1483.92), (20, 1516.06), (30, 1540.59),
    ], ids=["T0", "T10", "T20", "T30"])
    def test_shallow(self, T, c_ref):
        p = depth_to_pressure(10, 45.0)
        c = chen_millero.sound_speed(T, 30, p)
        assert c == pytest.approx(c_ref, abs=0.02)


class TestPhysics:
    """Physics consistency."""

    def test_temperature_increases_speed(self):
        assert chen_millero.sound_speed(25, 35, 0) > chen_millero.sound_speed(5, 35, 0)

    def test_salinity_increases_speed(self):
        assert chen_millero.sound_speed(10, 40, 0) > chen_millero.sound_speed(10, 0, 0)

    def test_pressure_increases_speed(self):
        assert chen_millero.sound_speed(10, 35, 5000) > chen_millero.sound_speed(10, 35, 0)

    def test_fresh_water_valid(self):
        """S=0 is within Chen-Millero validity range (unlike Del Grosso)."""
        c = chen_millero.sound_speed(20, 0, 0)
        assert 1480.0 < c < 1485.0


class TestCrossCheck:
    """Cross-check with Del Grosso."""

    def test_agrees_at_surface(self):
        c_cm = chen_millero.sound_speed(10, 35, 0)
        c_dg = del_grosso.sound_speed(10, 35, 0)
        assert c_cm == pytest.approx(c_dg, abs=0.5)

    def test_diverges_at_depth(self):
        """Chen-Millero overestimates by ~0.5 m/s at high pressure vs Del Grosso."""
        c_cm = chen_millero.sound_speed(0, 35, 10000)
        c_dg = del_grosso.sound_speed(0, 35, 10000)
        diff = abs(c_cm - c_dg)
        assert 0.1 < diff < 2.0
