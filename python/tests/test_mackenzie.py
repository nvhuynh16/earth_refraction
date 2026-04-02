"""Tests for Mackenzie (1981) sound speed equation.

Reference data:
    - SONAR.m computational tables [A] at S=30 (gorbatschow.github.io/SonarDocs).
      Mackenzie uses depth directly — no pressure conversion ambiguity — so all
      16 entries serve as exact check values (tolerance 0.01 m/s for rounding).
    - Self-consistency: c(T=0, S=35, Z=0) = C0 = 1448.96 m/s (the (S-35) and
      Z terms all vanish).
"""

import pytest

from refraction.ocean import del_grosso, mackenzie


# -- SONAR.m reference table [A], S=30 ----------------------------------------
# (Z_m, T_C, c_ref_m_s)
SONAR_S30 = [
    (   10,  0, 1442.42),
    (   10, 10, 1483.78),
    (   10, 20, 1515.95),
    (   10, 30, 1540.36),
    ( 1000,  0, 1458.73),
    ( 1000, 10, 1500.08),
    ( 1000, 20, 1532.24),
    ( 1000, 30, 1556.65),
    ( 2000,  0, 1475.53),
    ( 2000, 10, 1516.83),
    ( 2000, 20, 1548.94),
    ( 2000, 30, 1573.30),
    ( 5000,  0, 1527.95),
    ( 5000, 10, 1568.41),
    ( 5000, 20, 1599.69),
    ( 5000, 30, 1623.21),
]


class TestSONAR:
    """Validate against SONAR.m computational tables [A] at S=30.

    Mackenzie uses depth directly (no pressure conversion), so all entries
    should match to rounding precision.
    """

    @pytest.mark.parametrize(
        "Z, T, c_ref",
        SONAR_S30,
        ids=[f"Z{z}T{t}" for z, t, _ in SONAR_S30],
    )
    def test_table(self, Z, T, c_ref):
        c = mackenzie.sound_speed(T, 30, Z)
        assert c == pytest.approx(c_ref, abs=0.01)


class TestBaseConstant:
    """At T=0, S=35, Z=0: all (S-35), Z, and T terms vanish => c = C0."""

    def test_c0(self):
        assert mackenzie.sound_speed(0, 35, 0) == pytest.approx(1448.96, abs=0.001)


class TestPhysics:
    """Physics consistency."""

    def test_temperature_increases_speed(self):
        assert mackenzie.sound_speed(25, 35, 0) > mackenzie.sound_speed(5, 35, 0)

    def test_salinity_increases_speed(self):
        assert mackenzie.sound_speed(10, 40, 0) > mackenzie.sound_speed(10, 25, 0)

    def test_depth_increases_speed(self):
        assert mackenzie.sound_speed(10, 35, 4000) > mackenzie.sound_speed(10, 35, 0)

    def test_depth_monotonic(self):
        """Sound speed increases monotonically with depth (constant T, S)."""
        prev = mackenzie.sound_speed(10, 35, 0)
        for z in range(1000, 9000, 1000):
            c = mackenzie.sound_speed(10, 35, z)
            assert c > prev
            prev = c


class TestCrossCheck:
    """Cross-check with Del Grosso (Mackenzie was fitted to Del Grosso)."""

    def test_agrees_at_surface_cold(self):
        c_mk = mackenzie.sound_speed(10, 35, 0)
        c_dg = del_grosso.sound_speed(10, 35, 0)
        assert c_mk == pytest.approx(c_dg, abs=0.5)

    def test_agrees_at_surface_warm(self):
        c_mk = mackenzie.sound_speed(25, 35, 0)
        c_dg = del_grosso.sound_speed(25, 35, 0)
        assert c_mk == pytest.approx(c_dg, abs=0.5)
