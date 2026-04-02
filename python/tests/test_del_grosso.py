"""Tests for Del Grosso (1974) sound speed equation.

Reference data:
    - SONAR.m computational tables [A] at S=30 (gorbatschow.github.io/SonarDocs).
      These use depth in meters; Mackenzie confirms S=30, not S=35 as captioned.
      Del Grosso/Chen-Millero entries require depth-to-pressure conversion, so
      only the Z=10 m row (where pressure ambiguity is negligible) is used for
      tight tolerance.  Deeper rows are checked with 0.3 m/s tolerance to allow
      for differing depth-to-pressure formulas.
    - Self-consistency: C000 = 1402.392 m/s at T=0, S=0, P=0.
"""

import pytest

from refraction.ocean import del_grosso
from refraction.ocean.ocean_profile import depth_to_pressure


# -- SONAR.m reference table [A], S=30, lat=45 for depth-to-pressure ----------
# (Z_m, T_C, c_ref_m_s)
SONAR_S30 = [
    # Z=10 m: negligible pressure conversion ambiguity => tight tolerance
    (10,  0, 1442.55),
    (10, 10, 1483.85),
    (10, 20, 1516.04),
    (10, 30, 1540.44),
    # Z=1000 m
    (1000,  0, 1458.67),
    (1000, 10, 1500.30),
    (1000, 20, 1532.61),
    (1000, 30, 1556.65),
    # Z=2000 m
    (2000,  0, 1475.45),
    (2000, 10, 1517.18),
    (2000, 20, 1549.48),
    (2000, 30, 1573.14),
    # Z=5000 m
    (5000,  0, 1528.32),
    (5000, 10, 1569.16),
    (5000, 20, 1600.96),
    (5000, 30, 1623.67),
]


class TestSONAR:
    """Validate against SONAR.m computational tables [A] at S=30."""

    @pytest.mark.parametrize(
        "Z, T, c_ref",
        [(z, t, c) for z, t, c in SONAR_S30 if z == 10],
        ids=[f"Z{z}T{t}" for z, t, _ in SONAR_S30 if z == 10],
    )
    def test_shallow(self, Z, T, c_ref):
        """At 10 m depth, pressure conversion ambiguity is < 0.01 m/s."""
        p = depth_to_pressure(Z, 45.0)
        c = del_grosso.sound_speed(T, 30, p)
        assert c == pytest.approx(c_ref, abs=0.02)

    @pytest.mark.parametrize(
        "Z, T, c_ref",
        [(z, t, c) for z, t, c in SONAR_S30 if z > 10],
        ids=[f"Z{z}T{t}" for z, t, _ in SONAR_S30 if z > 10],
    )
    def test_deep(self, Z, T, c_ref):
        """Deeper points: allow 0.3 m/s for depth-to-pressure formula differences."""
        p = depth_to_pressure(Z, 45.0)
        c = del_grosso.sound_speed(T, 30, p)
        assert c == pytest.approx(c_ref, abs=0.3)


class TestBaseConstant:
    """The C000 base constant is recovered at T=0, S=0, P=0."""

    def test_c000(self):
        assert del_grosso.sound_speed(0, 0, 0) == pytest.approx(1402.392, abs=0.001)


class TestPhysics:
    """Physics consistency: monotonicity in T, S, P."""

    def test_temperature_increases_speed(self):
        assert del_grosso.sound_speed(25, 35, 0) > del_grosso.sound_speed(5, 35, 0)

    def test_salinity_increases_speed(self):
        assert del_grosso.sound_speed(10, 40, 0) > del_grosso.sound_speed(10, 30, 0)

    def test_pressure_increases_speed(self):
        assert del_grosso.sound_speed(10, 35, 5000) > del_grosso.sound_speed(10, 35, 0)

    def test_pressure_monotonic(self):
        """Sound speed increases monotonically with pressure."""
        prev = del_grosso.sound_speed(10, 35, 0)
        for p in range(1000, 11000, 1000):
            c = del_grosso.sound_speed(10, 35, p)
            assert c > prev
            prev = c


class TestRange:
    """Output values are in the physical range 1400-1600 m/s."""

    @pytest.mark.parametrize("T", [0, 10, 20, 30])
    @pytest.mark.parametrize("S", [30, 35, 40])
    def test_output_range(self, T, S):
        c = del_grosso.sound_speed(T, S, 0)
        assert 1400.0 < c < 1600.0
