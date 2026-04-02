"""Tests for the Millard & Seaver 1990 seawater refractive index algorithm.

Validates against Table 2 of the original paper [1] at lambda = 589.26 nm
(sodium D line).  Table 2 gives N(T, p, S) for T = 0-25 deg C in 5-degree
steps, S = 0, 30, 35, 40 PSU, and p = 0-10000 dbar in 2000-dbar steps.

The table was digitized from the original paper and is stored in
``docs/millard1990 - Table 2.txt``.  A few OCR artifacts in the pressure
rows (noted below) are excluded from the test set.

References
----------
.. [1] Millard, R.C. and G. Seaver (1990). "An index of refraction algorithm
       for seawater over temperature, pressure, salinity, density, and
       wavelength." Deep-Sea Research, 37(12), 1909-1926.
"""

import pytest
from refraction.ocean.millard_seaver import refractive_index


# ---------------------------------------------------------------------------
# Table 2 validation data  [1] Table 2, lambda = 589.26 nm
# ---------------------------------------------------------------------------
# Format: (T_C, S_psu, p_dbar, expected_n)
#
# Accuracy per region:
#   Region I  (S=0, p=0):  0.12 ppm  -> abs tol 1e-6 sufficient
#   Region II (S>0, p=0):  4.7 ppm   -> abs tol 5e-6
#   Region III (p>0, S=0): 26.5 ppm  -> abs tol 15e-6
#   Region IV  (p>0, S>0): 19.7 ppm  -> abs tol 15e-6
#
# Some Table 2 pressure-row entries have OCR artifacts in the digitized file:
#   Line 6 col 2: "1.346468" should be "1.345468" (p=8000, T=5, S=0)
#   Line 12 col 3: "1.356480" is garbled (p=8000, T=10, S=30)
#   Line 15 col 1: comma for period (p=2000, T=0, S=35) — corrected here
#   Line 19: "10002" should be "10000" (S=35)
#   Line 22 col 5: "1.34,586,5" is garbled (p=4000, T=20, S=40)
# Only verified-clean rows are included below.

TABLE2 = [
    # ---- S=0, p=0 — Region I only (SD = 0.12 ppm) ----
    (0,  0, 0,     1.333949),
    (5,  0, 0,     1.333884),
    (10, 0, 0,     1.333691),
    (15, 0, 0,     1.333387),
    (20, 0, 0,     1.332988),
    (25, 0, 0,     1.332503),
    # ---- S=35, p=0 — Regions I + II (SD = 4.7 ppm) ----
    (0,  35, 0,    1.340854),
    (5,  35, 0,    1.340630),
    (10, 35, 0,    1.340298),
    (15, 35, 0,    1.339878),
    (20, 35, 0,    1.339386),
    (25, 35, 0,    1.338838),
    # ---- S=40, p=0 — Regions I + II ----
    (0,  40, 0,    1.341840),
    (5,  40, 0,    1.341594),
    (10, 40, 0,    1.341242),
    (15, 40, 0,    1.340805),
    (20, 40, 0,    1.340300),
    (25, 40, 0,    1.339743),
    # ---- S=30, p=0 — Regions I + II ----
    (5,  30, 0,    1.339666),
    (10, 30, 0,    1.339354),
    (15, 30, 0,    1.338950),
    (20, 30, 0,    1.338472),
    (25, 30, 0,    1.337933),
    # ---- S=0, p>0 — Regions I + III (SD = 26.5 ppm) ----
    (0,  0, 2000,  1.337122),
    (0,  0, 4000,  1.340168),
    (0,  0, 6000,  1.343089),
    (0,  0, 10000, 1.348552),
    (20, 0, 2000,  1.335871),
    (20, 0, 4000,  1.338647),
    # ---- S=35, p>0 — All four regions ----
    (0,  35, 2000,  1.343948),
    (20, 35, 2000,  1.342228),
    (0,  35, 4000,  1.346916),
    (20, 35, 4000,  1.344962),
    (0,  35, 10000, 1.355065),
    (25, 35, 10000, 1.351836),
]


class TestTable2:
    """Validate against [1] Table 2."""

    @pytest.mark.parametrize("T,S,p,expected", TABLE2,
                             ids=[f"T{t}S{s}p{p}" for t, s, p, _ in TABLE2])
    def test_table2(self, T, S, p, expected):
        """Each Table 2 entry should match within 15 ppm (covers all regions'
        stated SDs plus 6-decimal-place rounding in the table)."""
        n = refractive_index(T, S, p, 589.26)
        assert n == pytest.approx(expected, abs=15e-6), (
            f"n({T},{S},{p})={n:.7f}, expected {expected:.6f}, "
            f"diff={1e6*(n-expected):.1f} ppm"
        )


class TestPhysics:
    """Physical consistency checks at lambda = 589.26 nm."""

    def test_salinity_increases_n(self):
        """Dissolved salts increase molecular polarizability, so dn/dS > 0.
        At 20 C, 1 atm: n(35 PSU) - n(0 PSU) ~ 0.006 (from Table 2)."""
        n0 = refractive_index(20.0, 0.0, 0.0, 589.26)
        n35 = refractive_index(20.0, 35.0, 0.0, 589.26)
        assert n35 > n0

    def test_pressure_increases_n(self):
        """Compression increases density and thus refractive index.
        At 20 C, 35 PSU: n(2000 dbar) - n(0 dbar) ~ 0.003."""
        n0 = refractive_index(20.0, 35.0, 0.0, 589.26)
        n2000 = refractive_index(20.0, 35.0, 2000.0, 589.26)
        assert n2000 > n0

    def test_temperature_decreases_n(self):
        """Above ~4 C, higher temperature decreases density and thus n.
        At 35 PSU, 1 atm: n(10 C) - n(25 C) ~ 0.0015."""
        n10 = refractive_index(10.0, 35.0, 0.0, 589.26)
        n25 = refractive_index(25.0, 35.0, 0.0, 589.26)
        assert n25 < n10

    def test_normal_dispersion(self):
        """Shorter wavelength -> higher n (normal dispersion in the visible).
        Consistent with Cauchy/Sellmeier theory for transparent liquids."""
        n500 = refractive_index(20.0, 35.0, 0.0, 500.0)
        n700 = refractive_index(20.0, 35.0, 0.0, 700.0)
        assert n500 > n700

    def test_pure_water_sodium_D(self):
        """Pure water at 20 C, sodium D: n ~ 1.333 (Tilton & Taylor 1938)."""
        n = refractive_index(20.0, 0.0, 0.0, 589.26)
        assert n == pytest.approx(1.333, abs=0.001)

    def test_seawater_sodium_D(self):
        """Standard seawater at 20 C, sodium D: n ~ 1.339 (Mehu & Johannin-Gilles 1968)."""
        n = refractive_index(20.0, 35.0, 0.0, 589.26)
        assert n == pytest.approx(1.339, abs=0.001)

    def test_deep_ocean(self):
        """At 10000 dbar (full ocean depth), 0 C, S=0: n ~ 1.349.
        The ~1.5% increase from surface is due to ~5% density increase."""
        n = refractive_index(0.0, 0.0, 10000.0, 589.26)
        assert n == pytest.approx(1.349, abs=0.001)
