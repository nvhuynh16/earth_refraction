"""Tests for the IAPWS R9-97 pure water refractive index.

Validates against:
    - IAPWS published check value [1]: n = 1.33285 at T = 298.15 K,
      rho = 997.047 kg/m^3, lambda = 0.5893 um.
    - Kell (1975) [2] density of pure water at 4 C and 20 C.

References
----------
.. [1] IAPWS (1997). "Release on the Refractive Index of Ordinary Water
       Substance as a Function of Wavelength, Temperature, and Pressure."
.. [2] Kell, G.S. (1975). J. Chem. Eng. Data, 20(1), 97-105.
"""

import pytest
from refraction.ocean.iapws_r9_97 import refractive_index, pure_water_density


def test_iapws_check_value_589nm_25C():
    """Official IAPWS Table 3 [1]: T=25 C (0.1 MPa ≈ 1 atm), lambda=589 nm.
    Table 3 value at 0 C, 0.1 MPa: n = 1.334344.  At 25 C (interpolated
    from Table 3 check values), the published single-point check gives
    n = 1.33285 at rho = 997.047 kg/m^3."""
    n = refractive_index(25.0, 589.3, rho_kg_m3=997.047)
    assert n == pytest.approx(1.33285, abs=0.00002)


def test_iapws_table3_589nm_0C():
    """IAPWS Table 3 [1]: lambda=589.0 nm, T=0 C, P=0.1 MPa -> n=1.334344."""
    # rho(0 C, 1 atm) = 999.84 kg/m^3
    n = refractive_index(0.0, 589.0, rho_kg_m3=999.84)
    assert n == pytest.approx(1.334344, abs=0.0005)


def test_iapws_table3_226nm_0C():
    """IAPWS Table 3 [1]: lambda=226.5 nm (UV), T=0 C, P=0.1 MPa -> n=1.394527."""
    n = refractive_index(0.0, 226.5, rho_kg_m3=999.84)
    assert n == pytest.approx(1.394527, abs=0.001)


def test_iapws_table3_1014nm_0C():
    """IAPWS Table 3 [1]: lambda=1013.98 nm (near-IR), T=0 C, P=0.1 MPa -> n=1.326135."""
    n = refractive_index(0.0, 1013.98, rho_kg_m3=999.84)
    assert n == pytest.approx(1.326135, abs=0.0005)


def test_pure_water_20C_589nm():
    """IAPWS formula at 20 C, 1 atm, 589 nm: n ~ 1.3333."""
    n = refractive_index(20.0, 589.3)
    assert n == pytest.approx(1.3333, abs=0.001)


def test_pressure_increases_n():
    """Higher pressure compresses water, increasing density and thus n."""
    n_atm = refractive_index(20.0, 589.3, p_dbar=0.0)
    n_deep = refractive_index(20.0, 589.3, p_dbar=5000.0)
    assert n_deep > n_atm


def test_pure_water_0C():
    """At 0 C, n should be higher than at 20 C (dn/dT < 0 for water)."""
    n = refractive_index(0.0, 589.3)
    assert n > refractive_index(20.0, 589.3)


def test_temperature_decreases_n():
    """dn/dT < 0 for liquid water above ~4 C."""
    assert refractive_index(25.0, 589.3) < refractive_index(5.0, 589.3)


def test_normal_dispersion():
    """Shorter wavelength -> higher n in the visible band."""
    assert refractive_index(20.0, 450.0) > refractive_index(20.0, 650.0)


def test_density_4C():
    """Kell [2]: rho(4 C) = 999.972 kg/m^3 (density maximum of water)."""
    rho = pure_water_density(4.0)
    assert rho == pytest.approx(1000.0, abs=0.1)


def test_density_20C():
    """Kell [2]: rho(20 C) = 998.204 kg/m^3."""
    rho = pure_water_density(20.0)
    assert rho == pytest.approx(998.2, abs=0.5)
