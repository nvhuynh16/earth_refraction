"""Tests for species-specific refractivity K functions."""

import pytest
from refraction import ciddor, species
from refraction.itu_r_p453 import itu_N, k2, k3


def test_optical_ciddor_crosscheck():
    sigma2 = 1.0 / (633.0 * 1e-3) / (633.0 * 1e-3)
    n_axs = ciddor.n_standard_air(sigma2)

    KB = 1.380649e-23
    n_total = 101325.0 / (KB * 288.15)
    n_N2 = species.f_N2 * n_total
    n_O2 = species.f_O2 * n_total
    n_Ar = species.f_Ar * n_total

    refractivity = (n_N2 * species.K_N2_optical(sigma2)
                    + n_O2 * species.K_O2_optical(sigma2)
                    + n_Ar * species.K_Ar_optical(sigma2))

    assert refractivity == pytest.approx(n_axs, rel=0.01)


def test_optical_species_sum_reproduces_ciddor_n():
    sigma2 = 1.0 / (633.0 * 1e-3) / (633.0 * 1e-3)
    n_ciddor = ciddor.ciddor_n(20.0, 101.325, 0.0, 633.0)
    refractivity_ciddor = n_ciddor - 1.0

    KB = 1.380649e-23
    T_K = 293.15
    n_total = 101325.0 / (KB * T_K)
    n_N2 = species.f_N2 * n_total
    n_O2 = species.f_O2 * n_total
    n_Ar = species.f_Ar * n_total

    refractivity = (n_N2 * species.K_N2_optical(sigma2)
                    + n_O2 * species.K_O2_optical(sigma2)
                    + n_Ar * species.K_Ar_optical(sigma2))

    assert refractivity == pytest.approx(refractivity_ciddor, rel=0.005)


def test_radio_dry_crosscheck():
    T_K = 288.15
    P_hPa = 1013.25
    N_itu = itu_N(T_K, P_hPa, 0.0)

    KB = 1.380649e-23
    n_total = (P_hPa * 100.0) / (KB * T_K)
    n_N2 = species.f_N2 * n_total
    n_O2 = species.f_O2 * n_total
    n_Ar = species.f_Ar * n_total

    N_species = (n_N2 + n_O2 + n_Ar) * species.K_radio_dry
    assert N_species == pytest.approx(N_itu, rel=0.01)


def test_radio_humid_crosscheck():
    T_K = 293.15
    e_hPa = 23.39

    N_wv_itu = k2 * e_hPa / T_K + k3 * e_hPa / (T_K * T_K)

    KB = 1.380649e-23
    e_Pa = e_hPa * 100.0
    n_w = e_Pa / (KB * T_K)
    N_wv_species = n_w * species.K_H2O_radio(T_K)
    assert N_wv_species == pytest.approx(N_wv_itu, rel=0.001)


def test_K_values_positive():
    sigma2 = 2.496
    assert species.K_N2_optical(sigma2) > 0
    assert species.K_O2_optical(sigma2) > 0
    assert species.K_Ar_optical(sigma2) > 0
    assert species.K_H2O_optical(sigma2) > 0
    assert species.K_radio_dry > 0
    assert species.K_H2O_radio(293.15) > 0


# ---- New tests ----


def test_K_H2O_radio_dipolar_temperature():
    """K_H2O_radio_dipolar decreases with T (1/T dependence)."""
    K_250 = species.K_H2O_radio_dipolar(250.0)
    K_300 = species.K_H2O_radio_dipolar(300.0)
    K_350 = species.K_H2O_radio_dipolar(350.0)
    assert K_250 > K_300 > K_350 > 0.0


def test_K_H2O_radio_is_sum():
    T_K = 300.0
    total = species.K_H2O_radio(T_K)
    parts = species.K_H2O_radio_density + species.K_H2O_radio_dipolar(T_K)
    assert total == pytest.approx(parts, rel=1e-14)


def test_optical_K_wavelength_dependence():
    """Dispersion: K at shorter wavelength > K at longer wavelength."""
    sigma2_400 = 1.0 / (0.4 ** 2)  # 400 nm
    sigma2_1000 = 1.0 / (1.0 ** 2)  # 1000 nm
    assert species.K_N2_optical(sigma2_400) > species.K_N2_optical(sigma2_1000)


def test_K_NO_optical_positive():
    sigma2 = 2.496
    K = species.K_NO_optical(sigma2)
    assert K > 0.0


def test_trace_species_smaller_than_N2():
    """Minor species have smaller polarizabilities than N2."""
    sigma2 = 2.496
    K_N2 = species.K_N2_optical(sigma2)
    assert species.K_He_optical(sigma2) < K_N2
    assert species.K_H_optical(sigma2) < K_N2
    assert species.K_O_optical(sigma2) < K_N2
    assert species.K_N_optical(sigma2) < K_N2
