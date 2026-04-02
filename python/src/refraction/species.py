"""
Species-specific refractivity K-factors for optical and radio modes.

Each atmospheric species contributes to the total refractivity via:

    n - 1 = sum_i( n_i * K_i(lambda) )

where n_i is the number density (m^-3) of species i and K_i is the
per-molecule refractivity contribution.  This module computes K_i for
each species in two modes:

**Optical (Ciddor-based)**:

    For major dry-air species (N2, O2, Ar), K_i is derived by partitioning
    the Ciddor standard-air refractivity proportionally to each species'
    molecular polarizability:

        K_i(sigma^2) = n_standard_air(sigma^2) * (alpha_i / alpha_air) / N_std

    where alpha_i is the static polarizability, alpha_air is the density-
    weighted average over N2/O2/Ar, and N_std is the number density at
    Ciddor standard conditions (15 degC, 101325 Pa).

    For minor species (He, H, O, N, NO), K is scaled from the N2 value
    by the polarizability ratio, since no species-specific dispersion
    data are available.

    For water vapor, K_H2O uses the Ciddor water-vapor dispersion
    equation (Eq. 3) referenced to the water-vapor standard density
    at 20 degC, 1333 Pa.

**Radio (ITU-R P.453-based)**:

    K_radio_dry = k1 * k_B * 0.01  (same for all dry species)
    K_H2O_radio = K_density + K_dipolar(T)

    The dipolar (k3) term is temperature-dependent because the permanent
    dipole contribution to refractivity scales as 1/T.

References
----------
.. [1] Ciddor 1996 (see ciddor.py docstring for full citation).
.. [2] ITU-R P.453-14 (see itu_r_p453.py docstring).
"""

from . import ciddor
from . import itu_r_p453 as itu
from .constants import kB, HPA_TO_PA

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
# Loschmidt number: ideal-gas number density at STP (0 degC, 101325 Pa).
# N_L = P / (k_B * T) = 101325 / (1.380649e-23 * 273.15)
N_L = 2.6867774e25  # m^-3

# Number density at Ciddor standard conditions (15 degC, 101325 Pa).
# N_std = P / (k_B * T) = 101325 / (1.380649e-23 * 288.15)
N_std = 2.54708e25  # m^-3

# ---------------------------------------------------------------------------
# Molecular polarizabilities
# ---------------------------------------------------------------------------
# Static polarizabilities in atomic units (a0^3), where a0 is the Bohr
# radius.  These determine how each species' refractivity relates to the
# bulk dry-air value.
a0_cubed = 1.6488e-41  # m^3, conversion factor (a0^3 -> SI, not used here)

alpha_N2 = 11.74   # a0^3
alpha_O2 = 10.67   # a0^3
alpha_Ar = 11.08   # a0^3
alpha_He = 1.38    # a0^3
alpha_H = 4.50     # a0^3
alpha_O = 5.40     # a0^3
alpha_N = 7.40     # a0^3
alpha_NO = 11.50   # a0^3

# Standard dry-air mole fractions (used to compute the weighted average
# polarizability of dry air for partitioning the bulk Ciddor refractivity).
f_N2 = 0.78084
f_O2 = 0.20946
f_Ar = 0.00934

# Mole-fraction-weighted average polarizability of dry air.
# alpha_air = sum(f_i * alpha_i) for the three major species.
alpha_air = f_N2 * alpha_N2 + f_O2 * alpha_O2 + f_Ar * alpha_Ar

# Water vapor reference number density at Ciddor reference conditions
# (20 degC, 1333 Pa).  Idealized: P/(k_B*T) = 3.2935e23; this value
# includes a small compressibility correction.
N_ws_ref = 3.2921e23  # m^-3

# kB, HPA_TO_PA imported from constants module

# ---------------------------------------------------------------------------
# Radio K-factors (wavelength-independent)
# ---------------------------------------------------------------------------
# K_radio_dry: per-molecule refractivity for any dry species, derived from
# ITU k1.  The factor 0.01 converts hPa -> Pa in N = k1*Pd/T, since
# Pd = n * kB * T gives K = k1 * kB * HPA_TO_PA.
K_radio_dry = itu.k1 * kB * HPA_TO_PA

# K_H2O_radio_density: the k2 (non-polar) water-vapor contribution,
# temperature-independent portion.
K_H2O_radio_density = itu.k2 * kB * HPA_TO_PA


# ---------------------------------------------------------------------------
# Optical K-factors (wavelength-dependent via sigma^2)
# ---------------------------------------------------------------------------
def K_N2_optical(sigma2):
    """
    Per-molecule optical refractivity of N2.

    Parameters
    ----------
    sigma2 : float
        Vacuum wavenumber squared (um^-2).

    Returns
    -------
    float
        K_N2 in m^3 (refractivity contribution per molecule).
    """
    return ciddor.n_standard_air(sigma2) * (alpha_N2 / alpha_air) / N_std


def K_O2_optical(sigma2):
    """
    Per-molecule optical refractivity of O2.

    Parameters
    ----------
    sigma2 : float
        Vacuum wavenumber squared (um^-2).

    Returns
    -------
    float
        K_O2 in m^3.
    """
    return ciddor.n_standard_air(sigma2) * (alpha_O2 / alpha_air) / N_std


def K_Ar_optical(sigma2):
    """
    Per-molecule optical refractivity of Ar.

    Parameters
    ----------
    sigma2 : float
        Vacuum wavenumber squared (um^-2).

    Returns
    -------
    float
        K_Ar in m^3.
    """
    return ciddor.n_standard_air(sigma2) * (alpha_Ar / alpha_air) / N_std


def K_He_optical(sigma2):
    """
    Per-molecule optical refractivity of He.

    Scaled from N2 by the polarizability ratio alpha_He / alpha_N2.
    No species-specific dispersion data available.

    Parameters
    ----------
    sigma2 : float
        Vacuum wavenumber squared (um^-2).

    Returns
    -------
    float
        K_He in m^3.
    """
    return K_N2_optical(sigma2) * (alpha_He / alpha_N2)


def K_H_optical(sigma2):
    """
    Per-molecule optical refractivity of atomic hydrogen.

    Scaled from N2 by the polarizability ratio alpha_H / alpha_N2.

    Parameters
    ----------
    sigma2 : float
        Vacuum wavenumber squared (um^-2).

    Returns
    -------
    float
        K_H in m^3.
    """
    return K_N2_optical(sigma2) * (alpha_H / alpha_N2)


def K_O_optical(sigma2):
    """
    Per-molecule optical refractivity of atomic oxygen.

    Scaled from N2 by the polarizability ratio alpha_O / alpha_N2.

    Parameters
    ----------
    sigma2 : float
        Vacuum wavenumber squared (um^-2).

    Returns
    -------
    float
        K_O in m^3.
    """
    return K_N2_optical(sigma2) * (alpha_O / alpha_N2)


def K_N_optical(sigma2):
    """
    Per-molecule optical refractivity of atomic nitrogen.

    Scaled from N2 by the polarizability ratio alpha_N / alpha_N2.

    Parameters
    ----------
    sigma2 : float
        Vacuum wavenumber squared (um^-2).

    Returns
    -------
    float
        K_N in m^3.
    """
    return K_N2_optical(sigma2) * (alpha_N / alpha_N2)


def K_NO_optical(sigma2):
    """
    Per-molecule optical refractivity of nitric oxide.

    Scaled from N2 by the polarizability ratio alpha_NO / alpha_N2.

    Parameters
    ----------
    sigma2 : float
        Vacuum wavenumber squared (um^-2).

    Returns
    -------
    float
        K_NO in m^3.
    """
    return K_N2_optical(sigma2) * (alpha_NO / alpha_N2)


def K_H2O_optical(sigma2):
    """
    Per-molecule optical refractivity of water vapor.

    Uses the Ciddor water-vapor dispersion equation (Eq. 3) normalized
    by the reference number density at 20 degC, 1333 Pa.

    Parameters
    ----------
    sigma2 : float
        Vacuum wavenumber squared (um^-2).

    Returns
    -------
    float
        K_H2O in m^3.
    """
    return ciddor.n_water_vapor(sigma2) / N_ws_ref


def K_H2O_radio_dipolar(T_K):
    """
    Temperature-dependent dipolar radio refractivity of water vapor.

    This is the k3 * e / T^2 term from ITU-R P.453, converted to
    a per-molecule K-factor:  K = k3 * kB * HPA_TO_PA / T.

    Parameters
    ----------
    T_K : float
        Temperature in kelvin.

    Returns
    -------
    float
        K_H2O_dipolar in m^3 (the 1/T is already evaluated).
    """
    return itu.k3 * kB * HPA_TO_PA / T_K


def K_H2O_radio(T_K):
    """
    Total per-molecule radio refractivity of water vapor.

    Sum of the density (k2) and dipolar (k3/T) contributions.

    Parameters
    ----------
    T_K : float
        Temperature in kelvin.

    Returns
    -------
    float
        Total K_H2O_radio in m^3.
    """
    return K_H2O_radio_density + K_H2O_radio_dipolar(T_K)
