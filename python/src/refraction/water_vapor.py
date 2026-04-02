"""
Water vapor thermodynamic functions for refractive index calculations.

Provides saturation vapor pressure (SVP), enhancement factor, mole fraction,
number density, and vertical profile functions used by the Ciddor 1996
optical model and the ITU-R radio model.

The SVP and enhancement factor formulas follow the BIPM formulation
(Giacomo 1982 [1], Davis 1992 [2]), which is what Ciddor adopted for
his density equation (Eq. 4).

References
----------
.. [1] Giacomo, P. (1982). "Equation for the determination of the density
       of moist air (1981)." Metrologia, 18, 33-40.
.. [2] Davis, R.S. (1992). "Equation for the determination of the density
       of moist air (1981/91)." Metrologia, 29, 67-70.
.. [3] Ciddor, P.E. (1996). "Refractive index of air: new equations for
       the visible and near infrared." Applied Optics, 35(9), 1566-1573.
"""

import math

import numpy as np

from .constants import kB


def svp_giacomo(T_K):
    """
    Saturation vapor pressure of water over liquid water.

    Implements the BIPM / Giacomo 1982 formula (Appendix A of Ciddor [3]):

        svp = exp(A*T^2 + B*T + C + D/T)   [Pa]

    Parameters
    ----------
    T_K : float
        Temperature in kelvin.

    Returns
    -------
    float
        Saturation vapor pressure in pascals.

    Notes
    -----
    Valid for temperatures above 0 degC (liquid water).  For temperatures
    below 0 degC (over ice), a different formula applies -- see Ciddor
    Appendix C, Eq. (13): log10(svp) = -2663.5/T + 12.537.  That case
    is not implemented here.
    """
    A = 1.2378847e-5   # K^-2
    B = -1.9121316e-2  # K^-1
    C = 33.93711047     # dimensionless
    D = -6.3431645e3   # K
    return math.exp(A * T_K * T_K + B * T_K + C + D / T_K)


def enhancement_factor(T_C, P_Pa):
    """
    Enhancement factor for water vapor in air.

    Implements the Giacomo 1982 formula (Appendix A of Ciddor [3]):

        f = alpha + beta * p + gamma * t^2

    The enhancement factor accounts for the increase in the effective
    vapor pressure of water when mixed with air, compared to pure water
    vapor.  It is typically close to 1.0 (e.g., ~1.004 at 20 degC, 1 atm).

    Parameters
    ----------
    T_C : float
        Temperature in degrees Celsius.
    P_Pa : float
        Total pressure in pascals.

    Returns
    -------
    float
        Enhancement factor f (dimensionless, slightly above 1.0).
    """
    alpha = 1.00062       # dimensionless
    beta = 3.14e-8        # Pa^-1
    gamma = 5.6e-7        # degC^-2
    return alpha + beta * P_Pa + gamma * T_C * T_C


def water_vapor_mole_fraction(T_C, P_Pa, RH):
    """
    Mole fraction of water vapor in moist air.

    Computes x_w = f * h * svp / p, following the BIPM formulation
    (Ciddor [3], text after Eq. 4).

    Parameters
    ----------
    T_C : float
        Temperature in degrees Celsius.
    P_Pa : float
        Total atmospheric pressure in pascals.
    RH : float
        Fractional relative humidity, 0.0 (dry) to 1.0 (saturated).

    Returns
    -------
    float
        Water vapor mole fraction x_w (dimensionless).  At 20 degC,
        101325 Pa, 50% RH, x_w ~ 0.012.
    """
    T_K = T_C + 273.15
    svp = svp_giacomo(T_K)
    f = enhancement_factor(T_C, P_Pa)
    return f * RH * svp / P_Pa


def water_vapor_number_density(T_K, P_Pa, RH):
    """
    Number density of water vapor molecules.

    Computes N_w = e / (k_B * T) where e = x_w * p is the partial
    pressure of water vapor.

    Parameters
    ----------
    T_K : float
        Temperature in kelvin.
    P_Pa : float
        Total atmospheric pressure in pascals.
    RH : float
        Fractional relative humidity, 0.0 to 1.0.

    Returns
    -------
    float
        Water vapor number density in m^-3.
    """
    T_C = T_K - 273.15
    x_w = water_vapor_mole_fraction(T_C, P_Pa, RH)
    e = x_w * P_Pa
    return e / (kB * T_K)


def water_vapor_profile(n_H2O_surface, h_km, h_surface_km, H_w=2.0):
    """
    Exponential decay profile for water vapor number density vs altitude.

    Models the rapid decrease of water vapor with altitude using a simple
    exponential scale height.  Below the surface altitude, the surface
    value is returned (no extrapolation underground).

    Parameters
    ----------
    n_H2O_surface : float
        Water vapor number density at the surface in m^-3.
    h_km : float or array_like
        Altitude(s) in kilometers at which to evaluate the profile.
    h_surface_km : float
        Surface altitude in kilometers (profile anchor point).
    H_w : float, optional
        Water vapor scale height in kilometers.  Default is 2.0 km,
        a typical tropospheric value.

    Returns
    -------
    numpy.ndarray
        Water vapor number density in m^-3 at each requested altitude.
    """
    h_km = np.asarray(h_km, dtype=float)
    dh = h_km - h_surface_km
    return np.where(dh < 0, n_H2O_surface,
                    n_H2O_surface * np.exp(-np.maximum(dh, 0.0) / H_w))
