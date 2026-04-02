"""
ITU-R P.453-14 radio refractivity model.

Computes the radio refractivity N = (n - 1) x 10^6 of moist air using the
standard three-term Smith-Weintraub formula with updated coefficients from
Rueger 2002 ("best average" values):

    N = k1 * Pd/T + k2 * e/T + k3 * e/T^2

where:
    Pd = partial pressure of dry air (hPa)
    e  = partial pressure of water vapor (hPa)
    T  = temperature (K)

The first term represents the non-polar (induced dipole) contribution of
dry gases.  The second and third terms represent the non-polar and polar
(permanent dipole) contributions of water vapor, respectively.

Unlike optical refractivity, radio N is independent of wavelength (valid
assumption for frequencies below ~1000 GHz).

References
----------
.. [1] ITU-R P.453-14 (2019). "The radio refractive index: its formula
       and refractivity data."
.. [2] Rueger, J.M. (2002). "Refractive index formulae for radio waves."
       FIG XXII International Congress, Washington D.C.
"""

from .water_vapor import water_vapor_mole_fraction

# ---------------------------------------------------------------------------
# Rueger 2002 "best average" coefficients
# ---------------------------------------------------------------------------
# These are the coefficients recommended by ITU-R P.453-14, superseding the
# older Smith-Weintraub (1953) values of k1=77.6, k2=72, k3=3.75e5.
k1 = 77.6890    # K hPa^-1  (dry term)
k2 = 71.2952    # K hPa^-1  (water vapor density term)
k3 = 375463.0   # K^2 hPa^-1  (water vapor dipolar term)


def itu_N(T_K, Pd_hPa, e_hPa):
    """
    Radio refractivity N from thermodynamic variables.

    Implements the ITU-R P.453-14 three-term formula:

        N = k1 * Pd/T + k2 * e/T + k3 * e/T^2

    Parameters
    ----------
    T_K : float
        Temperature in kelvin.
    Pd_hPa : float
        Partial pressure of dry air in hectopascals.
    e_hPa : float
        Partial pressure of water vapor in hectopascals.

    Returns
    -------
    float
        Radio refractivity N = (n - 1) x 10^6 (dimensionless N-units).
        Typical sea-level values: ~320 N-units (temperate, moderate
        humidity) to ~250 N-units (warm, dry).  Note: N increases
        with decreasing T for dry air (k1 * Pd / T) and increases
        strongly with humidity via the k3 dipolar term.
    """
    return k1 * Pd_hPa / T_K + k2 * e_hPa / T_K + k3 * e_hPa / (T_K * T_K)


def itu_N_from_surface(T_C, P_kPa, RH):
    """
    Radio refractivity N from surface meteorological observations.

    Convenience wrapper that converts surface observations to the partial
    pressures required by :func:`itu_N`.

    Parameters
    ----------
    T_C : float
        Temperature in degrees Celsius.
    P_kPa : float
        Total atmospheric pressure in kilopascals.
    RH : float
        Fractional relative humidity, 0.0 to 1.0.

    Returns
    -------
    float
        Radio refractivity N = (n - 1) x 10^6 in N-units.
    """
    T_K = T_C + 273.15
    P_Pa = P_kPa * 1000.0
    P_hPa = P_kPa * 10.0

    x_w = water_vapor_mole_fraction(T_C, P_Pa, RH)
    e_hPa = x_w * P_hPa
    Pd_hPa = P_hPa - e_hPa

    return itu_N(T_K, Pd_hPa, e_hPa)
