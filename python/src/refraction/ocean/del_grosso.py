"""
Sound speed in seawater — Del Grosso (1974) with ITS-90 coefficients.

Implements the Del Grosso (1974) [1] equation for the speed of sound in
seawater, using the ITS-90 temperature-scale revision by Wong & Zhu (1995) [2]:

    c(S, T, P) = C000 + dC_T(T) + dC_S(S) + dC_P(P) + dC_STP(S, T, P)

where dC_T, dC_S, dC_P are pure temperature, salinity, and pressure
contributions (3, 2, and 3 terms respectively), and dC_STP contains
10 cross-product terms.  Total: 19 coefficients.

Valid ranges (from [2]):
    Temperature:  0-30 deg C
    Salinity:     30-40 ppt (PSU)
    Pressure:     0-1000 kg/cm^2 (~0-9807 dbar)
    Accuracy:     +/- 0.05 m/s

Del Grosso is preferred over Chen-Millero for deep-ocean acoustics:
long-range tomography experiments (Dushaw et al. 1993 [3]) found Del Grosso
agrees with measured travel times to +/- 0.05 m/s, while Chen-Millero
diverges by ~0.5 m/s at depth.

References
----------
.. [1] Del Grosso, V.A. (1974). "New equation for the speed of sound in
       natural waters (with comparisons to other equations)."
       J. Acoust. Soc. Am., 56(4), 1084-1091.
.. [2] Wong, G.S.K. and S. Zhu (1995). "Speed of sound in seawater as a
       function of salinity, temperature and pressure."
       J. Acoust. Soc. Am., 97(3), 1732-1736.
.. [3] Dushaw, B.D. et al. (1993). "On equations for the speed of sound
       in seawater." J. Acoust. Soc. Am., 93(1), 255-275.

Notes
-----
Pressure is kg/cm^2 internally; the public function accepts dbar and converts
via _DBAR_TO_KGCM2.  All 19 coefficients are from [2] Table IV.
"""

# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------
# 1 dbar = 10^4 Pa; 1 kg/cm^2 = 98066.5 Pa  =>  1 dbar = 10^4/98066.5
_DBAR_TO_KGCM2 = 0.101972   # dbar -> kg/cm^2

# ---------------------------------------------------------------------------
# Coefficients — Wong & Zhu (1995) ITS-90 values [2] Table IV
# ---------------------------------------------------------------------------
_C000  = 1402.392            # base speed (m/s)

# Temperature terms
_CT1   =  5.012285           # T     (m/s/degC)           [2] Table IV
_CT2   = -0.0551184          # T^2   (m/s/degC^2)         [2] Table IV
_CT3   =  2.21649e-4         # T^3   (m/s/degC^3)         [2] Table IV

# Salinity terms
_CS1   =  1.329530           # S     (m/s/ppt)            [2] Table IV
_CS2   =  1.288598e-4        # S^2   (m/s/ppt^2)          [2] Table IV

# Pressure terms  (P in kg/cm^2)
_CP1   =  0.1560592          # P     (m/s/(kg/cm^2))      [2] Table IV
_CP2   =  2.449993e-5        # P^2   (m/s/(kg/cm^2)^2)    [2] Table IV
_CP3   = -8.833959e-9        # P^3   (m/s/(kg/cm^2)^3)    [2] Table IV

# Cross-product terms (10 terms)
_CST    = -1.275936e-2       # S*T                        [2] Table IV
_CTP    =  6.353509e-3       # T*P                        [2] Table IV
_CT2P2  =  2.656174e-8       # T^2*P^2                    [2] Table IV
_CTP2   = -1.593895e-6       # T*P^2                      [2] Table IV
_CTP3   =  5.222483e-10      # T*P^3                      [2] Table IV
_CT3P   = -4.383615e-7       # T^3*P                      [2] Table IV
_CS2P2  = -1.616745e-9       # S^2*P^2                    [2] Table IV
_CST2   =  9.688441e-5       # S*T^2                      [2] Table IV
_CS2TP  =  4.857614e-6       # S^2*T*P                    [2] Table IV
_CSTP   = -3.406824e-4       # S*T*P                      [2] Table IV


def sound_speed(T_C: float, S_ppt: float, p_dbar: float) -> float:
    """Sound speed of seawater via Del Grosso (1974) [1].

    Parameters
    ----------
    T_C : float
        Temperature in degrees Celsius (valid 0-30).
    S_ppt : float
        Practical salinity in ppt / PSU (valid 30-40).
    p_dbar : float
        Sea pressure in dbar (valid 0-9807).

    Returns
    -------
    float
        Sound speed in m/s (typical range 1400-1600).

    Notes
    -----
    Coefficients are the ITS-90 revision from [2] Table IV.  Pressure is
    converted internally from dbar to kg/cm^2.

    The equation structure is additive [2] Table IV::

        c = C000 + dC_T(T) + dC_S(S) + dC_P(P) + dC_STP(S, T, P)
    """
    T = T_C
    S = S_ppt
    P = p_dbar * _DBAR_TO_KGCM2   # dbar -> kg/cm^2

    dC_T = _CT1 * T + _CT2 * T * T + _CT3 * T * T * T
    dC_S = _CS1 * S + _CS2 * S * S
    dC_P = _CP1 * P + _CP2 * P * P + _CP3 * P * P * P

    dC_STP = (_CST   * S * T
            + _CTP   * T * P
            + _CT2P2 * T * T * P * P
            + _CTP2  * T * P * P
            + _CTP3  * T * P * P * P
            + _CT3P  * T * T * T * P
            + _CS2P2 * S * S * P * P
            + _CST2  * S * T * T
            + _CS2TP * S * S * T * P
            + _CSTP  * S * T * P)

    return _C000 + dC_T + dC_S + dC_P + dC_STP
