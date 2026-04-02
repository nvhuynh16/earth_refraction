"""
Sound speed in seawater — Mackenzie (1981).

Implements the Mackenzie (1981) [1] nine-term equation for the speed of
sound in the oceans:

    c = 1448.96 + 4.591*T - 5.304e-2*T^2 + 2.374e-4*T^3    [1] Eq. 1
      + 1.340*(S - 35) + 1.630e-2*Z + 1.675e-7*Z^2
      - 1.025e-2*T*(S - 35) - 7.139e-13*T*Z^3

Uses depth (meters) directly rather than pressure, making it convenient
for field use when depth is known.  The equation was fitted to the
Del Grosso (1974) equation over the stated range.

Valid ranges:
    Temperature:  2-30 deg C
    Salinity:     25-40 ppt (PSU)
    Depth:        0-8000 m
    Accuracy:     +/- 0.070 m/s

At depths > 8000 m accuracy degrades: +1.4 m/s at 10,000 m.

References
----------
.. [1] Mackenzie, K.V. (1981). "Nine-term equation for the sound speed in
       the oceans." J. Acoust. Soc. Am., 70(3), 807-812.
"""

# ---------------------------------------------------------------------------
# Coefficients — Mackenzie (1981) [1] Eq. 1
# ---------------------------------------------------------------------------
_C0    = 1448.96         # base speed (m/s)              [1] Eq. 1
_CT1   =    4.591        # T     (m/s/degC)             [1] Eq. 1
_CT2   =   -5.304e-2     # T^2   (m/s/degC^2)           [1] Eq. 1
_CT3   =    2.374e-4     # T^3   (m/s/degC^3)           [1] Eq. 1
_CS    =    1.340        # (S-35) (m/s/ppt)             [1] Eq. 1
_CZ1   =    1.630e-2     # Z     (m/s/m)               [1] Eq. 1
_CZ2   =    1.675e-7     # Z^2   (m/s/m^2)             [1] Eq. 1
_CTS   =   -1.025e-2     # T*(S-35) (m/s/degC/ppt)     [1] Eq. 1
_CTZ3  =   -7.139e-13    # T*Z^3 (m/s/degC/m^3)        [1] Eq. 1


def sound_speed(T_C: float, S_ppt: float, depth_m: float) -> float:
    """Sound speed of seawater via Mackenzie (1981) [1].

    Parameters
    ----------
    T_C : float
        Temperature in degrees Celsius (valid 2-30).
    S_ppt : float
        Practical salinity in ppt / PSU (valid 25-40).
    depth_m : float
        Depth in meters, positive downward (valid 0-8000).

    Returns
    -------
    float
        Sound speed in m/s (typical range 1400-1600).

    Notes
    -----
    This equation uses depth directly — no pressure conversion is needed.
    All 9 coefficients are from [1] Eq. 1.
    """
    T = T_C
    S = S_ppt
    Z = depth_m

    return (_C0
            + _CT1 * T
            + _CT2 * T * T
            + _CT3 * T * T * T
            + _CS  * (S - 35.0)
            + _CZ1 * Z
            + _CZ2 * Z * Z
            + _CTS * T * (S - 35.0)
            + _CTZ3 * T * Z * Z * Z)
