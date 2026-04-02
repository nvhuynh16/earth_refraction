"""
Refractive index of seawater over temperature, pressure, salinity, and wavelength.

Implements the 27-term polynomial algorithm of Millard & Seaver (1990) [1]:

    N(T, p, S, lambda) = N_I(T, lambda) + N_II(T, lambda, S)
                       + N_III(p, T, lambda) + N_IV(S, p, T)       [Eq. 5]

The four regions are fitted independently to different experimental datasets:
    - Region I:   Tilton & Taylor (1938) pure water at 1 atm (248 points)
    - Region II:  Mehu & Johannin-Gilles (1968) seawater at 1 atm (48 points)
    - Region III: Waxler et al. (1964) pure water at elevated P (100 points)
    - Region IV:  Stanley (1970) seawater at elevated P (32 points)

Valid ranges (from [1], p. 1920):
    Temperature:   0-30 deg C
    Salinity:      0-43 PSU
    Wavelength:    500-700 nm
    Pressure:      0-11,000 dbar
    Accuracy:      0.4 ppm (Region I) to 80 ppm (Region III/IV)

References
----------
.. [1] Millard, R.C. and G. Seaver (1990). "An index of refraction algorithm
       for seawater over temperature, pressure, salinity, density, and
       wavelength." Deep-Sea Research, 37(12), 1909-1926.
.. [2] Tilton, L. and J. Taylor (1938). "Refractive index and dispersion of
       distilled water." J. Res. Nat. Bur. Standards, 20, 419-477.
.. [3] Mehu, A. and A. Johannin-Gilles (1968). "Variation de l'indice de
       refraction de l'eau de Mer Etalon..." Cahiers Oceanogr., 20, 803-812.
.. [4] Waxler, R. et al. (1964). "Effect of pressure and temperature upon the
       optical dispersion of benzene, carbon tetrachloride, and water."
       J. Res. Nat. Bur. Standards, 68, 489-498.
.. [5] Stanley, E. (1970). "The refractive index of seawater as a function of
       temperature, pressure, and two wavelengths." Deep-Sea Res., 18, 833-840.

Notes
-----
Coefficients were extracted from [1] Table 1 and validated against [1] Table 2.
Four coefficients (T1, T2L, S1T, PLM2) required zero-count corrections from
the scanned PDF, documented inline.  All 27 coefficients reproduce Table 2 to
within the stated accuracy of each region.

Parameters
----------
All temperatures in degrees Celsius, pressures in dbar, salinities in PSU,
wavelengths in nm (converted internally to micrometers as used in [1]).
"""

# ---------------------------------------------------------------------------
# Region I — Pure water at atmospheric pressure
# ---------------------------------------------------------------------------
# 12 terms, SD = 0.12 ppm for 236 of 248 Tilton-Taylor [2] data points.
#
# N_I = A0 + L2*L^2 + LM2/L^2 + LM4/L^4 + LM6/L^6              [1] Table 1
#     + T1*T + T2*T^2 + T3*T^3 + T4*T^4
#     + TL*T*L + T2L*T^2*L + T3L*T^3*L
#
# L = wavelength in micrometers, T = temperature in deg C.
#
# Dispersion enters only through even powers of wavelength (+2, -2, -4, -6)
# and mixed T*L, T^2*L, T^3*L terms, consistent with the Cauchy/Sellmeier
# dispersion theory for transparent media [1, p. 1917].
# ---------------------------------------------------------------------------

_A0  =  1.3280657          # constant term                        EOV [5460]
_L2  = -4.5536802e-3       # L^2 coefficient (um^2)               EOV [ 274]
_LM2 =  2.5471707e-3       # 1/L^2 coefficient (um^-2)            EOV [ 194]
_LM4 =  7.501966e-6        # 1/L^4 coefficient (um^-4)            EOV [   2.4]
_LM6 =  2.802632e-6        # 1/L^6 coefficient (um^-6)            EOV [  10.5]
_T1  = -5.2883907e-6       # T coefficient (degC^-1)              EOV [  62.8]
# PDF correction: scanned as -.000052883907 (4 zeros) but validated
# against Table 2 as -.0000052883907 (5 zeros) = -5.2883907e-6.
_T2  = -3.0738272e-6       # T^2 coefficient (degC^-2)            EOV [ 451]
_T3  =  3.0124687e-8       # T^3 coefficient (degC^-3)            EOV [ 170]
_T4  = -2.0883178e-10      # T^4 coefficient (degC^-4)            EOV [ 114]
_TL  =  1.0508621e-5       # T*L coefficient (degC^-1 um)         EOV [  77.3]
_T2L =  2.1282248e-7       # T^2*L coefficient (degC^-2 um)       EOV [  19.9]
# PDF correction: scanned as .0000021282248 (5 zeros) but validated
# against Table 2 as .00000021282248 (6 zeros) = 2.1282248e-7.
_T3L = -1.705881e-10       # T^3*L coefficient (degC^-3 um)       EOV [   7.2]


# ---------------------------------------------------------------------------
# Region II — Salinity at atmospheric pressure
# ---------------------------------------------------------------------------
# 6 terms, SD = 4.7 ppm for all 48 Mehu-Gilles [3] data points.
#
# N_II = S0*S + S1LM2*S/L^2 + S1T*S*T + S1T2*S*T^2              [1] Table 1
#      + S1T3*S*T^3 + STL*S*T*L
#
# All terms are linear in S, confirming the Austin & Halikas [1, p. 1911]
# observation of linearity in the index-salinity relationship.
# ---------------------------------------------------------------------------

_S0    =  1.9029121e-4     # S coefficient (PSU^-1)               EOV [ 822]
_S1LM2 =  2.4239607e-6    # S/L^2 coefficient (PSU^-1 um^-2)     EOV [  32.9]
_S1T   = -7.3960297e-7    # S*T coefficient (PSU^-1 degC^-1)      EOV [  23.1]
# PDF correction: scanned as -.0000073960297 (5 zeros) but validated
# against Table 2 as -.00000073960297 (6 zeros) = -7.3960297e-7.
_S1T2  =  8.9818478e-9    # S*T^2 coefficient (PSU^-1 degC^-2)    EOV [   6.30]
_S1T3  =  1.2078804e-10   # S*T^3 coefficient (PSU^-1 degC^-3)    EOV [   3.84]
_STL   = -3.589495e-7     # S*T*L coefficient (PSU^-1 degC^-1 um) EOV [   7.60]


# ---------------------------------------------------------------------------
# Region III — Pressure for pure water
# ---------------------------------------------------------------------------
# 6 terms, SD = 26.5 ppm for 93 of 100 Waxler-Weir [4] data points.
#
# N_III = P1*p + P2*p^2 + PLM2*p/L^2 + PT*p*T + PT2*p*T^2       [1] Table 1
#       + P2T2*p^2*T^2
# ---------------------------------------------------------------------------

_P1   =  1.5868383e-6     # p coefficient (dbar^-1)               EOV [ 556]
_P2   = -1.574074e-11     # p^2 coefficient (dbar^-2)             EOV [  67]
_PLM2 =  1.0712063e-8     # p/L^2 coefficient (dbar^-1 um^-2)     EOV [  17.3]
# PDF correction: scanned as .0000010712063 (5 zeros) but validated
# against Table 2 as .000000010712063 (7 zeros) = 1.0712063e-8.
_PT   = -9.4834486e-9     # p*T coefficient (dbar^-1 degC^-1)     EOV [  38.3]
_PT2  =  1.0100326e-10    # p*T^2 coefficient (dbar^-1 degC^-2)   EOV [   8.56]
_P2T2 =  5.8085198e-15    # p^2*T^2 coefficient (dbar^-2 degC^-2) EOV [   8.56]


# ---------------------------------------------------------------------------
# Region IV — Pressure x salinity
# ---------------------------------------------------------------------------
# 3 terms, SD = 19.7 ppm for all 32 Stanley [5] data points.
#
# N_IV = P1S*p*S + PTS*p*T*S + PT2S*p*T^2*S                     [1] Table 1
# ---------------------------------------------------------------------------

_P1S  = -1.1177517e-9     # p*S coefficient (dbar^-1 PSU^-1)          EOV [ 37.2]
_PTS  =  5.7311268e-11    # p*T*S coefficient (dbar^-1 degC^-1 PSU^-1) EOV [  9.48]
_PT2S = -1.5460458e-12    # p*T^2*S coefficient (dbar^-1 degC^-2 PSU^-1) EOV [6.18]


# ---------------------------------------------------------------------------
# Region evaluation functions
# ---------------------------------------------------------------------------

def _n_region_I(T: float, L: float) -> float:
    """Pure water at atmospheric pressure (12 terms).  [1] Table 1, Region I."""
    L2 = L * L
    return (_A0
            + _L2 * L2
            + _LM2 / L2
            + _LM4 / (L2 * L2)
            + _LM6 / (L2 * L2 * L2)
            + _T1 * T
            + _T2 * T * T
            + _T3 * T * T * T
            + _T4 * T * T * T * T
            + _TL * T * L
            + _T2L * T * T * L
            + _T3L * T * T * T * L)


def _n_region_II(T: float, L: float, S: float) -> float:
    """Salinity correction at atmospheric pressure (6 terms).  [1] Table 1, Region II."""
    L2 = L * L
    return (_S0 * S
            + _S1LM2 * S / L2
            + _S1T * S * T
            + _S1T2 * S * T * T
            + _S1T3 * S * T * T * T
            + _STL * S * T * L)


def _n_region_III(p: float, T: float, L: float) -> float:
    """Pressure correction for pure water (6 terms).  [1] Table 1, Region III."""
    L2 = L * L
    return (_P1 * p
            + _P2 * p * p
            + _PLM2 * p / L2
            + _PT * p * T
            + _PT2 * p * T * T
            + _P2T2 * p * p * T * T)


def _n_region_IV(S: float, p: float, T: float) -> float:
    """Pressure-salinity cross terms (3 terms).  [1] Table 1, Region IV."""
    return (_P1S * p * S
            + _PTS * p * T * S
            + _PT2S * p * T * T * S)


def refractive_index(T_C: float, S_psu: float, p_dbar: float,
                     wavelength_nm: float) -> float:
    """Refractive index of seawater.

    Evaluates the 27-term polynomial [1] Eq. 5:

        N = N_I(T, lambda) + N_II(T, lambda, S) + N_III(p, T, lambda)
          + N_IV(S, p, T)

    Parameters
    ----------
    T_C : float
        Temperature in degrees Celsius (valid 0-30).
    S_psu : float
        Practical salinity in PSU (valid 0-43).
    p_dbar : float
        Sea pressure in dbar (valid 0-11,000).
    wavelength_nm : float
        Vacuum wavelength in nm (valid 500-700).

    Returns
    -------
    float
        Refractive index of seawater (dimensionless).
        Typical values: 1.333 (pure water, 1 atm) to 1.356 (40 PSU, 10000 dbar).

    Notes
    -----
    Wavelength is converted internally to micrometers as used in [1].
    The algorithm accuracy varies by region: 0.4 ppm for pure water at 1 atm
    (Region I) degrading to ~80 ppm at high pressure (Regions III/IV).
    """
    L = wavelength_nm * 1.0e-3   # nm -> micrometers
    T = T_C
    S = S_psu
    p = p_dbar
    return (_n_region_I(T, L)
            + _n_region_II(T, L, S)
            + _n_region_III(p, T, L)
            + _n_region_IV(S, p, T))
