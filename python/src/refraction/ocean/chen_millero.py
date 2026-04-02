"""
Sound speed in seawater — Chen & Millero (1977) with ITS-90 coefficients.

Implements the Chen & Millero (1977) [1] equation for the speed of sound in
seawater, using the ITS-90 temperature-scale revision by Wong & Zhu (1995) [2]:

    c(S, T, P) = Cw(T, P) + A(T, P)*S + B(T, P)*S^(3/2) + D(T, P)*S^2

where Cw is the pure-water sound speed (19 terms), A is the salinity linear
coefficient (17 terms), B is the S^(3/2) coefficient (4 terms), and D is the
S^2 coefficient (2 terms).  Total: 42 coefficients.

This was the UNESCO/IOC standard (Fofonoff & Millard 1983 [3]) until
superseded by TEOS-10 in 2010.  It has wider validity than Del Grosso
(T up to 40 degC, S down to 0 ppt) but is known to overestimate sound
speed at high pressure by ~0.5 m/s.

Valid ranges (from [2]):
    Temperature:  0-40 deg C
    Salinity:     0-40 ppt (PSU)
    Pressure:     0-1000 bar (~0-10000 dbar)
    Accuracy:     +/- 0.05 m/s (surface), ~0.5 m/s at depth

References
----------
.. [1] Chen, C.T. and F.J. Millero (1977). "Speed of sound in seawater at
       high pressures." J. Acoust. Soc. Am., 62(5), 1129-1135.
.. [2] Wong, G.S.K. and S. Zhu (1995). "Speed of sound in seawater as a
       function of salinity, temperature and pressure."
       J. Acoust. Soc. Am., 97(3), 1732-1736.
.. [3] Fofonoff, N.P. and R.C. Millard (1983). "Algorithms for computation
       of fundamental properties of seawater." UNESCO Tech. Paper 44.

Notes
-----
Pressure is bar internally; the public function accepts dbar and converts
via _DBAR_TO_BAR.  All 42 coefficients are from [2] Table III.
"""

import math

# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------
_DBAR_TO_BAR = 0.1   # 1 dbar = 0.1 bar

# ---------------------------------------------------------------------------
# Cw(T, P) — pure water sound speed [2] Table III
# ---------------------------------------------------------------------------
# Cw = sum_{i=0}^{3} sum_{j=0}^{Ni} Cij * P^i * T^j
# 19 coefficients total.

# P^0 terms (atmospheric)
_C00 =  1402.388        # (m/s)                   [2] Table III
_C01 =  5.03830         # T    (m/s/degC)         [2] Table III
_C02 = -5.81090e-2      # T^2  (m/s/degC^2)       [2] Table III
_C03 =  3.3432e-4       # T^3  (m/s/degC^3)       [2] Table III
_C04 = -1.47797e-6      # T^4  (m/s/degC^4)       [2] Table III
_C05 =  3.1419e-9       # T^5  (m/s/degC^5)       [2] Table III

# P^1 terms
_C10 =  0.153563        # P    (m/s/bar)          [2] Table III
_C11 =  6.8999e-4       # P*T                     [2] Table III
_C12 = -8.1829e-6       # P*T^2                   [2] Table III
_C13 =  1.3632e-7       # P*T^3                   [2] Table III
_C14 = -6.1260e-10      # P*T^4                   [2] Table III

# P^2 terms
_C20 =  3.1260e-5       # P^2                     [2] Table III
_C21 = -1.7111e-6       # P^2*T                   [2] Table III
_C22 =  2.5986e-8       # P^2*T^2                 [2] Table III
_C23 = -2.5353e-10      # P^2*T^3                 [2] Table III
_C24 =  1.0415e-12      # P^2*T^4                 [2] Table III

# P^3 terms
_C30 = -9.7729e-9       # P^3                     [2] Table III
_C31 =  3.8513e-10      # P^3*T                   [2] Table III
_C32 = -2.3654e-12      # P^3*T^2                 [2] Table III


# ---------------------------------------------------------------------------
# A(T, P) — salinity linear coefficient [2] Table III
# ---------------------------------------------------------------------------
# 17 coefficients total.

# P^0 terms
_A00 =  1.389           # (m/s/ppt)               [2] Table III
_A01 = -1.262e-2        # T                        [2] Table III
_A02 =  7.166e-5        # T^2                      [2] Table III
_A03 =  2.008e-6        # T^3                      [2] Table III
_A04 = -3.21e-8         # T^4                      [2] Table III

# P^1 terms
_A10 =  9.4742e-5       # P                        [2] Table III
_A11 = -1.2583e-5       # P*T                      [2] Table III
_A12 = -6.4928e-8       # P*T^2                    [2] Table III
_A13 =  1.0515e-8       # P*T^3                    [2] Table III
_A14 = -2.0142e-10      # P*T^4                    [2] Table III

# P^2 terms
_A20 = -3.9064e-7       # P^2                      [2] Table III
_A21 =  9.1061e-9       # P^2*T                    [2] Table III
_A22 = -1.6009e-10      # P^2*T^2                  [2] Table III
_A23 =  7.994e-12       # P^2*T^3                  [2] Table III

# P^3 terms
_A30 =  1.100e-10       # P^3                      [2] Table III
_A31 =  6.651e-12       # P^3*T                    [2] Table III
_A32 = -3.391e-13       # P^3*T^2                  [2] Table III


# ---------------------------------------------------------------------------
# B(T, P) — salinity S^(3/2) coefficient [2] Table III
# ---------------------------------------------------------------------------
_B00 = -1.922e-2        # (m/s/ppt^1.5)            [2] Table III
_B01 = -4.42e-5         # T                        [2] Table III
_B10 =  7.3637e-5       # P                        [2] Table III
_B11 =  1.7950e-7       # P*T                      [2] Table III


# ---------------------------------------------------------------------------
# D(T, P) — salinity S^2 coefficient [2] Table III
# ---------------------------------------------------------------------------
_D00 =  1.727e-3        # (m/s/ppt^2)              [2] Table III
_D10 = -7.9836e-6       # P                        [2] Table III


# ---------------------------------------------------------------------------
# Helper polynomials
# ---------------------------------------------------------------------------

def _c_w(T: float, P: float) -> float:
    """Pure water sound speed.  [2] Table III."""
    T2, T3, T4, T5 = T*T, T*T*T, T**4, T**5
    P2, P3 = P*P, P*P*P
    return ((_C00 + _C01*T + _C02*T2 + _C03*T3 + _C04*T4 + _C05*T5)
          + (_C10 + _C11*T + _C12*T2 + _C13*T3 + _C14*T4) * P
          + (_C20 + _C21*T + _C22*T2 + _C23*T3 + _C24*T4) * P2
          + (_C30 + _C31*T + _C32*T2) * P3)


def _A_coeff(T: float, P: float) -> float:
    """Salinity linear coefficient.  [2] Table III."""
    T2, T3, T4 = T*T, T*T*T, T**4
    P2, P3 = P*P, P*P*P
    return ((_A00 + _A01*T + _A02*T2 + _A03*T3 + _A04*T4)
          + (_A10 + _A11*T + _A12*T2 + _A13*T3 + _A14*T4) * P
          + (_A20 + _A21*T + _A22*T2 + _A23*T3) * P2
          + (_A30 + _A31*T + _A32*T2) * P3)


def _B_coeff(T: float, P: float) -> float:
    """Salinity S^(3/2) coefficient.  [2] Table III."""
    return (_B00 + _B01*T) + (_B10 + _B11*T) * P


def _D_coeff(T: float, P: float) -> float:
    """Salinity S^2 coefficient.  [2] Table III."""
    return _D00 + _D10 * P


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def sound_speed(T_C: float, S_ppt: float, p_dbar: float) -> float:
    """Sound speed of seawater via Chen & Millero (1977) [1].

    Parameters
    ----------
    T_C : float
        Temperature in degrees Celsius (valid 0-40).
    S_ppt : float
        Practical salinity in ppt / PSU (valid 0-40).
    p_dbar : float
        Sea pressure in dbar (valid 0-10000).

    Returns
    -------
    float
        Sound speed in m/s (typical range 1400-1600).

    Notes
    -----
    Coefficients are the ITS-90 revision from [2] Table III.  Pressure is
    converted internally from dbar to bar.

    The equation structure is [2] Table III::

        c = Cw(T, P) + A(T, P)*S + B(T, P)*S^(3/2) + D(T, P)*S^2
    """
    T = T_C
    S = S_ppt
    P = p_dbar * _DBAR_TO_BAR   # dbar -> bar

    S32 = S * math.sqrt(S)  # S^(3/2)

    return _c_w(T, P) + _A_coeff(T, P) * S + _B_coeff(T, P) * S32 \
         + _D_coeff(T, P) * S * S
