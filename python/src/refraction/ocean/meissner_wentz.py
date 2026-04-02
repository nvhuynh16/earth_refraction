"""
Complex permittivity and refractive index of seawater at microwave frequencies.

Implements the Meissner & Wentz double-Debye relaxation model for the complex
dielectric constant of pure and sea water:

    eps(f, T, S) = (eps_s - eps_1) / (1 - j*f/v1)
                 + (eps_1 - eps_inf) / (1 - j*f/v2)
                 + eps_inf
                 + j * sigma * f0 / f                               [Eq. 1]

where eps_s is the static dielectric constant, eps_1 and eps_inf are the
intermediate and high-frequency limits, v1 and v2 are the first and second
relaxation frequencies, sigma is the ionic conductivity, and f0 is a
dimensional constant converting conductivity to dielectric loss.

The five Debye parameters (eps_s, eps_1, eps_inf, v1, v2) are first
computed for pure water [1, Eqs. 5-9, Table III], then modified for
salinity [1, Eq. 17, Table VI] with 2012 updates [2, Table 7].
Conductivity follows the Stogryn (1971) model as given in [1, Eqs. 11-16].

Valid ranges (from [1], Section II-A):
    Temperature:   -2 to 29 deg C (extended to -30 C via SST clamping in [2])
    Salinity:      0 to 40 PSU
    Frequency:     up to ~90 GHz ([1] validated to 90 GHz)

Stated accuracy: brightness temperature residuals < 0.3 K below X-band [1].

References
----------
.. [1] Meissner, T. and F. Wentz (2004). "The complex dielectric constant of
       pure and sea water from microwave satellite observations." IEEE Trans.
       Geosci. Remote Sens., 42(9), 1836-1849.
.. [2] Meissner, T. and F. Wentz (2012). "The emissivity of the ocean surface
       between 6 and 90 GHz over a large range of wind speeds and earth
       incidence angles." IEEE Trans. Geosci. Remote Sens., 50(11), 4919-4932.
.. [3] Stogryn, A. (1971). "Equations for calculating the dielectric constant
       of saline water." IEEE Trans. Microwave Theory Tech., 19(8), 733-736.
.. [4] RSS Fortran implementation (2012 updates with errata corrections):
       github.com/Remote-Sensing-Systems/RSS-L-band-Ocean-Surface-Emission-Model
"""

import math
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_F0 = 17.97510
"""Conductivity-to-loss conversion factor (GHz m / S).  [1] Eq. 1.
   Equals 1 / (2 * pi * eps_0) in GHz units."""

# ---------------------------------------------------------------------------
# Pure-water Debye coefficients  [1] Table III
# ---------------------------------------------------------------------------
# These parameterize the five Debye parameters for S = 0 (pure water):
#   eps_s(T)   = (3.70886e4 - 8.2168e1*T) / (4.21854e2 + T)    [1] Eq. 5
#   eps_1(T)   = a[0] + a[1]*T + a[2]*T^2                       [1] Eq. 6
#   v1(T)      = (45 + T) / (a[3] + a[4]*T + a[5]*T^2)  (GHz)  [1] Eq. 7
#   eps_inf(T) = a[6] + a[7]*T                                  [1] Eq. 8
#   v2(T)      = (45 + T) / (a[8] + a[9]*T + a[10]*T^2) (GHz)  [1] Eq. 9
_A = (
    5.7230E+00,   # a0  eps_1 constant term
    2.2379E-02,   # a1  eps_1 linear in T (degC^-1)
    -7.1237E-04,  # a2  eps_1 quadratic in T (degC^-2)
    5.0478E+00,   # a3  v1 denominator constant
    -7.0315E-02,  # a4  v1 denominator linear in T
    6.0059E-04,   # a5  v1 denominator quadratic in T
    3.6143E+00,   # a6  eps_inf constant term
    2.8841E-02,   # a7  eps_inf linear in T (degC^-1)
    1.3652E-01,   # a8  v2 denominator constant
    1.4825E-03,   # a9  v2 denominator linear in T
    2.4166E-04,   # a10 v2 denominator quadratic in T
)

# ---------------------------------------------------------------------------
# Salinity-dependent coefficients  [1] Table VI
# ---------------------------------------------------------------------------
# These modify the pure-water Debye parameters for salinity S (PSU):
#   eps_s(T,S)   = eps_s(T,0)   * exp(b[0]*S + b[1]*S^2 + b[2]*T*S)  [Eq.17a]
#   v1(T,S)      = v1(T,0)      * (1 + S*(b[3] + b[4]*T + b[5]*T^2)) [Eq.17b]
#   eps_1(T,S)   = eps_1(T,0)   * exp(b[6]*S + b[7]*S^2 + b[8]*S*T)  [Eq.17c]
#   v2(T,S)      = v2(T,0)      * (1 + S*(b[9] + b[10]*T))           [Eq.17d]
#   eps_inf(T,S) = eps_inf(T,0) * (1 + S*(b[11] + b[12]*T))          [Eq.17e]
_B_2004 = (
    -3.56417E-03,  # b0   eps_s: salinity coefficient (PSU^-1)
    4.74868E-06,   # b1   eps_s: salinity^2 coefficient (PSU^-2)
    1.15574E-05,   # b2   eps_s: T*S cross-term (degC^-1 PSU^-1)
    2.39357E-03,   # b3   v1: salinity coefficient (PSU^-1)
    -3.13530E-05,  # b4   v1: S*T cross-term (degC^-1 PSU^-1)
    2.52477E-07,   # b5   v1: S*T^2 cross-term (degC^-2 PSU^-1)
    -6.28908E-03,  # b6   eps_1: salinity coefficient (PSU^-1)
    1.76032E-04,   # b7   eps_1: salinity^2 coefficient (PSU^-2)
    -9.22144E-05,  # b8   eps_1: S*T cross-term (degC^-1 PSU^-1)
    -1.99723E-02,  # b9   v2: salinity coefficient (PSU^-1)
    1.81176E-04,   # b10  v2: S*T cross-term (degC^-1 PSU^-1)
    -2.04265E-03,  # b11  eps_inf: salinity coefficient (PSU^-1)
    1.57883E-04,   # b12  eps_inf: S*T cross-term (degC^-1 PSU^-1)
)

# ---------------------------------------------------------------------------
# MW2012 coefficient updates  [2] Table 7, [4]
# ---------------------------------------------------------------------------
# The 2012 paper updated three salinity corrections based on improved
# L-band calibration from WindSat/SSM/I:

# eps_s: b[0] changed from -3.56417e-3 to -3.3330e-3  [2] Table 7, row 1
_B0_MW2012 = -0.33330E-02

# v1: replaced 3-term polynomial (b[3]-b[5]) with 5-term polynomial  [2] Table 7
# Note: d3 sign corrected per [4] errata (printed +, correct is -)
_V1_MW2012 = (
    0.23232E-02,   # d0 (PSU^-1)
    -0.79208E-04,  # d1 (degC^-1 PSU^-1)
    0.36764E-05,   # d2 (degC^-2 PSU^-1)
    -0.35594E-06,  # d3 (degC^-3 PSU^-1) — sign corrected per errata
    0.89795E-08,   # d4 (degC^-4 PSU^-1)
)
# High-temperature branch (SST > 30 C):  linear extrapolation  [2] Sec. IV
_V1_MW2012_HI_OFFSET = 9.1873715E-04   # PSU^-1
_V1_MW2012_HI_SLOPE = 1.5012396E-04    # degC^-1 PSU^-1

# v2: salinity correction changed from b[10]*T to b[10]*0.5*(T+30)  [2] Table 7
# (eps_1 and eps_inf corrections b[6]-b[12] are unchanged from 2004.)


# ---------------------------------------------------------------------------
# Conductivity coefficients  [3] via [1] Eqs. 11-16
# ---------------------------------------------------------------------------
# sigma(T, S) = sigma_35(T) * R_15(S) * [1 + alpha_0(S)*(T-15)/(alpha_1(S)+T)]
#
# sigma_35(T) = 2.903602 + 8.60700e-2*T + 4.738817e-4*T^2
#             - 2.991e-6*T^3 + 4.3047e-9*T^4                    [1] Eq. 12
#
# R_15(S) = S*(37.5109 + 5.45216*S + 0.014409*S^2)
#         / (1004.75 + 182.283*S + S^2)                          [1] Eq. 14
#
# alpha_0(S) = (6.9431 + 3.2841*S - 0.099486*S^2)
#            / (84.850 + 69.024*S + S^2)                         [1] Eq. 15
#
# alpha_1(S) = 49.843 - 0.2276*S + 0.00198*S^2                  [1] Eq. 16


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class ComplexPermittivity(NamedTuple):
    """Complex relative permittivity of seawater.

    Convention: eps = real - j*imag, where imag > 0 for lossy media.

    Attributes
    ----------
    real : float
        Real part of relative permittivity (dielectric constant).
    imag : float
        Imaginary part of relative permittivity (dielectric loss, positive).
    """
    real: float
    imag: float


class ComplexRefractiveIndex(NamedTuple):
    """Complex refractive index of seawater.

    Computed from permittivity via n = sqrt(eps), so n = n_real - j*n_imag.

    Attributes
    ----------
    n_real : float
        Real part of the refractive index.
    n_imag : float
        Imaginary part (extinction coefficient, positive for lossy media).
    """
    n_real: float
    n_imag: float


# ---------------------------------------------------------------------------
# Pure-water Debye parameters  [1] Eqs. 5-9, Table III
# ---------------------------------------------------------------------------

def eps_s_pure(T_C: float) -> float:
    """Static permittivity of pure water.

    Rational function fit from [1] Eq. 5.

    Parameters
    ----------
    T_C : float
        Temperature in degrees Celsius.

    Returns
    -------
    float
        Static permittivity (dimensionless).
        ~87.9 at 0 deg C, ~80.2 at 20 deg C, ~78.4 at 25 deg C.

    Notes
    -----
    These values are consistent with the CRC Handbook (2004) tabulation
    of the static dielectric constant of water.
    """
    # [1] Eq. 5, coefficients from Table III header
    return (3.70886E4 - 8.2168E1 * T_C) / (4.21854E2 + T_C)


def eps_1_pure(T_C: float) -> float:
    """Intermediate permittivity of pure water.

    Quadratic fit from [1] Eq. 6, coefficients a[0]-a[2] in Table III.

    Parameters
    ----------
    T_C : float
        Temperature in degrees Celsius.

    Returns
    -------
    float
        Intermediate permittivity (dimensionless), typically ~5.5-6.3.
    """
    return _A[0] + _A[1] * T_C + _A[2] * T_C * T_C


def v1_pure(T_C: float) -> float:
    """First (principal) relaxation frequency of pure water.

    Rational function from [1] Eq. 7, coefficients a[3]-a[5] in Table III.

    Parameters
    ----------
    T_C : float
        Temperature in degrees Celsius.

    Returns
    -------
    float
        Relaxation frequency in GHz.  ~9 GHz at 0 deg C, ~17 GHz at 20 deg C.
    """
    return (45.00 + T_C) / (_A[3] + _A[4] * T_C + _A[5] * T_C * T_C)


def eps_inf_pure(T_C: float) -> float:
    """High-frequency permittivity of pure water.

    Linear fit from [1] Eq. 8, coefficients a[6]-a[7] in Table III.

    Parameters
    ----------
    T_C : float
        Temperature in degrees Celsius.

    Returns
    -------
    float
        High-frequency permittivity (dimensionless), typically ~3.9-4.4.
    """
    return _A[6] + _A[7] * T_C


def v2_pure(T_C: float) -> float:
    """Second relaxation frequency of pure water.

    Rational function from [1] Eq. 9, coefficients a[8]-a[10] in Table III.

    Parameters
    ----------
    T_C : float
        Temperature in degrees Celsius.

    Returns
    -------
    float
        Relaxation frequency in GHz.  Typically ~100-300 GHz.
    """
    return (45.00 + T_C) / (_A[8] + _A[9] * T_C + _A[10] * T_C * T_C)


# ---------------------------------------------------------------------------
# Ionic conductivity  [3] via [1] Eqs. 11-16
# ---------------------------------------------------------------------------

def sigma35(T_C: float) -> float:
    """Conductivity of standard seawater at S = 35 PSU.

    Fourth-order polynomial from [1] Eq. 12 (Stogryn 1971 [3]).

    Parameters
    ----------
    T_C : float
        Temperature in degrees Celsius.

    Returns
    -------
    float
        Conductivity in S/m.  ~2.90 at 0 deg C, ~4.29 at 15 deg C,
        ~5.71 at 25 deg C.
    """
    T = T_C
    return (2.903602
            + 8.60700E-02 * T
            + 4.738817E-04 * T * T
            - 2.991E-06 * T * T * T
            + 4.3047E-09 * T * T * T * T)


def _R15(S_psu: float) -> float:
    """Conductivity ratio at 15 deg C relative to S=35.  [1] Eq. 14."""
    S = S_psu
    return S * (37.5109 + 5.45216 * S + 1.4409E-02 * S * S) / (
        1004.75 + 182.283 * S + S * S
    )


def _alpha0(S_psu: float) -> float:
    """Temperature correction coefficient alpha_0.  [1] Eq. 15."""
    S = S_psu
    return (6.9431 + 3.2841 * S - 9.9486E-02 * S * S) / (
        84.850 + 69.024 * S + S * S
    )


def _alpha1(S_psu: float) -> float:
    """Temperature correction coefficient alpha_1.  [1] Eq. 16."""
    S = S_psu
    return 49.843 - 0.2276 * S + 0.198E-02 * S * S


def conductivity(T_C: float, S_psu: float) -> float:
    """Ionic conductivity of seawater.

    Stogryn (1971) [3] model as parameterized in [1] Eqs. 11-16:

        sigma(T, S) = sigma_35(T) * R_15(S)
                    * [1 + alpha_0(S) * (T - 15) / (alpha_1(S) + T)]

    Parameters
    ----------
    T_C : float
        Temperature in degrees Celsius.
    S_psu : float
        Practical salinity in PSU.

    Returns
    -------
    float
        Conductivity in S/m.  Returns 0 for S <= 0 (pure water has
        negligible ionic conductivity).
    """
    if S_psu <= 0.0:
        return 0.0
    sig35 = sigma35(T_C)
    r15 = _R15(S_psu)
    a0 = _alpha0(S_psu)
    a1 = _alpha1(S_psu)
    # [1] Eq. 11
    return sig35 * r15 * (1.0 + (T_C - 15.0) * a0 / (a1 + T_C))


# ---------------------------------------------------------------------------
# Salinity-dependent Debye parameters  [1] Table VI + [2] updates
# ---------------------------------------------------------------------------

def eps_s_sea(T_C: float, S_psu: float) -> float:
    """Static permittivity of seawater.

    [1] Eq. 17a with [2] updated b0 coefficient (-3.3330e-3 replaces
    -3.56417e-3).  Salt ions reduce the static permittivity.

    Parameters
    ----------
    T_C : float
        Temperature in degrees Celsius.
    S_psu : float
        Practical salinity in PSU.

    Returns
    -------
    float
        Static permittivity (dimensionless).
    """
    S = S_psu
    # [1] Eq. 17a, with b[0] from [2] Table 7
    return eps_s_pure(T_C) * math.exp(
        _B0_MW2012 * S + _B_2004[1] * S * S + _B_2004[2] * T_C * S
    )


def v1_sea(T_C: float, S_psu: float) -> float:
    """First relaxation frequency of seawater.

    [2] Table 7 replaces the original [1] 3-term polynomial (b[3]-b[5])
    with a 5-term polynomial in SST, plus a linear extrapolation branch
    for SST > 30 deg C.

    Parameters
    ----------
    T_C : float
        Temperature in degrees Celsius.
    S_psu : float
        Practical salinity in PSU.

    Returns
    -------
    float
        Relaxation frequency in GHz.
    """
    v1_0 = v1_pure(T_C)
    S = S_psu
    T = T_C
    if T <= 30.0:
        # [2] Table 7, 5-term polynomial with d3 sign corrected per [4] errata
        c = _V1_MW2012
        factor = 1.0 + S * (
            c[0] + c[1] * T + c[2] * T * T
            + c[3] * T * T * T + c[4] * T * T * T * T
        )
    else:
        # [2] Sec. IV: linear extrapolation for SST > 30 C
        factor = 1.0 + S * (
            _V1_MW2012_HI_OFFSET + _V1_MW2012_HI_SLOPE * (T - 30.0)
        )
    return v1_0 * factor


def eps_1_sea(T_C: float, S_psu: float) -> float:
    """Intermediate permittivity of seawater.

    [1] Eq. 17c (unchanged in [2]).

    Parameters
    ----------
    T_C : float
        Temperature in degrees Celsius.
    S_psu : float
        Practical salinity in PSU.

    Returns
    -------
    float
        Intermediate permittivity (dimensionless).
    """
    S = S_psu
    b = _B_2004
    return eps_1_pure(T_C) * math.exp(
        b[6] * S + b[7] * S * S + b[8] * S * T_C
    )


def v2_sea(T_C: float, S_psu: float) -> float:
    """Second relaxation frequency of seawater.

    [1] Eq. 17d with [2] correction: the salinity-temperature cross-term
    uses 0.5*(T + 30) instead of T, improving L-band performance.

    Parameters
    ----------
    T_C : float
        Temperature in degrees Celsius.
    S_psu : float
        Practical salinity in PSU.

    Returns
    -------
    float
        Relaxation frequency in GHz.
    """
    b = _B_2004
    # [1] Eq. 17d with [2] modification to T dependence
    return v2_pure(T_C) * (1.0 + S_psu * (b[9] + b[10] * 0.5 * (T_C + 30.0)))


def eps_inf_sea(T_C: float, S_psu: float) -> float:
    """High-frequency permittivity of seawater.

    [1] Eq. 17e (unchanged in [2]).

    Parameters
    ----------
    T_C : float
        Temperature in degrees Celsius.
    S_psu : float
        Practical salinity in PSU.

    Returns
    -------
    float
        High-frequency permittivity (dimensionless).
    """
    b = _B_2004
    return eps_inf_pure(T_C) * (1.0 + S_psu * (b[11] + b[12] * T_C))


# ---------------------------------------------------------------------------
# Complex permittivity and refractive index
# ---------------------------------------------------------------------------

def permittivity(freq_ghz: float, T_C: float, S_psu: float) -> ComplexPermittivity:
    """Complex relative permittivity of seawater.

    Evaluates the double-Debye relaxation model [1] Eq. 1 with [2] updated
    coefficients and Stogryn [3] conductivity.  The complex permittivity is
    decomposed as:

        eps = Re(eps) - j*Im(eps)

    where the real part is the dielectric constant and the imaginary part
    (returned positive) is the dielectric loss.

    Parameters
    ----------
    freq_ghz : float
        Microwave frequency in GHz (typically 1-90).
    T_C : float
        Temperature in degrees Celsius (-2 to 29).
    S_psu : float
        Practical salinity in PSU (0 for pure water, 0-40).

    Returns
    -------
    ComplexPermittivity
        Named tuple (real, imag).
    """
    es = eps_s_sea(T_C, S_psu)
    e1 = eps_1_sea(T_C, S_psu)
    einf = eps_inf_sea(T_C, S_psu)
    nu1 = v1_sea(T_C, S_psu)
    nu2 = v2_sea(T_C, S_psu)
    sig = conductivity(T_C, S_psu)
    f = freq_ghz

    # First Debye term: (es - e1) / (1 - j*f/nu1)                 [1] Eq. 1
    # Rationalized: multiply by conjugate (1 + j*f/nu1) / (1 + (f/nu1)^2)
    r1 = f / nu1
    d1 = 1.0 + r1 * r1
    re1 = (es - e1) / d1
    im1 = (es - e1) * r1 / d1

    # Second Debye term: (e1 - einf) / (1 - j*f/nu2)              [1] Eq. 1
    r2 = f / nu2
    d2 = 1.0 + r2 * r2
    re2 = (e1 - einf) / d2
    im2 = (e1 - einf) * r2 / d2

    # Conductivity loss: j * sigma * f0 / f                       [1] Eq. 1
    im_cond = sig * _F0 / f if f > 0.0 else 0.0

    eps_real = re1 + re2 + einf
    eps_imag = im1 + im2 + im_cond

    return ComplexPermittivity(eps_real, eps_imag)


def refractive_index(freq_ghz: float, T_C: float, S_psu: float) -> ComplexRefractiveIndex:
    """Complex refractive index of seawater at microwave frequencies.

    Computed from complex permittivity via n = sqrt(eps).  For eps = a - j*b
    with a, b > 0:

        n_real = sqrt((|eps| + a) / 2)
        n_imag = sqrt((|eps| - a) / 2)

    Parameters
    ----------
    freq_ghz : float
        Microwave frequency in GHz.
    T_C : float
        Temperature in degrees Celsius.
    S_psu : float
        Practical salinity in PSU (0 for pure water).

    Returns
    -------
    ComplexRefractiveIndex
        Named tuple (n_real, n_imag) where n_imag >= 0 for lossy media.
    """
    eps = permittivity(freq_ghz, T_C, S_psu)
    # n = sqrt(eps_real - j*eps_imag)
    mag = math.sqrt(eps.real * eps.real + eps.imag * eps.imag)
    n_real = math.sqrt((mag + eps.real) / 2.0)
    n_imag = math.sqrt((mag - eps.real) / 2.0)
    return ComplexRefractiveIndex(n_real, n_imag)
