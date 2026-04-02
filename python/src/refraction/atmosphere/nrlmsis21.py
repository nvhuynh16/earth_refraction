"""
NRLMSIS 2.1 Atmosphere Model -- Python implementation.

Direct port of the NRLMSIS 2.1 Fortran code to Python/NumPy.  The model
computes neutral temperature and number densities of 10 atmospheric species
as functions of altitude, geographic location, time, and solar/geomagnetic
activity.

The temperature profile uses B-spline basis functions (order 4 for
temperature, up to order 6 for density integrals) in geopotential height
(zeta) from -15 km to 122.5 km, transitioning to a Bates exponential
profile above 122.5 km that asymptotes to the exospheric temperature T_ex.

Species number densities are computed via hydrostatic integration of
the temperature profile, with species-specific corrections for:
- Piecewise molecular weight transitions (mixed -> diffusive equilibrium)
- Chemical/photochemical production/loss (C, R terms)
- Low-altitude B-spline profiles for O and NO

Output array conventions:

    dn[0]  = unused (placeholder)
    dn[1]  = total mass density (kg/m^3) -- NOT a number density!
    dn[2]  = N2 number density (m^-3)
    dn[3]  = O2 number density (m^-3)
    dn[4]  = O  number density (m^-3)
    dn[5]  = He number density (m^-3)
    dn[6]  = H  number density (m^-3)
    dn[7]  = Ar number density (m^-3)
    dn[8]  = N  number density (m^-3)
    dn[9]  = anomalous O number density (m^-3)
    dn[10] = NO number density (m^-3)

Key altitude boundaries:

    ZETA_F  = 70.0 km   -- below this, major species use mixed composition
    ZETA_A  = 85.0 km   -- reference for O and H species
    ZETA_B  = 122.5 km  -- B-spline / Bates profile transition

Reference
---------
Emmert, J.T., et al. (2022). "NRLMSIS 2.1: An Empirical Model of Nitric
Oxide Incorporated into MSIS." J. Geophys. Res. Space Physics, 127,
e2022JA030896.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from . import _nrlmsis21_parm as parm

# =====================================================================
# Constants (from msis_constants.F90)
# =====================================================================
PI = math.pi
DEG2RAD = PI / 180.0     # degrees -> radians
DOY2RAD = 2.0 * PI / 365.0  # day of year -> radians (annual cycle)
LST2RAD = PI / 12.0      # local solar time (hours) -> radians
TANH1 = 0.7615941559557649  # tanh(1.0), pre-computed for speed

# Physical constants
KB = 1.380649e-23   # J/K, Boltzmann constant (SI 2019 exact)
NA = 6.02214076e23  # mol^-1, Avogadro constant
G0 = 9.80665        # m/s^2, standard gravitational acceleration

# Species indices (matching Fortran convention)
SPEC_MASS = 1   # Total mass density (kg/m^3, NOT a number density)
SPEC_N2 = 2     # Molecular nitrogen
SPEC_O2 = 3     # Molecular oxygen
SPEC_O = 4      # Atomic oxygen
SPEC_HE = 5     # Helium
SPEC_H = 6      # Atomic hydrogen
SPEC_AR = 7     # Argon
SPEC_N = 8      # Atomic nitrogen
SPEC_OA = 9     # Anomalous (hot) oxygen
SPEC_NO = 10    # Nitric oxide

# Per-molecule mass for each species (kg), computed as molar_mass / (1000 * NA).
# Index 0 and 1 are unused placeholders.
SPECMASS = [
    0.0, 0.0,
    28.0134 / (1.0e3 * NA),                  # N2
    31.9988 / (1.0e3 * NA),                  # O2
    31.9988 / 2.0 / (1.0e3 * NA),            # O (half of O2)
    4.0 / (1.0e3 * NA),                      # He
    1.0 / (1.0e3 * NA),                      # H
    39.948 / (1.0e3 * NA),                   # Ar
    28.0134 / 2.0 / (1.0e3 * NA),            # N (half of N2)
    31.9988 / 2.0 / (1.0e3 * NA),            # anomalous O
    (28.0134 + 31.9988) / 2.0 / (1.0e3 * NA),  # NO
]

# Mean molecular mass of air at sea level (kg)
MBAR = 28.96546 / (1.0e3 * NA)

# ln(volume mixing ratio) at the surface for well-mixed species.
# These are MSIS-internal fitted values and do not exactly equal
# ln() of the commonly cited standard mole fractions.
# Zero entries indicate species whose surface density is computed
# differently (O, H, N, anomalous O, NO).
LNVMR = [
    0.0, 0.0,
    -0.247359612336553,   # N2 (~78.1%)
    -1.563296139553406,   # O2 (~20.9%)
    0.0,                  # O (not well-mixed)
    -12.16717628831267,   # He (~5.2 ppm)
    0.0,                  # H (not well-mixed)
    -4.674383440483631,   # Ar (~0.93%)
    0.0, 0.0, 0.0,       # N, anomalous O, NO
]

# ln of reference surface pressure parameter: exp(LNP0) ~ 100269 Pa.
# This is an MSIS-internal fitted constant, not exactly ln(101325).
LNP0 = 11.515614

# Derived constants for hydrostatic integration.
# The 1e3 factor converts km -> m so that altitude in km can be used
# directly in hydrostatic integrals.
G0DIVKB = G0 / KB * 1.0e3
# MBARG0DIVKB = M_bar * g0 / kB * 1e3 ~ 34.16 km^-1, the characteristic
# inverse scale height of air (H = T / MBARG0DIVKB ~ 8.4 km at 288 K).
# Also used by surface_anchor.py for hydrostatic density propagation.
MBARG0DIVKB = MBAR * G0DIVKB

# Sentinel value for missing/below-minimum-altitude species densities
DMISSING = 9.999e-38

ND = 27
P_ORDER = 4
NL = ND - P_ORDER
NLS = 9

ZETA_F = 70.0
ZETA_B = 122.5
ZETA_A = 85.0
ZETA_GAMMA = 100.0
H_GAMMA = 1.0 / 30.0

H_OA = 4000.0 * KB / (SPECMASS[SPEC_OA] * G0) * 1.0e-3

IZFMX = 13
IZFX = 14
IZAX = 17
ITEX = NL
ITGB0 = NL - 1
ITB0 = NL - 2

NODES_TN = [
    -15., -10., -5., 0., 5., 10., 15., 20., 25., 30.,
    35., 40., 45., 50., 55., 60., 65., 70., 75., 80.,
    85., 92.5, 102.5, 112.5, 122.5, 132.5, 142.5, 152.5, 162.5, 172.5,
]

MBF = 383
MAXN = 6
MAXL = 3
MAXM = 2

CTIMEIND = 0
CINTANN = 7
CTIDE = 35
CSPW = 185
CSFX = 295
NSFX = 5
CEXTRA = 300
NSFXMOD = 5
NMAG = 54
NUT = 12
CNONLIN = 384
CSFXMOD = 384
CMAG = 389
CUT = 443

NDO1 = 13
NSPLO1 = NDO1 - 5
NODES_O1 = [
    35., 40., 45., 50., 55., 60., 65., 70., 75., 80., 85., 92.5, 102.5, 112.5,
]
ZETAREF_O1 = ZETA_A

NDNO = 13
NSPLNO = NDNO - 5
NODES_NO = [
    47.5, 55., 62.5, 70., 77.5, 85., 92.5, 100., 107.5, 115., 122.5, 130., 137.5, 145.,
]
ZETAREF_NO = ZETA_B
ZETAREF_OA = ZETA_B

C1O1 = [
    [1.75, -1.624999900076852],
    [-2.916666573405061, 21.458332647194382],
]
C1O1ADJ = [0.257142857142857, -0.102857142686844]

C1NO = [
    [1.5, 0.0],
    [-3.75, 15.0],
]
C1NOADJ = [0.166666666666667, -0.066666666666667]

WGHTAXDZ = [-0.102857142857, 0.0495238095238, 0.053333333333]

S5ZETA_B = [0.041666666666667, 0.458333333333333, 0.458333333333333, 0.041666666666667]
S6ZETA_B = [0.008771929824561, 0.216228070175439, 0.550000000000000, 0.216666666666667, 0.008333333333333]
S4ZETA_F = [0.166666666666667, 0.666666666666667, 0.166666666666667]
S5ZETA_F = [0.041666666666667, 0.458333333333333, 0.458333333333333, 0.041666666666667]
S5ZETA_0 = [0.458333333333333, 0.458333333333333, 0.041666666666667]
S4ZETA_A = [0.257142857142857, 0.653968253968254, 0.088888888888889]
S5ZETA_A = [0.085714285714286, 0.587590187590188, 0.313020313020313, 0.013675213675214]
S6ZETA_A = [0.023376623376623, 0.378732378732379, 0.500743700743701, 0.095538448479625, 0.001608848667672]

C2TN = [
    [1.0, 1.0, 1.0],
    [-10.0, 0.0, 10.0],
    [33.333333333333336, -16.666666666666668, 33.333333333333336],
]


# =====================================================================
# Utility functions
# =====================================================================
def _alt2gph(lat: float, alt: float) -> float:
    """
    Convert geodetic altitude to geopotential height.

    Uses the WGS-84 ellipsoidal gravity model to compute the exact
    geopotential height, accounting for latitude-dependent gravity
    and centrifugal effects.  This is more accurate than the simple
    approximation zeta = R_E * h / (R_E + h).

    Parameters
    ----------
    lat : float
        Geodetic latitude in degrees.
    alt : float
        Geodetic altitude in kilometers.

    Returns
    -------
    float
        Geopotential height (zeta) in kilometers.
    """
    a = 6378.1370e3
    finv = 298.257223563
    w = 7.292115e-5
    GM = 398600.4418e9

    f = 1.0 / finv
    esq = 2.0 * f - f * f
    e_val = 0.08181919084262149
    Elin = a * e_val
    Elinsq = Elin * Elin
    q0 = 7.33462578708548e-05
    U0 = -62636851.7149236
    g0 = 9.80665
    GMdivElin = GM / Elin

    x0sq = 2.0e7 * 2.0e7
    Hsq = 1.2e7 * 1.2e7

    altm = alt * 1000.0
    sinsqlat = math.sin(lat * DEG2RAD)
    sinsqlat = sinsqlat * sinsqlat

    v = a / math.sqrt(1.0 - esq * sinsqlat)
    xsq = (v + altm) * (v + altm) * (1.0 - sinsqlat)
    zsq = (v * (1.0 - esq) + altm) * (v * (1.0 - esq) + altm) * sinsqlat

    rsqminElinsq = xsq + zsq - Elinsq
    usq = rsqminElinsq / 2.0 + math.sqrt(
        rsqminElinsq * rsqminElinsq / 4.0 + Elinsq * zsq
    )
    cossqdelta = zsq / usq

    epru = Elin / math.sqrt(usq)
    atanepru = math.atan(epru)
    qq = ((1.0 + 3.0 / (epru * epru)) * atanepru - 3.0 / epru) / 2.0
    asq = a * a
    wsq = w * w
    U = -GMdivElin * atanepru - wsq * (asq * qq * (cossqdelta - 1.0 / 3.0) / q0) / 2.0

    if xsq <= x0sq:
        Vc = (wsq / 2.0) * xsq
    else:
        Vc = (wsq / 2.0) * (Hsq * math.tanh((xsq - x0sq) / Hsq) + x0sq)
    U = U - Vc

    return (U - U0) / g0 / 1000.0


def _dilog(x0: float) -> float:
    """Spence's dilogarithm Li_2(x) = -integral(ln(1-t)/t, 0, x).

    Uses a rational approximation valid for 0 <= x <= 1.  Appears in
    the analytic integral of the Bates temperature profile (W function).
    """
    pi2_6 = PI * PI / 6.0
    x = x0
    if x > 0.5:
        lnx = math.log(x)
        x = 1.0 - x
        xx = x * x
        x4 = 4.0 * x
        return (
            pi2_6 - lnx * math.log(x)
            - (4.0 * xx * (23.0 / 16.0 + x / 36.0 + xx / 576.0 + xx * x / 3600.0)
               + x4 + 3.0 * (1.0 - xx) * lnx)
            / (1.0 + x4 + xx)
        )
    else:
        xx = x * x
        x4 = 4.0 * x
        return (
            (4.0 * xx * (23.0 / 16.0 + x / 36.0 + xx / 576.0 + xx * x / 3600.0)
             + x4 + 3.0 * (1.0 - xx) * math.log(1.0 - x))
            / (1.0 + x4 + xx)
        )


def _solzen(ddd: float, lst: float, lat: float, lon: float) -> float:
    """Solar zenith angle in degrees.

    Computes the angle between the sun and local zenith using an
    approximate solar declination and equation of time.
    """
    humr = PI / 12.0
    p = [0.017203534, 0.034407068, 0.051610602, 0.068814136, 0.103221204]

    teqnx = ddd + 0.9369
    dec = (
        23.256 * math.sin(p[0] * (teqnx - 82.242))
        + 0.381 * math.sin(p[1] * (teqnx - 44.855))
        + 0.167 * math.sin(p[2] * (teqnx - 23.355))
        - 0.013 * math.sin(p[3] * (teqnx + 11.97))
        + 0.011 * math.sin(p[4] * (teqnx - 10.410))
        + 0.339137
    )
    dec *= DEG2RAD

    tf = teqnx - 0.5
    teqt = (
        -7.38 * math.sin(p[0] * (tf - 4.0))
        - 9.87 * math.sin(p[1] * (tf + 9.0))
        + 0.27 * math.sin(p[2] * (tf - 53.0))
        - 0.2 * math.cos(p[3] * (tf - 17.0))
    )

    phi = humr * (lst - 12.0) + teqt * DEG2RAD / 4.0
    rlat = lat * DEG2RAD
    cosx = (
        math.sin(rlat) * math.sin(dec)
        + math.cos(rlat) * math.cos(dec) * math.cos(phi)
    )
    if abs(cosx) > 1.0:
        cosx = 1.0 if cosx > 0 else -1.0
    return math.acos(cosx) / DEG2RAD


def _bspline_eval(z, nodes, nd, eta):
    """Evaluate B-spline basis functions of orders 2-6.

    Returns (iz, S3[6], S4[6], S5[6], S6[6]).
    """
    S3 = [0.0] * 6
    S4 = [0.0] * 6
    S5 = [0.0] * 6
    S6 = [0.0] * 6

    if z >= nodes[nd - 1]:
        return nd - 1, S3, S4, S5, S6
    if z <= nodes[0]:
        return -1, S3, S4, S5, S6

    low, high = 0, nd - 1
    iz = (low + high) // 2
    while z < nodes[iz] or z >= nodes[iz + 1]:
        if z < nodes[iz]:
            high = iz
        else:
            low = iz
        iz = (low + high) // 2

    i = iz

    # Order 2
    S2 = [0.0, 0.0]
    w0_ = (z - nodes[i]) * eta[0][i]
    S2[1] = w0_
    if i > 0:
        S2[0] = 1.0 - w0_
    if i >= nd - 1:
        S2[1] = 0.0

    # Order 3
    _S3 = [0.0, 0.0, 0.0]
    w3 = [0.0, 0.0]
    w3[1] = (z - nodes[i]) * eta[1][i]
    if i >= 1:
        w3[0] = (z - nodes[i - 1]) * eta[1][i - 1]

    if i < nd - 2:
        _S3[2] = w3[1] * S2[1]
    if i >= 1 and i - 1 < nd - 2:
        _S3[1] = w3[0] * S2[0] + (1.0 - w3[1]) * S2[1]
    if i >= 2:
        _S3[0] = (1.0 - w3[0]) * S2[0]

    S3[0] = _S3[0]
    S3[1] = _S3[1]
    S3[2] = _S3[2]

    # Order 4
    _S4 = [0.0, 0.0, 0.0, 0.0]
    w4 = [0.0, 0.0, 0.0]
    for l in range(0, -3, -1):
        j = i + l
        if 0 <= j < nd - 2:
            w4[l + 2] = (z - nodes[j]) * eta[2][j]

    if i < nd - 3:
        _S4[3] = w4[2] * _S3[2]
    for l in range(-1, -3, -1):
        if 0 <= i + l < nd - 3:
            _S4[l + 3] = w4[l + 2] * _S3[l + 2] + (1.0 - w4[l + 3]) * _S3[l + 3]
    if i >= 3:
        _S4[0] = (1.0 - w4[0]) * _S3[0]

    for j in range(4):
        S4[j] = _S4[j]

    # Order 5
    _S5 = [0.0, 0.0, 0.0, 0.0, 0.0]
    w5 = [0.0, 0.0, 0.0, 0.0]
    for l in range(0, -4, -1):
        j = i + l
        if 0 <= j < nd - 3:
            w5[l + 3] = (z - nodes[j]) * eta[3][j]

    if i < nd - 4:
        _S5[4] = w5[3] * _S4[3]
    for l in range(-1, -4, -1):
        if 0 <= i + l < nd - 4:
            _S5[l + 4] = w5[l + 3] * _S4[l + 3] + (1.0 - w5[l + 4]) * _S4[l + 4]
    if i >= 4:
        _S5[0] = (1.0 - w5[0]) * _S4[0]

    for j in range(5):
        S5[j] = _S5[j]

    # Order 6
    w6 = [0.0, 0.0, 0.0, 0.0, 0.0]
    for l in range(0, -5, -1):
        j = i + l
        if 0 <= j < nd - 4:
            w6[l + 4] = (z - nodes[j]) * eta[4][j]

    if i < nd - 5:
        S6[5] = w6[4] * _S5[4]
    for l in range(-1, -5, -1):
        if 0 <= i + l < nd - 5:
            S6[l + 5] = w6[l + 4] * _S5[l + 4] + (1.0 - w6[l + 5]) * _S5[l + 5]
    if i >= 5:
        S6[0] = (1.0 - w6[0]) * _S5[0]

    return iz, S3, S4, S5, S6


def _bspline_eval_o4(z, nodes, nd, eta):
    """Evaluate order-4 B-spline for O1/NO species.

    Returns (iz, S4[4]).
    """
    S4 = [0.0, 0.0, 0.0, 0.0]

    iz = 0
    for ii in range(nd - 1):
        if z >= nodes[ii]:
            iz = ii

    # Order 2
    S2 = [0.0, 0.0, 0.0, 0.0]
    w2 = (z - nodes[iz]) * eta[0][iz]
    S2[2] = w2
    S2[1] = 1.0 - w2

    # Order 3
    S3 = [0.0, 0.0, 0.0, 0.0]
    for l in range(0, -2, -1):
        j = iz + l
        if 0 <= j <= nd - 3:
            wl = (z - nodes[j]) * eta[1][j]
            idx = l + 2
            if l == 0:
                S3[idx] = wl * S2[2]
            if l == -1:
                S3[idx] = wl * S2[1] + (1.0 - (z - nodes[iz]) * eta[1][iz]) * S2[2]
    if iz - 2 >= 0 and iz - 1 <= nd - 3:
        wm1 = (z - nodes[iz - 1]) * eta[1][iz - 1]
        S3[0] = (1.0 - wm1) * S2[1]

    # Order 4
    for l in range(0, -4, -1):
        j = iz + l
        wl = 0.0
        if 0 <= j <= nd - 4:
            wl = (z - nodes[j]) * eta[2][j]

        wl1 = 0.0
        j1 = iz + l + 1
        if 0 <= j1 <= nd - 4:
            wl1 = (z - nodes[j1]) * eta[2][j1]

        s3_l = l + 2
        s3_l1 = l + 3
        s3_val = S3[s3_l] if 0 <= s3_l < 4 else 0.0
        s3_val1 = S3[s3_l1] if 0 <= s3_l1 < 4 else 0.0

        s4_idx = l + 3
        if 0 <= s4_idx < 4:
            S4[s4_idx] = wl * s3_val + (1.0 - wl1) * s3_val1

    return iz, S4


# =====================================================================
# Data classes
# =====================================================================
@dataclass
class MSISInput:
    """
    Input parameters for a single NRLMSIS 2.1 evaluation.

    Attributes
    ----------
    day : float
        Day of year (1-366).
    utsec : float
        Universal time in seconds since midnight.
    alt : float
        Geodetic altitude in kilometers.
    lat : float
        Geodetic latitude in degrees (-90 to 90).
    lon : float
        Geodetic longitude in degrees (-180 to 180).
    f107a : float
        81-day average F10.7 solar radio flux (SFU).
    f107 : float
        Daily F10.7 for the previous day (SFU).
    ap : list of float
        Seven geomagnetic activity indices:
        [daily_Ap, 3hr_Ap(t), 3hr_Ap(t-3h), 3hr_Ap(t-6h),
         3hr_Ap(t-9h), avg_Ap(t-12h..t-33h), avg_Ap(t-36h..t-57h)].
    """
    day: float = 172.0
    utsec: float = 29000.0
    alt: float = 0.0
    lat: float = 45.0
    lon: float = -75.0
    f107a: float = 150.0
    f107: float = 150.0
    ap: list = field(default_factory=lambda: [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0])


@dataclass
class MSISOutput:
    """
    NRLMSIS 2.1 output at a single altitude.

    Attributes
    ----------
    tn : float
        Neutral temperature (K) at the requested altitude.
    tex : float
        Exospheric temperature (K) -- the asymptotic temperature above
        ~500 km, independent of altitude.
    dn : list of float
        Number densities (m^-3) indexed by species.  See module docstring
        for index mapping.  dn[1] is total mass density (kg/m^3), not
        a number density.  Missing species are set to DMISSING (9.999e-38).
    """
    tn: float = 0.0
    tex: float = 0.0
    dn: list = field(default_factory=lambda: [DMISSING] * 11)


@dataclass
class MSISFullOutput:
    """
    NRLMSIS output with temperature derivative.

    Attributes
    ----------
    output : MSISOutput
        Standard NRLMSIS output (temperature and densities).
    dT_dz : float
        Temperature gradient dT/d(zeta) in K/km, where zeta is
        geopotential height.  Note: this is NOT dT/d(geodetic_alt);
        the Jacobian factor (R_E/(R_E+h))^2 causes ~6% deviation
        at 130 km.
    """
    output: MSISOutput = field(default_factory=MSISOutput)
    dT_dz: float = 0.0


@dataclass
class MSISProfileOutput:
    """
    NRLMSIS output for an array of altitudes.

    All altitude-independent work (basis function evaluation, parameter
    computation) is shared across the array for efficiency.

    Attributes
    ----------
    tn : numpy.ndarray
        Temperature array (K), shape (N,).
    tex : float
        Exospheric temperature (K), scalar (same for all altitudes).
    dn : numpy.ndarray
        Number density array, shape (N, 11).  See MSISOutput.dn for
        column indexing.
    dT_dz : numpy.ndarray
        Temperature gradient array (K/km), shape (N,).
    """
    tn: np.ndarray = field(default_factory=lambda: np.array([]))
    tex: float = 0.0
    dn: np.ndarray = field(default_factory=lambda: np.empty((0, 11)))
    dT_dz: np.ndarray = field(default_factory=lambda: np.array([]))


# =====================================================================
# Internal parameter structures
# =====================================================================
@dataclass
class _TnParm:
    cf: list = field(default_factory=lambda: [0.0] * 24)
    tzetaF: float = 0.0
    tzetaA: float = 0.0
    dlntdzA: float = 0.0
    tex: float = 0.0
    tgb0: float = 0.0
    tb0: float = 0.0
    sigma: float = 0.0
    sigmasq: float = 0.0
    b: float = 0.0
    beta: list = field(default_factory=lambda: [0.0] * 24)
    gamma: list = field(default_factory=lambda: [0.0] * 24)
    cVS: float = 0.0
    cVB: float = 0.0
    cWS: float = 0.0
    cWB: float = 0.0
    VzetaF: float = 0.0
    VzetaA: float = 0.0
    WzetaA: float = 0.0
    Vzeta0: float = 0.0
    lndtotF: float = 0.0


@dataclass
class _DnParm:
    ispec: int = 0
    lnPhiF: float = 0.0
    lndref: float = 0.0
    zref: float = 0.0
    zmin: float = 0.0
    zhyd: float = 0.0
    zetaM: float = 0.0
    HML: float = 0.0
    HMU: float = 0.0
    zetaMi: list = field(default_factory=lambda: [0.0] * 5)
    Mi: list = field(default_factory=lambda: [0.0] * 5)
    aMi: list = field(default_factory=lambda: [0.0] * 4)
    WMi: list = field(default_factory=lambda: [0.0] * 5)
    XMi: list = field(default_factory=lambda: [0.0] * 5)
    C: float = 0.0
    zetaC: float = 0.0
    HC: float = 1.0
    R: float = 0.0
    zetaR: float = 0.0
    HR: float = 0.0
    cf: list = field(default_factory=lambda: [0.0] * 10)
    Mzref: float = 0.0
    Izref: float = 0.0
    Tref: float = 0.0


# =====================================================================
# NRLMSIS 2.1 Class
# =====================================================================
class NRLMSIS21:
    """
    NRLMSIS 2.1 empirical atmosphere model.

    Computes neutral temperature and species number densities from the
    surface to the exobase.  The model uses pre-computed parameter arrays
    (in ``_nrlmsis21_parm``) and B-spline basis functions for altitude
    interpolation.

    The constructor pre-computes reciprocal knot differences (eta arrays)
    for the temperature and species-specific B-spline evaluations, as well
    as solar flux modulation flags.

    Methods
    -------
    msiscalc(inp)
        Single-altitude evaluation returning temperature and densities.
    msiscalc_with_derivative(inp)
        Single-altitude with temperature gradient dT/dz.
    msiscalc_profile(inp, altitudes_km)
        Multi-altitude evaluation sharing altitude-independent work.
    """
    def __init__(self):
        # Pre-compute eta arrays (reciprocal knot differences) for TN splines
        self._etaTN = [[0.0] * 30 for _ in range(5)]
        for k in range(5):
            for i in range(NL + 1):
                denom = NODES_TN[i + k + 1] - NODES_TN[i]
                self._etaTN[k][i] = 1.0 / denom if denom > 0.0 else 0.0

        self._etaO1 = [[0.0] * 30 for _ in range(3)]
        for k in range(3):
            for j in range(NDO1 - k - 1):
                denom = NODES_O1[j + k + 1] - NODES_O1[j]
                self._etaO1[k][j] = 1.0 / denom if denom > 0.0 else 0.0

        self._etaNO = [[0.0] * 30 for _ in range(3)]
        for k in range(3):
            for j in range(NDNO - k - 1):
                denom = NODES_NO[j + k + 1] - NODES_NO[j]
                self._etaNO[k][j] = 1.0 / denom if denom > 0.0 else 0.0

        self._zsfx = [False] * 384
        self._tsfx = [False] * 384
        self._psfx = [False] * 384
        self._zsfx[9] = self._zsfx[10] = True
        self._zsfx[13] = self._zsfx[14] = True
        self._zsfx[17] = self._zsfx[18] = True
        for i in range(CTIDE, min(CSPW, 384)):
            self._tsfx[i] = True
        for i in range(CSPW, min(CSPW + 60, 384)):
            self._psfx[i] = True

        self._smod = [False] * 24
        for ix in range(NL + 1):
            if (parm.TN_BETA[CSFXMOD, ix] != 0.0 or
                    parm.TN_BETA[CSFXMOD + 1, ix] != 0.0 or
                    parm.TN_BETA[CSFXMOD + 2, ix] != 0.0):
                self._smod[ix] = True

        gammaterm0 = math.tanh((ZETAREF_O1 - ZETA_GAMMA) * H_GAMMA)
        self._HRfactO1ref = 0.5 * (1.0 + gammaterm0)
        self._dHRfactO1ref = (
            (1.0 - (ZETAREF_O1 - ZETA_GAMMA) * (1.0 - gammaterm0) * H_GAMMA)
            / self._HRfactO1ref
        )

        gammaterm0 = math.tanh((ZETAREF_NO - ZETA_GAMMA) * H_GAMMA)
        self._HRfactNOref = 0.5 * (1.0 + gammaterm0)
        self._dHRfactNOref = (
            (1.0 - (ZETAREF_NO - ZETA_GAMMA) * (1.0 - gammaterm0) * H_GAMMA)
            / self._HRfactNOref
        )

    def _eval_at_altitude(self, zeta, tpro, dpros):
        """Altitude-dependent core of msiscalc. Returns (tn, dn[11], dT_dz)."""
        if zeta < ZETA_B:
            iz, S3, S4, S5, S6 = _bspline_eval(zeta, NODES_TN, ND + 2, self._etaTN)
        else:
            iz = NL
            S3 = [0.0] * 6
            S4 = [0.0] * 6
            S5 = [0.0] * 6
            S6 = [0.0] * 6

        tn = self._compute_temperature(zeta, tpro, iz, S4)

        delz = zeta - ZETA_B
        lndtotz = 0.0
        if zeta < ZETA_F:
            i_start = max(iz - 4, 0)
            j_start = -iz if iz < 4 else -4
            Vz = 0.0
            for l in range(j_start, 1):
                beta_idx = i_start + (l - j_start)
                Vz += tpro.beta[beta_idx] * S5[l + 4]
            Vz += tpro.cVS
            Wz = 0.0
            lnPz = LNP0 - MBARG0DIVKB * (Vz - tpro.Vzeta0)
            lndtotz = lnPz - math.log(KB * tn)
        elif zeta < ZETA_B:
            Vz = 0.0
            for k in range(5):
                idx = iz - 4 + k
                if 0 <= idx <= NL:
                    Vz += tpro.beta[idx] * S5[k]
            Vz += tpro.cVS
            Wz = self._compute_Wz(zeta, tpro, iz, S6)
        else:
            Vz = (delz + math.log(tn / tpro.tex) / tpro.sigma) / tpro.tex + tpro.cVB
            Wz = (
                (0.5 * delz * delz
                 + _dilog(tpro.b * math.exp(-tpro.sigma * delz)) / tpro.sigmasq)
                / tpro.tex + tpro.cVB * delz + tpro.cWB
            )

        HRfact = 0.5 * (1.0 + math.tanh(H_GAMMA * (zeta - ZETA_GAMMA)))

        dn = [DMISSING] * 11
        for ispec in range(2, 11):
            dn[ispec] = self._compute_dfnx(
                zeta, tn, lndtotz, Vz, Wz, HRfact, tpro, dpros[ispec]
            )

        dn[1] = 0.0
        for i in range(2, 10):
            if dn[i] != DMISSING and dn[i] > 0.0:
                dn[1] += dn[i] * SPECMASS[i]

        if zeta < ZETA_B:
            dT_dz = self._compute_dT_dz(zeta, tpro, iz, S3)
        else:
            dT_dz = tpro.sigma * (tpro.tex - tn)

        return tn, dn, dT_dz

    def _prepare_params(self, inp: MSISInput):
        """Compute altitude-independent parameters: bf, tpro, dpros."""
        bf = self._compute_globe(inp)
        tpro = self._compute_tfnparm(bf)
        dpros = [None, None]  # indices 0, 1 unused
        for ispec in range(2, 11):
            dpros.append(self._compute_dfnparm(ispec, bf, tpro))
        return bf, tpro, dpros

    def msiscalc(self, inp: MSISInput) -> MSISOutput:
        """
        Compute NRLMSIS output at a single altitude.

        Parameters
        ----------
        inp : MSISInput
            Input conditions (date, time, location, altitude, solar indices).

        Returns
        -------
        MSISOutput
            Temperature and species number densities.
        """
        bf, tpro, dpros = self._prepare_params(inp)
        zeta = _alt2gph(inp.lat, inp.alt)
        tn, dn, _ = self._eval_at_altitude(zeta, tpro, dpros)

        output = MSISOutput()
        output.tn = tn
        output.tex = tpro.tex
        output.dn = dn
        return output

    def msiscalc_with_derivative(self, inp: MSISInput) -> MSISFullOutput:
        """
        Compute NRLMSIS output with temperature altitude derivative.

        Parameters
        ----------
        inp : MSISInput
            Input conditions.

        Returns
        -------
        MSISFullOutput
            Temperature, densities, and dT/d(zeta) in K/km.
        """
        bf, tpro, dpros = self._prepare_params(inp)
        zeta = _alt2gph(inp.lat, inp.alt)
        tn, dn, dT_dz = self._eval_at_altitude(zeta, tpro, dpros)

        full = MSISFullOutput()
        full.output = MSISOutput()
        full.output.tn = tn
        full.output.tex = tpro.tex
        full.output.dn = dn
        full.dT_dz = dT_dz
        return full

    def msiscalc_profile(self, inp: MSISInput, altitudes_km: np.ndarray) -> MSISProfileOutput:
        """
        Compute NRLMSIS output for an array of altitudes.

        Shares all altitude-independent work (basis function evaluation,
        parameter fitting) across the altitude array, making this
        significantly faster than calling :meth:`msiscalc` in a loop.

        Parameters
        ----------
        inp : MSISInput
            Input conditions (altitude field is ignored; altitudes_km
            is used instead).
        altitudes_km : numpy.ndarray
            1-D array of geodetic altitudes in kilometers.

        Returns
        -------
        MSISProfileOutput
            Arrays of temperature, densities, and dT/dz at each altitude.
        """
        bf, tpro, dpros = self._prepare_params(inp)
        N = len(altitudes_km)

        tn_arr = np.empty(N)
        dn_arr = np.empty((N, 11))
        dT_dz_arr = np.empty(N)

        for i, alt_km in enumerate(altitudes_km):
            zeta = _alt2gph(inp.lat, alt_km)
            tn, dn, dT_dz = self._eval_at_altitude(zeta, tpro, dpros)
            tn_arr[i] = tn
            dn_arr[i] = dn
            dT_dz_arr[i] = dT_dz

        return MSISProfileOutput(
            tn=tn_arr, tex=tpro.tex, dn=dn_arr, dT_dz=dT_dz_arr
        )

    # -----------------------------------------------------------------
    # compute_globe
    # -----------------------------------------------------------------
    def _compute_globe(self, inp: MSISInput) -> np.ndarray:
        bf = np.zeros(512)

        lat_rad = inp.lat * DEG2RAD
        slat = math.sin(lat_rad)
        clat = math.cos(lat_rad)

        clat_leg = slat
        slat_leg = clat
        clat2 = clat_leg * clat_leg
        clat4 = clat2 * clat2
        slat2 = slat_leg * slat_leg

        plg = [[0.0] * 4 for _ in range(7)]
        plg[0][0] = 1.0
        plg[1][0] = clat_leg
        plg[2][0] = 0.5 * (3.0 * clat2 - 1.0)
        plg[3][0] = 0.5 * (5.0 * clat_leg * clat2 - 3.0 * clat_leg)
        plg[4][0] = (35.0 * clat4 - 30.0 * clat2 + 3.0) / 8.0
        plg[5][0] = (63.0 * clat2 * clat2 * clat_leg - 70.0 * clat2 * clat_leg + 15.0 * clat_leg) / 8.0
        plg[6][0] = (11.0 * clat_leg * plg[5][0] - 5.0 * plg[4][0]) / 6.0

        plg[1][1] = slat_leg
        plg[2][1] = 3.0 * clat_leg * slat_leg
        plg[3][1] = 1.5 * (5.0 * clat2 - 1.0) * slat_leg
        plg[4][1] = 2.5 * (7.0 * clat2 * clat_leg - 3.0 * clat_leg) * slat_leg
        plg[5][1] = 1.875 * (21.0 * clat4 - 14.0 * clat2 + 1.0) * slat_leg
        plg[6][1] = (11.0 * clat_leg * plg[5][1] - 6.0 * plg[4][1]) / 5.0

        plg[2][2] = 3.0 * slat2
        plg[3][2] = 15.0 * slat2 * clat_leg
        plg[4][2] = 7.5 * (7.0 * clat2 - 1.0) * slat2
        plg[5][2] = 3.0 * clat_leg * plg[4][2] - 2.0 * plg[3][2]
        plg[6][2] = (11.0 * clat_leg * plg[5][2] - 7.0 * plg[4][2]) / 4.0

        plg[3][3] = 15.0 * slat2 * slat_leg
        plg[4][3] = 105.0 * slat2 * slat_leg * clat_leg
        plg[5][3] = (9.0 * clat_leg * plg[4][3] - 7.0 * plg[3][3]) / 2.0
        plg[6][3] = (11.0 * clat_leg * plg[5][3] - 8.0 * plg[4][3]) / 3.0

        lst = inp.utsec / 3600.0 + inp.lon / 15.0
        lst = lst % 24.0
        if lst < 0.0:
            lst += 24.0

        cdoy1 = math.cos(DOY2RAD * inp.day)
        sdoy1 = math.sin(DOY2RAD * inp.day)
        cdoy2 = math.cos(DOY2RAD * inp.day * 2.0)
        sdoy2 = math.sin(DOY2RAD * inp.day * 2.0)

        clst1 = math.cos(LST2RAD * lst)
        slst1 = math.sin(LST2RAD * lst)
        clst2 = math.cos(LST2RAD * lst * 2.0)
        slst2 = math.sin(LST2RAD * lst * 2.0)
        clst3 = math.cos(LST2RAD * lst * 3.0)
        slst3 = math.sin(LST2RAD * lst * 3.0)

        clon1 = math.cos(DEG2RAD * inp.lon)
        slon1 = math.sin(DEG2RAD * inp.lon)
        clon2 = math.cos(DEG2RAD * inp.lon * 2.0)
        slon2 = math.sin(DEG2RAD * inp.lon * 2.0)

        c = CTIMEIND
        for n in range(MAXN + 1):
            bf[c] = plg[n][0]
            c += 1

        cdoy = [cdoy1, cdoy2]
        sdoy = [sdoy1, sdoy2]
        for s in range(2):
            for n in range(MAXN + 1):
                bf[c] = plg[n][0] * cdoy[s]
                bf[c + 1] = plg[n][0] * sdoy[s]
                c += 2

        clst = [clst1, clst2, clst3]
        slst = [slst1, slst2, slst3]
        for l in range(1, MAXL + 1):
            for n in range(l, MAXN + 1):
                bf[c] = plg[n][l] * clst[l - 1]
                bf[c + 1] = plg[n][l] * slst[l - 1]
                c += 2
            for s in range(2):
                for n in range(l, MAXN + 1):
                    bf[c] = plg[n][l] * clst[l - 1] * cdoy[s]
                    bf[c + 1] = plg[n][l] * slst[l - 1] * cdoy[s]
                    bf[c + 2] = plg[n][l] * clst[l - 1] * sdoy[s]
                    bf[c + 3] = plg[n][l] * slst[l - 1] * sdoy[s]
                    c += 4

        clon = [clon1, clon2]
        slon = [slon1, slon2]
        for m in range(1, MAXM + 1):
            for n in range(m, MAXN + 1):
                bf[c] = plg[n][m] * clon[m - 1]
                bf[c + 1] = plg[n][m] * slon[m - 1]
                c += 2
            for s in range(2):
                for n in range(m, MAXN + 1):
                    bf[c] = plg[n][m] * clon[m - 1] * cdoy[s]
                    bf[c + 1] = plg[n][m] * slon[m - 1] * cdoy[s]
                    bf[c + 2] = plg[n][m] * clon[m - 1] * sdoy[s]
                    bf[c + 3] = plg[n][m] * slon[m - 1] * sdoy[s]
                    c += 4

        dfa = inp.f107a - 150.0
        df = inp.f107 - inp.f107a
        bf[CSFX] = dfa
        bf[CSFX + 1] = dfa * dfa
        bf[CSFX + 2] = df
        bf[CSFX + 3] = df * df
        bf[CSFX + 4] = df * dfa

        sza = _solzen(inp.day, lst, inp.lat, inp.lon)
        bf[300] = -0.5 * math.tanh((sza - 98.0) / 6.0)
        bf[301] = -0.5 * math.tanh((sza - 101.5) / 20.0)
        bf[302] = dfa * bf[300]
        bf[303] = dfa * bf[301]
        bf[304] = dfa * plg[2][0]
        bf[305] = dfa * plg[4][0]
        bf[306] = dfa * plg[0][0] * cdoy1
        bf[307] = dfa * plg[0][0] * sdoy1
        bf[308] = dfa * plg[0][0] * cdoy2
        bf[309] = dfa * plg[0][0] * sdoy2
        sfluxavg_quad_cutoff = 150.0
        sfluxavgref = 150.0
        if inp.f107a <= sfluxavg_quad_cutoff:
            dfa_trunc = dfa * dfa
        else:
            dfa_trunc = (sfluxavg_quad_cutoff - sfluxavgref) * (
                2.0 * dfa - (sfluxavg_quad_cutoff - sfluxavgref)
            )
        bf[310] = dfa_trunc
        bf[311] = dfa_trunc * plg[2][0]
        bf[312] = dfa_trunc * plg[4][0]
        bf[313] = df * plg[2][0]
        bf[314] = df * plg[4][0]

        bf[CSFXMOD] = dfa
        bf[CSFXMOD + 1] = dfa * dfa
        bf[CSFXMOD + 2] = df
        bf[CSFXMOD + 3] = df * df
        bf[CSFXMOD + 4] = df * dfa

        for i in range(7):
            bf[CMAG + i] = inp.ap[i] - 4.0

        doy_rad = DOY2RAD * inp.day
        lon_rad = DEG2RAD * inp.lon
        ut_rad = LST2RAD * (inp.utsec / 3600.0)
        bf[CMAG + 8] = doy_rad
        bf[CMAG + 9] = LST2RAD * lst
        bf[CMAG + 10] = lon_rad
        bf[CMAG + 11] = ut_rad
        bf[CMAG + 12] = abs(inp.lat)

        bf[CMAG + 13] = plg[0][0]
        bf[CMAG + 14] = plg[1][0]
        bf[CMAG + 15] = plg[2][0]
        bf[CMAG + 16] = plg[3][0]
        bf[CMAG + 17] = plg[4][0]
        bf[CMAG + 18] = plg[5][0]
        bf[CMAG + 19] = plg[6][0]
        bf[CMAG + 20] = 0.0
        bf[CMAG + 21] = plg[1][1]
        bf[CMAG + 22] = plg[2][1]
        bf[CMAG + 23] = plg[3][1]
        bf[CMAG + 24] = plg[4][1]
        bf[CMAG + 25] = plg[5][1]
        bf[CMAG + 26] = plg[6][1]

        bf[CUT] = ut_rad
        bf[CUT + 1] = doy_rad
        bf[CUT + 2] = dfa
        bf[CUT + 3] = lon_rad
        bf[CUT + 4] = plg[1][0]
        bf[CUT + 5] = plg[3][0]
        bf[CUT + 6] = plg[5][0]
        bf[CUT + 7] = plg[3][2]
        bf[CUT + 8] = plg[5][2]

        return bf

    # -----------------------------------------------------------------
    # sfluxmod
    # -----------------------------------------------------------------
    def _sfluxmod(self, iz, gf, beta_arr, dffact):
        f1 = (beta_arr[CSFXMOD, iz] * gf[CSFXMOD]
              + (beta_arr[CSFX + 2, iz] * gf[CSFXMOD + 2]
                 + beta_arr[CSFX + 3, iz] * gf[CSFXMOD + 3]) * dffact)
        f2 = (beta_arr[CSFXMOD + 1, iz] * gf[CSFXMOD]
              + (beta_arr[CSFX + 2, iz] * gf[CSFXMOD + 2]
                 + beta_arr[CSFX + 3, iz] * gf[CSFXMOD + 3]) * dffact)
        f3 = beta_arr[CSFXMOD + 2, iz] * gf[CSFXMOD]

        total = 0.0
        for j in range(MBF + 1):
            if self._zsfx[j]:
                total += beta_arr[j, iz] * gf[j] * f1
            elif self._tsfx[j]:
                total += beta_arr[j, iz] * gf[j] * f2
            elif self._psfx[j]:
                total += beta_arr[j, iz] * gf[j] * f3
        return total

    def _sfluxmod_tn(self, iz, gf, dffact):
        return self._sfluxmod(iz, gf, parm.TN_BETA, dffact)

    # -----------------------------------------------------------------
    # geomag and UT dependence
    # -----------------------------------------------------------------
    @staticmethod
    def _geomag(beta_arr, level, bf):
        k00r = beta_arr[CMAG, level]
        k00s = beta_arr[CMAG + 1, level]
        if k00s == 0.0:
            return 0.0

        def G0fn(a, k00r_, k00s_):
            return a + (k00r_ - 1.0) * (a + (math.exp(-a * k00s_) - 1.0) / k00s_)

        delA = G0fn(bf[CMAG], k00r, k00s)
        plg0 = bf[CMAG + 13:CMAG + 20]
        plg1 = bf[CMAG + 20:CMAG + 27]

        p = [beta_arr[CMAG + i, level] for i in range(2, 26)]

        doy_rad = bf[CMAG + 8]
        lst_rad = bf[CMAG + 9]
        lon_rad = bf[CMAG + 10]
        ut_rad = bf[CMAG + 11]

        result = (
            (p[0] * plg0[0] + p[1] * plg0[2] + p[2] * plg0[4]
             + (p[3] * plg0[1] + p[4] * plg0[3] + p[5] * plg0[5]) * math.cos(doy_rad - p[6])
             + (p[7] * plg1[1] + p[8] * plg1[3] + p[9] * plg1[5]) * math.cos(lst_rad - p[10])
             + (1.0 + p[11] * plg0[1])
             * (p[12] * plg1[2] + p[13] * plg1[4] + p[14] * plg1[6]) * math.cos(lon_rad - p[15])
             + (p[16] * plg1[1] + p[17] * plg1[3] + p[18] * plg1[5]) * math.cos(lon_rad - p[19])
             * math.cos(doy_rad - p[6])
             + (p[20] * plg0[1] + p[21] * plg0[3] + p[22] * plg0[5]) * math.cos(ut_rad - p[23]))
            * delA
        )
        return result

    @staticmethod
    def _utdep(beta_arr, level, bf):
        p = [beta_arr[CUT + i, level] for i in range(12)]

        ut_rad = bf[CUT]
        doy_rad = bf[CUT + 1]
        dfa = bf[CUT + 2]
        lon_rad = bf[CUT + 3]
        plg10 = bf[CUT + 4]
        plg30 = bf[CUT + 5]
        plg50 = bf[CUT + 6]
        plg32 = bf[CUT + 7]
        plg52 = bf[CUT + 8]

        return (
            math.cos(ut_rad - p[0])
            * (1.0 + p[3] * plg10 * math.cos(doy_rad - p[1]))
            * (1.0 + p[4] * dfa) * (1.0 + p[5] * plg10)
            * (p[6] * plg10 + p[7] * plg30 + p[8] * plg50)
            + math.cos(ut_rad - p[2] + 2.0 * lon_rad) * (p[9] * plg32 + p[10] * plg52) * (1.0 + p[11] * dfa)
        )

    # -----------------------------------------------------------------
    # compute_tfnparm
    # -----------------------------------------------------------------
    def _compute_tfnparm(self, bf):
        tpro = _TnParm()

        for ix in range(ITB0):
            tpro.cf[ix] = float(np.dot(parm.TN_BETA[:MBF + 1, ix], bf[:MBF + 1]))
            if self._smod[ix]:
                dffact_ix = 1.0 / parm.TN_BETA[0, ix]
                tpro.cf[ix] += self._sfluxmod_tn(ix, bf, dffact_ix)

        tpro.tex = float(np.dot(parm.TN_BETA[:MBF + 1, ITEX], bf[:MBF + 1]))
        tpro.tex += self._sfluxmod_tn(ITEX, bf, 1.0 / parm.TN_BETA[0, ITEX])
        tpro.tex += self._geomag(parm.TN_BETA, ITEX, bf)
        tpro.tex += self._utdep(parm.TN_BETA, ITEX, bf)

        tpro.tgb0 = float(np.dot(parm.TN_BETA[:MBF + 1, ITGB0], bf[:MBF + 1]))
        if self._smod[ITGB0]:
            tpro.tgb0 += self._sfluxmod_tn(ITGB0, bf, 1.0 / parm.TN_BETA[0, ITGB0])
        tpro.tgb0 += self._geomag(parm.TN_BETA, ITGB0, bf)

        tpro.tb0 = float(np.dot(parm.TN_BETA[:MBF + 1, ITB0], bf[:MBF + 1]))
        if self._smod[ITB0]:
            tpro.tb0 += self._sfluxmod_tn(ITB0, bf, 1.0 / parm.TN_BETA[0, ITB0])
        tpro.tb0 += self._geomag(parm.TN_BETA, ITB0, bf)

        tpro.sigma = tpro.tgb0 / (tpro.tex - tpro.tb0)

        bc1 = 1.0 / tpro.tb0
        bc2 = -tpro.tgb0 / (tpro.tb0 * tpro.tb0)
        bc3 = -bc2 * (tpro.sigma + 2.0 * tpro.tgb0 / tpro.tb0)

        tpro.cf[ITB0] = bc1 * C2TN[0][0] + bc2 * C2TN[1][0] + bc3 * C2TN[2][0]
        tpro.cf[ITGB0] = bc1 * C2TN[0][1] + bc2 * C2TN[1][1] + bc3 * C2TN[2][1]
        tpro.cf[ITEX] = bc1 * C2TN[0][2] + bc2 * C2TN[1][2] + bc3 * C2TN[2][2]

        dot_zetaF = (tpro.cf[IZFX] * S4ZETA_F[0]
                     + tpro.cf[IZFX + 1] * S4ZETA_F[1]
                     + tpro.cf[IZFX + 2] * S4ZETA_F[2])
        tpro.tzetaF = 1.0 / dot_zetaF

        dot_zetaA = (tpro.cf[IZAX] * S4ZETA_A[0]
                     + tpro.cf[IZAX + 1] * S4ZETA_A[1]
                     + tpro.cf[IZAX + 2] * S4ZETA_A[2])
        tpro.tzetaA = 1.0 / dot_zetaA

        dfdz_A = (tpro.cf[IZAX] * WGHTAXDZ[0]
                  + tpro.cf[IZAX + 1] * WGHTAXDZ[1]
                  + tpro.cf[IZAX + 2] * WGHTAXDZ[2])
        tpro.dlntdzA = -tpro.tzetaA * dfdz_A

        tpro.beta[0] = tpro.cf[0] * (NODES_TN[4] - NODES_TN[0]) / 4.0
        for ix in range(1, NL + 1):
            wbeta_ix = (NODES_TN[ix + 4] - NODES_TN[ix]) / 4.0
            tpro.beta[ix] = tpro.beta[ix - 1] + tpro.cf[ix] * wbeta_ix

        tpro.gamma[0] = tpro.beta[0] * (NODES_TN[5] - NODES_TN[0]) / 5.0
        for ix in range(1, NL + 1):
            wgamma_ix = (NODES_TN[ix + 5] - NODES_TN[ix]) / 5.0
            tpro.gamma[ix] = tpro.gamma[ix - 1] + tpro.beta[ix] * wgamma_ix

        tpro.b = 1.0 - tpro.tb0 / tpro.tex
        tpro.sigmasq = tpro.sigma * tpro.sigma

        tpro.cVS = -(
            tpro.beta[ITB0 - 1] * S5ZETA_B[0]
            + tpro.beta[ITB0] * S5ZETA_B[1]
            + tpro.beta[ITB0 + 1] * S5ZETA_B[2]
            + tpro.beta[ITB0 + 2] * S5ZETA_B[3]
        )

        tpro.cWS = -(
            tpro.gamma[ITB0 - 2] * S6ZETA_B[0]
            + tpro.gamma[ITB0 - 1] * S6ZETA_B[1]
            + tpro.gamma[ITB0] * S6ZETA_B[2]
            + tpro.gamma[ITB0 + 1] * S6ZETA_B[3]
            + tpro.gamma[ITB0 + 2] * S6ZETA_B[4]
        )

        tpro.cVB = -math.log(1.0 - tpro.b) / (tpro.sigma * tpro.tex)
        tpro.cWB = -_dilog(tpro.b) / (tpro.sigmasq * tpro.tex)

        tpro.VzetaF = (
            tpro.beta[IZFX - 1] * S5ZETA_F[0]
            + tpro.beta[IZFX] * S5ZETA_F[1]
            + tpro.beta[IZFX + 1] * S5ZETA_F[2]
            + tpro.beta[IZFX + 2] * S5ZETA_F[3]
            + tpro.cVS
        )

        tpro.VzetaA = (
            tpro.beta[IZAX - 1] * S5ZETA_A[0]
            + tpro.beta[IZAX] * S5ZETA_A[1]
            + tpro.beta[IZAX + 1] * S5ZETA_A[2]
            + tpro.beta[IZAX + 2] * S5ZETA_A[3]
            + tpro.cVS
        )

        delzA = ZETA_A - ZETA_B
        tpro.WzetaA = (
            tpro.gamma[IZAX - 2] * S6ZETA_A[0]
            + tpro.gamma[IZAX - 1] * S6ZETA_A[1]
            + tpro.gamma[IZAX] * S6ZETA_A[2]
            + tpro.gamma[IZAX + 1] * S6ZETA_A[3]
            + tpro.gamma[IZAX + 2] * S6ZETA_A[4]
            + tpro.cVS * delzA + tpro.cWS
        )

        tpro.Vzeta0 = (
            tpro.beta[0] * S5ZETA_0[0]
            + tpro.beta[1] * S5ZETA_0[1]
            + tpro.beta[2] * S5ZETA_0[2]
            + tpro.cVS
        )

        tpro.lndtotF = LNP0 - MBARG0DIVKB * (tpro.VzetaF - tpro.Vzeta0) - math.log(KB * tpro.tzetaF)

        return tpro

    # -----------------------------------------------------------------
    # pwmp - piecewise molecular weight profile
    # -----------------------------------------------------------------
    @staticmethod
    def _pwmp(z, dpro):
        if z >= dpro.zetaMi[4]:
            return dpro.Mi[4]
        if z <= dpro.zetaMi[0]:
            return dpro.Mi[0]
        for i in range(4):
            if z < dpro.zetaMi[i + 1]:
                return dpro.Mi[i] + dpro.aMi[i] * (z - dpro.zetaMi[i])
        return dpro.Mi[4]

    # -----------------------------------------------------------------
    # compute_dfnparm
    # -----------------------------------------------------------------
    def _compute_dfnparm(self, ispec, bf, tpro):
        dpro = _DnParm()
        dpro.ispec = ispec

        def dot_bf(beta_arr, iz_):
            return float(np.dot(beta_arr[:MBF + 1, iz_], bf[:MBF + 1]))

        if ispec == SPEC_N2:
            dpro.lnPhiF = LNVMR[ispec]
            dpro.lndref = tpro.lndtotF + dpro.lnPhiF
            dpro.zref = ZETA_F
            dpro.zmin = -1.0
            dpro.zhyd = ZETA_F
            dpro.zetaM = dot_bf(parm.N2_BETA, 1)
            dpro.HML = parm.N2_BETA[0, 2]
            dpro.HMU = parm.N2_BETA[0, 3]
            dpro.C = 0.0
            dpro.zetaC = 0.0
            dpro.HC = 1.0
            dpro.R = 0.0
            dpro.zetaR = parm.N2_BETA[0, 8]
            dpro.HR = parm.N2_BETA[0, 9]

        elif ispec == SPEC_O2:
            dpro.lnPhiF = LNVMR[ispec]
            dpro.lndref = tpro.lndtotF + dpro.lnPhiF
            dpro.zref = ZETA_F
            dpro.zmin = -1.0
            dpro.zhyd = ZETA_F
            dpro.zetaM = parm.O2_BETA[0, 1]
            dpro.HML = parm.O2_BETA[0, 2]
            dpro.HMU = parm.O2_BETA[0, 3]
            dpro.C = 0.0
            dpro.zetaC = 0.0
            dpro.HC = 1.0
            dpro.R = dot_bf(parm.O2_BETA, 7)
            dpro.R += self._geomag(parm.O2_BETA, 7, bf)
            dpro.zetaR = parm.O2_BETA[0, 8]
            dpro.HR = parm.O2_BETA[0, 9]

        elif ispec == SPEC_O:
            dpro.lnPhiF = 0.0
            dpro.lndref = dot_bf(parm.O1_BETA, 0)
            dpro.zref = ZETAREF_O1
            dpro.zmin = NODES_O1[3]
            dpro.zhyd = ZETAREF_O1
            dpro.zetaM = parm.O1_BETA[0, 1]
            dpro.HML = parm.O1_BETA[0, 2]
            dpro.HMU = parm.O1_BETA[0, 3]
            dpro.C = dot_bf(parm.O1_BETA, 4)
            dpro.zetaC = parm.O1_BETA[0, 5]
            dpro.HC = parm.O1_BETA[0, 6]
            dpro.R = dot_bf(parm.O1_BETA, 7)
            dpro.R += self._sfluxmod(7, bf, parm.O1_BETA, 0.0)
            dpro.R += self._geomag(parm.O1_BETA, 7, bf)
            dpro.R += self._utdep(parm.O1_BETA, 7, bf)
            dpro.zetaR = parm.O1_BETA[0, 8]
            dpro.HR = parm.O1_BETA[0, 9]
            for k in range(NSPLO1):
                dpro.cf[k] = dot_bf(parm.O1_BETA, 10 + k)

        elif ispec == SPEC_HE:
            dpro.lnPhiF = LNVMR[ispec]
            dpro.lndref = tpro.lndtotF + dpro.lnPhiF
            dpro.zref = ZETA_F
            dpro.zmin = -1.0
            dpro.zhyd = ZETA_F
            dpro.zetaM = parm.HE_BETA[0, 1]
            dpro.HML = parm.HE_BETA[0, 2]
            dpro.HMU = parm.HE_BETA[0, 3]
            dpro.C = 0.0
            dpro.zetaC = 0.0
            dpro.HC = 1.0
            dpro.R = dot_bf(parm.HE_BETA, 7)
            dpro.R += self._sfluxmod(7, bf, parm.HE_BETA, 1.0)
            dpro.R += self._geomag(parm.HE_BETA, 7, bf)
            dpro.R += self._utdep(parm.HE_BETA, 7, bf)
            dpro.zetaR = parm.HE_BETA[0, 8]
            dpro.HR = parm.HE_BETA[0, 9]

        elif ispec == SPEC_H:
            dpro.lnPhiF = 0.0
            dpro.lndref = dot_bf(parm.H1_BETA, 0)
            dpro.zref = ZETA_A
            dpro.zmin = 75.0
            dpro.zhyd = ZETA_F
            dpro.zetaM = parm.H1_BETA[0, 1]
            dpro.HML = parm.H1_BETA[0, 2]
            dpro.HMU = parm.H1_BETA[0, 3]
            dpro.C = dot_bf(parm.H1_BETA, 4)
            dpro.zetaC = dot_bf(parm.H1_BETA, 5)
            dpro.HC = parm.H1_BETA[0, 6]
            dpro.R = dot_bf(parm.H1_BETA, 7)
            dpro.R += self._sfluxmod(7, bf, parm.H1_BETA, 0.0)
            dpro.R += self._geomag(parm.H1_BETA, 7, bf)
            dpro.R += self._utdep(parm.H1_BETA, 7, bf)
            dpro.zetaR = parm.H1_BETA[0, 8]
            dpro.HR = parm.H1_BETA[0, 9]

        elif ispec == SPEC_AR:
            dpro.lnPhiF = LNVMR[ispec]
            dpro.lndref = tpro.lndtotF + dpro.lnPhiF
            dpro.zref = ZETA_F
            dpro.zmin = -1.0
            dpro.zhyd = ZETA_F
            dpro.zetaM = parm.AR_BETA[0, 1]
            dpro.HML = parm.AR_BETA[0, 2]
            dpro.HMU = parm.AR_BETA[0, 3]
            dpro.C = 0.0
            dpro.zetaC = 0.0
            dpro.HC = 1.0
            dpro.R = dot_bf(parm.AR_BETA, 7)
            dpro.R += self._geomag(parm.AR_BETA, 7, bf)
            dpro.R += self._utdep(parm.AR_BETA, 7, bf)
            dpro.zetaR = parm.AR_BETA[0, 8]
            dpro.HR = parm.AR_BETA[0, 9]

        elif ispec == SPEC_N:
            dpro.lnPhiF = 0.0
            dpro.lndref = dot_bf(parm.N1_BETA, 0)
            dpro.lndref += self._sfluxmod(0, bf, parm.N1_BETA, 0.0)
            dpro.lndref += self._geomag(parm.N1_BETA, 0, bf)
            dpro.lndref += self._utdep(parm.N1_BETA, 0, bf)
            dpro.zref = ZETA_B
            dpro.zmin = 90.0
            dpro.zhyd = ZETA_F
            dpro.zetaM = parm.N1_BETA[0, 1]
            dpro.HML = parm.N1_BETA[0, 2]
            dpro.HMU = parm.N1_BETA[0, 3]
            dpro.C = parm.N1_BETA[0, 4]
            dpro.zetaC = parm.N1_BETA[0, 5]
            dpro.HC = parm.N1_BETA[0, 6]
            dpro.R = dot_bf(parm.N1_BETA, 7)
            dpro.zetaR = parm.N1_BETA[0, 8]
            dpro.HR = parm.N1_BETA[0, 9]

        elif ispec == SPEC_OA:
            dpro.lnPhiF = 0.0
            dpro.lndref = dot_bf(parm.OA_BETA, 0)
            dpro.lndref += self._geomag(parm.OA_BETA, 0, bf)
            dpro.zref = ZETAREF_OA
            dpro.zmin = 120.0
            dpro.zhyd = 0.0
            dpro.C = parm.OA_BETA[0, 4]
            dpro.zetaC = parm.OA_BETA[0, 5]
            dpro.HC = parm.OA_BETA[0, 6]
            return dpro

        elif ispec == SPEC_NO:
            dpro.lnPhiF = 0.0
            dpro.lndref = dot_bf(parm.NO_BETA, 0)
            dpro.lndref += self._geomag(parm.NO_BETA, 0, bf)
            dpro.zref = ZETAREF_NO
            dpro.zmin = 72.5
            dpro.zhyd = ZETAREF_NO
            dpro.zetaM = dot_bf(parm.NO_BETA, 1)
            dpro.HML = dot_bf(parm.NO_BETA, 2)
            dpro.HMU = dot_bf(parm.NO_BETA, 3)
            dpro.C = dot_bf(parm.NO_BETA, 4)
            dpro.C += self._geomag(parm.NO_BETA, 4, bf)
            dpro.zetaC = dot_bf(parm.NO_BETA, 5)
            dpro.HC = dot_bf(parm.NO_BETA, 6)
            dpro.R = dot_bf(parm.NO_BETA, 7)
            dpro.zetaR = dot_bf(parm.NO_BETA, 8)
            dpro.HR = dot_bf(parm.NO_BETA, 9)
            for k in range(NSPLNO):
                dpro.cf[k] = dot_bf(parm.NO_BETA, 10 + k)
                dpro.cf[k] += self._geomag(parm.NO_BETA, 10 + k, bf)
        else:
            return dpro

        if ispec != SPEC_OA:
            dpro.zetaMi[0] = dpro.zetaM - 2.0 * dpro.HML
            dpro.zetaMi[1] = dpro.zetaM - dpro.HML
            dpro.zetaMi[2] = dpro.zetaM
            dpro.zetaMi[3] = dpro.zetaM + dpro.HMU
            dpro.zetaMi[4] = dpro.zetaM + 2.0 * dpro.HMU

            dpro.Mi[0] = MBAR
            dpro.Mi[4] = SPECMASS[ispec]
            dpro.Mi[2] = (dpro.Mi[0] + dpro.Mi[4]) / 2.0
            delM = TANH1 * (dpro.Mi[4] - dpro.Mi[0]) / 2.0
            dpro.Mi[1] = dpro.Mi[2] - delM
            dpro.Mi[3] = dpro.Mi[2] + delM

            for i in range(4):
                dpro.aMi[i] = (dpro.Mi[i + 1] - dpro.Mi[i]) / (dpro.zetaMi[i + 1] - dpro.zetaMi[i])

            for i in range(5):
                delz = dpro.zetaMi[i] - ZETA_B
                if dpro.zetaMi[i] < ZETA_B:
                    iz_, S3_, S4_, S5_, S6_ = _bspline_eval(
                        dpro.zetaMi[i], NODES_TN, ND + 2, self._etaTN
                    )
                    W = 0.0
                    for j in range(6):
                        idx = iz_ - 5 + j
                        if 0 <= idx <= NL:
                            W += tpro.gamma[idx] * S6_[j]
                    dpro.WMi[i] = W + tpro.cVS * delz + tpro.cWS
                else:
                    dpro.WMi[i] = (
                        (0.5 * delz * delz
                         + _dilog(tpro.b * math.exp(-tpro.sigma * delz)) / tpro.sigmasq)
                        / tpro.tex + tpro.cVB * delz + tpro.cWB
                    )

            dpro.XMi[0] = -dpro.aMi[0] * dpro.WMi[0]
            for i in range(1, 4):
                dpro.XMi[i] = dpro.XMi[i - 1] - dpro.WMi[i] * (dpro.aMi[i] - dpro.aMi[i - 1])
            dpro.XMi[4] = dpro.XMi[3] + dpro.WMi[4] * dpro.aMi[3]

            if dpro.zref == ZETA_F:
                dpro.Mzref = MBAR
                dpro.Tref = tpro.tzetaF
                dpro.Izref = MBAR * tpro.VzetaF
            elif dpro.zref == ZETA_A:
                dpro.Mzref = self._pwmp(ZETA_A, dpro)
                dpro.Tref = tpro.tzetaA
                Izref_raw = dpro.Mzref * tpro.VzetaA
                dpro.Izref = Izref_raw
                if ZETA_A > dpro.zetaMi[0] and ZETA_A < dpro.zetaMi[4]:
                    seg = 0
                    for i in range(1, 4):
                        if ZETA_A >= dpro.zetaMi[i]:
                            seg = i
                    adj = dpro.aMi[seg] * tpro.WzetaA + dpro.XMi[seg]
                    dpro.Izref -= adj
                elif ZETA_A >= dpro.zetaMi[4]:
                    dpro.Izref -= dpro.XMi[4]
            elif dpro.zref == ZETA_B:
                dpro.Mzref = self._pwmp(ZETA_B, dpro)
                dpro.Tref = tpro.tb0
                dpro.Izref = 0.0
                if ZETA_B > dpro.zetaMi[0] and ZETA_B < dpro.zetaMi[4]:
                    seg = 0
                    for i in range(1, 4):
                        if ZETA_B >= dpro.zetaMi[i]:
                            seg = i
                    dpro.Izref -= dpro.XMi[seg]
                elif ZETA_B >= dpro.zetaMi[4]:
                    dpro.Izref -= dpro.XMi[4]

        if ispec == SPEC_O:
            Cterm = dpro.C * math.exp(-(dpro.zref - dpro.zetaC) / dpro.HC)
            Rterm0 = math.tanh((dpro.zref - dpro.zetaR) / (self._HRfactO1ref * dpro.HR))
            Rterm = dpro.R * (1.0 + Rterm0)

            bc0 = (dpro.lndref - Cterm + Rterm
                   - dpro.cf[NSPLO1 - 1] * C1O1ADJ[0])
            bc1 = (-dpro.Mzref * G0DIVKB / tpro.tzetaA
                   - tpro.dlntdzA
                   + Cterm / dpro.HC
                   + Rterm * (1.0 - Rterm0) / dpro.HR * self._dHRfactO1ref
                   - dpro.cf[NSPLO1 - 1] * C1O1ADJ[1])

            dpro.cf[NSPLO1] = bc0 * C1O1[0][0] + bc1 * C1O1[1][0]
            dpro.cf[NSPLO1 + 1] = bc0 * C1O1[0][1] + bc1 * C1O1[1][1]

        if ispec == SPEC_NO and dpro.lndref != 0.0:
            Cterm = dpro.C * math.exp(-(dpro.zref - dpro.zetaC) / dpro.HC)
            Rterm0 = math.tanh((dpro.zref - dpro.zetaR) / (self._HRfactNOref * dpro.HR))
            Rterm = dpro.R * (1.0 + Rterm0)

            bc0 = (dpro.lndref - Cterm + Rterm
                   - dpro.cf[NSPLNO - 1] * C1NOADJ[0])
            bc1 = (-dpro.Mzref * G0DIVKB / tpro.tb0
                   - tpro.tgb0 / tpro.tb0
                   + Cterm / dpro.HC
                   + Rterm * (1.0 - Rterm0) / dpro.HR * self._dHRfactNOref
                   - dpro.cf[NSPLNO - 1] * C1NOADJ[1])

            dpro.cf[NSPLNO] = bc0 * C1NO[0][0] + bc1 * C1NO[1][0]
            dpro.cf[NSPLNO + 1] = bc0 * C1NO[0][1] + bc1 * C1NO[1][1]

        return dpro

    # -----------------------------------------------------------------
    # compute_dfnx - species density at altitude
    # -----------------------------------------------------------------
    def _compute_dfnx(self, zeta, tn, lndtotz, Vz, Wz, HRfact, tpro, dpro):
        if zeta < dpro.zmin:
            return DMISSING

        if dpro.ispec == SPEC_OA:
            lnd = dpro.lndref - (zeta - dpro.zref) / H_OA
            if dpro.C != 0.0:
                lnd -= dpro.C * math.exp(-(zeta - dpro.zetaC) / dpro.HC)
            return math.exp(lnd)

        if dpro.ispec == SPEC_NO and dpro.lndref == 0.0:
            return DMISSING

        ccor = 0.0
        if dpro.ispec in (SPEC_N2, SPEC_O2, SPEC_HE, SPEC_AR):
            ccor = dpro.R * (1.0 + math.tanh((zeta - dpro.zetaR) / (HRfact * dpro.HR)))
        elif dpro.ispec in (SPEC_O, SPEC_H, SPEC_N, SPEC_NO):
            ccor = (
                -dpro.C * math.exp(-(zeta - dpro.zetaC) / dpro.HC)
                + dpro.R * (1.0 + math.tanh((zeta - dpro.zetaR) / (HRfact * dpro.HR)))
            )

        if zeta < dpro.zhyd:
            if dpro.ispec in (SPEC_N2, SPEC_O2, SPEC_HE, SPEC_AR):
                return math.exp(lndtotz + dpro.lnPhiF + ccor)
            if dpro.ispec == SPEC_O:
                iz_, S4_ = _bspline_eval_o4(zeta, NODES_O1, NDO1 + 1, self._etaO1)
                lnf = 0.0
                for k in range(4):
                    idx = iz_ - 3 + k
                    if 0 <= idx < NSPLO1 + 2:
                        lnf += dpro.cf[idx] * S4_[k]
                return math.exp(lnf)
            if dpro.ispec == SPEC_NO:
                iz_, S4_ = _bspline_eval_o4(zeta, NODES_NO, NDNO + 1, self._etaNO)
                lnf = 0.0
                for k in range(4):
                    idx = iz_ - 3 + k
                    if 0 <= idx < NSPLNO + 2:
                        lnf += dpro.cf[idx] * S4_[k]
                return math.exp(lnf)
            return DMISSING

        Mz = self._pwmp(zeta, dpro)
        Ihyd = Mz * Vz - dpro.Izref

        if zeta > dpro.zetaMi[0] and zeta < dpro.zetaMi[4]:
            seg = 0
            for i in range(1, 4):
                if zeta >= dpro.zetaMi[i]:
                    seg = i
            Ihyd -= (dpro.aMi[seg] * Wz + dpro.XMi[seg])
        elif zeta >= dpro.zetaMi[4]:
            Ihyd -= dpro.XMi[4]

        lnd = dpro.lndref - Ihyd * G0DIVKB + ccor
        return math.exp(lnd) * dpro.Tref / tn

    # -----------------------------------------------------------------
    # compute_temperature
    # -----------------------------------------------------------------
    @staticmethod
    def _compute_temperature(zeta, tpro, iz, S4):
        if zeta < ZETA_B:
            i = max(iz - 3, 0)
            j_start = -iz if iz < 3 else -3
            dot = 0.0
            for j in range(j_start, 1):
                dot += tpro.cf[i + j - j_start] * S4[j + 3]
            return 1.0 / dot
        else:
            return tpro.tex - (tpro.tex - tpro.tb0) * math.exp(-tpro.sigma * (zeta - ZETA_B))

    # -----------------------------------------------------------------
    # compute_Wz
    # -----------------------------------------------------------------
    def _compute_Wz(self, zeta, tpro, iz, S6):
        delz = zeta - ZETA_B
        if zeta < ZETA_B:
            W = 0.0
            for j in range(6):
                idx = iz - 5 + j
                if 0 <= idx <= NL:
                    W += tpro.gamma[idx] * S6[j]
            return W + tpro.cVS * delz + tpro.cWS
        else:
            return (
                (0.5 * delz * delz
                 + _dilog(tpro.b * math.exp(-tpro.sigma * delz)) / tpro.sigmasq)
                / tpro.tex + tpro.cVB * delz + tpro.cWB
            )

    # -----------------------------------------------------------------
    # compute_dT_dz
    # -----------------------------------------------------------------
    def _compute_dT_dz(self, zeta, tpro, iz, S3):
        if zeta >= ZETA_B:
            T = tpro.tex - (tpro.tex - tpro.tb0) * math.exp(-tpro.sigma * (zeta - ZETA_B))
            return tpro.sigma * (tpro.tex - T)

        dfdz = 0.0
        for j in range(3):
            m = iz - 2 + j
            if 1 <= m <= NL:
                dcf = tpro.cf[m] - tpro.cf[m - 1]
                dnode = NODES_TN[m + 3] - NODES_TN[m]
                if dnode > 0.0:
                    dfdz += dcf * 3.0 / dnode * S3[j]

        iz2, S3_2, S4_2, S5_2, S6_2 = _bspline_eval(zeta, NODES_TN, ND + 2, self._etaTN)
        T = self._compute_temperature(zeta, tpro, iz2, S4_2)

        return -T * T * dfdz
