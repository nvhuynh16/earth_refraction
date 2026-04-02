"""WGS-84 geodetic coordinate utilities.

Provides conversions between geodetic (latitude, longitude, height above
ellipsoid) and ECEF (Earth-Centered Earth-Fixed) Cartesian coordinates,
plus local ENU frame construction, principal radii of curvature, and the
Jacobian of the geodetic normal for the eikonal ray tracer.

All public functions accept angles in **degrees** and heights in **metres**.
"""

from __future__ import annotations

import math

import numpy as np

from .constants import WGS84_A, WGS84_B, WGS84_E2, WGS84_F


# ---------------------------------------------------------------------------
# Coordinate conversions
# ---------------------------------------------------------------------------

def geodetic_to_ecef(lat_deg: float, lon_deg: float, h_m: float) -> np.ndarray:
    """Convert geodetic coordinates to ECEF.

    Parameters
    ----------
    lat_deg : float
        Geodetic latitude (deg).
    lon_deg : float
        Geodetic longitude (deg).
    h_m : float
        Height above the WGS-84 ellipsoid (m).

    Returns
    -------
    numpy.ndarray
        ECEF position [X, Y, Z] in metres, shape (3,).
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    X = (N + h_m) * cos_lat * cos_lon
    Y = (N + h_m) * cos_lat * sin_lon
    Z = (N * (1.0 - WGS84_E2) + h_m) * sin_lat
    return np.array([X, Y, Z])


def ecef_to_geodetic(
    X: float, Y: float, Z: float
) -> tuple[float, float, float]:
    """Convert ECEF coordinates to geodetic.

    Uses Bowring's iterative method (2--3 iterations to sub-nanometre
    accuracy).

    Parameters
    ----------
    X, Y, Z : float
        ECEF position in metres.

    Returns
    -------
    tuple[float, float, float]
        (lat_deg, lon_deg, h_m).
    """
    lon = math.atan2(Y, X)
    p = math.sqrt(X * X + Y * Y)

    # Initial latitude estimate (Bowring)
    lat = math.atan2(Z, p * (1.0 - WGS84_E2))

    for _ in range(5):
        sin_lat = math.sin(lat)
        N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        lat = math.atan2(Z + WGS84_E2 * N * sin_lat, p)

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)

    if abs(cos_lat) > 1e-10:
        h = p / cos_lat - N
    else:
        h = abs(Z) / abs(sin_lat) - N * (1.0 - WGS84_E2)

    return math.degrees(lat), math.degrees(lon), h


# ---------------------------------------------------------------------------
# Geodetic normal and ENU frame
# ---------------------------------------------------------------------------

def _sincos_latlon(lat_deg: float, lon_deg: float):
    """Shared sin/cos computation for normal and ENU."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    return math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon)


def geodetic_normal(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Outward geodetic surface normal in ECEF coordinates.

    This is the direction perpendicular to the WGS-84 ellipsoid at the
    given geodetic latitude and longitude.  It does not depend on height.

    Parameters
    ----------
    lat_deg, lon_deg : float
        Geodetic latitude and longitude (deg).

    Returns
    -------
    numpy.ndarray
        Unit normal vector in ECEF, shape (3,).
    """
    sl, cl, sn, cn = _sincos_latlon(lat_deg, lon_deg)
    return np.array([cl * cn, cl * sn, sl])


def enu_frame(
    lat_deg: float, lon_deg: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """East-North-Up basis vectors in ECEF at a geodetic location.

    Parameters
    ----------
    lat_deg, lon_deg : float
        Geodetic latitude and longitude (deg).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        (E, N, U) each shape (3,).  E is East, N is North, U is Up
        (= geodetic normal).  The triple is right-handed: E x N = U.
    """
    sl, cl, sn, cn = _sincos_latlon(lat_deg, lon_deg)

    E = np.array([-sn, cn, 0.0])
    N = np.array([-sl * cn, -sl * sn, cl])
    U = np.array([cl * cn, cl * sn, sl])
    return E, N, U



# ---------------------------------------------------------------------------
# Radii of curvature
# ---------------------------------------------------------------------------

def principal_radii(lat_deg: float) -> tuple[float, float]:
    """Principal radii of curvature of the WGS-84 ellipsoid.

    Parameters
    ----------
    lat_deg : float
        Geodetic latitude (deg).

    Returns
    -------
    tuple[float, float]
        (M, N_rad) where M is the meridional radius and N_rad is the
        prime-vertical (transverse) radius, both in metres.

        M = a(1-e²) / (1-e²sin²φ)^(3/2)
        N = a / (1-e²sin²φ)^(1/2)
    """
    sin_lat = math.sin(math.radians(lat_deg))
    w = 1.0 - WGS84_E2 * sin_lat * sin_lat
    sqrt_w = math.sqrt(w)
    N_rad = WGS84_A / sqrt_w
    M = WGS84_A * (1.0 - WGS84_E2) / (w * sqrt_w)
    return M, N_rad


# ---------------------------------------------------------------------------
# Normal Jacobian (for sensitivity equations)
# ---------------------------------------------------------------------------

def normal_jacobian(
    lat_deg: float, lon_deg: float, h_m: float
) -> np.ndarray:
    r"""Jacobian of the geodetic normal w.r.t. ECEF position.

    Returns the 3×3 matrix :math:`\partial \hat{n} / \partial \mathbf{r}`
    where :math:`\hat{n}` is the outward geodetic normal and
    :math:`\mathbf{r} = (X, Y, Z)` is the ECEF position.

    .. math::

        \frac{\partial \hat{n}}{\partial \mathbf{r}}
        = \frac{\mathbf{e}_N \mathbf{e}_N^T}{M + h}
        + \frac{\mathbf{e}_E \mathbf{e}_E^T}{N_{\mathrm{rad}} + h}

    Parameters
    ----------
    lat_deg, lon_deg : float
        Geodetic latitude and longitude (deg).
    h_m : float
        Geodetic height (m).

    Returns
    -------
    numpy.ndarray
        Shape (3, 3).
    """
    E, N_vec, _U = enu_frame(lat_deg, lon_deg)
    M, N_rad = principal_radii(lat_deg)
    return (np.outer(N_vec, N_vec) / (M + h_m)
            + np.outer(E, E) / (N_rad + h_m))
