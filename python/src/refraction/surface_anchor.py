"""
Surface anchoring: tie NRLMSIS 2.1 profiles to surface observations.

NRLMSIS 2.1 is a climatological model -- its output represents average
conditions for a given date, location, and solar activity.  To match
actual surface observations (temperature, pressure, humidity), this module
computes correction factors that smoothly blend the observed surface state
into the NRLMSIS profile with altitude.

Two corrections are applied to the dry species:

**Temperature offset** (Gaussian taper):

    dT(h) = dT_0 * exp(-(h - h_s)^2 / (2 * H_T^2))

    where dT_0 = T_obs - T_model at the surface and H_T is the taper
    scale height (``SurfaceObservation.temperature_taper_km``, default
    2 km).  A surface temperature anomaly is mostly a boundary-layer
    feature, so the correction is confined to the lowest few km.

**Density scaling** (hydrostatic propagation):

    S(h) = S_P * exp(I(h)) * T_model(h) / T_anch(h)

    where T_anch = T_model + dT and

    S_P  = (P_obs - e_obs) / P_model     dry partial-pressure ratio
    I(h) = (M_bar * g0 / k_B) * integral_{zeta_s}^{zeta(h)}
               [1/T_model - 1/T_anch] dzeta'

    I(h) is the exact hydrostatic correction: both the model and the
    anchored dry air satisfy d(ln P)/d(zeta) = -M_bar * g0 / (k_B * T),
    so their log-pressure ratio changes by the difference of the two
    integrands.  zeta is geopotential height, matching how NRLMSIS
    itself integrates the hydrostatic equation.  The T_model / T_anch
    factor converts the pressure ratio to a number-density ratio
    (n = P / (k_B * T)).  At the surface this reduces to

        S_0 = (P_obs - e_obs) / P_model * T_model / T_obs

    so the anchored dry air plus water vapor reproduce the observed
    total pressure and temperature exactly.

    ln S(h) and its exact altitude derivative are tabulated at
    construction and evaluated by cubic Hermite interpolation, so S(h)
    is continuously differentiable (needed for smooth dn/dh).

**Water vapor**: modeled as an exponential decay from the surface value
with scale height H_w = 2.0 km (typical tropospheric value), independent
of the NRLMSIS dry-species profiles.  Below the surface the gradient is
tapered smoothly to zero (Gaussian scale 0.3 km), keeping n, dn/dh and
d^2n/dh^2 continuous at the station; see
:func:`~refraction.water_vapor.water_vapor_profile`.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from .constants import kB
from .water_vapor import (
    H2O_BELOW_TAPER_KM, water_vapor_number_density, water_vapor_profile,
    water_vapor_profile_derivative,
)
from .atmosphere.nrlmsis21 import (
    NRLMSIS21, MSISInput, MBARG0DIVKB, _alt2gph, _dgph_dalt,
)


# Hydrostatic table extent: the temperature offset is below
# exp(-18) ~ 1.5e-8 of its surface value beyond 6 taper scale heights.
_TABLE_HALF_WIDTH_TAPERS = 6.0
# NRLMSIS temperatures are unreliable well below sea level, so the table
# does not extend more than this far below min(h_s, 0).
_TABLE_BELOW_SURFACE_KM = 1.0
# Table node spacing: at most this, and at most H_T / 40.
_TABLE_MAX_STEP_KM = 0.05


@dataclass
class SurfaceObservation:
    """
    Surface meteorological observation used to anchor NRLMSIS profiles.

    Attributes
    ----------
    altitude_km : float
        Surface altitude (station elevation) in kilometers.
    temperature_C : float
        Observed surface temperature in degrees Celsius.
    pressure_kPa : float
        Observed surface total (moist-air) pressure in kilopascals.
    relative_humidity : float
        Fractional relative humidity, 0.0 to 1.0.
    temperature_taper_km : float, optional
        Gaussian scale height H_T (km) over which the surface temperature
        offset decays back to the NRLMSIS profile.  Default 2.0 km, the
        typical depth of the atmospheric boundary layer.
    """
    altitude_km: float
    temperature_C: float
    pressure_kPa: float
    relative_humidity: float  # 0.0 to 1.0
    temperature_taper_km: float = 2.0


@dataclass
class AnchorParams:
    """
    Computed anchoring parameters derived from surface observation vs model.

    These parameters fully determine how the NRLMSIS profile is corrected
    at any altitude via :func:`density_scale_at`, :func:`temperature_offset_at`,
    and :func:`anchored_H2O_density`.

    Attributes
    ----------
    density_scale : float
        Surface number-density scale factor
        S_0 = (P_obs - e_obs) / P_model * T_model / T_obs (dimensionless).
    pressure_scale : float
        Surface dry partial-pressure ratio S_P = (P_obs - e_obs) / P_model
        (dimensionless).
    temperature_offset : float
        Surface temperature correction dT_0 = T_obs - T_model (K).
    n_H2O_surface : float
        Water vapor number density at the surface (m^-3).
    h_surface_km : float
        Surface altitude (km), anchor point for all corrections.
    H_w : float
        Water vapor exponential scale height (km).
    H_w_below : float
        Gaussian scale (km) over which the water-vapor gradient tapers to
        zero below the surface.
    H_T : float
        Temperature taper scale height (km).
    table_h_km : numpy.ndarray
        Altitude nodes (km) of the tabulated density scale.
    table_ln_scale : numpy.ndarray
        ln S(h) at each node.
    table_dln_scale : numpy.ndarray
        d(ln S)/dh at each node (km^-1).
    """
    density_scale: float
    pressure_scale: float
    temperature_offset: float
    n_H2O_surface: float
    h_surface_km: float
    H_w: float
    H_w_below: float
    H_T: float
    table_h_km: np.ndarray = field(repr=False)
    table_ln_scale: np.ndarray = field(repr=False)
    table_dln_scale: np.ndarray = field(repr=False)


def compute_anchor(obs, msis, base_input):
    """
    Compute anchor parameters from a surface observation and NRLMSIS model.

    Evaluates NRLMSIS on an altitude grid around the surface, compares
    the model temperature and pressure to observations, and integrates
    the hydrostatic correction to the density scale factor.

    Parameters
    ----------
    obs : SurfaceObservation
        Observed surface conditions.
    msis : NRLMSIS21
        Initialized NRLMSIS 2.1 model instance.
    base_input : MSISInput
        MSIS input with date, time, location, and solar/geomagnetic indices.

    Returns
    -------
    AnchorParams
        Correction parameters for use with :func:`density_scale_at`,
        :func:`temperature_offset_at`, and :func:`anchored_H2O_density`.

    Raises
    ------
    ValueError
        If ``obs.temperature_taper_km`` is not positive.

    Notes
    -----
    The model pressure is reconstructed from NRLMSIS number densities via
    the ideal gas law: P_model = sum(n_i) * k_B * T_model, summing over
    species indices 2-10 (N2, O2, O, He, H, Ar, N, anomalous O, NO).
    Index 1 (total mass density) is excluded as it is not a number density.

    NRLMSIS models dry air only, so it is compared with the observed dry
    partial pressure P_obs - e_obs; water vapor is added separately by
    :func:`anchored_H2O_density`.
    """
    H_T = float(obs.temperature_taper_km)
    if not H_T > 0.0:
        raise ValueError(f"temperature_taper_km must be positive; got {H_T}")

    h_s = float(obs.altitude_km)
    T_obs_K = obs.temperature_C + 273.15
    P_obs_Pa = obs.pressure_kPa * 1000.0

    # Altitude grid with the surface exactly on node i_s
    step = min(_TABLE_MAX_STEP_KM, H_T / 40.0)
    h_lo = max(h_s - _TABLE_HALF_WIDTH_TAPERS * H_T,
               min(h_s, 0.0) - _TABLE_BELOW_SURFACE_KM)
    h_hi = h_s + _TABLE_HALF_WIDTH_TAPERS * H_T
    n_below = int(math.ceil((h_s - h_lo) / step - 1e-9))
    n_above = int(math.ceil((h_hi - h_s) / step - 1e-9))
    h = h_s + step * np.arange(-n_below, n_above + 1, dtype=float)
    i_s = n_below

    out = msis.msiscalc_profile(base_input, h)
    T_m = out.tn
    dTm_dzeta = out.dT_dz  # K per geopotential km

    # Model dry pressure at the surface.  Sum number densities of all
    # species (indices 2-10); skip index 1 (total mass density, kg/m^3).
    dn_s = out.dn[i_s]
    n_model_total = sum(dn_s[i] for i in range(2, 11) if 0.0 < dn_s[i] < 1e30)
    T_model_K = T_m[i_s]
    P_model_Pa = n_model_total * kB * T_model_K

    n_H2O_surface = water_vapor_number_density(T_obs_K, P_obs_Pa, obs.relative_humidity)
    e_obs_Pa = n_H2O_surface * kB * T_obs_K

    temperature_offset = T_obs_K - T_model_K
    pressure_scale = (P_obs_Pa - e_obs_Pa) / P_model_Pa

    # Gaussian temperature taper and its altitude derivative
    dh = h - h_s
    dT = temperature_offset * np.exp(-dh * dh / (2.0 * H_T * H_T))
    ddT_dh = -dT * dh / (H_T * H_T)
    T_a = T_m + dT

    # Geopotential height and d(zeta)/dh at each node
    lat = base_input.lat
    zeta = np.array([_alt2gph(lat, x) for x in h])
    dzeta_dh = np.array([_dgph_dalt(lat, x) for x in h])

    # Hydrostatic integrand d(I)/d(zeta), integrated by the trapezoidal
    # rule outward from the surface node in both directions.
    f = MBARG0DIVKB * (1.0 / T_m - 1.0 / T_a)
    seg = 0.5 * (f[1:] + f[:-1]) * np.diff(zeta)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    I = cum - cum[i_s]

    ln_T_ratio = np.log(T_m / T_a)
    ln_scale = math.log(pressure_scale) + I + ln_T_ratio

    dTm_dh = dTm_dzeta * dzeta_dh
    dln_scale = (f * dzeta_dh
                 + dTm_dh / T_m
                 - (dTm_dh + ddT_dh) / T_a)

    return AnchorParams(
        density_scale=pressure_scale * T_model_K / T_obs_K,
        pressure_scale=pressure_scale,
        temperature_offset=temperature_offset,
        n_H2O_surface=n_H2O_surface,
        h_surface_km=h_s,
        H_w=2.0,
        H_w_below=H2O_BELOW_TAPER_KM,
        H_T=H_T,
        table_h_km=h,
        table_ln_scale=ln_scale,
        table_dln_scale=dln_scale,
    )


def _hermite_interp(x, xp, fp, dfp):
    """
    Cubic Hermite interpolation on a uniform grid with known derivatives.

    Below the grid, extrapolates linearly using the end slope; above the
    grid, holds the end value (the hydrostatic correction has converged).

    Parameters
    ----------
    x : numpy.ndarray
        Evaluation point(s).
    xp, fp, dfp : numpy.ndarray
        Uniformly spaced nodes, values, and derivatives at the nodes.

    Returns
    -------
    numpy.ndarray
        Interpolated values, same shape as ``x``.
    """
    step = xp[1] - xp[0]
    u = (x - xp[0]) / step
    i = np.clip(np.floor(u).astype(int), 0, xp.size - 2)
    t = np.clip(u - i, 0.0, 1.0)
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    y = (h00 * fp[i] + h10 * step * dfp[i]
         + h01 * fp[i + 1] + h11 * step * dfp[i + 1])
    return np.where(x < xp[0], fp[0] + dfp[0] * (x - xp[0]), y)


def density_scale_at(h_km, p):
    """
    Density scale factor at altitude with hydrostatic propagation.

    Implements S(h) = S_P * exp(I(h)) * T_model(h) / T_anch(h) by cubic
    Hermite interpolation of the table built in :func:`compute_anchor`.

    Parameters
    ----------
    h_km : float or array_like
        Altitude(s) in kilometers.
    p : AnchorParams
        Anchoring parameters from :func:`compute_anchor`.

    Returns
    -------
    float or numpy.ndarray
        Dimensionless density scale factor(s).  Multiply NRLMSIS number
        densities by this value to obtain anchored densities.
    """
    x = np.asarray(h_km, dtype=float)
    ln_s = _hermite_interp(x, p.table_h_km, p.table_ln_scale, p.table_dln_scale)
    return np.exp(ln_s)[()]


def temperature_offset_at(h_km, p):
    """
    Temperature offset at altitude with Gaussian taper.

    Implements dT(h) = dT_0 * exp(-(h - h_s)^2 / (2 * H_T^2)).

    Parameters
    ----------
    h_km : float or array_like
        Altitude(s) in kilometers.
    p : AnchorParams
        Anchoring parameters from :func:`compute_anchor`.

    Returns
    -------
    float or numpy.ndarray
        Temperature correction in kelvin.  Add to NRLMSIS temperature
        to obtain the anchored temperature.
    """
    dh = np.asarray(h_km, dtype=float) - p.h_surface_km
    return p.temperature_offset * np.exp(-dh * dh / (2.0 * p.H_T * p.H_T))


def anchored_H2O_density(h_km, p):
    """
    Water vapor number density at altitude from surface observation.

    Uses an exponential decay profile anchored to the surface value,
    independent of NRLMSIS (which does not model water vapor).

    Parameters
    ----------
    h_km : float or array_like
        Altitude(s) in kilometers.
    p : AnchorParams
        Anchoring parameters from :func:`compute_anchor`.

    Returns
    -------
    float or numpy.ndarray
        Water vapor number density in m^-3.
    """
    return water_vapor_profile(p.n_H2O_surface, h_km, p.h_surface_km,
                               p.H_w, p.H_w_below)


def anchored_H2O_gradient(h_km, p):
    """
    Altitude derivative of :func:`anchored_H2O_density`.

    Parameters
    ----------
    h_km : float or array_like
        Altitude(s) in kilometers.
    p : AnchorParams
        Anchoring parameters from :func:`compute_anchor`.

    Returns
    -------
    float or numpy.ndarray
        d(n_H2O)/dh in m^-3 km^-1.
    """
    return water_vapor_profile_derivative(p.n_H2O_surface, h_km, p.h_surface_km,
                                          p.H_w, p.H_w_below)
