"""
Surface anchoring: tie NRLMSIS 2.1 profiles to surface observations.

NRLMSIS 2.1 is a climatological model -- its output represents average
conditions for a given date, location, and solar activity.  To match
actual surface observations (temperature, pressure, humidity), this module
computes correction factors that smoothly blend the observed surface state
into the NRLMSIS profile with altitude.

Two corrections are applied:

**Temperature offset** (Gaussian taper):

    dT(h) = dT_0 * exp(-(h - h_s)^2 / (2 * H_T^2))

    where dT_0 = T_obs - T_model at the surface and H_T = 15 km is the
    taper scale height.  The correction is strongest at the surface and
    decays to zero over ~30 km.

**Density scaling** (hydrostatic propagation):

    S(h) = S_0 * exp(C * erf(dh / (H_T * sqrt(2))))

    where S_0 = P_obs / P_model is the surface density scale factor and
    C = (M_bar * g0) / (k_B) * dT * H_T * sqrt(pi/2) / T_s^2.

    This propagates the surface pressure correction upward using the
    hydrostatic equation, accounting for the temperature offset's effect
    on scale height.  The erf profile ensures the correction transitions
    smoothly and asymptotes at altitude.

**Water vapor**: modeled as an exponential decay from the surface value
with scale height H_w = 2.0 km (typical tropospheric value), independent
of the NRLMSIS dry-species profiles.
"""

import math
from dataclasses import dataclass

import numpy as np

from .constants import kB
from .water_vapor import water_vapor_number_density, water_vapor_profile
from .atmosphere.nrlmsis21 import NRLMSIS21, MSISInput, MBARG0DIVKB


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
        Observed surface pressure in kilopascals.
    relative_humidity : float
        Fractional relative humidity, 0.0 to 1.0.
    """
    altitude_km: float
    temperature_C: float
    pressure_kPa: float
    relative_humidity: float  # 0.0 to 1.0


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
        Surface density scale factor S_0 = P_obs / P_model (dimensionless).
    temperature_offset : float
        Surface temperature correction dT_0 = T_obs - T_model (K).
    n_H2O_surface : float
        Water vapor number density at the surface (m^-3).
    h_surface_km : float
        Surface altitude (km), anchor point for all corrections.
    H_w : float
        Water vapor exponential scale height (km).
    H_T : float
        Temperature/density taper scale height (km).
    hydro_C : float
        Hydrostatic propagation constant C (dimensionless), encoding the
        integrated effect of the temperature offset on pressure.
    """
    density_scale: float
    temperature_offset: float
    n_H2O_surface: float
    h_surface_km: float
    H_w: float
    H_T: float
    hydro_C: float


def compute_anchor(obs, msis, base_input):
    """
    Compute anchor parameters from a surface observation and NRLMSIS model.

    Evaluates NRLMSIS at the surface altitude and compares the model
    temperature and pressure to observations to derive correction factors.

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

    Notes
    -----
    The model pressure is reconstructed from NRLMSIS number densities via
    the ideal gas law: P_model = sum(n_i) * k_B * T_model, summing over
    species indices 2-10 (N2, O2, O, He, H, Ar, N, anomalous O, NO).
    Index 1 (total mass density) is excluded as it is not a number density.

    The hydrostatic constant C is derived from the requirement that the
    density correction remain consistent with hydrostatic equilibrium
    given the Gaussian temperature taper:

        C = (M_bar * g0 / k_B) * dT_0 * H_T * sqrt(pi/2) / T_s^2

    where M_bar is the mean molecular mass, g0 = 9.80665 m/s^2, and
    T_s is the observed surface temperature.
    """
    # kB imported from constants module

    inp = MSISInput(
        day=base_input.day, utsec=base_input.utsec,
        alt=obs.altitude_km, lat=base_input.lat, lon=base_input.lon,
        f107a=base_input.f107a, f107=base_input.f107, ap=list(base_input.ap),
    )
    out = msis.msiscalc(inp)

    T_model_K = out.tn
    T_obs_K = obs.temperature_C + 273.15

    # Sum number densities of all species (indices 2-10) to get total
    # number density.  Skip index 1 (total mass density, kg/m^3).
    n_model_total = 0.0
    for i in range(2, 11):
        if out.dn[i] > 0.0 and out.dn[i] < 1e30:
            n_model_total += out.dn[i]

    P_model_Pa = n_model_total * kB * T_model_K
    P_obs_Pa = obs.pressure_kPa * 1000.0

    n_H2O_surface = water_vapor_number_density(T_obs_K, P_obs_Pa, obs.relative_humidity)

    temperature_offset = T_obs_K - T_model_K
    H_T = 15.0  # km, taper scale height
    T_s = T_obs_K
    # MBARG0DIVKB = M_bar * g0 / k_B (with km -> m conversion built in)
    hydro_C = MBARG0DIVKB * temperature_offset * H_T * math.sqrt(math.pi / 2.0) / (T_s * T_s)

    return AnchorParams(
        density_scale=P_obs_Pa / P_model_Pa,
        temperature_offset=temperature_offset,
        n_H2O_surface=n_H2O_surface,
        h_surface_km=obs.altitude_km,
        H_w=2.0,
        H_T=H_T,
        hydro_C=hydro_C,
    )


def _np_erf(x):
    """
    Scalar/array-compatible error function without scipy dependency.

    Parameters
    ----------
    x : float or array_like
        Argument(s) to the error function.

    Returns
    -------
    float or numpy.ndarray
        erf(x), matching the input shape.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        return np.float64(math.erf(float(x)))
    return np.array([math.erf(float(v)) for v in x])


def density_scale_at(h_km, p):
    """
    Density scale factor at altitude with hydrostatic propagation.

    Implements S(h) = S_0 * exp(C * erf(dh / (H_T * sqrt(2)))).

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
    dh = np.asarray(h_km, dtype=float) - p.h_surface_km
    erf_arg = dh / (p.H_T * math.sqrt(2.0))
    erf_val = _np_erf(erf_arg)
    return p.density_scale * np.exp(p.hydro_C * erf_val)


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
    return water_vapor_profile(p.n_H2O_surface, h_km, p.h_surface_km, p.H_w)
