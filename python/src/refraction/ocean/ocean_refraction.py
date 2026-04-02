"""
Ocean refractive index profile computation.

Provides :class:`OceanRefractionProfile`, the main user-facing class for
computing the refractive index of seawater as a function of depth.

Modes:
    - **MeissnerWentz**: radio/microwave permittivity (double-Debye).
    - **MillardSeaver**: optical with full pressure support (500-700 nm).
    - **QuanFry**: optical, atmospheric pressure only (400-700 nm).
    - **IAPWS**: pure water, wide wavelength range (200-1100 nm).
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from . import meissner_wentz as mw
from . import millard_seaver as ms
from . import quan_fry as qf
from . import iapws_r9_97 as iapws
from .ocean_profile import (
    DEFAULT_LATITUDE_DEG,
    OceanConditions, OceanPreset, _OceanProfileBase,
    depth_to_pressure, pressure_to_depth,
)


class OceanMode(Enum):
    """Computation mode for ocean refractive index."""
    MeissnerWentz = "MeissnerWentz"
    MillardSeaver = "MillardSeaver"
    QuanFry = "QuanFry"
    IAPWS = "IAPWS"


@dataclass
class OceanRefractivityResult:
    """Result of a single-depth ocean refractive index computation.

    Attributes
    ----------
    depth_m : float
        Depth below surface (m, positive downward).
    pressure_dbar : float
        Hydrostatic pressure (dbar).
    temperature_C : float
        Water temperature at this depth (deg C).
    salinity_psu : float
        Salinity at this depth (PSU).
    n_real : float
        Real part of the refractive index.
    n_imag : float
        Imaginary part (extinction coefficient); nonzero only for radio mode.
    eps_real : float
        Real permittivity (radio mode only).
    eps_imag : float
        Imaginary permittivity (radio mode only).
    dn_dz : float
        Derivative dn_real/dz (per meter, z positive downward).
    """
    depth_m: float
    pressure_dbar: float
    temperature_C: float
    salinity_psu: float
    n_real: float
    n_imag: float = 0.0
    eps_real: float = 0.0
    eps_imag: float = 0.0
    dn_dz: float = 0.0


@dataclass
class VectorOceanRefractivityResult:
    """Vectorized result for array-input ocean computations.

    All fields are numpy arrays of the same length.
    """
    depth_m: np.ndarray
    pressure_dbar: np.ndarray
    temperature_C: np.ndarray
    salinity_psu: np.ndarray
    n_real: np.ndarray
    n_imag: np.ndarray
    eps_real: np.ndarray
    eps_imag: np.ndarray
    dn_dz: np.ndarray


class OceanRefractionProfile(_OceanProfileBase):
    """Compute ocean refractive index as a function of depth.

    Parameters
    ----------
    cond : OceanConditions
        Ocean conditions (SST, SSS, profile shape, frequency/wavelength).

    Examples
    --------
    >>> cond = OceanConditions(sst_C=25.0, sss_psu=35.0, freq_ghz=10.0)
    >>> prof = OceanRefractionProfile(cond)
    >>> result = prof.compute(100.0, OceanMode.MeissnerWentz)
    >>> print(result.n_real, result.eps_real)
    """

    _vector_result_cls = VectorOceanRefractivityResult

    @classmethod
    def from_preset(cls, preset: OceanPreset, *,
                    freq_ghz: float = 10.0,
                    wavelength_nm: float = 550.0,
                    latitude_deg: float = DEFAULT_LATITUDE_DEG) -> "OceanRefractionProfile":
        """Create from a preset ocean profile.

        Parameters
        ----------
        preset : OceanPreset
            One of FLORIDA_GULF_SUMMER, CALIFORNIA_SUMMER, etc.
        freq_ghz : float
            Microwave frequency.
        wavelength_nm : float
            Optical wavelength.
        latitude_deg : float
            Latitude for depth-pressure conversion.
        """
        return super().from_preset(preset, latitude_deg=latitude_deg,
                                   freq_ghz=freq_ghz,
                                   wavelength_nm=wavelength_nm)

    @classmethod
    def from_arrays(cls, depths_m: np.ndarray, temperatures_C: np.ndarray,
                    salinities_psu: np.ndarray, *,
                    freq_ghz: float = 10.0,
                    wavelength_nm: float = 550.0,
                    latitude_deg: float = DEFAULT_LATITUDE_DEG) -> "OceanRefractionProfile":
        """Create from user-supplied T(z), S(z) arrays.

        Linear interpolation between supplied data points.

        Parameters
        ----------
        depths_m : array_like
            Depth values (m, positive downward, must be sorted ascending).
        temperatures_C : array_like
            Temperature at each depth (deg C).
        salinities_psu : array_like
            Salinity at each depth (PSU).
        """
        return super().from_arrays(depths_m, temperatures_C, salinities_psu,
                                   latitude_deg=latitude_deg,
                                   freq_ghz=freq_ghz,
                                   wavelength_nm=wavelength_nm)

    # ── Internal hooks ────────────────────────────────────────────────

    def _n_at_depth(self, depth_m: float, mode: OceanMode):
        """Compute n (and optionally permittivity) at a single depth."""
        T, S = self._ts_at_depth(depth_m)
        p = depth_to_pressure(depth_m, self._cond.latitude_deg)
        c = self._cond

        if mode == OceanMode.MeissnerWentz:
            eps = mw.permittivity(c.freq_ghz, T, S)
            n = mw.refractive_index(c.freq_ghz, T, S)
            return n.n_real, n.n_imag, eps.real, eps.imag
        elif mode == OceanMode.MillardSeaver:
            n_val = ms.refractive_index(T, S, p, c.wavelength_nm)
            return n_val, 0.0, 0.0, 0.0
        elif mode == OceanMode.QuanFry:
            n_val = qf.refractive_index(T, S, c.wavelength_nm)
            return n_val, 0.0, 0.0, 0.0
        elif mode == OceanMode.IAPWS:
            n_val = iapws.refractive_index(T, c.wavelength_nm, p_dbar=p)
            return n_val, 0.0, 0.0, 0.0
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _compute_at_depth(self, depth_m: float, mode: OceanMode):
        return self._n_at_depth(depth_m, mode)

    def _scalar_value(self, depth_m: float, mode: OceanMode) -> float:
        """Return n_real at the given depth (for FD gradient)."""
        return self._n_at_depth(depth_m, mode)[0]

    def _make_result(self, depth_m, p, T, S, value, gradient):
        n_real, n_imag, eps_r, eps_i = value
        return OceanRefractivityResult(
            depth_m=depth_m,
            pressure_dbar=p,
            temperature_C=T,
            salinity_psu=S,
            n_real=n_real,
            n_imag=n_imag,
            eps_real=eps_r,
            eps_imag=eps_i,
            dn_dz=gradient,
        )
