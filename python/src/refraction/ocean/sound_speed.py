"""
Ocean sound speed profile computation.

Provides :class:`SoundSpeedProfile`, the main user-facing class for computing
sound speed vs. depth in seawater.  Three modes are available via
:class:`SoundMode`:

- **DelGrosso**: Del Grosso (1974) / Wong & Zhu (1995) ITS-90.
  Most accurate at depth.  T=0-30 C, S=30-40 ppt, P=0-9807 dbar.
- **ChenMillero**: Chen & Millero (1977) / Wong & Zhu (1995) ITS-90.
  Wider validity range.  T=0-40 C, S=0-40 ppt, P=0-10000 dbar.
- **Mackenzie**: Mackenzie (1981).  Simple 9-term equation using depth
  directly.  T=2-30 C, S=25-40 ppt, Z=0-8000 m.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from . import del_grosso as _dg
from . import chen_millero as _cm
from . import mackenzie as _mk
from .ocean_profile import (
    DEFAULT_LATITUDE_DEG,
    OceanConditions, OceanPreset, _OceanProfileBase,
    depth_to_pressure,
)


class SoundMode(Enum):
    """Computation mode for ocean sound speed."""
    DelGrosso = "DelGrosso"
    ChenMillero = "ChenMillero"
    Mackenzie = "Mackenzie"


@dataclass
class SoundSpeedResult:
    """Single-depth sound speed computation result.

    Attributes
    ----------
    depth_m : float
        Depth in meters (positive downward).
    pressure_dbar : float
        Sea pressure in dbar (via Saunders 1981).
    temperature_C : float
        Water temperature at depth (deg C).
    salinity_psu : float
        Salinity at depth (PSU).
    sound_speed_m_s : float
        Sound speed in m/s.
    dc_dz : float
        Vertical gradient dc/dz in (m/s)/m, z positive downward.
    """
    depth_m: float
    pressure_dbar: float
    temperature_C: float
    salinity_psu: float
    sound_speed_m_s: float
    dc_dz: float = 0.0


@dataclass
class VectorSoundSpeedResult:
    """Array sound speed computation result.

    All fields are numpy arrays of the same length.
    """
    depth_m: np.ndarray
    pressure_dbar: np.ndarray
    temperature_C: np.ndarray
    salinity_psu: np.ndarray
    sound_speed_m_s: np.ndarray
    dc_dz: np.ndarray


class SoundSpeedProfile(_OceanProfileBase):
    """Sound speed vs. depth for a given ocean column.

    Mirrors :class:`OceanRefractionProfile` for acoustic applications.
    Three constructors are available:

    1. Direct: ``SoundSpeedProfile(cond)`` from :class:`OceanConditions`.
    2. Preset: ``SoundSpeedProfile.from_preset(preset, ...)``.
    3. Arrays: ``SoundSpeedProfile.from_arrays(depths, temps, sals, ...)``.

    Parameters
    ----------
    cond : OceanConditions
        Ocean conditions dataclass.  The ``freq_ghz`` and ``wavelength_nm``
        fields are ignored (sound speed is independent of EM wavelength).
    """

    _vector_result_cls = VectorSoundSpeedResult

    @classmethod
    def from_preset(cls, preset: OceanPreset, *,
                    latitude_deg: float = DEFAULT_LATITUDE_DEG) -> "SoundSpeedProfile":
        """Create profile from an ocean preset.

        Parameters
        ----------
        preset : OceanPreset
            One of FLORIDA_GULF_SUMMER, FLORIDA_GULF_WINTER,
            CALIFORNIA_SUMMER, CALIFORNIA_WINTER.
        latitude_deg : float, optional
            Latitude for depth-pressure conversion (default 45).
        """
        return super().from_preset(preset, latitude_deg=latitude_deg)

    @classmethod
    def from_arrays(cls, depths_m: np.ndarray, temperatures_C: np.ndarray,
                    salinities_psu: np.ndarray, *,
                    latitude_deg: float = DEFAULT_LATITUDE_DEG) -> "SoundSpeedProfile":
        """Create profile from user-supplied T(z), S(z) arrays.

        Parameters
        ----------
        depths_m : array_like
            Depth values in meters (sorted ascending).
        temperatures_C : array_like
            Temperature at each depth (deg C).
        salinities_psu : array_like
            Salinity at each depth (PSU).
        latitude_deg : float, optional
            Latitude for depth-pressure conversion (default 45).
        """
        return super().from_arrays(depths_m, temperatures_C, salinities_psu,
                                   latitude_deg=latitude_deg)

    # ── Internal hooks ────────────────────────────────────────────────

    def _c_at_depth(self, depth_m: float, mode: SoundMode) -> float:
        """Compute sound speed at a single depth."""
        T, S = self._ts_at_depth(depth_m)

        if mode == SoundMode.Mackenzie:
            return _mk.sound_speed(T, S, depth_m)

        # Del Grosso and Chen-Millero need pressure
        p = depth_to_pressure(depth_m, self._cond.latitude_deg)

        if mode == SoundMode.DelGrosso:
            return _dg.sound_speed(T, S, p)
        else:  # ChenMillero
            return _cm.sound_speed(T, S, p)

    def _compute_at_depth(self, depth_m: float, mode: SoundMode):
        return self._c_at_depth(depth_m, mode)

    def _scalar_value(self, depth_m: float, mode: SoundMode) -> float:
        """Return sound speed at the given depth (for FD gradient)."""
        return self._c_at_depth(depth_m, mode)

    def _make_result(self, depth_m, p, T, S, value, gradient):
        return SoundSpeedResult(
            depth_m=depth_m,
            pressure_dbar=p,
            temperature_C=T,
            salinity_psu=S,
            sound_speed_m_s=value,
            dc_dz=gradient,
        )
