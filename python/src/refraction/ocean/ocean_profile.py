"""
Ocean depth-pressure conversion and parametric T(z), S(z) profiles.

Depth-pressure conversion uses the Saunders (1981) [1] formula as given in
Fofonoff & Millard (1983) [2]:

    z = (1 - c1) * p - c2 * p^2

where c1 = (5.92 + 5.25*sin^2(phi)) * 1e-3 accounts for the latitude
dependence of gravity and c2 = 2.21e-6 accounts for the compressibility
of seawater.  depth_to_pressure inverts this iteratively.

Parametric temperature and salinity profiles use a hyperbolic tangent
thermocline model, producing smooth transitions between surface (mixed
layer) and deep-water values.

Preset profiles for Florida and California waters are based on World Ocean
Atlas (2023) [3] climatological data and CalCOFI [4] observations.

References
----------
.. [1] Saunders, P. (1981). "Practical conversion of pressure to depth."
       J. Phys. Oceanogr., 11, 573-574.
.. [2] Fofonoff, N.P. and R.C. Millard (1983). "Algorithms for computation
       of fundamental properties of seawater." UNESCO Tech. Papers in Marine
       Science, No. 44.
.. [3] NOAA NCEI (2023). "World Ocean Atlas 2023."
       https://www.ncei.noaa.gov/products/world-ocean-atlas
.. [4] CalCOFI (California Cooperative Oceanic Fisheries Investigations).
       http://sccoos.ucsd.edu/data/cast/calcofi/

Parameters
----------
All depths in meters (positive downward), pressures in dbar, latitudes
in degrees.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from typing import NamedTuple

# ---------------------------------------------------------------------------
# Preset ocean profiles
# ---------------------------------------------------------------------------

class OceanPreset(NamedTuple):
    """Parametric ocean profile preset.

    Attributes
    ----------
    sst_C : float
        Sea surface temperature (deg C).
    sss_psu : float
        Sea surface salinity (PSU).
    mld_m : float
        Mixed layer depth (m).
    t_deep_C : float
        Deep water temperature (deg C).
    s_deep_psu : float
        Deep water salinity (PSU).
    d_thermo_m : float
        Thermocline / halocline thickness scale (m).
    """
    sst_C: float
    sss_psu: float
    mld_m: float
    t_deep_C: float
    s_deep_psu: float
    d_thermo_m: float


# Climatological values from [3] and regional literature
FLORIDA_GULF_SUMMER = OceanPreset(30.0, 36.0, 25.0, 4.25, 35.0, 50.0)
"""Gulf of Mexico, summer.  SST ~30 C, S ~36 PSU.  Shallow MLD (25 m)."""

FLORIDA_GULF_WINTER = OceanPreset(23.0, 36.2, 75.0, 4.25, 35.0, 50.0)
"""Gulf of Mexico, winter.  SST ~23 C, deep MLD (75 m)."""

CALIFORNIA_SUMMER = OceanPreset(19.0, 33.4, 20.0, 3.5, 34.5, 40.0)
"""Southern California Bight, summer.  Upwelling influence, low salinity."""

CALIFORNIA_WINTER = OceanPreset(14.0, 33.4, 45.0, 3.5, 34.5, 40.0)
"""Southern California Bight, winter.  Deep MLD (45 m) from [4] CalCOFI."""

# Default latitude for depth-pressure conversion (degrees)
DEFAULT_LATITUDE_DEG = 45.0

# Finite difference step for ocean gradients (meters)
FD_STEP_OCEAN_M = 0.1


# ---------------------------------------------------------------------------
# Ocean conditions (shared by OceanRefractionProfile and SoundSpeedProfile)
# ---------------------------------------------------------------------------

@dataclass
class OceanConditions:
    """Input conditions for ocean profile computation.

    Parameters
    ----------
    sst_C : float
        Sea surface temperature (deg C).
    sss_psu : float
        Sea surface salinity (PSU).
    latitude_deg : float
        Geodetic latitude for depth-pressure conversion.
    mld_m : float
        Mixed layer depth (m).
    t_deep_C : float
        Deep water temperature (deg C).
    s_deep_psu : float
        Deep water salinity (PSU).
    d_thermo_m : float
        Thermocline thickness scale (m).
    d_halo_m : float
        Halocline thickness scale (m).
    freq_ghz : float
        Microwave frequency (GHz) for MeissnerWentz mode.
    wavelength_nm : float
        Optical wavelength (nm) for optical modes.
    """
    sst_C: float
    sss_psu: float
    latitude_deg: float = DEFAULT_LATITUDE_DEG
    mld_m: float = 50.0
    t_deep_C: float = 4.0
    s_deep_psu: float = 35.0
    d_thermo_m: float = 50.0
    d_halo_m: float = 50.0
    freq_ghz: float = 10.0
    wavelength_nm: float = 550.0


# ---------------------------------------------------------------------------
# Depth <-> Pressure  [1] via [2]
# ---------------------------------------------------------------------------

def _saunders_coefficients(latitude_deg: float):
    """Saunders (1981) [1] coefficients for depth-pressure conversion.

    Returns (c1, c2) where c1 is the gravity-latitude correction and
    c2 is the seawater compressibility correction.
    """
    sin_lat = math.sin(math.radians(latitude_deg))
    sin2 = sin_lat * sin_lat
    c1 = (5.92 + 5.25 * sin2) * 1.0e-3
    c2 = 2.21e-6
    return c1, c2


def pressure_to_depth(p_dbar: float, latitude_deg: float) -> float:
    """Convert sea pressure to depth.

    Saunders (1981) [1] formula as given in [2] (UNESCO 44, p. 28):

        z = (1 - c1) * p - c2 * p^2

    Parameters
    ----------
    p_dbar : float
        Sea pressure in dbar.
    latitude_deg : float
        Geodetic latitude in degrees.

    Returns
    -------
    float
        Depth in meters (positive downward).

    Notes
    -----
    The full Fofonoff & Millard (1983) 4-term formula gives z = 9712.653 m
    at p = 10000 dbar, lat = 30 deg.  This simplified 2-term Saunders
    formula gives ~9707 m (within 6 m of the full formula).
    """
    c1, c2 = _saunders_coefficients(latitude_deg)
    return (1.0 - c1) * p_dbar - c2 * p_dbar * p_dbar


def depth_to_pressure(depth_m: float, latitude_deg: float) -> float:
    """Convert depth to sea pressure.

    Iteratively inverts :func:`pressure_to_depth` using Newton's method.

    Parameters
    ----------
    depth_m : float
        Depth below sea surface in meters (positive downward).
    latitude_deg : float
        Geodetic latitude in degrees.

    Returns
    -------
    float
        Sea pressure in dbar.
    """
    if depth_m <= 0.0:
        return 0.0

    c1, c2 = _saunders_coefficients(latitude_deg)

    # Initial guess: p ~ z / (1 - c1)
    p = depth_m / (1.0 - c1)

    for _ in range(10):
        z = (1.0 - c1) * p - c2 * p * p
        err = z - depth_m
        if abs(err) < 1.0e-6:
            break
        # dz/dp = (1 - c1) - 2*c2*p
        dzdp = (1.0 - c1) - 2.0 * c2 * p
        p -= err / dzdp

    return p


# ---------------------------------------------------------------------------
# Parametric T(z), S(z) profiles
# ---------------------------------------------------------------------------

def temperature_at_depth(depth_m: float, sst_C: float, t_deep_C: float,
                         mld_m: float, d_thermo_m: float) -> float:
    """Temperature at depth using tanh thermocline model.

    .. math::

        T(z) = T_{deep} + (T_{sst} - T_{deep})
               \\cdot \\frac{1 - \\tanh((z - z_{MLD}) / D)}{2}

    At the mixed layer depth (z = MLD), the temperature equals the midpoint
    of SST and T_deep.

    Parameters
    ----------
    depth_m : float
        Depth below surface (m, positive downward).
    sst_C : float
        Sea surface temperature (deg C).
    t_deep_C : float
        Deep water temperature (deg C).
    mld_m : float
        Mixed layer depth (m).
    d_thermo_m : float
        Thermocline thickness scale (m).  Larger values give a more
        gradual transition.

    Returns
    -------
    float
        Temperature in degrees Celsius.
    """
    return t_deep_C + (sst_C - t_deep_C) * 0.5 * (
        1.0 - math.tanh((depth_m - mld_m) / d_thermo_m)
    )


def salinity_at_depth(depth_m: float, sss_psu: float, s_deep_psu: float,
                      mld_m: float, d_halo_m: float) -> float:
    """Salinity at depth using tanh halocline model.

    Same functional form as :func:`temperature_at_depth`.

    Parameters
    ----------
    depth_m : float
        Depth below surface (m, positive downward).
    sss_psu : float
        Sea surface salinity (PSU).
    s_deep_psu : float
        Deep water salinity (PSU).
    mld_m : float
        Mixed layer depth (m).
    d_halo_m : float
        Halocline thickness scale (m).

    Returns
    -------
    float
        Salinity in PSU.
    """
    return s_deep_psu + (sss_psu - s_deep_psu) * 0.5 * (
        1.0 - math.tanh((depth_m - mld_m) / d_halo_m)
    )


# ---------------------------------------------------------------------------
# Base class for ocean profile computations
# ---------------------------------------------------------------------------

class _OceanProfileBase:
    """Internal base class for ocean profile classes.

    Provides shared T/S interpolation, finite-difference gradient computation,
    profile iteration, and the ``compute``/``__call__``/``from_preset``/
    ``from_arrays`` template methods.

    Subclasses must:
    - Set ``_vector_result_cls`` to the vectorized result dataclass.
    - Override ``_scalar_value(depth_m, mode)`` (used by FD gradient).
    - Override ``_compute_at_depth(depth_m, mode)`` (returns raw value(s)).
    - Override ``_make_result(depth_m, p, T, S, value, gradient)`` (assembles
      a scalar result dataclass).
    """

    _vector_result_cls = None

    def __init__(self, cond: OceanConditions):
        self._cond = cond
        self._user_depths: Optional[np.ndarray] = None
        self._user_temps: Optional[np.ndarray] = None
        self._user_sals: Optional[np.ndarray] = None

    def _init_from_arrays(self, depths: np.ndarray, temps: np.ndarray,
                          sals: np.ndarray) -> None:
        """Store user-supplied profile arrays."""
        self._user_depths = np.asarray(depths, dtype=float)
        self._user_temps = np.asarray(temps, dtype=float)
        self._user_sals = np.asarray(sals, dtype=float)

    def _ts_at_depth(self, depth_m: float) -> tuple:
        """Get (T, S) at a given depth."""
        if self._user_depths is not None:
            T = float(np.interp(depth_m, self._user_depths, self._user_temps))
            S = float(np.interp(depth_m, self._user_depths, self._user_sals))
            return T, S
        c = self._cond
        T = temperature_at_depth(depth_m, c.sst_C, c.t_deep_C,
                                 c.mld_m, c.d_thermo_m)
        S = salinity_at_depth(depth_m, c.sss_psu, c.s_deep_psu,
                              c.mld_m, c.d_halo_m)
        return T, S

    def _scalar_value(self, depth_m: float, mode) -> float:
        """Return the primary scalar value at *depth_m* for the given mode.

        Must be overridden by subclasses.
        """
        raise NotImplementedError

    def _compute_at_depth(self, depth_m: float, mode):
        """Return computed value(s) at *depth_m* for the given mode.

        Must be overridden by subclasses.  The return type is
        subclass-specific (e.g. a 4-tuple for refraction, a scalar for
        sound speed).
        """
        raise NotImplementedError

    def _make_result(self, depth_m, p, T, S, value, gradient):
        """Assemble a scalar result dataclass from computed components.

        Must be overridden by subclasses.
        """
        raise NotImplementedError

    def _compute_gradient(self, depth_m: float, mode) -> float:
        """Central finite-difference gradient of the primary scalar value."""
        dz = FD_STEP_OCEAN_M
        z_lo = max(0.0, depth_m - dz)
        z_hi = depth_m + dz
        v_lo = self._scalar_value(z_lo, mode)
        v_hi = self._scalar_value(z_hi, mode)
        return (v_hi - v_lo) / (z_hi - z_lo)

    # ── Template methods (shared compute / __call__ / factories) ──────

    def compute(self, depth_m: float, mode):
        """Compute result at a single depth.

        Parameters
        ----------
        depth_m : float
            Depth below surface (m, positive downward).
        mode
            Computation mode (subclass-specific enum).
        """
        T, S = self._ts_at_depth(depth_m)
        p = depth_to_pressure(depth_m, self._cond.latitude_deg)
        value = self._compute_at_depth(depth_m, mode)
        gradient = self._compute_gradient(depth_m, mode)
        return self._make_result(depth_m, p, T, S, value, gradient)

    def __call__(self, depth_m, mode):
        """Compute at scalar or array depths.

        Parameters
        ----------
        depth_m : float or array_like
            Depth(s) below surface (m).
        mode
            Computation mode (subclass-specific enum).
        """
        if np.ndim(depth_m) == 0:
            return self.compute(float(depth_m), mode)
        depths = np.asarray(depth_m, dtype=float)
        results = [self.compute(float(z), mode) for z in depths]
        return self._vectorize_results(results, self._vector_result_cls)

    @classmethod
    def from_preset(cls, preset: OceanPreset, *,
                    latitude_deg: float = DEFAULT_LATITUDE_DEG,
                    **kwargs) -> "_OceanProfileBase":
        """Create from a preset ocean profile.

        Parameters
        ----------
        preset : OceanPreset
            One of FLORIDA_GULF_SUMMER, CALIFORNIA_SUMMER, etc.
        latitude_deg : float
            Latitude for depth-pressure conversion.
        **kwargs
            Extra fields forwarded to OceanConditions (e.g. freq_ghz,
            wavelength_nm).
        """
        cond = OceanConditions(
            sst_C=preset.sst_C,
            sss_psu=preset.sss_psu,
            latitude_deg=latitude_deg,
            mld_m=preset.mld_m,
            t_deep_C=preset.t_deep_C,
            s_deep_psu=preset.s_deep_psu,
            d_thermo_m=preset.d_thermo_m,
            d_halo_m=preset.d_thermo_m,
            **kwargs,
        )
        return cls(cond)

    @classmethod
    def from_arrays(cls, depths_m: np.ndarray, temperatures_C: np.ndarray,
                    salinities_psu: np.ndarray, *,
                    latitude_deg: float = DEFAULT_LATITUDE_DEG,
                    **kwargs) -> "_OceanProfileBase":
        """Create from user-supplied T(z), S(z) arrays.

        Parameters
        ----------
        depths_m : array_like
            Depth values (m, positive downward, must be sorted ascending).
        temperatures_C : array_like
            Temperature at each depth (deg C).
        salinities_psu : array_like
            Salinity at each depth (PSU).
        latitude_deg : float
            Latitude for depth-pressure conversion.
        **kwargs
            Extra fields forwarded to OceanConditions (e.g. freq_ghz,
            wavelength_nm).
        """
        cond = OceanConditions(
            sst_C=float(np.asarray(temperatures_C)[0]),
            sss_psu=float(np.asarray(salinities_psu)[0]),
            latitude_deg=latitude_deg,
            **kwargs,
        )
        obj = cls(cond)
        obj._init_from_arrays(depths_m, temperatures_C, salinities_psu)
        return obj

    def profile(self, z_min_m: float, z_max_m: float, dz_m: float,
                mode) -> list:
        """Compute profile over a depth range.

        Parameters
        ----------
        z_min_m, z_max_m : float
            Depth range (m).
        dz_m : float
            Depth step (m).
        mode
            Computation mode (subclass-specific enum).

        Returns
        -------
        list
            List of result dataclasses.
        """
        depths = np.arange(z_min_m, z_max_m + dz_m * 0.5, dz_m)
        return [self.compute(float(z), mode) for z in depths]

    @staticmethod
    def _vectorize_results(results, result_cls):
        """Assemble a list of result dataclasses into a vectorized result."""
        fields = result_cls.__dataclass_fields__
        return result_cls(**{f: np.array([getattr(r, f) for r in results])
                             for f in fields})
