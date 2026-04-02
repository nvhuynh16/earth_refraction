"""
Refraction profile: species-sum refractivity from surface to 122 km.

This is the main integration module.  It combines:

- NRLMSIS 2.1 for species number densities vs altitude
- Surface anchoring for temperature/density corrections
- Species K-factors (optical or radio) for refractivity per molecule
- Water vapor exponential profile

to compute the refractive index n, refractivity N = (n-1)*10^6, and their
altitude derivatives at any height.

The core formula is:

    n - 1 = sum_i( n_i(h) * K_i(lambda, T) )

In **Ciddor mode**, K_i is wavelength-dependent (via sigma^2) and
temperature-independent.  In **ITU-R mode**, K_i is wavelength-independent
but the water-vapor dipolar term K_H2O depends on 1/T.

The altitude derivative dn/dh is computed by:
- Finite differences (central, step = 0.01 km) on NRLMSIS densities
- Analytic derivative for water vapor (exponential decay)
- Chain rule for the radio dipolar term (dK/dT * dT/dh)
"""

import warnings
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .atmosphere.nrlmsis21 import NRLMSIS21, MSISInput, MSISProfileOutput
from . import species
from . import itu_r_p453 as itu
from .constants import HPA_TO_PA, REFRACTIVITY_SCALE, kB
from .surface_anchor import (
    SurfaceObservation, AnchorParams, compute_anchor,
    density_scale_at, temperature_offset_at, anchored_H2O_density,
)
from .water_vapor import water_vapor_profile

# Finite difference step for atmospheric density derivatives (km)
_FD_STEP_KM = 0.01


class Mode(Enum):
    """Refractivity computation mode.

    Attributes
    ----------
    Ciddor : str
        Optical mode using Ciddor 1996 dispersion equations.
        Wavelength-dependent, valid 300-1700 nm.
    ITU_R_P453 : str
        Radio mode using ITU-R P.453-14 three-term formula.
        Wavelength-independent.
    """
    Ciddor = "Ciddor"
    ITU_R_P453 = "ITU_R_P453"


@dataclass
class AtmosphericConditions:
    """
    Space weather and geolocation inputs for NRLMSIS 2.1.

    These parameters control the climatological state of the atmosphere
    (independent of the surface observation used for anchoring).

    Attributes
    ----------
    day_of_year : float
        Day of year (1-366).
    ut_seconds : float
        Universal time in seconds since midnight.
    latitude_deg : float
        Geodetic latitude in degrees (-90 to 90).
    longitude_deg : float
        Geodetic longitude in degrees (-180 to 180).
    f107a : float
        81-day average F10.7 solar radio flux (SFU).
    f107 : float
        Daily F10.7 solar radio flux for the previous day (SFU).
    ap : list of float
        Geomagnetic activity indices (7 values):
        [daily Ap, 3-hr Ap at t, t-3h, t-6h, t-9h,
         avg(t-12h..t-33h), avg(t-36h..t-57h)].
    """
    day_of_year: float = 172.0
    ut_seconds: float = 29000.0
    latitude_deg: float = 45.0
    longitude_deg: float = -75.0
    f107a: float = 150.0
    f107: float = 150.0
    ap: list = field(default_factory=lambda: [4, 4, 4, 4, 4, 4, 4])


@dataclass
class RefractivityResult:
    """
    Refractivity result at a single altitude.

    Attributes
    ----------
    h_km : float
        Altitude in kilometers.
    n : float
        Refractive index (typically ~1.000270 at sea level, visible).
    N : float
        Refractivity N = (n - 1) * 10^6 in N-units.
    dn_dh : float
        Altitude derivative of n (km^-1).
    dN_dh : float
        Altitude derivative of N (N-units km^-1).
    temperature_K : float
        Anchored temperature at this altitude (K).
    """
    h_km: float = 0.0
    n: float = 1.0
    N: float = 0.0
    dn_dh: float = 0.0
    dN_dh: float = 0.0
    temperature_K: float = 0.0


@dataclass
class VectorRefractivityResult:
    """
    Refractivity results across an array of altitudes.

    Same fields as :class:`RefractivityResult` but all values are
    numpy arrays of shape (N,).

    Attributes
    ----------
    h_km : numpy.ndarray
        Altitudes in kilometers.
    n : numpy.ndarray
        Refractive index at each altitude.
    N : numpy.ndarray
        Refractivity N = (n - 1) * 10^6 at each altitude.
    dn_dh : numpy.ndarray
        Altitude derivative of n (km^-1).
    dN_dh : numpy.ndarray
        Altitude derivative of N (N-units km^-1).
    temperature_K : numpy.ndarray
        Anchored temperature at each altitude (K).
    """
    h_km: np.ndarray = field(default_factory=lambda: np.array([]))
    n: np.ndarray = field(default_factory=lambda: np.array([]))
    N: np.ndarray = field(default_factory=lambda: np.array([]))
    dn_dh: np.ndarray = field(default_factory=lambda: np.array([]))
    dN_dh: np.ndarray = field(default_factory=lambda: np.array([]))
    temperature_K: np.ndarray = field(default_factory=lambda: np.array([]))


class RefractionProfile:
    """
    Altitude-resolved refractivity profile anchored to surface observations.

    This is the primary user-facing class.  It initializes NRLMSIS 2.1,
    computes anchoring parameters from the surface observation, and
    provides methods to evaluate refractivity at any altitude.

    Parameters
    ----------
    atm : AtmosphericConditions
        Space weather and geolocation inputs.
    obs : SurfaceObservation, optional
        Surface meteorological observation for anchoring.  If ``None``,
        the NRLMSIS model's own surface prediction is used (no anchoring
        correction).  This gives the raw model output for the given date,
        location, and solar conditions.

    Examples
    --------
    >>> atm = AtmosphericConditions(day_of_year=172, latitude_deg=45.0)
    >>> prof = RefractionProfile(atm)  # unanchored model
    >>> r = prof(10.0, Mode.Ciddor, wavelength_nm=633.0)
    >>> print(f"n = {r.n:.10f}, N = {r.N:.4f}")
    """
    def __init__(self, atm, obs=None):
        self._msis = NRLMSIS21()

        self._base_input = MSISInput(
            day=atm.day_of_year,
            utsec=atm.ut_seconds,
            alt=0.0,
            lat=atm.latitude_deg,
            lon=atm.longitude_deg,
            f107a=atm.f107a,
            f107=atm.f107,
            ap=list(atm.ap),
        )

        if obs is None:
            # No anchoring: derive a synthetic observation from NRLMSIS
            # surface values so the anchoring correction is identity
            # (temperature_offset=0, density_scale=1).
            out = self._msis.msiscalc(self._base_input)
            T_model_K = out.tn
            n_total = sum(out.dn[i] for i in range(2, 11)
                          if 0 < out.dn[i] < 1e30)
            P_model_Pa = n_total * kB * T_model_K
            obs = SurfaceObservation(
                altitude_km=0.0,
                temperature_C=T_model_K - 273.15,
                pressure_kPa=P_model_Pa / 1000.0,
                relative_humidity=0.0,
            )

        self._obs = obs
        self._anchor = compute_anchor(obs, self._msis, self._base_input)

    _CALL_SENTINEL = object()

    def __call__(self, h_km, mode, wavelength_nm=_CALL_SENTINEL):
        """
        Compute refractivity at one or more altitudes.

        Parameters
        ----------
        h_km : float or array_like
            Altitude(s) in kilometers.
        mode : Mode
            Computation mode (Ciddor optical or ITU-R radio).
        wavelength_nm : float, optional
            Vacuum wavelength in nanometers (Ciddor mode only).
            Default is 633.0 nm.  Ignored in ITU-R mode.

        Returns
        -------
        RefractivityResult or VectorRefractivityResult
            Scalar result for scalar input, vectorized result for array input.

        Warns
        -----
        UserWarning
            If Ciddor mode is used outside 300-1700 nm, or if wavelength_nm
            is explicitly passed in ITU-R mode (where it is ignored).
        """
        if wavelength_nm is self._CALL_SENTINEL:
            wavelength_nm = 633.0
            wl_explicit = False
        else:
            wl_explicit = True

        if mode == Mode.Ciddor and (wavelength_nm < 300.0 or wavelength_nm > 1700.0):
            warnings.warn(
                f"Ciddor model is valid for optical wavelengths (300\u20131700 nm); "
                f"got {wavelength_nm} nm"
            )
        if mode == Mode.ITU_R_P453 and wl_explicit:
            warnings.warn(
                "ITU-R P.453 radio refractivity is wavelength-independent; "
                "wavelength_nm argument is ignored"
            )

        scalar = isinstance(h_km, (int, float))
        if scalar:
            return self.compute(h_km, mode, wavelength_nm)

        return self._compute_batch(np.asarray(h_km, dtype=float), mode, wavelength_nm)

    # ── Shared refractivity helpers ─────────────────────────────────────

    @staticmethod
    def _optical_k_factors(sigma2):
        """Compute 8 optical K-factors for wavenumber squared *sigma2*."""
        return (
            species.K_N2_optical(sigma2),
            species.K_O2_optical(sigma2),
            species.K_Ar_optical(sigma2),
            species.K_He_optical(sigma2),
            species.K_O_optical(sigma2),
            species.K_N_optical(sigma2),
            species.K_H_optical(sigma2),
            species.K_H2O_optical(sigma2),
        )

    @staticmethod
    def _refractivity_optical(n_species, dn_species, n_H2O, dn_H2O, k_factors):
        """Compute optical refractivity and its derivative.

        Works for both scalar and numpy array inputs.

        Returns (n_minus_1, dn_dh).
        """
        K_N2, K_O2, K_Ar, K_He, K_O, K_N_, K_H, K_H2O = k_factors
        n_N2, n_O2, n_Ar, n_He, n_O, n_N, n_H = n_species
        dn_N2, dn_O2, dn_Ar, dn_He, dn_O, dn_N, dn_H = dn_species

        n_minus_1 = (n_N2 * K_N2 + n_O2 * K_O2 + n_Ar * K_Ar
                     + n_He * K_He + n_O * K_O + n_N * K_N_ + n_H * K_H
                     + n_H2O * K_H2O)

        dn_dh = (K_N2 * dn_N2 + K_O2 * dn_O2 + K_Ar * dn_Ar
                 + K_He * dn_He + K_O * dn_O + K_N_ * dn_N
                 + K_H * dn_H + K_H2O * dn_H2O)

        return n_minus_1, dn_dh

    @staticmethod
    def _refractivity_radio(n_species, dn_species, n_H2O, dn_H2O, T_K, dT_dz):
        """Compute radio refractivity and its derivative.

        Works for both scalar and numpy array inputs.

        Returns (N_val, dN_dh).
        """
        n_N2, n_O2, n_Ar, n_He, n_O, n_N, n_H = n_species
        dn_N2, dn_O2, dn_Ar, dn_He, dn_O, dn_N, dn_H = dn_species

        K_dry = species.K_radio_dry
        n_dry = n_N2 + n_O2 + n_Ar + n_He + n_O + n_N + n_H
        K_H2O_total = species.K_H2O_radio(T_K)
        N_val = n_dry * K_dry + n_H2O * K_H2O_total

        dn_dry_dh = dn_N2 + dn_O2 + dn_Ar + dn_He + dn_O + dn_N + dn_H

        K_dens = species.K_H2O_radio_density
        K_dip = species.K_H2O_radio_dipolar(T_K)
        dK_dip_dh = -itu.k3 * species.kB * HPA_TO_PA / (T_K * T_K) * dT_dz

        dN_dh = (K_dry * dn_dry_dh
                 + K_dens * dn_H2O
                 + K_dip * dn_H2O
                 + n_H2O * dK_dip_dh)

        return N_val, dN_dh

    def _gaussian_taper_dT(self, dh):
        """Derivative of Gaussian temperature-offset taper.

        Works for both scalar and numpy array *dh*.
        """
        H_T = self._anchor.H_T
        return (self._anchor.temperature_offset
                * (-dh / (H_T * H_T))
                * np.exp(-dh * dh / (2.0 * H_T * H_T)))

    # ── Scalar computation ──────────────────────────────────────────────

    def compute(self, h_km, mode, wavelength_nm=633.0):
        """
        Compute refractivity at a single altitude.

        This is the scalar path.  For arrays, use :meth:`__call__` or
        :meth:`_compute_batch` directly.

        Parameters
        ----------
        h_km : float
            Altitude in kilometers.
        mode : Mode
            Computation mode.
        wavelength_nm : float, optional
            Vacuum wavelength in nm (default 633.0, Ciddor mode only).

        Returns
        -------
        RefractivityResult
            Refractivity, refractive index, and derivatives at ``h_km``.

        Notes
        -----
        Species density derivatives are computed via central finite
        differences (step = 0.01 km) on the anchored NRLMSIS output.
        The NRLMSIS model is evaluated three times per call (center,
        +step, -step).

        For radio mode, the dipolar water-vapor term introduces a
        temperature-dependent K-factor whose altitude derivative
        requires dT/dz from NRLMSIS.
        """
        inp = MSISInput(
            day=self._base_input.day, utsec=self._base_input.utsec,
            alt=h_km, lat=self._base_input.lat, lon=self._base_input.lon,
            f107a=self._base_input.f107a, f107=self._base_input.f107,
            ap=list(self._base_input.ap),
        )
        full = self._msis.msiscalc_with_derivative(inp)
        out = full.output

        S = density_scale_at(h_km, self._anchor)
        dT = temperature_offset_at(h_km, self._anchor)
        T_K = out.tn + dT
        dT_dz = full.dT_dz

        # Add the derivative of the Gaussian temperature taper
        dh = h_km - self._anchor.h_surface_km
        dT_dz += self._gaussian_taper_dT(dh)

        # Anchored species number densities (m^-3)
        # NRLMSIS indices: 2=N2, 3=O2, 4=O, 5=He, 6=H, 7=Ar, 8=N
        n_N2 = out.dn[2] * S
        n_O2 = out.dn[3] * S
        n_O = out.dn[4] * S
        n_He = out.dn[5] * S
        n_H = out.dn[6] * S
        n_Ar = out.dn[7] * S
        n_N = out.dn[8] * S

        # Central finite differences for density derivatives
        fd_dh = _FD_STEP_KM
        def get_densities(alt):
            inp2 = MSISInput(
                day=self._base_input.day, utsec=self._base_input.utsec,
                alt=alt, lat=self._base_input.lat, lon=self._base_input.lon,
                f107a=self._base_input.f107a, f107=self._base_input.f107,
                ap=list(self._base_input.ap),
            )
            out2 = self._msis.msiscalc(inp2)
            S2 = density_scale_at(alt, self._anchor)
            return [
                out2.dn[2] * S2, out2.dn[3] * S2, out2.dn[4] * S2,
                out2.dn[5] * S2, out2.dn[6] * S2, out2.dn[7] * S2,
                out2.dn[8] * S2,
            ]

        dp = get_densities(h_km + fd_dh)
        dm = get_densities(h_km - fd_dh)
        inv_2dh = 1.0 / (2.0 * fd_dh)
        dn_N2_dh = (dp[0] - dm[0]) * inv_2dh
        dn_O2_dh = (dp[1] - dm[1]) * inv_2dh
        dn_O_dh = (dp[2] - dm[2]) * inv_2dh
        dn_He_dh = (dp[3] - dm[3]) * inv_2dh
        dn_H_dh = (dp[4] - dm[4]) * inv_2dh
        dn_Ar_dh = (dp[5] - dm[5]) * inv_2dh
        dn_N_dh = (dp[6] - dm[6]) * inv_2dh

        # Water vapor (exponential decay, independent of NRLMSIS)
        n_H2O = anchored_H2O_density(h_km, self._anchor)
        if h_km >= self._anchor.h_surface_km:
            dn_H2O_dh = -n_H2O / self._anchor.H_w
        else:
            dn_H2O_dh = 0.0

        result = RefractivityResult()
        result.h_km = h_km
        result.temperature_K = T_K

        n_spec = (n_N2, n_O2, n_Ar, n_He, n_O, n_N, n_H)
        dn_spec = (dn_N2_dh, dn_O2_dh, dn_Ar_dh, dn_He_dh,
                   dn_O_dh, dn_N_dh, dn_H_dh)

        if mode == Mode.Ciddor:
            sigma2 = 1.0 / (wavelength_nm * 1e-3) ** 2
            k_factors = self._optical_k_factors(sigma2)
            n_minus_1, dn_dh = self._refractivity_optical(
                n_spec, dn_spec, n_H2O, dn_H2O_dh, k_factors)

            result.n = 1.0 + n_minus_1
            result.N = n_minus_1 * REFRACTIVITY_SCALE
            result.dn_dh = dn_dh
            result.dN_dh = dn_dh * REFRACTIVITY_SCALE

        else:  # ITU_R_P453
            N_val, dN_dh = self._refractivity_radio(
                n_spec, dn_spec, n_H2O, dn_H2O_dh, T_K, dT_dz)

            result.n = 1.0 + N_val / REFRACTIVITY_SCALE
            result.N = N_val
            result.dN_dh = dN_dh
            result.dn_dh = dN_dh / REFRACTIVITY_SCALE

        return result

    def _compute_batch(self, h_km, mode, wavelength_nm):
        """
        Vectorized profile computation over an array of altitudes.

        Uses :meth:`NRLMSIS21.msiscalc_profile` to share all altitude-
        independent MSIS work (basis function evaluation, parameter
        computation) across the array.

        Parameters
        ----------
        h_km : numpy.ndarray
            1-D array of altitudes in kilometers.
        mode : Mode
            Computation mode.
        wavelength_nm : float
            Vacuum wavelength in nm (Ciddor mode only).

        Returns
        -------
        VectorRefractivityResult
            Arrays of n, N, dn/dh, dN/dh, and temperature at each altitude.
        """
        N = h_km.size
        fd_dh = _FD_STEP_KM

        # Build all altitudes: main, +fd, -fd
        all_alts = np.concatenate([h_km, h_km + fd_dh, h_km - fd_dh])
        msis_out = self._msis.msiscalc_profile(self._base_input, all_alts)

        # Split results
        tn_main = msis_out.tn[:N]
        dn_main = msis_out.dn[:N]
        dT_dz = msis_out.dT_dz[:N]
        dn_plus = msis_out.dn[N:2*N]
        dn_minus = msis_out.dn[2*N:3*N]

        # Vectorized surface anchor corrections
        S = density_scale_at(h_km, self._anchor)
        Sp = density_scale_at(h_km + fd_dh, self._anchor)
        Sm = density_scale_at(h_km - fd_dh, self._anchor)
        dT = temperature_offset_at(h_km, self._anchor)
        T_K = tn_main + dT

        # Temperature offset derivative (Gaussian taper)
        dh = h_km - self._anchor.h_surface_km
        dT_dz = dT_dz + self._gaussian_taper_dT(dh)

        # Species densities (indices 2-8: N2, O2, O, He, H, Ar, N)
        species_idx = [2, 3, 4, 5, 6, 7, 8]
        n_species = dn_main[:, species_idx] * S[:, None]

        # FD density derivatives
        inv_2dh = 1.0 / (2.0 * fd_dh)
        dn_species_dh = (dn_plus[:, species_idx] * Sp[:, None]
                         - dn_minus[:, species_idx] * Sm[:, None]) * inv_2dh

        n_N2, n_O2, n_O, n_He, n_H, n_Ar, n_N = [n_species[:, i] for i in range(7)]
        dn_N2, dn_O2, dn_O, dn_He, dn_H, dn_Ar, dn_N = [dn_species_dh[:, i] for i in range(7)]

        # Water vapor (vectorized exponential decay)
        n_H2O = water_vapor_profile(self._anchor.n_H2O_surface, h_km,
                                    self._anchor.h_surface_km, self._anchor.H_w)
        dn_H2O_dh = np.where(h_km >= self._anchor.h_surface_km,
                             -n_H2O / self._anchor.H_w, 0.0)

        result = VectorRefractivityResult()
        result.h_km = h_km
        result.temperature_K = T_K

        n_spec = (n_N2, n_O2, n_Ar, n_He, n_O, n_N, n_H)
        dn_spec = (dn_N2, dn_O2, dn_Ar, dn_He, dn_O, dn_N, dn_H)

        if mode == Mode.Ciddor:
            sigma2 = 1.0 / (wavelength_nm * 1e-3) ** 2
            k_factors = self._optical_k_factors(sigma2)
            n_minus_1, dn_dh_val = self._refractivity_optical(
                n_spec, dn_spec, n_H2O, dn_H2O_dh, k_factors)

            result.n = 1.0 + n_minus_1
            result.N = n_minus_1 * REFRACTIVITY_SCALE
            result.dn_dh = dn_dh_val
            result.dN_dh = dn_dh_val * REFRACTIVITY_SCALE

        else:  # ITU_R_P453
            N_val, dN_dh = self._refractivity_radio(
                n_spec, dn_spec, n_H2O, dn_H2O_dh, T_K, dT_dz)

            result.n = 1.0 + N_val / REFRACTIVITY_SCALE
            result.N = N_val
            result.dN_dh = dN_dh
            result.dn_dh = dN_dh / REFRACTIVITY_SCALE

        return result

    def profile(self, h_min_km, h_max_km, dh_km, mode, wavelength_nm=633.0):
        """
        Compute refractivity on a uniform altitude grid.

        Parameters
        ----------
        h_min_km : float
            Starting altitude in kilometers.
        h_max_km : float
            Ending altitude in kilometers (inclusive).
        dh_km : float
            Altitude step in kilometers.
        mode : Mode
            Computation mode.
        wavelength_nm : float, optional
            Vacuum wavelength in nm (default 633.0, Ciddor mode only).

        Returns
        -------
        list of RefractivityResult
            One result per altitude step from h_min_km to h_max_km.
        """
        results = []
        h = h_min_km
        while h <= h_max_km + 0.5 * dh_km:
            results.append(self.compute(h, mode, wavelength_nm))
            h += dh_km
        return results
