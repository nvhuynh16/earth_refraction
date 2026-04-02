"""
Atmospheric refractive index computation (surface to 122 km).

This package computes the refractive index of air as a function of altitude,
wavelength, and atmospheric conditions using a species-sum decomposition:

    n - 1 = sum_i( n_i * K_i )

where n_i is the number density of species i (from NRLMSIS 2.1) and K_i is
the per-molecule refractivity contribution, which depends on the mode:

- **Ciddor (optical)**: wavelength-dependent K_i derived from Ciddor 1996
  dispersion equations and molecular polarizabilities.  Valid 300-1700 nm.
- **ITU-R P.453 (radio)**: wavelength-independent K_i from the ITU-R P.453-14
  three-term radio refractivity formula.

NRLMSIS 2.1 profiles are anchored to surface observations via temperature
offset (Gaussian taper) and density scaling (hydrostatic propagation).

Typical usage::

    from refraction import (
        AtmosphericConditions, SurfaceObservation,
        RefractionProfile, Mode,
    )

    atm = AtmosphericConditions(day_of_year=172, latitude_deg=45.0)
    obs = SurfaceObservation(
        altitude_km=0.0, temperature_C=20.0,
        pressure_kPa=101.325, relative_humidity=0.5,
    )
    profile = RefractionProfile(atm, obs)
    result = profile(h_km=10.0, mode=Mode.Ciddor, wavelength_nm=633.0)
    print(result.n, result.N)
"""

from .water_vapor import (
    svp_giacomo, enhancement_factor, water_vapor_mole_fraction,
    water_vapor_number_density, water_vapor_profile,
)
from .ciddor import ciddor_n, n_standard_air, n_water_vapor, compressibility_Z
from .itu_r_p453 import itu_N, itu_N_from_surface
from .species import (
    K_N2_optical, K_O2_optical, K_Ar_optical, K_He_optical,
    K_H_optical, K_O_optical, K_N_optical, K_NO_optical,
    K_H2O_optical, K_radio_dry, K_H2O_radio, K_H2O_radio_dipolar,
)
from .surface_anchor import (
    SurfaceObservation, AnchorParams, compute_anchor,
    density_scale_at, temperature_offset_at, anchored_H2O_density,
)
from .profile import (
    Mode, AtmosphericConditions, RefractivityResult, VectorRefractivityResult,
    RefractionProfile,
)
from .ray_trace import (
    EikonalTracer, EikonalInput, EikonalStopCondition,
    EikonalResult, SensitivityResult,
    BatchInput, BatchResult,
    speed_from_eta, SPEED_OF_LIGHT,
)
from .geodetic import (
    geodetic_to_ecef, ecef_to_geodetic, geodetic_normal,
    enu_frame, principal_radii, normal_jacobian,
)
from .atmosphere import NRLMSIS21, MSISInput, MSISOutput, MSISFullOutput
from .atmosphere.nrlmsis21 import MSISProfileOutput

# Auto-select fastest available tracer
try:
    from .native import NativeTracer as Tracer
except ImportError:
    Tracer = EikonalTracer

__all__ = [
    "svp_giacomo", "enhancement_factor", "water_vapor_mole_fraction",
    "water_vapor_number_density", "water_vapor_profile",
    "ciddor_n", "n_standard_air", "n_water_vapor", "compressibility_Z",
    "itu_N", "itu_N_from_surface",
    "K_N2_optical", "K_O2_optical", "K_Ar_optical", "K_He_optical",
    "K_H_optical", "K_O_optical", "K_N_optical", "K_NO_optical",
    "K_H2O_optical", "K_radio_dry", "K_H2O_radio", "K_H2O_radio_dipolar",
    "SurfaceObservation", "AnchorParams", "compute_anchor",
    "density_scale_at", "temperature_offset_at", "anchored_H2O_density",
    "Mode", "AtmosphericConditions", "RefractivityResult",
    "VectorRefractivityResult", "RefractionProfile",
    "NRLMSIS21", "MSISInput", "MSISOutput", "MSISFullOutput",
    "MSISProfileOutput",
    "EikonalTracer", "EikonalInput", "EikonalStopCondition",
    "EikonalResult", "SensitivityResult",
    "BatchInput", "BatchResult",
    "speed_from_eta", "SPEED_OF_LIGHT",
    "geodetic_to_ecef", "ecef_to_geodetic", "geodetic_normal",
    "enu_frame", "principal_radii", "normal_jacobian",
    "Tracer",
]
