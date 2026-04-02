"""
Ocean water refractive index and sound speed computation.

This sub-package computes the refractive index and sound speed of seawater
as a function of depth (pressure), temperature, salinity, and
wavelength/frequency.

Refractive index modes:
    - **MeissnerWentz**: radio/microwave permittivity (Meissner & Wentz 2004/2012).
    - **MillardSeaver**: optical with pressure (Millard & Seaver 1990, 500-700 nm).
    - **QuanFry**: optical, surface only (Quan & Fry 1995, 400-700 nm).
    - **IAPWS**: pure water, wide wavelength range (IAPWS R9-97, 200-1100 nm).

Sound speed modes:
    - **DelGrosso**: Del Grosso (1974) / Wong & Zhu (1995) ITS-90.
    - **ChenMillero**: Chen & Millero (1977) / Wong & Zhu (1995) ITS-90.
    - **Mackenzie**: Mackenzie (1981), 9-term equation using depth directly.
"""

from .ocean_refraction import (
    OceanMode,
    OceanRefractivityResult,
    VectorOceanRefractivityResult,
    OceanRefractionProfile,
)
from .ocean_profile import (
    OceanConditions,
    OceanPreset,
    FLORIDA_GULF_SUMMER,
    FLORIDA_GULF_WINTER,
    CALIFORNIA_SUMMER,
    CALIFORNIA_WINTER,
    depth_to_pressure,
    pressure_to_depth,
)
from .sound_speed import (
    SoundMode,
    SoundSpeedResult,
    VectorSoundSpeedResult,
    SoundSpeedProfile,
)

__all__ = [
    "OceanMode",
    "OceanConditions",
    "OceanRefractivityResult",
    "VectorOceanRefractivityResult",
    "OceanRefractionProfile",
    "OceanPreset",
    "FLORIDA_GULF_SUMMER",
    "FLORIDA_GULF_WINTER",
    "CALIFORNIA_SUMMER",
    "CALIFORNIA_WINTER",
    "depth_to_pressure",
    "pressure_to_depth",
    "SoundMode",
    "SoundSpeedResult",
    "VectorSoundSpeedResult",
    "SoundSpeedProfile",
]
