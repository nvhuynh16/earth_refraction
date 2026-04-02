"""Shared fixtures for refraction tests."""

import pytest
from refraction.atmosphere.nrlmsis21 import NRLMSIS21, MSISInput
from refraction.surface_anchor import SurfaceObservation
from refraction.profile import AtmosphericConditions, RefractionProfile
from refraction.ocean import OceanConditions


@pytest.fixture
def msis():
    return NRLMSIS21()


@pytest.fixture
def standard_input():
    return MSISInput(
        day=172, utsec=29000, alt=0.0, lat=45.0, lon=-75.0,
        f107a=150.0, f107=150.0, ap=[4, 4, 4, 4, 4, 4, 4],
    )


@pytest.fixture
def standard_obs():
    return SurfaceObservation(
        altitude_km=0.0, temperature_C=20.0,
        pressure_kPa=101.325, relative_humidity=0.5,
    )


@pytest.fixture
def standard_atm():
    return AtmosphericConditions(
        day_of_year=172, ut_seconds=29000,
        latitude_deg=45.0, longitude_deg=-75.0,
    )


@pytest.fixture
def standard_profile(standard_atm, standard_obs):
    return RefractionProfile(standard_atm, standard_obs)


# ── Ocean fixtures ───────────────────────────────────────────────────

@pytest.fixture
def ocean_cond():
    """Basic ocean conditions (no EM frequency/wavelength)."""
    return OceanConditions(sst_C=25.0, sss_psu=35.0)


@pytest.fixture
def ocean_cond_radio():
    """Ocean conditions with frequency and wavelength for all EM modes."""
    return OceanConditions(sst_C=25.0, sss_psu=35.0, freq_ghz=10.0,
                           wavelength_nm=589.26)
