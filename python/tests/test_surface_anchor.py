"""Tests for surface anchoring."""

import math
import pytest
from refraction.surface_anchor import (
    SurfaceObservation, compute_anchor, density_scale_at,
    temperature_offset_at, anchored_H2O_density,
)


def test_density_scale_at_surface(msis, standard_input, standard_obs):
    params = compute_anchor(standard_obs, msis, standard_input)
    s = density_scale_at(0.0, params)
    assert s == pytest.approx(params.density_scale, abs=1e-10)


def test_density_scale_hydrostatic(msis, standard_input, standard_obs):
    params = compute_anchor(standard_obs, msis, standard_input)

    # Scale should change monotonically from S₀ toward asymptote
    s_0 = density_scale_at(0.0, params)
    s_20 = density_scale_at(20.0, params)
    s_40 = density_scale_at(40.0, params)
    s_100 = density_scale_at(100.0, params)

    # Monotonic: if hydro_C > 0, scale increases; if < 0, decreases
    if params.hydro_C > 0:
        assert s_20 > s_0
        assert s_40 > s_20
    elif params.hydro_C < 0:
        assert s_20 < s_0
        assert s_40 < s_20

    # Convergence: 40 km and 100 km should be close (erf saturates)
    asymptote = params.density_scale * math.exp(params.hydro_C)
    assert s_100 == pytest.approx(asymptote, rel=1e-6)


def test_density_scale_constant_when_no_temp_offset(msis, standard_input):
    """When dT=0, hydro_C=0, so S should be constant (= S₀) at all altitudes."""
    # Get model temperature at surface so we can match it exactly
    from refraction.atmosphere.nrlmsis21 import NRLMSIS21, MSISInput
    inp = MSISInput(
        day=standard_input.day, utsec=standard_input.utsec,
        alt=0.0, lat=standard_input.lat, lon=standard_input.lon,
        f107a=standard_input.f107a, f107=standard_input.f107,
        ap=list(standard_input.ap),
    )
    out = msis.msiscalc(inp)
    T_model_C = out.tn - 273.15

    obs = SurfaceObservation(altitude_km=0.0, temperature_C=T_model_C,
                             pressure_kPa=101.325, relative_humidity=0.0)
    params = compute_anchor(obs, msis, standard_input)

    assert params.hydro_C == pytest.approx(0.0, abs=1e-15)
    for h in [0.0, 10.0, 30.0, 60.0, 100.0]:
        assert density_scale_at(h, params) == pytest.approx(params.density_scale, rel=1e-14)


def test_density_scale_hydrostatic_asymptote(msis, standard_input, standard_obs):
    """At very high altitude, S → S₀·exp(C)."""
    params = compute_anchor(standard_obs, msis, standard_input)
    asymptote = params.density_scale * math.exp(params.hydro_C)
    s_200 = density_scale_at(200.0, params)
    assert s_200 == pytest.approx(asymptote, rel=1e-10)


def test_temperature_offset_at_surface(msis, standard_input, standard_obs):
    params = compute_anchor(standard_obs, msis, standard_input)
    dT = temperature_offset_at(0.0, params)
    assert dT == pytest.approx(params.temperature_offset, abs=1e-10)


def test_temperature_offset_tapers(msis, standard_input, standard_obs):
    params = compute_anchor(standard_obs, msis, standard_input)
    dT_30 = temperature_offset_at(30.0, params)
    expected = params.temperature_offset * math.exp(-30.0 * 30.0 / (2.0 * 15.0 * 15.0))
    assert dT_30 == pytest.approx(expected, abs=1e-10)


def test_water_vapor_surface_value(msis, standard_input, standard_obs):
    params = compute_anchor(standard_obs, msis, standard_input)
    assert params.n_H2O_surface > 0.0
    n_H2O_surface = anchored_H2O_density(0.0, params)
    assert n_H2O_surface == pytest.approx(params.n_H2O_surface, abs=1e-10)


def test_water_vapor_decay(msis, standard_input):
    obs = SurfaceObservation(altitude_km=0.0, temperature_C=20.0,
                             pressure_kPa=101.325, relative_humidity=1.0)
    params = compute_anchor(obs, msis, standard_input)
    n_10 = anchored_H2O_density(10.0, params)
    expected = params.n_H2O_surface * math.exp(-10.0 / 2.0)
    assert n_10 == pytest.approx(expected, rel=1e-10)
    n_15 = anchored_H2O_density(15.0, params)
    assert n_15 < params.n_H2O_surface * 1e-3


def test_dry_air_zero_humidity(msis, standard_input):
    obs = SurfaceObservation(altitude_km=0.0, temperature_C=20.0,
                             pressure_kPa=101.325, relative_humidity=0.0)
    params = compute_anchor(obs, msis, standard_input)
    assert params.n_H2O_surface == pytest.approx(0.0, abs=1e-10)


def test_scale_factor_reasonable(msis, standard_input):
    obs = SurfaceObservation(altitude_km=0.0, temperature_C=15.0,
                             pressure_kPa=101.325, relative_humidity=0.0)
    params = compute_anchor(obs, msis, standard_input)
    assert 0.8 < params.density_scale < 1.2


# ---- Array input tests ----

import numpy as np


def test_density_scale_at_array(msis, standard_input, standard_obs):
    params = compute_anchor(standard_obs, msis, standard_input)
    h_arr = np.array([0.0, 5.0, 10.0, 30.0, 60.0])
    s_arr = density_scale_at(h_arr, params)
    for i, h in enumerate(h_arr):
        s_scalar = float(density_scale_at(h, params))
        assert s_arr[i] == pytest.approx(s_scalar, rel=1e-14)


def test_temperature_offset_at_array(msis, standard_input, standard_obs):
    params = compute_anchor(standard_obs, msis, standard_input)
    h_arr = np.array([0.0, 5.0, 15.0, 40.0])
    dT_arr = temperature_offset_at(h_arr, params)
    for i, h in enumerate(h_arr):
        dT_scalar = float(temperature_offset_at(h, params))
        assert dT_arr[i] == pytest.approx(dT_scalar, rel=1e-14)


def test_anchored_H2O_density_array(msis, standard_input, standard_obs):
    params = compute_anchor(standard_obs, msis, standard_input)
    h_arr = np.array([0.0, 2.0, 5.0, 10.0])
    n_arr = anchored_H2O_density(h_arr, params)
    for i, h in enumerate(h_arr):
        n_scalar = float(anchored_H2O_density(h, params))
        assert n_arr[i] == pytest.approx(n_scalar, rel=1e-14)


def test_compute_anchor_elevated_station(msis, standard_input):
    obs = SurfaceObservation(altitude_km=1.5, temperature_C=10.0,
                             pressure_kPa=85.0, relative_humidity=0.3)
    params = compute_anchor(obs, msis, standard_input)
    assert 0.8 < params.density_scale < 1.2
    assert params.h_surface_km == 1.5
    assert params.n_H2O_surface > 0.0


def test_compute_anchor_extreme_humidity(msis, standard_input):
    obs_dry = SurfaceObservation(altitude_km=0.0, temperature_C=25.0,
                                  pressure_kPa=101.325, relative_humidity=0.0)
    obs_sat = SurfaceObservation(altitude_km=0.0, temperature_C=25.0,
                                  pressure_kPa=101.325, relative_humidity=1.0)
    p_dry = compute_anchor(obs_dry, msis, standard_input)
    p_sat = compute_anchor(obs_sat, msis, standard_input)
    assert p_dry.n_H2O_surface == pytest.approx(0.0, abs=1e-10)
    assert p_sat.n_H2O_surface > 1e22  # physically reasonable for 25C saturated
