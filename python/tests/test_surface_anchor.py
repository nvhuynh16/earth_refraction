"""Tests for surface anchoring."""

import math

import numpy as np
import pytest
from refraction.surface_anchor import (
    SurfaceObservation, compute_anchor, density_scale_at,
    temperature_offset_at, anchored_H2O_density,
)


def test_density_scale_at_surface(msis, standard_input, standard_obs):
    params = compute_anchor(standard_obs, msis, standard_input)
    s = density_scale_at(0.0, params)
    assert s == pytest.approx(params.density_scale, abs=1e-10)


def _model_surface_T_C(msis, standard_input, alt=0.0):
    from refraction.atmosphere.nrlmsis21 import MSISInput
    inp = MSISInput(
        day=standard_input.day, utsec=standard_input.utsec,
        alt=alt, lat=standard_input.lat, lon=standard_input.lon,
        f107a=standard_input.f107a, f107=standard_input.f107,
        ap=list(standard_input.ap),
    )
    return msis.msiscalc(inp).tn - 273.15


def _anchored_pressure(msis, standard_input, params, h):
    """Dry pressure of the anchored NRLMSIS species at altitude h (Pa)."""
    from refraction.atmosphere.nrlmsis21 import MSISInput
    from refraction.constants import kB
    inp = MSISInput(
        day=standard_input.day, utsec=standard_input.utsec,
        alt=h, lat=standard_input.lat, lon=standard_input.lon,
        f107a=standard_input.f107a, f107=standard_input.f107,
        ap=list(standard_input.ap),
    )
    out = msis.msiscalc(inp)
    n_model = sum(out.dn[i] for i in range(2, 11) if 0 < out.dn[i] < 1e30)
    T_a = out.tn + temperature_offset_at(h, params)
    return n_model * density_scale_at(h, params) * kB * T_a, n_model * kB * out.tn


@pytest.mark.parametrize("T_C, P_kPa, RH, h", [
    (20.0, 101.325, 0.5, 0.0),
    (35.0, 100.0, 0.9, 0.0),
    (-5.0, 102.0, 0.0, 0.0),
    (10.0, 85.0, 0.3, 1.5),
])
def test_anchored_surface_reproduces_observation(msis, standard_input, T_C, P_kPa, RH, h):
    """Anchored dry air + water vapor must give the observed T and total P."""
    from refraction.constants import kB
    obs = SurfaceObservation(altitude_km=h, temperature_C=T_C,
                             pressure_kPa=P_kPa, relative_humidity=RH)
    params = compute_anchor(obs, msis, standard_input)
    P_dry, _ = _anchored_pressure(msis, standard_input, params, h)
    T_K = T_C + 273.15
    e = anchored_H2O_density(h, params) * kB * T_K
    assert P_dry + e == pytest.approx(P_kPa * 1000.0, rel=1e-10)
    T_model_C = _model_surface_T_C(msis, standard_input, h)
    assert T_model_C + 273.15 + temperature_offset_at(h, params) == pytest.approx(T_K, rel=1e-12)


def test_density_scale_includes_temperature_ratio(msis, standard_input):
    """Warmer air at the same pressure must be less dense."""
    cold = compute_anchor(SurfaceObservation(0.0, 0.0, 101.325, 0.0), msis, standard_input)
    hot = compute_anchor(SurfaceObservation(0.0, 30.0, 101.325, 0.0), msis, standard_input)
    assert hot.density_scale / cold.density_scale == pytest.approx(273.15 / 303.15, rel=1e-12)


def test_density_scale_excludes_vapor_pressure(msis, standard_input):
    """At equal T and total P, humid air has less dry air (Dalton)."""
    from refraction.constants import kB
    dry = compute_anchor(SurfaceObservation(0.0, 25.0, 101.325, 0.0), msis, standard_input)
    wet = compute_anchor(SurfaceObservation(0.0, 25.0, 101.325, 1.0), msis, standard_input)
    e = wet.n_H2O_surface * kB * 298.15
    assert wet.pressure_scale / dry.pressure_scale == pytest.approx(
        (101325.0 - e) / 101325.0, rel=1e-12)


@pytest.mark.parametrize("T_C, H_T", [(35.0, 2.0), (35.0, 15.0), (-10.0, 5.0)])
def test_density_scale_hydrostatic(msis, standard_input, T_C, H_T):
    """ln(P_anch/P_model) must match an independent hydrostatic integral."""
    from refraction.atmosphere.nrlmsis21 import (
        MSISInput, MBARG0DIVKB, _alt2gph)
    obs = SurfaceObservation(0.0, T_C, 101.325, 0.0, temperature_taper_km=H_T)
    params = compute_anchor(obs, msis, standard_input)
    P_a0, P_m0 = _anchored_pressure(msis, standard_input, params, 0.0)

    # Dense independent integral with Simpson's rule in geopotential height
    hs = np.linspace(0.0, 8.0 * H_T, 4001)
    inp = MSISInput(day=standard_input.day, utsec=standard_input.utsec, alt=0.0,
                    lat=standard_input.lat, lon=standard_input.lon,
                    f107a=standard_input.f107a, f107=standard_input.f107,
                    ap=list(standard_input.ap))
    T_m = msis.msiscalc_profile(inp, hs).tn
    T_a = T_m + temperature_offset_at(hs, params)
    zeta = np.array([_alt2gph(standard_input.lat, x) for x in hs])
    f = MBARG0DIVKB * (1.0 / T_m - 1.0 / T_a)
    for k in [len(hs) // 8, len(hs) // 4, len(hs) // 2, len(hs) - 1]:
        z, y = zeta[:k + 1], f[:k + 1]
        integral = np.trapezoid(y, z) if hasattr(np, "trapezoid") else np.trapz(y, z)
        P_a, P_m = _anchored_pressure(msis, standard_input, params, hs[k])
        lhs = math.log(P_a / P_m) - math.log(P_a0 / P_m0)
        assert lhs == pytest.approx(integral, abs=1e-5)

    # Warm anomaly: anchored pressure decays more slowly than the model's
    if T_C + 273.15 > T_m[0]:
        assert math.log(P_a / P_m) > math.log(P_a0 / P_m0)


def test_density_scale_constant_when_no_temp_offset(msis, standard_input):
    """When dT=0, the hydrostatic correction vanishes and S = S₀ everywhere."""
    T_model_C = _model_surface_T_C(msis, standard_input)
    obs = SurfaceObservation(altitude_km=0.0, temperature_C=T_model_C,
                             pressure_kPa=101.325, relative_humidity=0.0)
    params = compute_anchor(obs, msis, standard_input)
    assert params.temperature_offset == pytest.approx(0.0, abs=1e-12)
    for h in [-0.5, 0.0, 10.0, 30.0, 60.0, 100.0]:
        assert density_scale_at(h, params) == pytest.approx(params.density_scale, rel=1e-12)


def test_density_scale_converges_above_taper(msis, standard_input, standard_obs):
    """Once the temperature offset has decayed, S stops changing."""
    params = compute_anchor(standard_obs, msis, standard_input)
    top = params.h_surface_km + 6.0 * params.H_T
    s_top = density_scale_at(top, params)
    for h in [top + 1.0, 60.0, 100.0, 200.0]:
        assert density_scale_at(h, params) == pytest.approx(s_top, rel=1e-7)


def test_density_scale_smooth(msis, standard_input):
    """Hermite table: d(ln S)/dh is continuous and matches the tabulated slope."""
    obs = SurfaceObservation(0.0, 35.0, 101.325, 0.5)
    params = compute_anchor(obs, msis, standard_input)
    # One-sided slopes on both sides of each node equal the analytic
    # table derivative, so d(ln S)/dh is continuous across nodes.
    nodes = params.table_h_km[10:-10:20]
    idx = np.searchsorted(params.table_h_km, nodes)
    eps = 1e-6  # truncation error eps*|f''|/2 < 1e-7
    ln_0 = np.log(density_scale_at(nodes, params))
    right = (np.log(density_scale_at(nodes + eps, params)) - ln_0) / eps
    left = (ln_0 - np.log(density_scale_at(nodes - eps, params))) / eps
    assert np.allclose(right, params.table_dln_scale[idx], rtol=0, atol=1e-7)
    assert np.allclose(left, params.table_dln_scale[idx], rtol=0, atol=1e-7)


def test_temperature_offset_at_surface(msis, standard_input, standard_obs):
    params = compute_anchor(standard_obs, msis, standard_input)
    dT = temperature_offset_at(0.0, params)
    assert dT == pytest.approx(params.temperature_offset, abs=1e-10)


def test_temperature_offset_tapers(msis, standard_input, standard_obs):
    params = compute_anchor(standard_obs, msis, standard_input)
    assert params.H_T == 2.0  # default taper scale
    dT_3 = temperature_offset_at(3.0, params)
    expected = params.temperature_offset * math.exp(-3.0 * 3.0 / (2.0 * 2.0 * 2.0))
    assert dT_3 == pytest.approx(expected, abs=1e-10)


def test_temperature_taper_configurable(msis, standard_input):
    obs = SurfaceObservation(0.0, 20.0, 101.325, 0.5, temperature_taper_km=5.0)
    params = compute_anchor(obs, msis, standard_input)
    assert params.H_T == 5.0
    assert temperature_offset_at(5.0, params) == pytest.approx(
        params.temperature_offset * math.exp(-0.5), rel=1e-12)


@pytest.mark.parametrize("H_T", [0.0, -1.0])
def test_temperature_taper_must_be_positive(msis, standard_input, H_T):
    obs = SurfaceObservation(0.0, 20.0, 101.325, 0.5, temperature_taper_km=H_T)
    with pytest.raises(ValueError):
        compute_anchor(obs, msis, standard_input)


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
