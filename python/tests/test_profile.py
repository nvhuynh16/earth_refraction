"""Tests for refraction profile."""

import numpy as np
import pytest
from refraction.profile import Mode


def test_ciddor_surface_reasonable(standard_profile):
    r = standard_profile.compute(0.0, Mode.Ciddor, 633.0)
    assert 250.0 < r.N < 300.0
    assert 1.00025 < r.n < 1.00030


def test_itu_surface_reasonable(standard_profile):
    r = standard_profile.compute(0.0, Mode.ITU_R_P453)
    assert 280.0 < r.N < 360.0


def test_profile_monotonically_decreasing(standard_profile):
    results = standard_profile.profile(0.0, 100.0, 5.0, Mode.Ciddor, 633.0)
    for i in range(1, len(results)):
        assert results[i].N < results[i - 1].N


def test_profile_approaches_vacuum(standard_profile):
    r = standard_profile.compute(100.0, Mode.Ciddor, 633.0)
    assert r.N < 1.0


def test_dn_dh_negative(standard_profile):
    for alt in [0.0, 5.0, 10.0, 20.0, 50.0]:
        r = standard_profile.compute(alt, Mode.Ciddor, 633.0)
        assert r.dn_dh < 0.0, f"dn_dh at {alt} km = {r.dn_dh} should be negative"


def test_radio_surface_gradient(standard_profile):
    r = standard_profile.compute(0.0, Mode.ITU_R_P453)
    assert -80.0 < r.dN_dh < -20.0


def test_ciddor_dn_dh_finite_difference(standard_profile):
    dh = 0.001
    for alt in [1.0, 5.0, 10.0, 30.0, 50.0]:
        r = standard_profile.compute(alt, Mode.Ciddor, 633.0)
        r_plus = standard_profile.compute(alt + dh, Mode.Ciddor, 633.0)
        r_minus = standard_profile.compute(alt - dh, Mode.Ciddor, 633.0)
        dn_dh_fd = (r_plus.n - r_minus.n) / (2.0 * dh)
        tol = max(abs(dn_dh_fd) * 0.05, 1e-12)
        assert abs(r.dn_dh - dn_dh_fd) < tol, (
            f"at {alt} km: analytical={r.dn_dh} FD={dn_dh_fd}"
        )


def test_itu_dn_dh_finite_difference(standard_profile):
    dh = 0.001
    for alt in [1.0, 5.0, 10.0, 30.0, 50.0]:
        r = standard_profile.compute(alt, Mode.ITU_R_P453)
        r_plus = standard_profile.compute(alt + dh, Mode.ITU_R_P453)
        r_minus = standard_profile.compute(alt - dh, Mode.ITU_R_P453)
        dN_dh_fd = (r_plus.N - r_minus.N) / (2.0 * dh)
        tol = max(abs(dN_dh_fd) * 0.05, 1e-6)
        assert abs(r.dN_dh - dN_dh_fd) < tol, (
            f"at {alt} km: analytical={r.dN_dh} FD={dN_dh_fd}"
        )


def test_ciddor_vs_itu_surface(standard_profile):
    r_opt = standard_profile.compute(0.0, Mode.Ciddor, 550.0)
    r_radio = standard_profile.compute(0.0, Mode.ITU_R_P453)
    assert r_opt.N == pytest.approx(r_radio.N, rel=0.30)


def test_batch_vs_scalar_ciddor(standard_profile):
    """Batch (vectorized) path must match scalar compute() for Ciddor."""
    alts = np.array([0.0, 1.0, 5.0, 10.0, 30.0, 50.0, 80.0, 100.0])
    batch = standard_profile(alts, Mode.Ciddor, 633.0)
    for i, h in enumerate(alts):
        s = standard_profile.compute(h, Mode.Ciddor, 633.0)
        assert batch.n[i] == pytest.approx(s.n, rel=1e-12), f"n mismatch at {h} km"
        assert batch.N[i] == pytest.approx(s.N, rel=1e-12), f"N mismatch at {h} km"
        assert batch.dn_dh[i] == pytest.approx(s.dn_dh, rel=1e-10), f"dn_dh mismatch at {h} km"
        assert batch.temperature_K[i] == pytest.approx(s.temperature_K, rel=1e-12)


def test_batch_vs_scalar_itu(standard_profile):
    """Batch (vectorized) path must match scalar compute() for ITU."""
    alts = np.array([0.0, 1.0, 5.0, 10.0, 30.0, 50.0, 80.0, 100.0])
    batch = standard_profile(alts, Mode.ITU_R_P453)
    for i, h in enumerate(alts):
        s = standard_profile.compute(h, Mode.ITU_R_P453)
        assert batch.n[i] == pytest.approx(s.n, rel=1e-12), f"n mismatch at {h} km"
        assert batch.N[i] == pytest.approx(s.N, rel=1e-12), f"N mismatch at {h} km"
        assert batch.dN_dh[i] == pytest.approx(s.dN_dh, rel=1e-10), f"dN_dh mismatch at {h} km"
        assert batch.temperature_K[i] == pytest.approx(s.temperature_K, rel=1e-12)


# ---- New __call__ and profile() tests ----

import warnings
from refraction.profile import RefractivityResult, VectorRefractivityResult


def test_call_scalar_returns_scalar_result(standard_profile):
    r = standard_profile(5.0, Mode.Ciddor, 633.0)
    assert isinstance(r, RefractivityResult)
    assert not isinstance(r, VectorRefractivityResult)


def test_call_single_element_array(standard_profile):
    r = standard_profile(np.array([5.0]), Mode.Ciddor, 633.0)
    assert isinstance(r, VectorRefractivityResult)
    assert r.n.shape == (1,)
    assert r.N.shape == (1,)


def test_call_empty_array(standard_profile):
    r = standard_profile(np.array([]), Mode.Ciddor, 633.0)
    assert isinstance(r, VectorRefractivityResult)
    assert r.n.shape == (0,)


def test_call_wavelength_warning_ciddor(standard_profile):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        standard_profile(0.0, Mode.Ciddor, 200.0)
        assert len(w) == 1
        assert "300" in str(w[0].message) or "1700" in str(w[0].message)


def test_call_wavelength_warning_itu(standard_profile):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        standard_profile(0.0, Mode.ITU_R_P453, 550.0)
        assert len(w) == 1
        assert "ignored" in str(w[0].message).lower()


def test_profile_method_basic(standard_profile):
    results = standard_profile.profile(0.0, 50.0, 10.0, Mode.Ciddor, 633.0)
    assert len(results) == 6  # 0, 10, 20, 30, 40, 50
    assert all(isinstance(r, RefractivityResult) for r in results)
    for i in range(1, len(results)):
        assert results[i].N < results[i - 1].N


def test_profile_method_matches_compute(standard_profile):
    results = standard_profile.profile(0.0, 20.0, 10.0, Mode.Ciddor, 633.0)
    for r in results:
        r2 = standard_profile.compute(r.h_km, Mode.Ciddor, 633.0)
        assert r.n == pytest.approx(r2.n, rel=1e-14)
        assert r.N == pytest.approx(r2.N, rel=1e-14)


def test_profile_high_altitude(standard_profile):
    r = standard_profile.compute(120.0, Mode.Ciddor, 633.0)
    assert r.N < 1.0
    assert r.temperature_K > 200.0


# ---- Rigorous dn/dh tests ----

from scipy.integrate import simpson


def test_fd_step_convergence(standard_profile):
    """Sweep FD step sizes to verify optimal step; middle steps must agree within 1%."""
    alt = 5.0
    steps = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    dn_dh_values = []
    for dh in steps:
        rp = standard_profile.compute(alt + dh, Mode.Ciddor, 633.0)
        rm = standard_profile.compute(alt - dh, Mode.Ciddor, 633.0)
        dn_dh_values.append((rp.n - rm.n) / (2.0 * dh))

    # The 3 middle steps (0.01, 0.001, 0.0001) should all agree within 1%
    ref = dn_dh_values[2]  # dh=0.001
    for i in [1, 3]:  # dh=0.01 and dh=0.0001
        assert abs(dn_dh_values[i] - ref) < abs(ref) * 0.01, (
            f"FD step {steps[i]} km gave {dn_dh_values[i]}, "
            f"reference (dh=0.001) = {ref}"
        )


_FD_ALTITUDES = [0.5, 1.0, 3.0, 5.0, 10.0, 20.0, 50.0, 80.0]


@pytest.mark.parametrize("alt", _FD_ALTITUDES)
@pytest.mark.parametrize("mode_wl", [
    (Mode.Ciddor, 633.0),
    (Mode.ITU_R_P453, None),
])
def test_dn_dh_optimal_fd(standard_profile, alt, mode_wl):
    """FD validation with near-optimal step (dh=0.0005 km), 2% tolerance."""
    mode, wl = mode_wl
    kwargs = {"wavelength_nm": wl} if wl is not None else {}
    dh = 0.0005

    r = standard_profile.compute(alt, mode, **kwargs)
    rp = standard_profile.compute(alt + dh, mode, **kwargs)
    rm = standard_profile.compute(alt - dh, mode, **kwargs)

    if mode == Mode.Ciddor:
        fd = (rp.n - rm.n) / (2.0 * dh)
        analytical = r.dn_dh
    else:
        fd = (rp.N - rm.N) / (2.0 * dh)
        analytical = r.dN_dh

    tol = max(abs(fd) * 0.02, 1e-12)
    assert abs(analytical - fd) < tol, (
        f"at {alt} km ({mode.value}): analytical={analytical}, FD={fd}"
    )


_INTEGRATION_RANGES = [
    (0.0, 5.0),   # crosses water vapor H_w=2 km
    (5.0, 15.0),  # mid troposphere
    (15.0, 50.0), # stratosphere
    (50.0, 90.0), # mesosphere
]


@pytest.mark.parametrize("h_start,h_end", _INTEGRATION_RANGES)
@pytest.mark.parametrize("mode_wl", [
    (Mode.Ciddor, 633.0),
    (Mode.ITU_R_P453, None),
])
def test_dn_dh_integration(standard_profile, h_start, h_end, mode_wl):
    """Integrate dn/dh with Simpson's rule; compare to n(h_end) - n(h_start)."""
    mode, wl = mode_wl
    kwargs = {"wavelength_nm": wl} if wl is not None else {}
    npts = 201

    alts = np.linspace(h_start, h_end, npts)
    batch = standard_profile(alts, mode, **kwargs)

    if mode == Mode.Ciddor:
        integral = simpson(batch.dn_dh, x=alts)
        delta = batch.n[-1] - batch.n[0]
    else:
        integral = simpson(batch.dN_dh, x=alts)
        delta = batch.N[-1] - batch.N[0]

    tol = max(abs(delta) * 0.005, 1e-12)
    assert abs(integral - delta) < tol, (
        f"{mode.value} [{h_start}-{h_end} km]: "
        f"integral={integral}, delta={delta}"
    )


@pytest.mark.parametrize("mode_wl", [
    (Mode.Ciddor, 633.0),
    (Mode.ITU_R_P453, None),
])
def test_dn_dh_water_vapor_integration(standard_profile, mode_wl):
    """Dense integration over 0-10 km water vapor region, 0.2% tolerance."""
    mode, wl = mode_wl
    kwargs = {"wavelength_nm": wl} if wl is not None else {}
    npts = 401
    alts = np.linspace(0.0, 10.0, npts)
    batch = standard_profile(alts, mode, **kwargs)

    if mode == Mode.Ciddor:
        integral = simpson(batch.dn_dh, x=alts)
        delta = batch.n[-1] - batch.n[0]
    else:
        integral = simpson(batch.dN_dh, x=alts)
        delta = batch.N[-1] - batch.N[0]

    tol = max(abs(delta) * 0.002, 1e-12)
    assert abs(integral - delta) < tol, (
        f"{mode.value} water vapor region: "
        f"integral={integral}, delta={delta}"
    )


@pytest.mark.parametrize("mode_wl", [
    (Mode.Ciddor, 633.0),
    (Mode.ITU_R_P453, None),
])
def test_dn_dh_smoothness(standard_profile, mode_wl):
    """No jumps >2x or sign flips in dn/dh over 0.5-100 km."""
    mode, wl = mode_wl
    kwargs = {"wavelength_nm": wl} if wl is not None else {}
    npts = 501
    alts = np.linspace(0.5, 100.0, npts)  # start at 0.5 to avoid surface edge
    batch = standard_profile(alts, mode, **kwargs)

    deriv = batch.dn_dh if mode == Mode.Ciddor else batch.dN_dh

    # All derivatives must be strictly negative (density always decreases)
    assert np.all(deriv < 0), (
        f"{mode.value}: expected all dn/dh < 0, "
        f"found {np.sum(deriv >= 0)} non-negative at "
        f"h={alts[deriv >= 0]} km"
    )

    # No adjacent values should jump by more than 2x
    for i in range(len(deriv) - 1):
        ratio = abs(deriv[i + 1] / deriv[i])
        assert ratio < 2.0, (
            f"{mode.value} at h={alts[i]:.2f} km: "
            f"dn/dh jump ratio={ratio:.3f} "
            f"({deriv[i]:.6e} -> {deriv[i+1]:.6e})"
        )
