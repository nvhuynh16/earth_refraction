"""Tests for the 3D ECEF eikonal ray tracer.

Organised in phases:
- Phase 1: Basic trace (constant speed, invariants, Bennett)
- Phase 2: Derivative properties
- Phase 3: Sensitivity derivatives (analytical + FD validation)
"""

import math

import numpy as np
import pytest

from refraction.ray_trace import (
    EikonalInput,
    EikonalResult,
    EikonalStopCondition,
    EikonalTracer,
    SensitivityResult,
    speed_from_eta,
)
from refraction.geodetic import (
    ecef_to_geodetic,
    enu_frame,
    geodetic_normal,
    geodetic_to_ecef,
    principal_radii,
)


# ===================================================================
# Helpers
# ===================================================================

def _constant_tracer(v0=1.0 / 1.000315):
    """Tracer with constant propagation speed (v = 1/eta with c_ref=1)."""
    return EikonalTracer(lambda _h: v0, lambda _h: 0.0)


def _exponential_tracer(n0=1.000315, a=-39e-9):
    """Tracer from exponential refractive index (c_ref = 1).

    eta(h) = n0 * exp(a*h), v(h) = 1/eta(h).
    """
    v_func, dv_func = speed_from_eta(
        lambda h: n0 * math.exp(a * h),
        lambda h: a * n0 * math.exp(a * h),
        1.0,  # c_ref = 1 for clean test values
    )
    return EikonalTracer(v_func, dv_func)


def _standard_input(elevation=30.0, azimuth=0.0):
    """Standard input at mid-latitude."""
    return EikonalInput(
        lat_deg=45.0, lon_deg=0.0, h_m=0.0,
        elevation_deg=elevation, azimuth_deg=azimuth,
    )


# ===================================================================
# Phase 1: Basic trace
# ===================================================================

class TestConstantSpeed:
    """Constant propagation speed -> straight line in ECEF."""

    V0 = 1.0 / 1.000315  # v = 1/eta with c_ref = 1

    @pytest.mark.parametrize("elevation", [10.0, 30.0, 45.0, 85.0])
    @pytest.mark.parametrize("azimuth", [0.0, 90.0, 225.0])
    def test_straight_line(self, elevation, azimuth):
        tracer = _constant_tracer(self.V0)
        inp = _standard_input(elevation, azimuth)
        stop = EikonalStopCondition(max_arc_length_m=100_000.0,
                                     detect_ground_hit=False)
        result = tracer.trace(inp, stop)

        r0 = result.r[0]
        sigma0 = 1.0 / self.V0
        t0 = result.p[0] / sigma0
        for k in range(len(result.s)):
            r_expected = r0 + t0 * result.s[k]
            np.testing.assert_allclose(
                result.r[k], r_expected, rtol=1e-11, atol=1e-4
            )

    def test_momentum_constant(self):
        tracer = _constant_tracer(self.V0)
        inp = _standard_input(30.0, 45.0)
        stop = EikonalStopCondition(max_arc_length_m=50_000.0,
                                     detect_ground_hit=False)
        result = tracer.trace(inp, stop)

        p0 = result.p[0]
        for k in range(len(result.s)):
            np.testing.assert_allclose(result.p[k], p0, atol=1e-12)

    @pytest.mark.parametrize("elevation", [10.0, 45.0, 85.0])
    def test_momentum_magnitude(self, elevation):
        tracer = _constant_tracer(self.V0)
        inp = _standard_input(elevation)
        stop = EikonalStopCondition(max_arc_length_m=50_000.0,
                                     detect_ground_hit=False)
        result = tracer.trace(inp, stop)

        np.testing.assert_allclose(
            np.abs(result.momentum_error), 0.0, atol=1e-13
        )


class TestMomentumPreservation:
    """For any profile, |p| = 1/v should be preserved."""

    @pytest.mark.parametrize("elevation", [5.0, 30.0, 60.0])
    def test_exponential_profile(self, elevation):
        tracer = _exponential_tracer()
        inp = _standard_input(elevation)
        stop = EikonalStopCondition(
            target_altitude_m=100_000.0,
            max_arc_length_m=500_000.0,
            detect_ground_hit=False,
        )
        result = tracer.trace(inp, stop)

        np.testing.assert_allclose(
            np.abs(result.momentum_error), 0.0, atol=1e-8
        )


class TestGradSigma:
    """Verify the slowness gradient computation."""

    def test_grad_vs_fd(self):
        """Compare analytical grad(sigma) against FD perturbation."""
        n0 = 1.000315
        a = -39e-9
        # v = 1/eta with c_ref = 1
        v_func, _ = speed_from_eta(
            lambda h: n0 * math.exp(a * h),
            lambda h: a * n0 * math.exp(a * h),
            1.0,
        )

        lat, lon, h = 45.0, 30.0, 10_000.0
        r = geodetic_to_ecef(lat, lon, h)
        n_hat = geodetic_normal(lat, lon)
        sigma = 1.0 / v_func(h)
        # dsigma/dh = -v'/v^2
        v = v_func(h)
        eta = n0 * math.exp(a * h)
        deta = a * eta
        dsigma = deta  # since sigma = eta when c_ref = 1
        grad_analytical = dsigma * n_hat

        delta = 1.0
        grad_fd = np.zeros(3)
        for j in range(3):
            r_p = r.copy()
            r_p[j] += delta
            _, _, h_p = ecef_to_geodetic(*r_p)
            r_m = r.copy()
            r_m[j] -= delta
            _, _, h_m = ecef_to_geodetic(*r_m)
            sigma_p = 1.0 / v_func(max(h_p, 0.0))
            sigma_m = 1.0 / v_func(max(h_m, 0.0))
            grad_fd[j] = (sigma_p - sigma_m) / (2.0 * delta)

        np.testing.assert_allclose(grad_analytical, grad_fd, rtol=1e-5)


class TestBouguerInvariant:

    def test_bouguer_conserved(self):
        tracer = _exponential_tracer(n0=1.000315, a=-39e-9)
        inp = _standard_input(elevation=10.0, azimuth=0.0)
        stop = EikonalStopCondition(
            target_altitude_m=80_000.0,
            max_arc_length_m=500_000.0,
            detect_ground_hit=False,
        )
        result = tracer.trace(inp, stop)

        bouguer = np.empty(len(result.s))
        for k in range(len(result.s)):
            r_k = result.r[k]
            p_k = result.p[k]
            sigma_k = 1.0 / result.v[k]
            r_mag = np.linalg.norm(r_k)
            t_k = p_k / sigma_k
            sin_elev = np.dot(r_k, t_k) / r_mag
            cos_elev = math.sqrt(max(0.0, 1.0 - sin_elev ** 2))
            bouguer[k] = sigma_k * r_mag * cos_elev

        bouguer_rel = bouguer / bouguer[0]
        np.testing.assert_allclose(bouguer_rel, 1.0, atol=0.005)


class TestStopConditions:

    def test_target_altitude(self):
        tracer = _constant_tracer()
        inp = _standard_input(30.0)
        target = 50_000.0
        stop = EikonalStopCondition(target_altitude_m=target,
                                     max_arc_length_m=500_000.0,
                                     detect_ground_hit=False)
        result = tracer.trace(inp, stop)
        assert result.terminated_by == "altitude"
        assert result.h_m[-1] == pytest.approx(target, rel=1e-3)

    def test_ground_hit(self):
        tracer = _constant_tracer()
        inp = EikonalInput(lat_deg=45.0, lon_deg=0.0, h_m=10_000.0,
                           elevation_deg=-30.0, azimuth_deg=0.0)
        stop = EikonalStopCondition(max_arc_length_m=500_000.0,
                                     detect_ground_hit=True)
        result = tracer.trace(inp, stop)
        assert result.terminated_by == "ground"
        assert result.h_m[-1] == pytest.approx(0.0, abs=5.0)

    def test_max_arc_length(self):
        tracer = _constant_tracer()
        inp = _standard_input(30.0)
        s_max = 50_000.0
        stop = EikonalStopCondition(max_arc_length_m=s_max,
                                     detect_ground_hit=False)
        result = tracer.trace(inp, stop)
        assert result.terminated_by == "arc_length"
        assert result.s[-1] == pytest.approx(s_max, rel=1e-6)


class TestAzimuth:

    def test_north_stays_on_meridian(self):
        tracer = _constant_tracer()
        inp = EikonalInput(lat_deg=30.0, lon_deg=45.0, h_m=0.0,
                           elevation_deg=30.0, azimuth_deg=0.0)
        stop = EikonalStopCondition(max_arc_length_m=100_000.0,
                                     detect_ground_hit=False)
        result = tracer.trace(inp, stop)
        np.testing.assert_allclose(result.lon_deg, 45.0, atol=1e-4)


class TestFromProfile:

    @pytest.fixture
    def standard_profile(self):
        from refraction import (
            AtmosphericConditions, SurfaceObservation, RefractionProfile,
        )
        atm = AtmosphericConditions(day_of_year=172, latitude_deg=45.0)
        obs = SurfaceObservation(
            altitude_km=0.0, temperature_C=15.0,
            pressure_kPa=101.325, relative_humidity=0.5,
        )
        return RefractionProfile(atm, obs)

    def test_traces_successfully(self, standard_profile):
        from refraction import Mode
        tracer = EikonalTracer.from_profile(standard_profile, Mode.Ciddor)
        inp = _standard_input(30.0)
        stop = EikonalStopCondition(target_altitude_m=100_000.0,
                                     max_arc_length_m=500_000.0,
                                     detect_ground_hit=False)
        result = tracer.trace(inp, stop)
        assert len(result.s) > 10
        assert result.h_m[-1] > 50_000.0

    def test_momentum_preserved(self, standard_profile):
        from refraction import Mode
        tracer = EikonalTracer.from_profile(standard_profile, Mode.Ciddor)
        inp = _standard_input(15.0)
        stop = EikonalStopCondition(target_altitude_m=50_000.0,
                                     max_arc_length_m=500_000.0,
                                     detect_ground_hit=False)
        result = tracer.trace(inp, stop)
        np.testing.assert_allclose(
            np.abs(result.momentum_error), 0.0, atol=1e-8
        )


class TestBennett:

    def test_bending_45deg(self):
        N0 = 315e-6
        H = 7350.0
        v_func, dv_func = speed_from_eta(
            lambda h: 1.0 + N0 * math.exp(-h / H),
            lambda h: -N0 / H * math.exp(-h / H),
            1.0,
        )
        tracer = EikonalTracer(v_func, dv_func)
        inp = _standard_input(elevation=45.0, azimuth=0.0)
        stop = EikonalStopCondition(target_altitude_m=120_000.0,
                                     max_arc_length_m=1_000_000.0,
                                     detect_ground_hit=False)
        result = tracer.trace(inp, stop)

        bending_arcmin = abs(result.bending_deg) * 60.0
        assert bending_arcmin == pytest.approx(1.0, rel=0.25)


class TestFromDepthSpeedTable:
    """Test creating a tracer from measured depth vs speed data."""

    def test_basic_trace(self):
        depths = [0, 50, 100, 200, 500, 1000, 2000]
        speeds = [1520, 1515, 1490, 1495, 1500, 1510, 1520]
        tracer = EikonalTracer.from_depth_speed_table(depths, speeds)

        inp = EikonalInput(lat_deg=30.0, lon_deg=-80.0, h_m=-1.0,
                           elevation_deg=-10.0, azimuth_deg=90.0)
        stop = EikonalStopCondition(max_arc_length_m=5000.0,
                                     detect_ground_hit=False)
        result = tracer.trace(inp, stop)

        # Speed should be ~1520 m/s near surface
        assert 1400 < result.v[0] < 1600
        # Ray goes deeper
        assert result.h_m[-1] < result.h_m[0]

    def test_speed_at_surface(self):
        """Speed at depth=0 should match speed at h=0."""
        depths = [0, 100, 500]
        speeds = [1520, 1490, 1500]
        tracer = EikonalTracer.from_depth_speed_table(depths, speeds)
        # v_func at h=0 should be 1520 (depth=0)
        assert tracer._v(0.0) == pytest.approx(1520.0, rel=1e-6)


class TestFromOceanProfiles:
    """Integration with ocean profile modules."""

    def test_from_sound_speed_profile(self):
        from refraction.ocean.sound_speed import SoundSpeedProfile, SoundMode
        from refraction.ocean.ocean_profile import OceanConditions

        cond = OceanConditions(sst_C=25.0, sss_psu=35.0)
        profile = SoundSpeedProfile(cond)
        tracer = EikonalTracer.from_sound_speed_profile(
            profile, SoundMode.DelGrosso
        )

        inp = EikonalInput(lat_deg=30.0, lon_deg=-80.0, h_m=-1.0,
                           elevation_deg=-10.0, azimuth_deg=90.0)
        stop = EikonalStopCondition(max_arc_length_m=5000.0,
                                     detect_ground_hit=False)
        result = tracer.trace(inp, stop)

        # Speed should be ~1500 m/s
        assert 1400 < result.v[0] < 1600
        # Momentum preserved (ocean FD gradient has ~1e-4 drift over 5 km)
        assert np.max(np.abs(result.momentum_error)) < 1e-3
        # Ray should go deeper (h decreases)
        assert result.h_m[-1] < result.h_m[0]

    def test_from_ocean_refraction_profile(self):
        from refraction.ocean.ocean_refraction import (
            OceanRefractionProfile, OceanMode,
        )
        from refraction.ocean.ocean_profile import OceanConditions
        from refraction.ray_trace import SPEED_OF_LIGHT

        cond = OceanConditions(sst_C=25.0, sss_psu=35.0, wavelength_nm=550.0)
        profile = OceanRefractionProfile(cond)
        tracer = EikonalTracer.from_ocean_refraction_profile(
            profile, OceanMode.QuanFry
        )

        inp = EikonalInput(lat_deg=30.0, lon_deg=-80.0, h_m=-1.0,
                           elevation_deg=-10.0, azimuth_deg=90.0)
        stop = EikonalStopCondition(max_arc_length_m=1000.0,
                                     detect_ground_hit=False)
        result = tracer.trace(inp, stop)

        # n_real for water ~1.33, so v ~ c/1.33 ~ 2.25e8 m/s
        assert result.v[0] < SPEED_OF_LIGHT
        assert result.v[0] > 2e8
        # Momentum preserved
        assert np.max(np.abs(result.momentum_error)) < 1e-6


class TestSpeedFromEta:
    """Test the helper function."""

    def test_roundtrip(self):
        """v = c_ref/eta, so eta = c_ref/v."""
        c_ref = 299_792_458.0
        eta = lambda h: 1.000315 - 39e-9 * h
        deta = lambda h: -39e-9
        v_func, dv_func = speed_from_eta(eta, deta, c_ref)

        h = 5000.0
        assert v_func(h) == pytest.approx(c_ref / eta(h), rel=1e-12)
        # v' = -c_ref * eta' / eta^2
        expected_dv = -c_ref * deta(h) / eta(h)**2
        assert dv_func(h) == pytest.approx(expected_dv, rel=1e-12)


# ===================================================================
# Analytical closed-form ray path tests
# ===================================================================

class TestReciprocalLinearArc:
    """v(h) = 1 + b*h (reciprocal-linear eta) gives exact circular arcs.

    At short range (~10 km) the flat-earth analytical solution matches
    the 3D ECEF tracer closely.
    """

    @pytest.mark.parametrize("elevation_deg", [5.0, 15.0, 45.0])
    def test_circular_arc(self, elevation_deg):
        b = 1e-6  # v increases with height -> ray bends down
        # v(h) = 1 + b*h, dv/dh = b
        tracer = EikonalTracer(lambda h: 1.0 + b * h, lambda _h: b)

        theta0 = math.radians(elevation_deg)
        sigma0 = 1.0 / (1.0 + b * 0.0)  # = 1 at h=0
        kappa = sigma0 * math.cos(theta0)
        R = 1.0 / (b * kappa)  # note: b here is -sigma'/sigma^2...
        # For v = 1+bh: sigma = 1/(1+bh), sigma' = -b/(1+bh)^2 = -b*sigma^2
        # dsigma at h=0: -b. Curvature = -b*kappa = -b*cos(theta0).
        # So dtheta/ds = -b*kappa (constant), R = 1/(b*kappa).
        # theta(s) = theta0 - b*kappa*s

        # Limit trace to 5 km so flat-earth approx is tight
        s_max = 5000.0

        inp = EikonalInput(lat_deg=0.0, lon_deg=0.0, h_m=0.0,
                           elevation_deg=elevation_deg, azimuth_deg=0.0)
        stop = EikonalStopCondition(max_arc_length_m=s_max,
                                     detect_ground_hit=False)
        result = tracer.trace(inp, stop)

        # Analytical: h(s) = R*(cos(theta0 - b*kappa*s) - cos(theta0))
        for k in range(len(result.s)):
            s = result.s[k]
            phi = theta0 - b * kappa * s
            h_analytical = R * (math.cos(phi) - math.cos(theta0))
            assert result.h_m[k] == pytest.approx(h_analytical, abs=3.0)


class TestExponentialLogTrig:
    """v(h) = 1/(n0*exp(a*h)) (exponential eta) has log-trig closed form.

    Key property: theta is linear in ground range w.
    """

    @pytest.mark.parametrize("elevation_deg", [5.0, 15.0, 45.0])
    def test_log_trig(self, elevation_deg):
        n0 = 1.0003
        a = -1e-6  # eta decreases with height

        v_func, dv_func = speed_from_eta(
            lambda h: n0 * math.exp(a * h),
            lambda h: a * n0 * math.exp(a * h),
            1.0,
        )
        tracer = EikonalTracer(v_func, dv_func)

        theta0 = math.radians(elevation_deg)
        # Limit trace to 5 km so flat-earth approx is tight
        s_max = 5000.0

        inp = EikonalInput(lat_deg=0.0, lon_deg=0.0, h_m=0.0,
                           elevation_deg=elevation_deg, azimuth_deg=0.0)
        stop = EikonalStopCondition(max_arc_length_m=s_max,
                                     detect_ground_hit=False)
        result = tracer.trace(inp, stop)

        # Analytical: L(s) = (sec(theta0)+tan(theta0))*exp(a*s)
        # sin(theta(s)) = (L^2-1)/(L^2+1), cos(theta(s)) = 2L/(L^2+1)
        # h(s) = (1/a)*ln(cos(theta0)/cos(theta(s)))
        L0 = 1.0 / math.cos(theta0) + math.tan(theta0)
        for k in range(len(result.s)):
            s = result.s[k]
            L = L0 * math.exp(a * s)
            cos_th = 2.0 * L / (L * L + 1.0)
            h_analytical = (1.0 / a) * math.log(math.cos(theta0) / cos_th)
            assert result.h_m[k] == pytest.approx(h_analytical, abs=3.0)


class TestPiecewiseLayers:
    """10 piecewise-constant refraction layers with Snell's law.

    The exact solution: straight-line segments joined by Snell's law at
    each boundary.  The tracer uses a smooth sigmoid approximation.
    """

    def test_ten_layers(self):
        # 10 layers from 0 to 10 km, eta decreasing linearly
        n_layers = 10
        boundaries = [i * 1000.0 for i in range(n_layers + 1)]  # 0..10000 m
        eta_values = [1.000300 - i * 0.000025 for i in range(n_layers)]
        # eta goes from 1.000300 at bottom to 1.000075 at top

        # Smooth profile: sigmoid transitions (width = 50 m)
        eps = 50.0

        def smooth_eta(h):
            eta = eta_values[0]
            for i in range(n_layers - 1):
                delta_eta = eta_values[i + 1] - eta_values[i]
                z = (h - boundaries[i + 1]) / eps
                z = max(-20.0, min(20.0, z))  # clamp to avoid overflow
                eta += delta_eta / (1.0 + math.exp(-z))
            return eta

        def smooth_deta(h):
            deta = 0.0
            for i in range(n_layers - 1):
                delta_eta = eta_values[i + 1] - eta_values[i]
                z = (h - boundaries[i + 1]) / eps
                z = max(-20.0, min(20.0, z))
                sig = 1.0 / (1.0 + math.exp(-z))
                deta += delta_eta * sig * (1.0 - sig) / eps
            return deta

        v_func, dv_func = speed_from_eta(smooth_eta, smooth_deta, 1.0)
        tracer = EikonalTracer(v_func, dv_func)

        elevation_deg = 30.0
        theta0 = math.radians(elevation_deg)

        inp = EikonalInput(lat_deg=0.0, lon_deg=0.0, h_m=0.0,
                           elevation_deg=elevation_deg, azimuth_deg=0.0)
        stop = EikonalStopCondition(target_altitude_m=10500.0,
                                     max_arc_length_m=50000.0,
                                     detect_ground_hit=False)
        result = tracer.trace(inp, stop)

        # Compute exact piecewise solution using discrete Snell's law
        # kappa = eta_0 * cos(theta_0)
        kappa = eta_values[0] * math.cos(theta0)

        # For each layer, compute the angle and accumulate (h, w, s)
        h_exact = [0.0]
        w_exact = [0.0]
        s_exact = [0.0]
        theta_cur = theta0

        for i in range(n_layers):
            eta_i = eta_values[i]
            cos_theta = kappa / eta_i
            if abs(cos_theta) > 1.0:
                break  # turning point
            sin_theta = math.sqrt(1.0 - cos_theta ** 2)
            if theta_cur < 0:
                sin_theta = -sin_theta

            h_top = boundaries[i + 1]
            dh = h_top - h_exact[-1]
            if abs(sin_theta) < 1e-15:
                break
            ds = dh / sin_theta
            dw = ds * cos_theta

            h_exact.append(h_top)
            w_exact.append(w_exact[-1] + dw)
            s_exact.append(s_exact[-1] + ds)

            # Snell's law at boundary -> angle in next layer
            if i + 1 < n_layers:
                cos_next = kappa / eta_values[i + 1]
                if abs(cos_next) > 1.0:
                    break
                theta_cur = math.acos(cos_next)

        # Compare tracer h(s) at the exact boundary s values
        # (only at boundaries where the sigmoid has fully transitioned)
        for j in range(1, len(s_exact)):
            s_target = s_exact[j]
            h_target = h_exact[j]
            # Find closest tracer point
            k = np.argmin(np.abs(result.s - s_target))
            # ~10 m from sigmoid smoothing + 3D vs flat-earth curvature
            assert result.h_m[k] == pytest.approx(
                h_target, abs=15.0
            ), f"Layer {j}: h={result.h_m[k]:.1f} vs exact={h_target:.1f} at s={s_target:.1f}"

        # Verify the ray reaches above the last boundary
        assert result.h_m[-1] > 10000.0


# ===================================================================
# Derivative properties
# ===================================================================

class TestDerivatives:

    def test_r_prime_unit_length(self):
        tracer = _exponential_tracer()
        inp = _standard_input(30.0)
        stop = EikonalStopCondition(target_altitude_m=50_000.0,
                                     max_arc_length_m=500_000.0,
                                     detect_ground_hit=False)
        result = tracer.trace(inp, stop)
        norms = np.linalg.norm(result.r_prime, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_r_prime_equals_p_times_v(self):
        """r' = p * v = p / sigma."""
        tracer = _exponential_tracer()
        inp = _standard_input(30.0)
        stop = EikonalStopCondition(target_altitude_m=50_000.0,
                                     max_arc_length_m=500_000.0,
                                     detect_ground_hit=False)
        result = tracer.trace(inp, stop)
        for k in range(len(result.s)):
            sigma = 1.0 / result.v[k]
            expected = result.p[k] / sigma
            np.testing.assert_allclose(result.r_prime[k], expected, atol=1e-12)

    def test_dh_ds_equals_sin_theta(self):
        tracer = _exponential_tracer()
        inp = _standard_input(30.0)
        stop = EikonalStopCondition(target_altitude_m=50_000.0,
                                     max_arc_length_m=500_000.0,
                                     detect_ground_hit=False)
        result = tracer.trace(inp, stop)
        sin_theta = np.sin(np.radians(result.theta_deg))
        np.testing.assert_allclose(result.dh_ds, sin_theta, atol=1e-10)

    def test_curvature_zero_constant_speed(self):
        tracer = _constant_tracer()
        inp = _standard_input(30.0, 45.0)
        stop = EikonalStopCondition(max_arc_length_m=50_000.0,
                                     detect_ground_hit=False)
        result = tracer.trace(inp, stop)
        np.testing.assert_allclose(result.curvature, 0.0, atol=1e-13)
        np.testing.assert_allclose(result.r_double_prime, 0.0, atol=1e-13)

    def test_curvature_formula(self):
        """Curvature = |v'/v| * cos(theta)."""
        n0 = 1.000315
        a = -39e-9
        tracer = _exponential_tracer(n0, a)
        inp = _standard_input(30.0)
        stop = EikonalStopCondition(target_altitude_m=50_000.0,
                                     max_arc_length_m=500_000.0,
                                     detect_ground_hit=False)
        result = tracer.trace(inp, stop)

        # v(h) = 1/eta = 1/(n0*exp(a*h)), v'(h) = -a*v(h)
        cos_theta = np.cos(np.radians(result.theta_deg))
        # |v'/v| = |a| (for exponential eta with c_ref=1)
        expected = abs(a) * cos_theta
        np.testing.assert_allclose(result.curvature, expected, rtol=1e-6)

    def test_bending_zero_constant_speed(self):
        tracer = _constant_tracer()
        inp = _standard_input(30.0)
        stop = EikonalStopCondition(max_arc_length_m=50_000.0,
                                     detect_ground_hit=False)
        result = tracer.trace(inp, stop)
        assert result.bending_deg == pytest.approx(0.0, abs=1e-10)

    def test_p_prime_direction(self):
        """p' should be parallel to n_hat (geodetic normal)."""
        tracer = _exponential_tracer()
        inp = _standard_input(30.0)
        stop = EikonalStopCondition(target_altitude_m=50_000.0,
                                     max_arc_length_m=500_000.0,
                                     detect_ground_hit=False)
        result = tracer.trace(inp, stop)
        for k in range(0, len(result.s), max(1, len(result.s) // 20)):
            n_hat = geodetic_normal(result.lat_deg[k], result.lon_deg[k])
            p_pr = result.p_prime[k]
            p_pr_norm = np.linalg.norm(p_pr)
            if p_pr_norm > 1e-15:
                direction = p_pr / p_pr_norm
                assert abs(abs(np.dot(direction, n_hat)) - 1.0) < 1e-8

    def test_travel_time_increases(self):
        """T should be strictly increasing."""
        tracer = _exponential_tracer()
        inp = _standard_input(30.0)
        stop = EikonalStopCondition(target_altitude_m=50_000.0,
                                     max_arc_length_m=500_000.0,
                                     detect_ground_hit=False)
        result = tracer.trace(inp, stop)
        assert np.all(np.diff(result.T) > 0)


# ===================================================================
# Sensitivity derivatives
# ===================================================================

class TestSensitivityConstantSpeed:
    """For constant speed, sensitivities have analytical solutions."""

    def test_dr_dh0_equals_normal(self):
        tracer = _constant_tracer()
        inp = _standard_input(30.0, 45.0)
        stop = EikonalStopCondition(max_arc_length_m=50_000.0,
                                     detect_ground_hit=False)
        result, sens = tracer.trace(inp, stop, compute_sensitivities=True)
        n_hat = geodetic_normal(inp.lat_deg, inp.lon_deg)
        for k in range(len(result.s)):
            np.testing.assert_allclose(sens.dr_dh0[k], n_hat, atol=1e-8)

    def test_dr_dtheta0_linear_in_s(self):
        """dr/dtheta0 = s * dt0/dtheta0."""
        tracer = _constant_tracer()
        inp = _standard_input(30.0, 45.0)
        stop = EikonalStopCondition(max_arc_length_m=50_000.0,
                                     detect_ground_hit=False)
        result, sens = tracer.trace(inp, stop, compute_sensitivities=True)
        theta = math.radians(inp.elevation_deg)
        alpha = math.radians(inp.azimuth_deg)
        E, N, U = enu_frame(inp.lat_deg, inp.lon_deg)
        dt_dtheta = (-math.sin(theta) * math.sin(alpha) * E
                     - math.sin(theta) * math.cos(alpha) * N
                     + math.cos(theta) * U)
        for k in range(1, len(result.s)):
            expected = result.s[k] * dt_dtheta
            np.testing.assert_allclose(
                sens.dr_dtheta0[k], expected, rtol=1e-8, atol=1e-4
            )

    def test_dr_dalpha0_linear_in_s(self):
        tracer = _constant_tracer()
        inp = _standard_input(30.0, 45.0)
        stop = EikonalStopCondition(max_arc_length_m=50_000.0,
                                     detect_ground_hit=False)
        result, sens = tracer.trace(inp, stop, compute_sensitivities=True)
        theta = math.radians(inp.elevation_deg)
        alpha = math.radians(inp.azimuth_deg)
        E, N, U = enu_frame(inp.lat_deg, inp.lon_deg)
        dt_dalpha = (math.cos(theta) * math.cos(alpha) * E
                     - math.cos(theta) * math.sin(alpha) * N)
        for k in range(1, len(result.s)):
            expected = result.s[k] * dt_dalpha
            np.testing.assert_allclose(
                sens.dr_dalpha0[k], expected, rtol=1e-8, atol=1e-4
            )

    def test_dr_dphi0_constant_speed(self):
        tracer = _constant_tracer()
        inp = _standard_input(30.0, 45.0)
        stop = EikonalStopCondition(max_arc_length_m=50_000.0,
                                     detect_ground_hit=False)
        result, sens = tracer.trace(inp, stop, compute_sensitivities=True)
        theta = math.radians(inp.elevation_deg)
        alpha = math.radians(inp.azimuth_deg)
        E, N, U = enu_frame(inp.lat_deg, inp.lon_deg)
        M, _Nr = principal_radii(inp.lat_deg)
        dt_dphi = math.sin(theta) * N - math.cos(theta) * math.cos(alpha) * U
        for k in range(1, len(result.s)):
            expected = (M + inp.h_m) * N + result.s[k] * dt_dphi
            np.testing.assert_allclose(
                sens.dr_dphi0[k], expected, rtol=1e-7, atol=1e-2
            )

    def test_dr_dlambda0_constant_speed(self):
        tracer = _constant_tracer()
        inp = _standard_input(30.0, 45.0)
        stop = EikonalStopCondition(max_arc_length_m=50_000.0,
                                     detect_ground_hit=False)
        result, sens = tracer.trace(inp, stop, compute_sensitivities=True)
        theta = math.radians(inp.elevation_deg)
        alpha = math.radians(inp.azimuth_deg)
        phi = math.radians(inp.lat_deg)
        E, N, U = enu_frame(inp.lat_deg, inp.lon_deg)
        _M, Nr = principal_radii(inp.lat_deg)
        sin_phi, cos_phi = math.sin(phi), math.cos(phi)
        sin_th, cos_th = math.sin(theta), math.cos(theta)
        sin_al, cos_al = math.sin(alpha), math.cos(alpha)
        dt_dlambda = ((sin_th * cos_phi - cos_th * cos_al * sin_phi) * E
                      + cos_th * sin_al * sin_phi * N
                      - cos_th * sin_al * cos_phi * U)
        for k in range(1, len(result.s)):
            expected = (Nr + inp.h_m) * cos_phi * E + result.s[k] * dt_dlambda
            np.testing.assert_allclose(
                sens.dr_dlambda0[k], expected, rtol=1e-7, atol=1e-2
            )


def _perturb_h0(inp, d):
    return EikonalInput(inp.lat_deg, inp.lon_deg, inp.h_m + d,
                        inp.elevation_deg, inp.azimuth_deg)

def _perturb_theta0(inp, d):
    return EikonalInput(inp.lat_deg, inp.lon_deg, inp.h_m,
                        inp.elevation_deg + d, inp.azimuth_deg)

def _perturb_alpha0(inp, d):
    return EikonalInput(inp.lat_deg, inp.lon_deg, inp.h_m,
                        inp.elevation_deg, inp.azimuth_deg + d)

def _perturb_phi0(inp, d):
    return EikonalInput(inp.lat_deg + d, inp.lon_deg, inp.h_m,
                        inp.elevation_deg, inp.azimuth_deg)

def _perturb_lambda0(inp, d):
    return EikonalInput(inp.lat_deg, inp.lon_deg + d, inp.h_m,
                        inp.elevation_deg, inp.azimuth_deg)


class TestSensitivityFiniteDifference:
    """Validate variational equations against FD perturbation."""

    @pytest.mark.parametrize("param_name,perturb_fn,sens_field,delta,azimuth", [
        ("h0",      _perturb_h0,      "dr_dh0",      1.0,  0.0),
        ("theta0",  _perturb_theta0,  "dr_dtheta0",  1e-4, 0.0),
        ("alpha0",  _perturb_alpha0,  "dr_dalpha0",  1e-4, 45.0),
        ("phi0",    _perturb_phi0,    "dr_dphi0",    1e-4, 45.0),
        ("lambda0", _perturb_lambda0, "dr_dlambda0", 1e-4, 45.0),
    ])
    def test_sensitivity_fd(self, param_name, perturb_fn, sens_field,
                            delta, azimuth):
        tracer = _exponential_tracer()
        inp = _standard_input(30.0, azimuth)
        stop = EikonalStopCondition(target_altitude_m=50_000.0,
                                     max_arc_length_m=500_000.0,
                                     detect_ground_hit=False)
        result, sens = tracer.trace(inp, stop, compute_sensitivities=True)

        # For angular parameters, delta is in degrees; convert to radians
        if param_name == "h0":
            delta_physical = delta  # metres
        else:
            delta_physical = math.radians(delta)  # radians

        result_p = tracer.trace(perturb_fn(inp, delta), stop)
        result_m = tracer.trace(perturb_fn(inp, -delta), stop)

        # Compare at an early point where s-grids align well
        k_ref = max(1, len(result.s) // 5)
        s_test = result.s[k_ref]
        k_p = np.argmin(np.abs(result_p.s - s_test))
        k_m = np.argmin(np.abs(result_m.s - s_test))

        dr_fd = (result_p.r[k_p] - result_m.r[k_m]) / (2.0 * delta_physical)
        dr_var = getattr(sens, sens_field)[k_ref]
        np.testing.assert_allclose(dr_var, dr_fd, rtol=0.05, atol=10.0)
