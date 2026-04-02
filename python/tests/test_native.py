"""Tests for the nanobind native ray tracer.

Cross-validates C++ results against pure Python, tests batch tracing,
table interpolation, and travel-time derivatives.
"""

import math

import numpy as np
import pytest

try:
    from refraction.native import (
        NativeTracer,
        TableSpeedProfile,
        geodetic_to_ecef,
        ecef_to_geodetic,
    )
    from refraction import (
        EikonalInput,
        EikonalStopCondition,
        EikonalResult,
        BatchInput,
        BatchResult,
    )
    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False

pytestmark = pytest.mark.skipif(not HAS_NATIVE, reason="native module not built")


# ===================================================================
# Geodetic utilities
# ===================================================================

class TestNativeGeodetic:

    def test_to_ecef_equator(self):
        X, Y, Z = geodetic_to_ecef(0.0, 0.0, 0.0)
        assert X == pytest.approx(6_378_137.0, abs=1)
        assert Y == pytest.approx(0.0, abs=1)
        assert Z == pytest.approx(0.0, abs=1)

    def test_roundtrip(self):
        X, Y, Z = geodetic_to_ecef(45.0, 30.0, 5000.0)
        lat, lon, h = ecef_to_geodetic(X, Y, Z)
        assert lat == pytest.approx(45.0, abs=1e-9)
        assert lon == pytest.approx(30.0, abs=1e-9)
        assert h == pytest.approx(5000.0, abs=1e-3)


# ===================================================================
# Single trace
# ===================================================================

class TestNativeSingleTrace:

    def test_constant_speed_momentum(self):
        v0 = 1.0 / 1.000315
        tracer = NativeTracer(lambda h: v0, lambda h: 0.0)
        inp = EikonalInput(lat_deg=45.0, lon_deg=0.0, h_m=0.0,
                           elevation_deg=30.0, azimuth_deg=0.0)
        stop = EikonalStopCondition(max_arc_length_m=50_000.0,
                                    detect_ground_hit=False)
        result = tracer.trace(inp, stop)
        assert isinstance(result, EikonalResult)
        assert np.max(np.abs(result.momentum_error)) < 1e-12

    def test_exponential_profile_momentum(self):
        from refraction.ray_trace import speed_from_eta
        n0, a = 1.000315, -39e-9
        v_func, dv_func = speed_from_eta(
            lambda h: n0 * math.exp(a * h),
            lambda h: a * n0 * math.exp(a * h),
            1.0,
        )
        tracer = NativeTracer(v_func, dv_func)
        inp = EikonalInput(lat_deg=45.0, lon_deg=0.0, h_m=0.0,
                           elevation_deg=30.0, azimuth_deg=0.0)
        stop = EikonalStopCondition(target_altitude_m=50_000.0,
                                    max_arc_length_m=500_000.0,
                                    detect_ground_hit=False)
        result = tracer.trace(inp, stop)
        assert isinstance(result, EikonalResult)
        assert np.max(np.abs(result.momentum_error)) < 1e-7

    def test_trace_returns_all_fields(self):
        """trace() should return EikonalResult with all fields."""
        v0 = 1.0 / 1.000315
        tracer = NativeTracer(lambda h: v0, lambda h: 0.0)
        inp = EikonalInput(lat_deg=45.0, lon_deg=0.0, h_m=0.0,
                           elevation_deg=30.0, azimuth_deg=0.0)
        stop = EikonalStopCondition(max_arc_length_m=10_000.0,
                                    detect_ground_hit=False)
        r = tracer.trace(inp, stop)
        n = len(r.s)
        assert r.r.shape == (n, 3)
        assert r.r_prime.shape == (n, 3)
        assert r.p_prime.shape == (n, 3)
        assert r.r_double_prime.shape == (n, 3)
        assert r.dr_dT.shape == (n, 3)
        assert r.d2r_dT2.shape == (n, 3)
        assert r.dp_dT.shape == (n, 3)
        assert r.curvature.shape == (n,)
        assert isinstance(r.bending_deg, float)
        assert isinstance(r.terminated_by, str)


# ===================================================================
# Table speed profile
# ===================================================================

class TestTableSpeedProfile:

    def test_constant_table(self):
        h = np.linspace(0, 200_000, 100)
        v = np.full_like(h, 1.0 / 1.000315)
        table = TableSpeedProfile(h, v)
        tracer = NativeTracer.from_table(table)
        inp = EikonalInput(lat_deg=45.0, lon_deg=0.0, h_m=0.0,
                           elevation_deg=30.0, azimuth_deg=0.0)
        stop = EikonalStopCondition(max_arc_length_m=50_000.0,
                                    detect_ground_hit=False)
        result = tracer.trace(inp, stop)
        assert np.max(np.abs(result.momentum_error)) < 1e-10

    def test_from_depth(self):
        """TableSpeedProfile.from_depth: depth-positive-down convention."""
        depths = np.array([0, 50, 100, 500, 1000, 2000], dtype=float)
        speeds = np.array([1520, 1515, 1490, 1500, 1510, 1520], dtype=float)
        table = TableSpeedProfile.from_depth(depths, speeds)
        # Speed at surface (h=0, depth=0)
        assert table.speed(0.0) == pytest.approx(1520.0, rel=1e-6)
        # Speed at 500m depth (h=-500)
        assert table.speed(-500.0) == pytest.approx(1500.0, rel=1e-3)

    def test_exponential_table(self):
        h = np.linspace(0, 200_000, 2000)
        n0, a = 1.000315, -39e-9
        v = 1.0 / (n0 * np.exp(a * h))
        table = TableSpeedProfile(h, v)
        tracer = NativeTracer.from_table(table)
        inp = EikonalInput(lat_deg=45.0, lon_deg=0.0, h_m=0.0,
                           elevation_deg=30.0, azimuth_deg=0.0)
        stop = EikonalStopCondition(target_altitude_m=50_000.0,
                                    max_arc_length_m=500_000.0,
                                    detect_ground_hit=False)
        result = tracer.trace(inp, stop)
        assert np.max(np.abs(result.momentum_error)) < 1e-6


# ===================================================================
# Batch tracing
# ===================================================================

class TestBatchTrace:

    def _make_tracer(self):
        v0 = 1.0 / 1.000315
        return NativeTracer(lambda h: v0, lambda h: 0.0)

    def test_batch_result_shapes(self):
        tracer = self._make_tracer()
        n = 10
        inp = BatchInput(
            lat_deg=np.full(n, 45.0),
            lon_deg=np.zeros(n),
            h_m=np.zeros(n),
            elevation_deg=np.linspace(10, 80, n),
            azimuth_deg=np.zeros(n),
            travel_time_s=np.full(n, 0.0001),
        )
        result = tracer.trace_batch(inp)

        assert isinstance(result, BatchResult)
        assert result.r.shape == (n, 3)
        assert result.p.shape == (n, 3)
        assert result.s.shape == (n,)
        assert result.T.shape == (n,)
        assert result.v.shape == (n,)
        assert result.r_prime.shape == (n, 3)
        assert result.p_prime.shape == (n, 3)
        assert result.r_double_prime.shape == (n, 3)
        assert result.dr_dT.shape == (n, 3)
        assert result.d2r_dT2.shape == (n, 3)
        assert result.dp_dT.shape == (n, 3)
        assert result.curvature.shape == (n,)
        assert result.dh_ds.shape == (n,)
        assert result.dv_ds.shape == (n,)
        assert result.lat_deg.shape == (n,)
        assert result.lon_deg.shape == (n,)
        assert result.h_m.shape == (n,)
        assert result.momentum_error.shape == (n,)

    def test_batch_single_matches_trace(self):
        tracer = self._make_tracer()
        inp = BatchInput(
            lat_deg=np.array([45.0]),
            lon_deg=np.array([0.0]),
            h_m=np.array([0.0]),
            elevation_deg=np.array([30.0]),
            azimuth_deg=np.array([0.0]),
            travel_time_s=np.array([0.0001]),
        )
        result = tracer.trace_batch(inp)

        single = tracer.trace(
            EikonalInput(45.0, 0.0, 0.0, 30.0, 0.0),
            EikonalStopCondition(target_travel_time_s=0.0001,
                                 detect_ground_hit=False),
        )
        np.testing.assert_allclose(result.r[0], single.r[-1], rtol=1e-8)

    def test_batch_ecef_positions_near_earth(self):
        tracer = self._make_tracer()
        n = 20
        inp = BatchInput(
            lat_deg=np.full(n, 45.0),
            lon_deg=np.zeros(n),
            h_m=np.zeros(n),
            elevation_deg=np.linspace(5, 85, n),
            azimuth_deg=np.zeros(n),
            travel_time_s=np.full(n, 0.0001),
        )
        result = tracer.trace_batch(inp)
        distances = np.linalg.norm(result.r, axis=1)
        assert np.all(distances > 6e6)
        assert np.all(distances < 7e6)

    def test_batch_with_table(self):
        h = np.linspace(0, 200_000, 2000)
        n0, a = 1.000315, -39e-9
        v = 1.0 / (n0 * np.exp(a * h))
        table = TableSpeedProfile(h, v)
        tracer = NativeTracer.from_table(table)

        inp = BatchInput(
            lat_deg=np.full(5, 45.0),
            lon_deg=np.zeros(5),
            h_m=np.zeros(5),
            elevation_deg=np.array([10, 20, 30, 45, 60], dtype=float),
            azimuth_deg=np.zeros(5),
            travel_time_s=np.full(5, 0.0001),
        )
        result = tracer.trace_batch(inp)
        assert result.r.shape == (5, 3)
        assert np.all(result.s > 0)

    def test_batch_full_returns_list(self):
        """endpoint=False returns list[EikonalResult]."""
        tracer = self._make_tracer()
        inp = BatchInput(
            lat_deg=np.full(3, 45.0),
            lon_deg=np.zeros(3),
            h_m=np.zeros(3),
            elevation_deg=np.array([10.0, 45.0, 80.0]),
            azimuth_deg=np.zeros(3),
            travel_time_s=np.full(3, 0.0001),
        )
        results = tracer.trace_batch(inp, endpoint=False)
        assert isinstance(results, list)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, EikonalResult)
            assert r.r.shape[1] == 3
            assert r.dr_dT.shape[1] == 3


# ===================================================================
# Travel-time derivatives
# ===================================================================

class TestBatchDerivatives:

    def test_dr_dT_magnitude_equals_speed(self):
        """dr/dT = v * t_hat, so |dr/dT| = v."""
        v0 = 1.0 / 1.000315
        tracer = NativeTracer(lambda h: v0, lambda h: 0.0)
        inp = BatchInput(
            lat_deg=np.full(5, 45.0),
            lon_deg=np.zeros(5),
            h_m=np.zeros(5),
            elevation_deg=np.array([10, 30, 45, 60, 80], dtype=float),
            azimuth_deg=np.zeros(5),
            travel_time_s=np.full(5, 0.0001),
        )
        result = tracer.trace_batch(inp)
        dr_dT_mag = np.linalg.norm(result.dr_dT, axis=1)
        np.testing.assert_allclose(dr_dT_mag, result.v, rtol=1e-8)

    def test_d2r_dT2_zero_constant_speed(self):
        """Acceleration = 0 when v is constant."""
        v0 = 1.0 / 1.000315
        tracer = NativeTracer(lambda h: v0, lambda h: 0.0)
        inp = BatchInput(
            lat_deg=np.full(3, 45.0),
            lon_deg=np.zeros(3),
            h_m=np.zeros(3),
            elevation_deg=np.array([10, 45, 80], dtype=float),
            azimuth_deg=np.zeros(3),
            travel_time_s=np.full(3, 0.0001),
        )
        result = tracer.trace_batch(inp)
        np.testing.assert_allclose(result.d2r_dT2, 0.0, atol=1e-10)

    def test_dp_dT_zero_constant_speed(self):
        """dp/dT = 0 when v is constant (v' = 0)."""
        v0 = 1.0 / 1.000315
        tracer = NativeTracer(lambda h: v0, lambda h: 0.0)
        inp = BatchInput(
            lat_deg=np.full(3, 45.0),
            lon_deg=np.zeros(3),
            h_m=np.zeros(3),
            elevation_deg=np.array([10, 45, 80], dtype=float),
            azimuth_deg=np.zeros(3),
            travel_time_s=np.full(3, 0.0001),
        )
        result = tracer.trace_batch(inp)
        np.testing.assert_allclose(result.dp_dT, 0.0, atol=1e-10)
