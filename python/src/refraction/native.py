"""Native (C++) eikonal ray tracer with batch support.

This module provides a C++-backed ray tracer via nanobind, with the
same API as the pure Python :class:`EikonalTracer`.

Usage::

    from refraction import NativeTracer, BatchInput, TableSpeedProfile
    from refraction import EikonalInput, EikonalStopCondition
    import numpy as np

    # Option 1: Python callables (flexible, slower)
    tracer = NativeTracer(v_func, dv_dh_func)

    # Option 2: Table interpolation (pure C++, fastest)
    h = np.linspace(0, 120_000, 1000)
    v = 299_792_458.0 / (1.000315 * np.exp(-39e-9 * h))
    table = TableSpeedProfile(h, v)
    tracer = NativeTracer.from_table(table)

    # Single trace (same interface as EikonalTracer)
    inp = EikonalInput(lat_deg=45.0, lon_deg=0.0, h_m=0.0,
                       elevation_deg=30.0, azimuth_deg=0.0)
    stop = EikonalStopCondition(target_travel_time_s=0.001)
    result = tracer.trace(inp, stop)  # → EikonalResult

    # Batch trace: N rays in a single C++ call
    batch_inp = BatchInput(
        lat_deg=np.full(100, 45.0),
        lon_deg=np.zeros(100),
        h_m=np.zeros(100),
        elevation_deg=np.linspace(1, 89, 100),
        azimuth_deg=np.zeros(100),
        travel_time_s=np.full(100, 0.001),
    )
    result = tracer.trace_batch(batch_inp)       # → BatchResult (endpoint)
    results = tracer.trace_batch(batch_inp, endpoint=False)  # → list[EikonalResult]
"""

from __future__ import annotations

import numpy as np

from .ray_trace import (
    EikonalInput,
    EikonalStopCondition,
    EikonalResult,
    BatchInput,
    BatchResult,
)
try:
    from .refraction_native import (
        EikonalTracer as _NativeTracer,
        TableSpeedProfile as _TableSpeedProfile,
        geodetic_to_ecef,
        ecef_to_geodetic,
        speed_from_eta,
    )
    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False

_NATIVE_MISSING_MSG = (
    "Native C++ extension not available. "
    "Install from a pre-built wheel or build with cmake. "
    "Use EikonalTracer for the pure Python alternative."
)

if HAS_NATIVE:
    class TableSpeedProfile(_TableSpeedProfile):
        """Precomputed speed-vs-altitude table with linear interpolation in C++.

        Parameters
        ----------
        h_array : numpy.ndarray
            Geodetic heights (m), must be sorted ascending.
            Negative values for underwater.
        v_array : numpy.ndarray
            Propagation speed at each height (m/s).
        """

        @staticmethod
        def from_depth(depths, speeds):
            """Create table from depth (positive downward) and speed arrays.

            Parameters
            ----------
            depths : array_like
                Depth values (m, positive downward).
            speeds : array_like
                Propagation speed at each depth (m/s).

            Returns
            -------
            TableSpeedProfile

            Examples
            --------
            >>> depths = [0, 50, 100, 500, 1000]
            >>> speeds = [1520, 1515, 1490, 1500, 1510]
            >>> table = TableSpeedProfile.from_depth(depths, speeds)
            """
            depths = np.asarray(depths, dtype=np.float64)
            speeds = np.asarray(speeds, dtype=np.float64)
            order = np.argsort(depths)
            h = -depths[order][::-1].copy()   # ascending h (negative = underwater)
            v = speeds[order][::-1].copy()
            return TableSpeedProfile(h, v)
else:
    class TableSpeedProfile:  # type: ignore[no-redef]
        """Stub — native C++ extension not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError(_NATIVE_MISSING_MSG)
        @staticmethod
        def from_depth(*args, **kwargs):
            raise ImportError(_NATIVE_MISSING_MSG)


def _dict_to_eikonal_result(raw: dict) -> EikonalResult:
    """Convert a C++ result dict to an EikonalResult dataclass."""
    n = len(raw['s'])
    return EikonalResult(
        s=np.array(raw['s']),
        r=np.array(raw['r']).reshape(n, 3),
        p=np.array(raw['p']).reshape(n, 3),
        T=np.array(raw['T']),
        r_prime=np.array(raw['r_prime']).reshape(n, 3),
        p_prime=np.array(raw['p_prime']).reshape(n, 3),
        dh_ds=np.array(raw['dh_ds']),
        dv_ds=np.array(raw['dv_ds']),
        r_double_prime=np.array(raw['r_double_prime']).reshape(n, 3),
        curvature=np.array(raw['curvature']),
        lat_deg=np.array(raw['lat_deg']),
        lon_deg=np.array(raw['lon_deg']),
        h_m=np.array(raw['h_m']),
        v=np.array(raw['v']),
        theta_deg=np.array(raw['theta_deg']),
        azimuth_deg=np.array(raw['azimuth_deg']),
        bending_deg=raw['bending_deg'],
        dr_dT=np.array(raw['dr_dT']).reshape(n, 3),
        d2r_dT2=np.array(raw['d2r_dT2']).reshape(n, 3),
        dp_dT=np.array(raw['dp_dT']).reshape(n, 3),
        momentum_error=np.array(raw['momentum_error']),
        terminated_by=raw['terminated_by'],
    )


def _dict_to_batch_result(raw: dict) -> BatchResult:
    """Convert a C++ batch result dict to a BatchResult dataclass."""
    n = raw['n']
    return BatchResult(
        s=np.array(raw['s']),
        r=np.array(raw['r']).reshape(n, 3),
        p=np.array(raw['p']).reshape(n, 3),
        T=np.array(raw['T']),
        r_prime=np.array(raw['r_prime']).reshape(n, 3),
        p_prime=np.array(raw['p_prime']).reshape(n, 3),
        dh_ds=np.array(raw['dh_ds']),
        dv_ds=np.array(raw['dv_ds']),
        r_double_prime=np.array(raw['r_double_prime']).reshape(n, 3),
        curvature=np.array(raw['curvature']),
        lat_deg=np.array(raw['lat_deg']),
        lon_deg=np.array(raw['lon_deg']),
        h_m=np.array(raw['h_m']),
        v=np.array(raw['v']),
        theta_deg=np.array(raw['theta_deg']),
        azimuth_deg=np.array(raw['azimuth_deg']),
        bending_deg=np.array(raw['bending_deg']),
        dr_dT=np.array(raw['dr_dT']).reshape(n, 3),
        d2r_dT2=np.array(raw['d2r_dT2']).reshape(n, 3),
        dp_dT=np.array(raw['dp_dT']).reshape(n, 3),
        momentum_error=np.array(raw['momentum_error']),
    )


# ---------------------------------------------------------------------------
# NativeTracer wrapper
# ---------------------------------------------------------------------------

class NativeTracer:
    """C++-backed eikonal ray tracer with batch support.

    Same interface as :class:`~refraction.ray_trace.EikonalTracer`.

    Parameters
    ----------
    v_func : callable
        Propagation speed as a function of geodetic height (m) → m/s.
    dv_dh_func : callable
        Derivative of speed w.r.t. height → 1/s.
    """

    def __init__(self, v_func, dv_dh_func):
        if not HAS_NATIVE:
            raise ImportError(_NATIVE_MISSING_MSG)
        self._native = _NativeTracer(v_func, dv_dh_func)
        self._v_func = v_func
        self._dv_dh_func = dv_dh_func

    @classmethod
    def from_table(cls, table: TableSpeedProfile) -> NativeTracer:
        """Create from a :class:`TableSpeedProfile` (pure C++, no callbacks).

        Parameters
        ----------
        table : TableSpeedProfile
            Precomputed (altitude, speed) table.
        """
        obj = cls.__new__(cls)
        obj._native = _NativeTracer.from_table(table)
        obj._v_func = table.speed
        obj._dv_dh_func = table.dspeed
        return obj

    def trace(
        self,
        inp: EikonalInput,
        stop: EikonalStopCondition,
        *,
        rtol: float = 1e-10,
        atol: float = 1e-12,
        max_step: float = 100.0,
    ) -> EikonalResult:
        """Trace a single ray.  Returns an :class:`EikonalResult`."""
        raw = self._native.trace(
            inp.lat_deg, inp.lon_deg, inp.h_m,
            inp.elevation_deg, inp.azimuth_deg,
            stop.target_altitude_m, stop.target_travel_time_s,
            stop.max_arc_length_m, stop.detect_ground_hit,
            rtol, atol, max_step,
        )
        return _dict_to_eikonal_result(raw)

    def trace_batch(
        self,
        inputs: BatchInput,
        *,
        endpoint: bool = True,
        rtol: float = 1e-10,
        atol: float = 1e-12,
        max_step: float = 100.0,
    ) -> BatchResult | list[EikonalResult]:
        """Trace N rays in a single C++ call.

        Parameters
        ----------
        inputs : BatchInput
            Per-ray initial conditions (all arrays length N).
        endpoint : bool
            If True (default), return a :class:`BatchResult` with only
            the endpoint state of each ray.  If False, return a list
            of :class:`EikonalResult` (one per ray, full path).
        rtol, atol : float
            ODE solver tolerances.
        max_step : float
            Maximum step size (m).

        Returns
        -------
        BatchResult or list[EikonalResult]
        """
        arr = np.column_stack([
            np.asarray(inputs.lat_deg, dtype=np.float64),
            np.asarray(inputs.lon_deg, dtype=np.float64),
            np.asarray(inputs.h_m, dtype=np.float64),
            np.asarray(inputs.elevation_deg, dtype=np.float64),
            np.asarray(inputs.azimuth_deg, dtype=np.float64),
            np.asarray(inputs.travel_time_s, dtype=np.float64),
        ])

        if not endpoint:
            raw_list = self._native.trace_batch_full(arr, rtol, atol, max_step)
            return [_dict_to_eikonal_result(d) for d in raw_list]

        raw = self._native.trace_batch(arr, rtol, atol, max_step)
        return _dict_to_batch_result(raw)


__all__ = [
    "NativeTracer",
    "TableSpeedProfile",
    "geodetic_to_ecef",
    "ecef_to_geodetic",
    "speed_from_eta",
]
