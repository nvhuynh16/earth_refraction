"""3D ECEF eikonal ray tracer.

Traces rays through a refractive medium using the eikonal equation
in Earth-Centered Earth-Fixed (ECEF) coordinates, parameterised by
the local propagation speed v(h):

    dr/ds = p / sigma       (position follows momentum)
    dp/ds = sigma' * n_hat  (momentum changes by slowness gradient)
    dT/ds = sigma           (travel time accumulates)

where r = (X, Y, Z) is the ECEF position, p = (1/v) * dr/ds is the
slowness momentum (|p| = 1/v), T is the accumulated travel time, and
v(h) is the propagation speed at geodetic height h.

For electromagnetic (radio) applications, use :func:`speed_from_eta` to
convert a refractive-index profile eta(h) and reference speed c into the
(v_func, dv_dh_func) pair expected by the constructor.

References
----------
[1] Huynh, N. "Deriving Atmospheric and Acoustic Refraction from
    Fermat's Principle." (internal report).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .constants import SPEED_OF_LIGHT
from .geodetic import (
    ecef_to_geodetic,
    enu_frame,
    geodetic_normal,
    geodetic_to_ecef,
    principal_radii,
)


# ---------------------------------------------------------------------------
# Helper: convert refractive index to speed
# ---------------------------------------------------------------------------


def speed_from_eta(eta_func, deta_dh_func, c_ref):
    """Convert refractive-index functions to propagation-speed functions.

    Parameters
    ----------
    eta_func : callable
        Refractive index as a function of geodetic height (m).
    deta_dh_func : callable
        Derivative of eta w.r.t. geodetic height (per metre).
    c_ref : float
        Reference speed (m/s).  Speed of light for electromagnetic rays.

    Returns
    -------
    tuple[callable, callable]
        ``(v_func, dv_dh_func)`` where ``v = c_ref / eta`` and
        ``v' = -c_ref * eta' / eta**2``.
    """
    def v_func(h):
        return c_ref / eta_func(h)

    def dv_dh_func(h):
        eta = eta_func(h)
        return -c_ref * deta_dh_func(h) / (eta * eta)

    return v_func, dv_dh_func


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EikonalInput:
    """Initial conditions for the 3D eikonal tracer.

    Parameters
    ----------
    lat_deg : float
        Geodetic latitude (deg).
    lon_deg : float
        Geodetic longitude (deg).
    h_m : float
        Height above the WGS-84 ellipsoid (m).
    elevation_deg : float
        Elevation angle from local horizontal (deg). Positive = up.
    azimuth_deg : float
        Azimuth from North, clockwise (deg). 0 = North, 90 = East.
    """

    lat_deg: float
    lon_deg: float
    h_m: float
    elevation_deg: float
    azimuth_deg: float


@dataclass
class EikonalStopCondition:
    """Stopping criteria for the 3D eikonal tracer.

    Parameters
    ----------
    target_altitude_m : float or None
        Stop when geodetic height reaches this value (m).
    target_travel_time_s : float or None
        Stop when accumulated travel time T = integral of ds/v
        reaches this value (s).  This is what a radio or sonar
        measures as round_trip_time / 2.
    max_arc_length_m : float
        Maximum arc length (m). Default 500 km.
    detect_ground_hit : bool
        Stop when geodetic height drops to zero. Default True.
    """

    target_altitude_m: float | None = None
    target_travel_time_s: float | None = None
    max_arc_length_m: float = 500_000.0
    detect_ground_hit: bool = True


@dataclass
class EikonalResult:
    """Result of a 3D eikonal ray trace.

    Organised to match the document structure: core state from the ODE,
    first and second derivatives computed from the state, and application
    quantities computed last.

    Parameters
    ----------
    s : numpy.ndarray
        Arc length (m), shape (N,).

    r : numpy.ndarray
        ECEF position (m), shape (N, 3).  [ODE state]
    p : numpy.ndarray
        Slowness momentum (= (1/v) * t_hat), shape (N, 3).  [ODE state]
    T : numpy.ndarray
        Accumulated travel time = integral of ds/v (s), shape (N,).
        [ODE state]

    r_prime : numpy.ndarray
        dr/ds = p*v = t_hat, shape (N, 3).  [first derivative]
    p_prime : numpy.ndarray
        dp/ds = -(v'/v^2) * n_hat, shape (N, 3).  [first derivative]
    dh_ds : numpy.ndarray
        Altitude rate = n_hat . t_hat = sin(theta), shape (N,).
    dv_ds : numpy.ndarray
        Speed rate = v'(h) * sin(theta), shape (N,).

    r_double_prime : numpy.ndarray
        Curvature vector = -(v'/v) * n_hat_perp, shape (N, 3).
        [second derivative]
    curvature : numpy.ndarray
        Curvature magnitude = |v'/v| * cos(theta) = 1/R, shape (N,).

    lat_deg : numpy.ndarray
        Geodetic latitude (deg), shape (N,).
    lon_deg : numpy.ndarray
        Geodetic longitude (deg), shape (N,).
    h_m : numpy.ndarray
        Geodetic height (m), shape (N,).
    v : numpy.ndarray
        Propagation speed (m/s), shape (N,).

    theta_deg : numpy.ndarray
        Local elevation angle (deg), shape (N,).  [application]
    azimuth_deg : numpy.ndarray
        Local azimuth from North (deg), shape (N,).  [application]
    bending_deg : float
        Elevation-only refraction (deg), measured in the starting ENU frame.
        Negative means the ray exits at a shallower angle than it entered
        (standard atmospheric refraction bends rays downward).
        Only elevation changes because the optical force acts only in the
        geodetic normal direction --- it has no East or North component,
        so it cannot deflect the azimuth.

    dr_dT : numpy.ndarray
        First derivative of r w.r.t. travel time = v * t_hat (m/s),
        shape (N, 3).
    d2r_dT2 : numpy.ndarray
        Second derivative of r w.r.t. travel time (m/s^2), shape (N, 3).
        = v * v' * (2*sin(theta)*t_hat - n_hat).
    dp_dT : numpy.ndarray
        First derivative of p w.r.t. travel time = -(v'/v) * n_hat (1/s),
        shape (N, 3).

    momentum_error : numpy.ndarray
        ``|p| - 1/v`` diagnostic (should be ~0), shape (N,).
    terminated_by : str
        Reason for termination.
    """

    # Independent variable
    s: np.ndarray

    # Core state (ODE)
    r: np.ndarray
    p: np.ndarray
    T: np.ndarray

    # First derivatives
    r_prime: np.ndarray
    p_prime: np.ndarray
    dh_ds: np.ndarray
    dv_ds: np.ndarray

    # Second derivatives
    r_double_prime: np.ndarray
    curvature: np.ndarray

    # Geodetic position
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    h_m: np.ndarray
    v: np.ndarray

    # Application
    theta_deg: np.ndarray
    azimuth_deg: np.ndarray
    bending_deg: float

    # Travel-time derivatives
    dr_dT: np.ndarray
    d2r_dT2: np.ndarray
    dp_dT: np.ndarray

    # Diagnostics
    momentum_error: np.ndarray
    terminated_by: str


@dataclass
class SensitivityResult:
    """Sensitivity derivatives along the ray path.

    Each field is shape (N, 3).  Five initial-condition parameters:
    h_0 (height), theta_0 (elevation), alpha_0 (azimuth),
    phi_0 (latitude), lambda_0 (longitude).
    """

    dr_dh0: np.ndarray
    dp_dh0: np.ndarray
    dr_dtheta0: np.ndarray
    dp_dtheta0: np.ndarray
    dr_dalpha0: np.ndarray
    dp_dalpha0: np.ndarray
    dr_dphi0: np.ndarray
    dp_dphi0: np.ndarray
    dr_dlambda0: np.ndarray
    dp_dlambda0: np.ndarray


# ---------------------------------------------------------------------------
# Batch I/O dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BatchInput:
    """Input for batch ray tracing.  All arrays must have the same length N.

    Parameters
    ----------
    lat_deg : numpy.ndarray
        Geodetic latitude (deg), shape (N,).
    lon_deg : numpy.ndarray
        Geodetic longitude (deg), shape (N,).
    h_m : numpy.ndarray
        Geodetic height (m), shape (N,).
    elevation_deg : numpy.ndarray
        Elevation angle from horizontal (deg), shape (N,).
    azimuth_deg : numpy.ndarray
        Azimuth from North, clockwise (deg), shape (N,).
    travel_time_s : numpy.ndarray
        One-way travel time to stop at (s), shape (N,).
    """

    lat_deg: np.ndarray
    lon_deg: np.ndarray
    h_m: np.ndarray
    elevation_deg: np.ndarray
    azimuth_deg: np.ndarray
    travel_time_s: np.ndarray


@dataclass
class BatchResult:
    """Result of batch ray tracing (endpoint only).

    All arrays have shape (N,) or (N, 3) and correspond to the
    endpoint state of each ray at its target travel time.
    """

    # Core state at endpoint
    s: np.ndarray
    r: np.ndarray
    p: np.ndarray
    T: np.ndarray
    # First derivatives at endpoint
    r_prime: np.ndarray
    p_prime: np.ndarray
    dh_ds: np.ndarray
    dv_ds: np.ndarray
    # Second derivatives at endpoint
    r_double_prime: np.ndarray
    curvature: np.ndarray
    # Geodetic at endpoint
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    h_m: np.ndarray
    v: np.ndarray
    # Application at endpoint
    theta_deg: np.ndarray
    azimuth_deg: np.ndarray
    bending_deg: np.ndarray
    # Travel-time derivatives at endpoint
    dr_dT: np.ndarray
    d2r_dT2: np.ndarray
    dp_dT: np.ndarray
    # Diagnostics
    momentum_error: np.ndarray


# ---------------------------------------------------------------------------
# Eikonal tracer
# ---------------------------------------------------------------------------

_FD_STEP_M = 10.0  # step for FD computation of sigma''(h)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class EikonalTracer:
    """3D ECEF eikonal ray tracer.

    Parameters
    ----------
    v_func : callable
        Propagation speed as a function of geodetic height (m):
        ``v_func(h_m) -> float`` in m/s.
    dv_dh_func : callable
        Derivative of speed w.r.t. geodetic height:
        ``dv_dh_func(h_m) -> float`` in (m/s)/m = 1/s.

    Notes
    -----
    Internally the ODE uses the slowness sigma = 1/v and its derivative
    sigma' = -v'/v^2.  The second derivative sigma'' needed for sensitivity
    equations is computed via central finite difference with step 10 m.

    For electromagnetic applications where the refractive index eta and
    reference speed c are given, use :func:`speed_from_eta` to convert::

        v_func, dv_dh_func = speed_from_eta(eta_func, deta_dh_func, c)
        tracer = EikonalTracer(v_func, dv_dh_func)
    """

    def __init__(self, v_func, dv_dh_func):
        self._v = v_func
        self._dv_dh = dv_dh_func

    # -- internal: slowness from speed --------------------------------------

    def _sigma(self, h: float) -> float:
        """Slowness sigma = 1/v."""
        return 1.0 / self._v(h)

    def _dsigma_dh(self, h: float) -> float:
        """sigma' = -v'/v^2."""
        v = self._v(h)
        return -self._dv_dh(h) / (v * v)

    def _d2sigma_dh2(self, h: float) -> float:
        """sigma'' via central FD on sigma'."""
        d = _FD_STEP_M
        return (self._dsigma_dh(h + d) - self._dsigma_dh(h - d)) / (2.0 * d)

    # -- ODE ----------------------------------------------------------------

    def _compute_base(self, r, p):
        """Shared base computation for both RHS variants.

        Returns (dy_base, sigma, dsigma, n_hat, E, N, grad_sigma, lat, lon, h).
        Uses enu_frame() so the ENU frame is available to the
        sensitivity RHS without a second trig pass.
        """
        lat, lon, h = ecef_to_geodetic(r[0], r[1], r[2])
        sigma = self._sigma(h)
        dsigma = self._dsigma_dh(h)
        E, N_vec, U = enu_frame(lat, lon)  # U = n_hat
        grad_sigma = dsigma * U
        dy_base = np.concatenate([p / sigma, grad_sigma, [sigma]])
        return dy_base, sigma, dsigma, U, E, N_vec, grad_sigma, lat, lon, h

    def _rhs_base(self, s: float, y: np.ndarray) -> np.ndarray:
        """Right-hand side of the 7D eikonal ODE (r, p, T)."""
        dy_base, *_ = self._compute_base(y[0:3], y[3:6])
        return dy_base

    def _rhs_with_sensitivity(self, s: float, y: np.ndarray) -> np.ndarray:
        """RHS of the 37D coupled ODE (7 base + 5x6 variational)."""
        r, p = y[0:3], y[3:6]
        dy_base, sigma, dsigma, n_hat, E_vec, N_vec, grad_sigma, lat, lon, h \
            = self._compute_base(r, p)

        # Jacobian blocks for the variational equations
        A11 = -np.outer(p, grad_sigma) / (sigma * sigma)
        A12 = np.eye(3) / sigma
        M, N_rad = principal_radii(lat)
        d2sigma = self._d2sigma_dh2(h)
        H_sigma = (d2sigma * np.outer(n_hat, n_hat)
                   + dsigma * np.outer(N_vec, N_vec) / (M + h)
                   + dsigma * np.outer(E_vec, E_vec) / (N_rad + h))

        # Variational equations for each of the 5 parameters
        dy_sens = np.empty(30)
        for i in range(5):
            dr_da = y[7 + 6 * i: 7 + 6 * i + 3]
            dp_da = y[7 + 6 * i + 3: 7 + 6 * i + 6]
            d_dr_da = A11 @ dr_da + A12 @ dp_da
            d_dp_da = H_sigma @ dr_da
            dy_sens[6 * i: 6 * i + 3] = d_dr_da
            dy_sens[6 * i + 3: 6 * i + 6] = d_dp_da

        return np.concatenate([dy_base, dy_sens])

    # -- initial conditions -------------------------------------------------

    def _initial_state(self, inp: EikonalInput) -> np.ndarray:
        """Compute initial [r0, p0, T0] from EikonalInput."""
        r0 = geodetic_to_ecef(inp.lat_deg, inp.lon_deg, inp.h_m)
        E, N, U = enu_frame(inp.lat_deg, inp.lon_deg)
        theta = math.radians(inp.elevation_deg)
        alpha = math.radians(inp.azimuth_deg)
        t0 = (math.cos(theta) * math.sin(alpha) * E
              + math.cos(theta) * math.cos(alpha) * N
              + math.sin(theta) * U)
        sigma0 = self._sigma(inp.h_m)
        p0 = sigma0 * t0
        return np.concatenate([r0, p0, [0.0]])

    def _initial_sensitivities(self, inp: EikonalInput) -> np.ndarray:
        """Compute initial conditions for the 5 variational equations."""
        E, N, U = enu_frame(inp.lat_deg, inp.lon_deg)
        theta = math.radians(inp.elevation_deg)
        alpha = math.radians(inp.azimuth_deg)
        phi = math.radians(inp.lat_deg)
        sigma0 = self._sigma(inp.h_m)
        dsigma0 = self._dsigma_dh(inp.h_m)
        sin_th, cos_th = math.sin(theta), math.cos(theta)
        sin_al, cos_al = math.sin(alpha), math.cos(alpha)
        sin_phi, cos_phi = math.sin(phi), math.cos(phi)
        t0 = (cos_th * sin_al * E
              + cos_th * cos_al * N
              + sin_th * U)

        # h₀ sensitivity
        dr_dh0 = U.copy()
        dp_dh0 = dsigma0 * t0

        # θ₀ sensitivity
        dr_dtheta0 = np.zeros(3)
        dt_dtheta0 = (-sin_th * sin_al * E
                      - sin_th * cos_al * N
                      + cos_th * U)
        dp_dtheta0 = sigma0 * dt_dtheta0

        # α₀ sensitivity
        dr_dalpha0 = np.zeros(3)
        dt_dalpha0 = cos_th * cos_al * E - cos_th * sin_al * N
        dp_dalpha0 = sigma0 * dt_dalpha0

        # φ₀ and λ₀ sensitivities
        M, Nr = principal_radii(inp.lat_deg)

        dr_dphi0 = (M + inp.h_m) * N
        dt_dphi0 = sin_th * N - cos_th * cos_al * U
        dp_dphi0 = sigma0 * dt_dphi0

        dr_dlambda0 = (Nr + inp.h_m) * cos_phi * E
        dt_dlambda0 = ((sin_th * cos_phi - cos_th * cos_al * sin_phi) * E
                       + cos_th * sin_al * sin_phi * N
                       - cos_th * sin_al * cos_phi * U)
        dp_dlambda0 = sigma0 * dt_dlambda0

        return np.concatenate([
            dr_dh0, dp_dh0,
            dr_dtheta0, dp_dtheta0,
            dr_dalpha0, dp_dalpha0,
            dr_dphi0, dp_dphi0,
            dr_dlambda0, dp_dlambda0,
        ])

    # -- trace --------------------------------------------------------------

    def trace(
        self,
        inp: EikonalInput,
        stop: EikonalStopCondition,
        *,
        compute_sensitivities: bool = False,
        rtol: float = 1e-10,
        atol: float = 1e-12,
        max_step: float = 100.0,
    ) -> EikonalResult | tuple[EikonalResult, SensitivityResult]:
        """Trace a ray through the atmosphere.

        Parameters
        ----------
        inp : EikonalInput
            Initial geodetic position and direction.
        stop : EikonalStopCondition
            Stopping criteria.
        compute_sensitivities : bool
            If True, also solve the variational equations and return
            a ``SensitivityResult``.
        rtol, atol : float
            ODE solver tolerances.
        max_step : float
            Maximum step size (m).

        Returns
        -------
        EikonalResult or tuple[EikonalResult, SensitivityResult]
        """
        # Initial state
        y0_base = self._initial_state(inp)

        if compute_sensitivities:
            y0_sens = self._initial_sensitivities(inp)
            y0 = np.concatenate([y0_base, y0_sens])
            rhs = self._rhs_with_sensitivity
        else:
            y0 = y0_base
            rhs = self._rhs_base

        # Events
        events = []
        event_names: list[str] = []

        if stop.detect_ground_hit:
            def ground_event(s, y):
                _, _, h = ecef_to_geodetic(y[0], y[1], y[2])
                return h

            ground_event.terminal = True  # type: ignore[attr-defined]
            ground_event.direction = -1  # type: ignore[attr-defined]
            events.append(ground_event)
            event_names.append("ground")

        if stop.target_altitude_m is not None:
            target_h = stop.target_altitude_m

            def altitude_event(s, y):
                _, _, h = ecef_to_geodetic(y[0], y[1], y[2])
                return h - target_h

            altitude_event.terminal = True  # type: ignore[attr-defined]
            events.append(altitude_event)
            event_names.append("altitude")

        if stop.target_travel_time_s is not None:
            target_T = stop.target_travel_time_s

            def travel_time_event(s, y):
                return y[6] - target_T  # y[6] = T(s)

            travel_time_event.terminal = True  # type: ignore[attr-defined]
            events.append(travel_time_event)
            event_names.append("travel_time")

        # Integrate
        sol = solve_ivp(
            rhs,
            (0.0, stop.max_arc_length_m),
            y0,
            method="RK45",
            events=events if events else None,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            dense_output=False,
        )

        if sol.status == -1:
            raise RuntimeError(f"ODE solver failed: {sol.message}")

        # Determine termination reason
        terminated_by = "arc_length"
        for i, name in enumerate(event_names):
            if sol.t_events[i].size > 0:
                terminated_by = name
                break

        # ==============================================================
        # Post-processing: compute derivatives and application quantities
        # from the ODE state (r, p, T) at each solver step.
        # ==============================================================
        n_pts = sol.t.size
        s_arr = sol.t
        r_arr = sol.y[0:3, :].T  # (N, 3)
        p_arr = sol.y[3:6, :].T  # (N, 3)
        T_arr = sol.y[6, :]       # (N,)

        # Allocate output arrays
        r_prime = np.empty((n_pts, 3))
        p_prime = np.empty((n_pts, 3))
        dh_ds_arr = np.empty(n_pts)
        dv_ds_arr = np.empty(n_pts)
        r_dbl_prime = np.empty((n_pts, 3))
        curv_arr = np.empty(n_pts)
        lat_arr = np.empty(n_pts)
        lon_arr = np.empty(n_pts)
        h_arr = np.empty(n_pts)
        v_arr = np.empty(n_pts)
        theta_arr = np.empty(n_pts)
        az_arr = np.empty(n_pts)
        dr_dT_arr = np.empty((n_pts, 3))
        d2r_dT2_arr = np.empty((n_pts, 3))
        dp_dT_arr = np.empty((n_pts, 3))
        mom_err = np.empty(n_pts)

        for k in range(n_pts):
            # Geodetic conversion
            lat_k, lon_k, h_k = ecef_to_geodetic(
                r_arr[k, 0], r_arr[k, 1], r_arr[k, 2]
            )
            lat_arr[k] = lat_k
            lon_arr[k] = lon_k
            h_arr[k] = h_k

            # Speed and slowness
            v_k = self._v(h_k)
            dv_k = self._dv_dh(h_k)
            sigma_k = 1.0 / v_k
            dsigma_k = -dv_k / (v_k * v_k)
            v_arr[k] = v_k

            # Geodetic normal and ENU
            n_hat = geodetic_normal(lat_k, lon_k)
            E_k, N_k, U_k = enu_frame(lat_k, lon_k)

            # --- First derivatives ---
            t_hat = p_arr[k] / sigma_k              # r' = p/σ = t̂
            p_pr = dsigma_k * n_hat                  # p' = σ'n̂
            sin_th = float(np.dot(n_hat, t_hat))     # dh/ds = sinθ
            dv_ds_k = dv_k * sin_th                  # dv/ds = v'sinθ

            r_prime[k] = t_hat
            p_prime[k] = p_pr
            dh_ds_arr[k] = sin_th
            dv_ds_arr[k] = dv_ds_k

            # --- Second derivatives ---
            n_perp = n_hat - sin_th * t_hat          # n̂_⊥
            r_dbl_prime[k] = (dsigma_k / sigma_k) * n_perp  # = -(v'/v)n̂_⊥
            cos_th = math.sqrt(max(0.0, 1.0 - sin_th * sin_th))
            curv_arr[k] = abs(dv_k / v_k) * cos_th  # = |v'/v|cosθ

            # --- Travel-time derivatives ---
            dr_dT_arr[k] = v_k * t_hat
            d2r_dT2_arr[k] = v_k * dv_k * (2.0 * sin_th * t_hat - n_hat)
            dp_dT_arr[k] = -(dv_k / v_k) * n_hat

            # --- Application ---
            theta_arr[k] = math.degrees(math.asin(
                _clamp(sin_th, -1.0, 1.0)
            ))
            az_arr[k] = math.degrees(math.atan2(
                float(np.dot(p_arr[k], E_k)),
                float(np.dot(p_arr[k], N_k)),
            )) % 360.0

            # --- Diagnostic ---
            mom_err[k] = float(np.linalg.norm(p_arr[k])) - sigma_k

        # Bending angle — elevation only, measured in the starting ENU frame.
        E0, _N0, U0 = enu_frame(lat_arr[0], lon_arr[0])
        sigma_final = 1.0 / v_arr[-1]
        t_final = p_arr[-1] / sigma_final
        theta_final_in_start = math.degrees(math.asin(
            _clamp(float(np.dot(t_final, U0)), -1.0, 1.0)
        ))
        bending = theta_final_in_start - theta_arr[0]

        result = EikonalResult(
            s=s_arr,
            r=r_arr,
            p=p_arr,
            T=T_arr,
            r_prime=r_prime,
            p_prime=p_prime,
            dh_ds=dh_ds_arr,
            dv_ds=dv_ds_arr,
            r_double_prime=r_dbl_prime,
            curvature=curv_arr,
            lat_deg=lat_arr,
            lon_deg=lon_arr,
            h_m=h_arr,
            v=v_arr,
            theta_deg=theta_arr,
            azimuth_deg=az_arr,
            bending_deg=bending,
            dr_dT=dr_dT_arr,
            d2r_dT2=d2r_dT2_arr,
            dp_dT=dp_dT_arr,
            momentum_error=mom_err,
            terminated_by=terminated_by,
        )

        if not compute_sensitivities:
            return result

        # Extract sensitivity arrays (5 parameters × 6 components = 30)
        sens_raw = sol.y[7:37, :].T  # (N, 30)
        sens = SensitivityResult(
            dr_dh0=sens_raw[:, 0:3],
            dp_dh0=sens_raw[:, 3:6],
            dr_dtheta0=sens_raw[:, 6:9],
            dp_dtheta0=sens_raw[:, 9:12],
            dr_dalpha0=sens_raw[:, 12:15],
            dp_dalpha0=sens_raw[:, 15:18],
            dr_dphi0=sens_raw[:, 18:21],
            dp_dphi0=sens_raw[:, 21:24],
            dr_dlambda0=sens_raw[:, 24:27],
            dp_dlambda0=sens_raw[:, 27:30],
        )
        return result, sens

    # -- batch ---------------------------------------------------------------

    def trace_batch(
        self,
        inputs: BatchInput,
        *,
        endpoint: bool = True,
        rtol: float = 1e-10,
        atol: float = 1e-12,
        max_step: float = 100.0,
    ) -> BatchResult | list[EikonalResult]:
        """Trace N rays.

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
        n = len(inputs.lat_deg)
        results: list[EikonalResult] = []
        for i in range(n):
            inp = EikonalInput(
                lat_deg=float(inputs.lat_deg[i]),
                lon_deg=float(inputs.lon_deg[i]),
                h_m=float(inputs.h_m[i]),
                elevation_deg=float(inputs.elevation_deg[i]),
                azimuth_deg=float(inputs.azimuth_deg[i]),
            )
            stop = EikonalStopCondition(
                target_travel_time_s=float(inputs.travel_time_s[i]),
                detect_ground_hit=False,
            )
            results.append(self.trace(inp, stop, rtol=rtol, atol=atol,
                                      max_step=max_step))

        if not endpoint:
            return results

        return BatchResult(
            s=np.array([r.s[-1] for r in results]),
            r=np.array([r.r[-1] for r in results]),
            p=np.array([r.p[-1] for r in results]),
            T=np.array([r.T[-1] for r in results]),
            r_prime=np.array([r.r_prime[-1] for r in results]),
            p_prime=np.array([r.p_prime[-1] for r in results]),
            dh_ds=np.array([r.dh_ds[-1] for r in results]),
            dv_ds=np.array([r.dv_ds[-1] for r in results]),
            r_double_prime=np.array([r.r_double_prime[-1] for r in results]),
            curvature=np.array([r.curvature[-1] for r in results]),
            lat_deg=np.array([r.lat_deg[-1] for r in results]),
            lon_deg=np.array([r.lon_deg[-1] for r in results]),
            h_m=np.array([r.h_m[-1] for r in results]),
            v=np.array([r.v[-1] for r in results]),
            theta_deg=np.array([r.theta_deg[-1] for r in results]),
            azimuth_deg=np.array([r.azimuth_deg[-1] for r in results]),
            bending_deg=np.array([r.bending_deg for r in results]),
            dr_dT=np.array([r.dr_dT[-1] for r in results]),
            d2r_dT2=np.array([r.d2r_dT2[-1] for r in results]),
            dp_dT=np.array([r.dp_dT[-1] for r in results]),
            momentum_error=np.array([r.momentum_error[-1] for r in results]),
        )

    # -- factory ------------------------------------------------------------

    @classmethod
    def from_profile(
        cls,
        profile,
        mode,
        *,
        wavelength_nm: float = 633.0,
    ) -> EikonalTracer:
        """Create an EikonalTracer from a :class:`RefractionProfile`.

        Uses the speed of light as the reference speed to convert the
        atmospheric refractive index profile to propagation speed.

        Parameters
        ----------
        profile : RefractionProfile
            An existing atmospheric refraction profile.
        mode : Mode
            Computation mode (Ciddor or ITU_R_P453).
        wavelength_nm : float
            Wavelength for Ciddor mode (nm).

        Returns
        -------
        EikonalTracer
        """
        def eta_func(h_m: float) -> float:
            h_km = max(0.0, min(h_m / 1000.0, 122.0))
            return profile.compute(h_km, mode, wavelength_nm).n

        def deta_dh_func(h_m: float) -> float:
            h_km = max(0.0, min(h_m / 1000.0, 122.0))
            return profile.compute(h_km, mode, wavelength_nm).dn_dh / 1000.0

        v_func, dv_dh_func = speed_from_eta(
            eta_func, deta_dh_func, SPEED_OF_LIGHT
        )
        return cls(v_func, dv_dh_func)

    @classmethod
    def from_sound_speed_profile(cls, profile, mode) -> EikonalTracer:
        """Create an EikonalTracer from a :class:`SoundSpeedProfile`.

        Handles the depth-to-height sign flip: ocean modules use depth
        (positive downward) while the ray tracer uses geodetic height
        (positive upward).

        Parameters
        ----------
        profile : SoundSpeedProfile
            An ocean sound speed profile.
        mode : SoundMode
            Sound speed model (DelGrosso, ChenMillero, or Mackenzie).

        Returns
        -------
        EikonalTracer
        """
        def v_func(h_m: float) -> float:
            depth = max(0.0, -h_m)
            return profile.compute(depth, mode).sound_speed_m_s

        def dv_dh_func(h_m: float) -> float:
            depth = max(0.0, -h_m)
            return -profile.compute(depth, mode).dc_dz

        return cls(v_func, dv_dh_func)

    @classmethod
    def from_ocean_refraction_profile(
        cls, profile, mode, *, c_ref: float = SPEED_OF_LIGHT,
    ) -> EikonalTracer:
        """Create an EikonalTracer from an :class:`OceanRefractionProfile`.

        Works with all modes (MeissnerWentz, MillardSeaver, QuanFry, IAPWS).
        The ray path is determined by the real refractive index n_real;
        the imaginary part (absorption) does not affect the path.

        Handles the depth-to-height sign flip.

        Parameters
        ----------
        profile : OceanRefractionProfile
            An ocean refraction profile.
        mode : OceanMode
            Refraction model (QuanFry, MillardSeaver, or IAPWS).
        c_ref : float
            Reference speed (m/s). Default: speed of light.

        Returns
        -------
        EikonalTracer
        """
        def eta_func(h_m: float) -> float:
            depth = max(0.0, -h_m)
            return profile.compute(depth, mode).n_real

        def deta_dh_func(h_m: float) -> float:
            depth = max(0.0, -h_m)
            return -profile.compute(depth, mode).dn_dz

        v_func, dv_dh_func = speed_from_eta(eta_func, deta_dh_func, c_ref)
        return cls(v_func, dv_dh_func)

    @classmethod
    def from_depth_speed_table(cls, depths, speeds) -> EikonalTracer:
        """Create an EikonalTracer from measured depth vs speed data.

        Handles the depth-to-height conversion (depth positive downward,
        h positive upward) and computes the speed derivative via finite
        differences on the table.

        Parameters
        ----------
        depths : array_like
            Depth values (m, positive downward).
        speeds : array_like
            Propagation speed at each depth (m/s).

        Returns
        -------
        EikonalTracer

        Examples
        --------
        >>> depths = [0, 50, 100, 500, 1000]
        >>> speeds = [1520, 1515, 1490, 1500, 1510]
        >>> tracer = EikonalTracer.from_depth_speed_table(depths, speeds)
        """
        from scipy.interpolate import interp1d

        depths = np.asarray(depths, dtype=float)
        speeds = np.asarray(speeds, dtype=float)

        # Sort by depth ascending, convert to h ascending
        order = np.argsort(depths)
        h = -depths[order][::-1]   # negative depth, reversed → ascending h
        v = speeds[order][::-1]

        # Compute dv/dh via numpy gradient
        dv = np.gradient(v, h)

        # Interpolate
        v_interp = interp1d(h, v, fill_value="extrapolate", kind="linear")
        dv_interp = interp1d(h, dv, fill_value="extrapolate", kind="linear")

        return cls(lambda hm: float(v_interp(hm)),
                   lambda hm: float(dv_interp(hm)))
