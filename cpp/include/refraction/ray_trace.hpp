#pragma once

// 3D ECEF eikonal ray tracer with adaptive Dormand-Prince RK45.
//
// The ODE is parameterised by propagation speed v(h):
//
//   dr/ds = p / sigma       (position follows momentum)
//   dp/ds = sigma' * n_hat  (momentum changes by slowness gradient)
//   dT/ds = sigma           (travel time accumulates)
//
// where sigma = 1/v is the slowness, p = sigma * t_hat is the slowness
// momentum (|p| = sigma = 1/v), and T is the travel time.
//
// For EM (radio), use speed_from_eta to convert eta(h) + c_ref to v(h).

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "constants.hpp"
#include "geodetic.hpp"

namespace refraction {

// -----------------------------------------------------------------
// Helper: convert refractive index to speed
// -----------------------------------------------------------------

template <typename EtaFunc, typename DEtaFunc>
inline auto speed_from_eta(EtaFunc eta_func, DEtaFunc deta_dh_func,
                           double c_ref) {
    auto v_func = [=](double h) { return c_ref / eta_func(h); };
    auto dv_func = [=](double h) {
        double eta = eta_func(h);
        return -c_ref * deta_dh_func(h) / (eta * eta);
    };
    return std::pair{v_func, dv_func};
}

// -----------------------------------------------------------------
// Data structures
// -----------------------------------------------------------------

struct EikonalInput {
    double lat_deg, lon_deg, h_m;
    double elevation_deg, azimuth_deg;
};

struct EikonalStopCondition {
    std::optional<double> target_altitude_m;
    std::optional<double> target_travel_time_s;
    double max_arc_length_m = 500'000.0;
    bool detect_ground_hit = true;
};

struct SensitivityResult {
    std::vector<Vec3> dr_dh0, dp_dh0;
    std::vector<Vec3> dr_dtheta0, dp_dtheta0;
    std::vector<Vec3> dr_dalpha0, dp_dalpha0;
    std::vector<Vec3> dr_dphi0, dp_dphi0;
    std::vector<Vec3> dr_dlambda0, dp_dlambda0;
};

struct EikonalResult {
    std::vector<double> s;
    std::vector<Vec3> r, p;
    std::vector<double> T;
    // First derivatives
    std::vector<Vec3> r_prime, p_prime;
    std::vector<double> dh_ds, dv_ds;
    // Second derivatives
    std::vector<Vec3> r_double_prime;
    std::vector<double> curvature;
    // Geodetic
    std::vector<double> lat_deg, lon_deg, h_m, v;
    // Application
    std::vector<double> theta_deg, azimuth_deg;
    double bending_deg = 0.0;
    // Travel-time derivatives
    std::vector<Vec3> dr_dT, d2r_dT2, dp_dT;
    // Diagnostics
    std::vector<double> momentum_error;
    std::string terminated_by;
};

// -----------------------------------------------------------------
// Dormand-Prince RK45 (adaptive, internal)
// -----------------------------------------------------------------

namespace detail {

// Dormand-Prince coefficients
inline constexpr double DP_A2 = 1.0/5, DP_A3 = 3.0/10, DP_A4 = 4.0/5,
    DP_A5 = 8.0/9, DP_A6 = 1.0, DP_A7 = 1.0;

inline constexpr double DP_B21 = 1.0/5;
inline constexpr double DP_B31 = 3.0/40, DP_B32 = 9.0/40;
inline constexpr double DP_B41 = 44.0/45, DP_B42 = -56.0/15, DP_B43 = 32.0/9;
inline constexpr double DP_B51 = 19372.0/6561, DP_B52 = -25360.0/2187,
    DP_B53 = 64448.0/6561, DP_B54 = -212.0/729;
inline constexpr double DP_B61 = 9017.0/3168, DP_B62 = -355.0/33,
    DP_B63 = 46732.0/5247, DP_B64 = 49.0/176, DP_B65 = -5103.0/18656;
inline constexpr double DP_B71 = 35.0/384, DP_B73 = 500.0/1113,
    DP_B74 = 125.0/192, DP_B75 = -2187.0/6784, DP_B76 = 11.0/84;

// Error coefficients (5th - 4th order)
inline constexpr double DP_E1 = 71.0/57600, DP_E3 = -71.0/16695,
    DP_E4 = 71.0/1920, DP_E5 = -17253.0/339200, DP_E6 = 22.0/525,
    DP_E7 = -1.0/40;

template <int N>
using State = std::array<double, N>;

template <int N>
inline State<N> state_add(const State<N>& a, const State<N>& b) {
    State<N> r;
    for (int i = 0; i < N; ++i) r[i] = a[i] + b[i];
    return r;
}

template <int N>
inline State<N> state_scale(double c, const State<N>& a) {
    State<N> r;
    for (int i = 0; i < N; ++i) r[i] = c * a[i];
    return r;
}

template <int N>
inline State<N> state_axpy(double c, const State<N>& x, const State<N>& y) {
    // y + c*x
    State<N> r;
    for (int i = 0; i < N; ++i) r[i] = y[i] + c * x[i];
    return r;
}

// Single Dormand-Prince step. Returns (y_new, error_norm).
template <int N, typename F>
std::pair<State<N>, double>
dp45_step(F&& f, double s, const State<N>& y, double h,
          double rtol, double atol) {
    auto k1 = f(s, y);
    auto k2 = f(s + DP_A2*h, [&]{
        State<N> r; for(int i=0;i<N;++i) r[i]=y[i]+h*(DP_B21*k1[i]); return r;
    }());
    auto k3 = f(s + DP_A3*h, [&]{
        State<N> r; for(int i=0;i<N;++i) r[i]=y[i]+h*(DP_B31*k1[i]+DP_B32*k2[i]); return r;
    }());
    auto k4 = f(s + DP_A4*h, [&]{
        State<N> r; for(int i=0;i<N;++i) r[i]=y[i]+h*(DP_B41*k1[i]+DP_B42*k2[i]+DP_B43*k3[i]); return r;
    }());
    auto k5 = f(s + DP_A5*h, [&]{
        State<N> r; for(int i=0;i<N;++i) r[i]=y[i]+h*(DP_B51*k1[i]+DP_B52*k2[i]+DP_B53*k3[i]+DP_B54*k4[i]); return r;
    }());
    auto k6 = f(s + DP_A6*h, [&]{
        State<N> r; for(int i=0;i<N;++i) r[i]=y[i]+h*(DP_B61*k1[i]+DP_B62*k2[i]+DP_B63*k3[i]+DP_B64*k4[i]+DP_B65*k5[i]); return r;
    }());

    // 5th-order solution
    State<N> y_new;
    for (int i = 0; i < N; ++i)
        y_new[i] = y[i] + h * (DP_B71*k1[i] + DP_B73*k3[i] + DP_B74*k4[i]
                                + DP_B75*k5[i] + DP_B76*k6[i]);

    // Error estimate (5th - 4th order)
    auto k7 = f(s + h, y_new);
    double err_norm = 0.0;
    for (int i = 0; i < N; ++i) {
        double ei = h * (DP_E1*k1[i] + DP_E3*k3[i] + DP_E4*k4[i]
                         + DP_E5*k5[i] + DP_E6*k6[i] + DP_E7*k7[i]);
        double sc = atol + rtol * std::max(std::abs(y[i]), std::abs(y_new[i]));
        err_norm += (ei / sc) * (ei / sc);
    }
    err_norm = std::sqrt(err_norm / N);

    return {y_new, err_norm};
}

}  // namespace detail

// -----------------------------------------------------------------
// EikonalTracer
// -----------------------------------------------------------------

template <typename VFunc, typename DVFunc>
class EikonalTracer {
public:
    EikonalTracer(VFunc v_func, DVFunc dv_dh_func)
        : v_(v_func), dv_(dv_dh_func) {}

    EikonalResult trace(const EikonalInput& inp,
                        const EikonalStopCondition& stop,
                        double rtol = 1e-10,
                        double atol = 1e-12,
                        double max_step = 100.0) const
    {
        using State7 = std::array<double, 7>;

        // Initial state
        auto r0 = geodetic_to_ecef(inp.lat_deg, inp.lon_deg, inp.h_m);
        auto [E0, N0, U0] = enu_frame(inp.lat_deg, inp.lon_deg);
        double theta = inp.elevation_deg * DEG_TO_RAD;
        double alpha = inp.azimuth_deg * DEG_TO_RAD;
        double ct = std::cos(theta), st = std::sin(theta);
        double ca = std::cos(alpha), sa = std::sin(alpha);
        Vec3 t0 = vec::add(vec::add(vec::scale(ct*sa, E0),
                                     vec::scale(ct*ca, N0)),
                           vec::scale(st, U0));
        double sigma0 = 1.0 / v_(inp.h_m);
        Vec3 p0 = vec::scale(sigma0, t0);

        State7 y0 = {r0[0], r0[1], r0[2],
                      p0[0], p0[1], p0[2], 0.0};

        // RHS
        auto rhs = [this](double s, const State7& y) -> State7 {
            auto g = ecef_to_geodetic(y[0], y[1], y[2]);
            double h = g.h_m;  // may be negative for underwater rays
            double v_h = v_(h);
            double dv_h = dv_(h);
            double sigma = 1.0 / v_h;
            double dsigma = -dv_h / (v_h * v_h);
            auto n_hat = geodetic_normal(g.lat_deg, g.lon_deg);
            return {y[3]/sigma, y[4]/sigma, y[5]/sigma,
                    dsigma*n_hat[0], dsigma*n_hat[1], dsigma*n_hat[2],
                    sigma};
        };

        // Integrate with adaptive stepping
        std::vector<double> s_out;
        std::vector<State7> y_out;
        s_out.push_back(0.0);
        y_out.push_back(y0);

        double s = 0.0;
        State7 y = y0;
        double h_step = std::min(max_step, stop.max_arc_length_m * 0.001);
        std::string terminated_by = "arc_length";

        while (s < stop.max_arc_length_m) {
            double s_rem = stop.max_arc_length_m - s;
            h_step = std::min(h_step, s_rem);
            h_step = std::min(h_step, max_step);
            if (h_step < 1e-12) break;

            auto [y_new, err] = detail::dp45_step<7>(rhs, s, y, h_step,
                                                      rtol, atol);

            if (err <= 1.0) {
                // Check events with linear interpolation for precise crossing
                auto g_old = ecef_to_geodetic(y[0], y[1], y[2]);
                auto g_new = ecef_to_geodetic(y_new[0], y_new[1], y_new[2]);

                // Ground hit: h crosses zero from above
                if (stop.detect_ground_hit &&
                    g_old.h_m > 0.0 && g_new.h_m <= 0.0) {
                    double frac = g_old.h_m / (g_old.h_m - g_new.h_m);
                    double s_cross = s + frac * h_step;
                    State7 y_cross;
                    for (int i = 0; i < 7; ++i)
                        y_cross[i] = y[i] + frac * (y_new[i] - y[i]);
                    s_out.push_back(s_cross);
                    y_out.push_back(y_cross);
                    terminated_by = "ground";
                    break;
                }

                // Altitude target: h crosses target from below
                if (stop.target_altitude_m &&
                    g_old.h_m < *stop.target_altitude_m &&
                    g_new.h_m >= *stop.target_altitude_m) {
                    double target = *stop.target_altitude_m;
                    double frac = (target - g_old.h_m) /
                                  (g_new.h_m - g_old.h_m);
                    double s_cross = s + frac * h_step;
                    State7 y_cross;
                    for (int i = 0; i < 7; ++i)
                        y_cross[i] = y[i] + frac * (y_new[i] - y[i]);
                    s_out.push_back(s_cross);
                    y_out.push_back(y_cross);
                    terminated_by = "altitude";
                    break;
                }

                // Travel time target
                if (stop.target_travel_time_s &&
                    y[6] < *stop.target_travel_time_s &&
                    y_new[6] >= *stop.target_travel_time_s) {
                    double target = *stop.target_travel_time_s;
                    double frac = (target - y[6]) / (y_new[6] - y[6]);
                    double s_cross = s + frac * h_step;
                    State7 y_cross;
                    for (int i = 0; i < 7; ++i)
                        y_cross[i] = y[i] + frac * (y_new[i] - y[i]);
                    s_out.push_back(s_cross);
                    y_out.push_back(y_cross);
                    terminated_by = "travel_time";
                    break;
                }

                // No event — accept step normally
                s += h_step;
                y = y_new;
                s_out.push_back(s);
                y_out.push_back(y);
            }

            // Adjust step size
            double factor = (err > 0) ? 0.9 * std::pow(err, -0.2) : 5.0;
            factor = std::clamp(factor, 0.2, 5.0);
            h_step *= factor;
        }

        // Post-processing
        return postprocess(s_out, y_out, terminated_by, inp);
    }

    std::pair<EikonalResult, SensitivityResult>
    trace_with_sensitivities(const EikonalInput& inp,
                             const EikonalStopCondition& stop,
                             double rtol = 1e-10,
                             double atol = 1e-12,
                             double max_step = 100.0) const
    {
        using State37 = std::array<double, 37>;

        // Initial state (same as trace())
        auto r0 = geodetic_to_ecef(inp.lat_deg, inp.lon_deg, inp.h_m);
        auto [E0, N0, U0] = enu_frame(inp.lat_deg, inp.lon_deg);
        double theta = inp.elevation_deg * DEG_TO_RAD;
        double alpha = inp.azimuth_deg * DEG_TO_RAD;
        double ct = std::cos(theta), st = std::sin(theta);
        double ca = std::cos(alpha), sa = std::sin(alpha);
        double phi = inp.lat_deg * DEG_TO_RAD;
        double sp = std::sin(phi), cp = std::cos(phi);
        Vec3 t0 = vec::add(vec::add(vec::scale(ct*sa, E0),
                                     vec::scale(ct*ca, N0)),
                           vec::scale(st, U0));
        double sigma0 = sigma_(inp.h_m);
        double dsigma0 = dsigma_(inp.h_m);
        Vec3 p0 = vec::scale(sigma0, t0);

        // Build 37D initial state: [r, p, T, sens_h0(6), sens_theta0(6),
        //   sens_alpha0(6), sens_phi0(6), sens_lambda0(6)]
        State37 y0{};
        y0[0]=r0[0]; y0[1]=r0[1]; y0[2]=r0[2];
        y0[3]=p0[0]; y0[4]=p0[1]; y0[5]=p0[2];
        y0[6]=0.0;

        // h0: dr=U0, dp=dsigma0*t0
        Vec3 dp_dh0 = vec::scale(dsigma0, t0);
        y0[7]=U0[0]; y0[8]=U0[1]; y0[9]=U0[2];
        y0[10]=dp_dh0[0]; y0[11]=dp_dh0[1]; y0[12]=dp_dh0[2];

        // theta0: dr=0, dp=sigma0*dt/dtheta0
        Vec3 dt_dth = vec::add(vec::add(vec::scale(-st*sa, E0),
                                         vec::scale(-st*ca, N0)),
                               vec::scale(ct, U0));
        Vec3 dp_dth = vec::scale(sigma0, dt_dth);
        y0[13]=0; y0[14]=0; y0[15]=0;
        y0[16]=dp_dth[0]; y0[17]=dp_dth[1]; y0[18]=dp_dth[2];

        // alpha0: dr=0, dp=sigma0*dt/dalpha0
        Vec3 dt_dal = vec::sub(vec::scale(ct*ca, E0), vec::scale(ct*sa, N0));
        Vec3 dp_dal = vec::scale(sigma0, dt_dal);
        y0[19]=0; y0[20]=0; y0[21]=0;
        y0[22]=dp_dal[0]; y0[23]=dp_dal[1]; y0[24]=dp_dal[2];

        // phi0: dr=(M+h0)*N, dp=sigma0*dt/dphi0
        auto [M0, Nr0] = principal_radii(inp.lat_deg);
        Vec3 dr_dphi = vec::scale(M0 + inp.h_m, N0);
        Vec3 dt_dph = vec::sub(vec::scale(st, N0), vec::scale(ct*ca, U0));
        Vec3 dp_dph = vec::scale(sigma0, dt_dph);
        y0[25]=dr_dphi[0]; y0[26]=dr_dphi[1]; y0[27]=dr_dphi[2];
        y0[28]=dp_dph[0]; y0[29]=dp_dph[1]; y0[30]=dp_dph[2];

        // lambda0: dr=(Nr+h0)*cos(phi)*E, dp=sigma0*dt/dlambda0
        Vec3 dr_dlam = vec::scale((Nr0 + inp.h_m) * cp, E0);
        Vec3 dt_dlam = vec::add(
            vec::add(vec::scale(st*cp - ct*ca*sp, E0),
                     vec::scale(ct*sa*sp, N0)),
            vec::scale(-ct*sa*cp, U0));
        Vec3 dp_dlam = vec::scale(sigma0, dt_dlam);
        y0[31]=dr_dlam[0]; y0[32]=dr_dlam[1]; y0[33]=dr_dlam[2];
        y0[34]=dp_dlam[0]; y0[35]=dp_dlam[1]; y0[36]=dp_dlam[2];

        // 37D RHS
        auto rhs37 = [this](double s, const State37& y) -> State37 {
            Vec3 r = {y[0], y[1], y[2]};
            Vec3 p = {y[3], y[4], y[5]};
            auto g = ecef_to_geodetic(r[0], r[1], r[2]);
            double h = g.h_m;
            double sig = sigma_(h);
            double dsig = dsigma_(h);
            auto [Ev, Nv, Uv] = enu_frame(g.lat_deg, g.lon_deg);
            Vec3 grad_sig = vec::scale(dsig, Uv);

            State37 dy{};
            // Base: dr/ds, dp/ds, dT/ds
            dy[0]=p[0]/sig; dy[1]=p[1]/sig; dy[2]=p[2]/sig;
            dy[3]=grad_sig[0]; dy[4]=grad_sig[1]; dy[5]=grad_sig[2];
            dy[6]=sig;

            // Jacobian blocks
            Mat3 A11 = vec::matscale(-1.0/(sig*sig), vec::outer(p, grad_sig));
            auto [Mv, Nrv] = principal_radii(g.lat_deg);
            double d2s = d2sigma_(h);
            Mat3 Hsig = vec::matadd(
                vec::matadd(
                    vec::matscale(d2s, vec::outer(Uv, Uv)),
                    vec::matscale(dsig/(Mv+h), vec::outer(Nv, Nv))),
                vec::matscale(dsig/(Nrv+h), vec::outer(Ev, Ev)));

            // Variational equations for 5 parameters
            for (int i = 0; i < 5; ++i) {
                Vec3 dr_da = {y[7+6*i], y[8+6*i], y[9+6*i]};
                Vec3 dp_da = {y[10+6*i], y[11+6*i], y[12+6*i]};
                Vec3 d_dr = vec::add(vec::matvec(A11, dr_da),
                                      vec::scale(1.0/sig, dp_da));
                Vec3 d_dp = vec::matvec(Hsig, dr_da);
                dy[7+6*i]=d_dr[0]; dy[8+6*i]=d_dr[1]; dy[9+6*i]=d_dr[2];
                dy[10+6*i]=d_dp[0]; dy[11+6*i]=d_dp[1]; dy[12+6*i]=d_dp[2];
            }
            return dy;
        };

        // Integrate (same adaptive loop as trace())
        std::vector<double> s_out;
        std::vector<State37> y_out;
        s_out.push_back(0.0);
        y_out.push_back(y0);

        double s = 0.0;
        State37 y = y0;
        double h_step = std::min(max_step, stop.max_arc_length_m * 0.001);
        std::string terminated_by = "arc_length";

        while (s < stop.max_arc_length_m) {
            double s_rem = stop.max_arc_length_m - s;
            h_step = std::min(h_step, s_rem);
            h_step = std::min(h_step, max_step);
            if (h_step < 1e-12) break;

            auto [y_new, err] = detail::dp45_step<37>(rhs37, s, y, h_step,
                                                       rtol, atol);
            if (err <= 1.0) {
                auto g_old = ecef_to_geodetic(y[0], y[1], y[2]);
                auto g_new = ecef_to_geodetic(y_new[0], y_new[1], y_new[2]);

                if (stop.detect_ground_hit &&
                    g_old.h_m > 0.0 && g_new.h_m <= 0.0) {
                    double frac = g_old.h_m / (g_old.h_m - g_new.h_m);
                    State37 yc{}; for(int i=0;i<37;++i) yc[i]=y[i]+frac*(y_new[i]-y[i]);
                    s_out.push_back(s + frac * h_step);
                    y_out.push_back(yc);
                    terminated_by = "ground"; break;
                }
                if (stop.target_altitude_m &&
                    g_old.h_m < *stop.target_altitude_m &&
                    g_new.h_m >= *stop.target_altitude_m) {
                    double frac = (*stop.target_altitude_m - g_old.h_m) / (g_new.h_m - g_old.h_m);
                    State37 yc{}; for(int i=0;i<37;++i) yc[i]=y[i]+frac*(y_new[i]-y[i]);
                    s_out.push_back(s + frac * h_step);
                    y_out.push_back(yc);
                    terminated_by = "altitude"; break;
                }
                if (stop.target_travel_time_s &&
                    y[6] < *stop.target_travel_time_s &&
                    y_new[6] >= *stop.target_travel_time_s) {
                    double frac = (*stop.target_travel_time_s - y[6]) / (y_new[6] - y[6]);
                    State37 yc{}; for(int i=0;i<37;++i) yc[i]=y[i]+frac*(y_new[i]-y[i]);
                    s_out.push_back(s + frac * h_step);
                    y_out.push_back(yc);
                    terminated_by = "travel_time"; break;
                }
                s += h_step;
                y = y_new;
                s_out.push_back(s);
                y_out.push_back(y);
            }
            double factor = (err > 0) ? 0.9 * std::pow(err, -0.2) : 5.0;
            factor = std::clamp(factor, 0.2, 5.0);
            h_step *= factor;
        }

        // Post-process base result (extract first 7 components)
        int n = static_cast<int>(s_out.size());
        std::vector<std::array<double, 7>> y7(n);
        for (int k = 0; k < n; ++k)
            for (int i = 0; i < 7; ++i) y7[k][i] = y_out[k][i];
        auto res = postprocess(s_out, y7, terminated_by, inp);

        // Extract sensitivity arrays
        SensitivityResult sens;
        sens.dr_dh0.resize(n); sens.dp_dh0.resize(n);
        sens.dr_dtheta0.resize(n); sens.dp_dtheta0.resize(n);
        sens.dr_dalpha0.resize(n); sens.dp_dalpha0.resize(n);
        sens.dr_dphi0.resize(n); sens.dp_dphi0.resize(n);
        sens.dr_dlambda0.resize(n); sens.dp_dlambda0.resize(n);
        for (int k = 0; k < n; ++k) {
            const auto& yk = y_out[k];
            sens.dr_dh0[k] = {yk[7], yk[8], yk[9]};
            sens.dp_dh0[k] = {yk[10], yk[11], yk[12]};
            sens.dr_dtheta0[k] = {yk[13], yk[14], yk[15]};
            sens.dp_dtheta0[k] = {yk[16], yk[17], yk[18]};
            sens.dr_dalpha0[k] = {yk[19], yk[20], yk[21]};
            sens.dp_dalpha0[k] = {yk[22], yk[23], yk[24]};
            sens.dr_dphi0[k] = {yk[25], yk[26], yk[27]};
            sens.dp_dphi0[k] = {yk[28], yk[29], yk[30]};
            sens.dr_dlambda0[k] = {yk[31], yk[32], yk[33]};
            sens.dp_dlambda0[k] = {yk[34], yk[35], yk[36]};
        }

        return {res, sens};
    }

private:
    VFunc v_;
    DVFunc dv_;

    static constexpr double FD_STEP_M = 10.0;

    double sigma_(double h) const { return 1.0 / v_(h); }
    double dsigma_(double h) const {
        double v = v_(h);
        return -dv_(h) / (v * v);
    }
    double d2sigma_(double h) const {
        return (dsigma_(h + FD_STEP_M) - dsigma_(h - FD_STEP_M))
               / (2.0 * FD_STEP_M);
    }

    EikonalResult postprocess(
        const std::vector<double>& s_vec,
        const std::vector<std::array<double, 7>>& y_vec,
        const std::string& terminated_by,
        const EikonalInput& inp) const
    {
        int n = static_cast<int>(s_vec.size());
        EikonalResult res;
        res.s = s_vec;
        res.terminated_by = terminated_by;

        res.r.resize(n); res.p.resize(n); res.T.resize(n);
        res.r_prime.resize(n); res.p_prime.resize(n);
        res.dh_ds.resize(n); res.dv_ds.resize(n);
        res.r_double_prime.resize(n); res.curvature.resize(n);
        res.lat_deg.resize(n); res.lon_deg.resize(n);
        res.h_m.resize(n); res.v.resize(n);
        res.theta_deg.resize(n); res.azimuth_deg.resize(n);
        res.dr_dT.resize(n); res.d2r_dT2.resize(n); res.dp_dT.resize(n);
        res.momentum_error.resize(n);

        for (int k = 0; k < n; ++k) {
            const auto& y = y_vec[k];
            Vec3 rk = {y[0], y[1], y[2]};
            Vec3 pk = {y[3], y[4], y[5]};
            res.r[k] = rk;
            res.p[k] = pk;
            res.T[k] = y[6];

            auto g = ecef_to_geodetic(y[0], y[1], y[2]);
            double hk = g.h_m;  // may be negative for underwater rays
            res.lat_deg[k] = g.lat_deg;
            res.lon_deg[k] = g.lon_deg;
            res.h_m[k] = hk;

            double vk = v_(hk);
            double dvk = dv_(hk);
            double sigma_k = 1.0 / vk;
            double dsigma_k = -dvk / (vk * vk);
            res.v[k] = vk;

            auto n_hat = geodetic_normal(g.lat_deg, g.lon_deg);
            auto [Ek, Nk, Uk] = enu_frame(g.lat_deg, g.lon_deg);

            // First derivatives
            Vec3 t_hat = vec::scale(1.0 / sigma_k, pk);
            Vec3 p_pr = vec::scale(dsigma_k, n_hat);
            double sin_th = vec::dot(n_hat, t_hat);
            res.r_prime[k] = t_hat;
            res.p_prime[k] = p_pr;
            res.dh_ds[k] = sin_th;
            res.dv_ds[k] = dvk * sin_th;

            // Second derivatives
            Vec3 n_perp = vec::sub(n_hat, vec::scale(sin_th, t_hat));
            res.r_double_prime[k] = vec::scale(dsigma_k / sigma_k, n_perp);
            double cos_th = std::sqrt(std::max(0.0, 1.0 - sin_th * sin_th));
            res.curvature[k] = std::abs(dvk / vk) * cos_th;

            // Application
            res.theta_deg[k] = std::asin(std::clamp(sin_th, -1.0, 1.0))
                                / DEG_TO_RAD;
            double pE = vec::dot(pk, Ek);
            double pN = vec::dot(pk, Nk);
            double az = std::atan2(pE, pN) / DEG_TO_RAD;
            if (az < 0) az += 360.0;
            res.azimuth_deg[k] = az;

            // Travel-time derivatives
            res.dr_dT[k] = vec::scale(vk, t_hat);
            res.d2r_dT2[k] = vec::scale(vk * dvk,
                vec::sub(vec::scale(2.0 * sin_th, t_hat), n_hat));
            res.dp_dT[k] = vec::scale(-(dvk / vk), n_hat);

            // Diagnostic
            res.momentum_error[k] = vec::norm(pk) - sigma_k;
        }

        // Bending — elevation-only, in the starting ENU frame
        auto [E0, N0_vec, U0] = enu_frame(inp.lat_deg, inp.lon_deg);
        double sigma_final = 1.0 / res.v.back();
        Vec3 t_final = vec::scale(1.0 / sigma_final, res.p.back());
        double sin_final = std::clamp(vec::dot(t_final, U0), -1.0, 1.0);
        double theta_final_in_start = std::asin(sin_final) / DEG_TO_RAD;
        res.bending_deg = theta_final_in_start - res.theta_deg[0];

        return res;
    }
};

// Deduction guide
template <typename V, typename DV>
EikonalTracer(V, DV) -> EikonalTracer<V, DV>;

// -----------------------------------------------------------------
// Factory: from atmospheric profile
// Include profile.hpp before ray_trace.hpp to use this function.
// -----------------------------------------------------------------

template <typename Profile, typename ModeT>
inline auto tracer_from_profile(const Profile& profile, ModeT mode,
                                double wavelength_nm = 633.0) {
    auto m = mode;
    auto wl = wavelength_nm;
    auto eta = [&profile, m, wl](double h_m) {
        double h_km = std::clamp(h_m / 1000.0, 0.0, 122.0);
        return profile.compute(h_km, m, wl).n;
    };
    auto deta = [&profile, m, wl](double h_m) {
        double h_km = std::clamp(h_m / 1000.0, 0.0, 122.0);
        return profile.compute(h_km, m, wl).dn_dh / 1000.0;
    };
    auto [v, dv] = speed_from_eta(eta, deta, SPEED_OF_LIGHT);
    return EikonalTracer{v, dv};
}

// -----------------------------------------------------------------
// Factory: from ocean sound speed profile
// -----------------------------------------------------------------

template <typename Profile, typename ModeT>
inline auto tracer_from_sound_speed_profile(const Profile& profile, ModeT mode) {
    auto m = mode;
    auto v = [&profile, m](double h_m) {
        double depth = std::max(0.0, -h_m);
        return profile.compute(depth, m).sound_speed_m_s;
    };
    auto dv = [&profile, m](double h_m) {
        double depth = std::max(0.0, -h_m);
        return -profile.compute(depth, m).dc_dz;
    };
    return EikonalTracer{v, dv};
}

// -----------------------------------------------------------------
// Factory: from ocean refraction profile (optical + radio)
// -----------------------------------------------------------------

template <typename Profile, typename ModeT>
inline auto tracer_from_ocean_refraction_profile(
    const Profile& profile, ModeT mode, double c_ref = SPEED_OF_LIGHT) {
    auto m = mode;
    auto eta = [&profile, m](double h_m) {
        double depth = std::max(0.0, -h_m);
        return profile.compute(depth, m).n_real;
    };
    auto deta = [&profile, m](double h_m) {
        double depth = std::max(0.0, -h_m);
        return -profile.compute(depth, m).dn_dz;
    };
    auto [v, dv] = speed_from_eta(eta, deta, c_ref);
    return EikonalTracer{v, dv};
}

}  // namespace refraction
