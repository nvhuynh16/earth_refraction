#pragma once

// Type-erased eikonal tracer + batch tracing for nanobind.
//
// EikonalTracerErased wraps EikonalTracer with std::function so it can
// be bound by nanobind without template instantiation issues.
//
// TableSpeedProfile provides pure-C++ speed lookup from a precomputed
// (altitude, speed) table with linear interpolation — no Python callbacks.
//
// trace_batch() traces N rays in a C++ loop, returning endpoint states.

#include <functional>
#include <vector>

#include "interpolation.hpp"
#include "ray_trace.hpp"

namespace refraction {

// -----------------------------------------------------------------
// Table-based speed profile (pure C++, no Python callbacks)
// -----------------------------------------------------------------

class TableSpeedProfile {
public:
    /// Construct from sorted (ascending h) arrays of altitude and speed.
    TableSpeedProfile(const double* h, const double* v, int n)
        : h_(h, h + n), v_(v, v + n), dv_(n)
    {
        // Precompute dv/dh via central finite differences
        for (int i = 0; i < n; ++i) {
            if (i == 0)
                dv_[i] = (v_[1] - v_[0]) / (h_[1] - h_[0]);
            else if (i == n - 1)
                dv_[i] = (v_[n-1] - v_[n-2]) / (h_[n-1] - h_[n-2]);
            else
                dv_[i] = (v_[i+1] - v_[i-1]) / (h_[i+1] - h_[i-1]);
        }
    }

    /// Linearly interpolate speed at height h.
    double speed(double h) const {
        return interp(h, h_, v_);
    }

    /// Linearly interpolate dv/dh at height h.
    double dspeed(double h) const {
        return interp(h, h_, dv_);
    }

    /// Construct from depth (positive downward) and speed arrays.
    static TableSpeedProfile from_depth(const double* depths,
                                         const double* speeds, int n) {
        // Sort by depth ascending, negate to h, reverse for ascending h
        std::vector<std::pair<double, double>> pairs(n);
        for (int i = 0; i < n; ++i) pairs[i] = {depths[i], speeds[i]};
        std::sort(pairs.begin(), pairs.end());
        std::vector<double> h(n), v(n);
        for (int i = 0; i < n; ++i) {
            h[n - 1 - i] = -pairs[i].first;   // negate + reverse
            v[n - 1 - i] = pairs[i].second;
        }
        return TableSpeedProfile(h.data(), v.data(), n);
    }

private:
    std::vector<double> h_, v_, dv_;
};

// -----------------------------------------------------------------
// Type-erased tracer (wraps EikonalTracer<function, function>)
// -----------------------------------------------------------------

using SpeedFunc = std::function<double(double)>;

class EikonalTracerErased {
public:
    EikonalTracerErased(SpeedFunc v, SpeedFunc dv)
        : tracer_(std::move(v), std::move(dv)) {}

    /// Construct from a table speed profile (pure C++, no callbacks).
    explicit EikonalTracerErased(const TableSpeedProfile& table)
        : tracer_(
            [table](double h) { return table.speed(h); },
            [table](double h) { return table.dspeed(h); }
          ) {}

    EikonalResult trace(const EikonalInput& inp,
                        const EikonalStopCondition& stop,
                        double rtol = 1e-10,
                        double atol = 1e-12,
                        double max_step = 100.0) const {
        return tracer_.trace(inp, stop, rtol, atol, max_step);
    }

    std::pair<EikonalResult, SensitivityResult>
    trace_with_sensitivities(const EikonalInput& inp,
                             const EikonalStopCondition& stop,
                             double rtol = 1e-10,
                             double atol = 1e-12,
                             double max_step = 100.0) const {
        return tracer_.trace_with_sensitivities(inp, stop, rtol, atol, max_step);
    }

private:
    EikonalTracer<SpeedFunc, SpeedFunc> tracer_;
};

// -----------------------------------------------------------------
// Batch I/O
// -----------------------------------------------------------------

struct BatchInput {
    double lat_deg, lon_deg, h_m;
    double elevation_deg, azimuth_deg;
    double travel_time_s;
};

struct BatchOutput {
    // Core state at endpoint
    Vec3 r, p;
    double s, T;
    // First derivatives at endpoint
    Vec3 r_prime, p_prime;
    double dh_ds, dv_ds;
    // Second derivatives at endpoint
    Vec3 r_double_prime;
    double curvature;
    // Geodetic at endpoint
    double lat_deg, lon_deg, h_m, v;
    // Application
    double theta_deg, azimuth_deg, bending_deg;
    // Travel-time derivatives
    Vec3 dr_dT, d2r_dT2, dp_dT;
    // Diagnostics
    double momentum_error;
};

// -----------------------------------------------------------------
// Batch trace — C++ loop, no Python GIL
// -----------------------------------------------------------------

inline std::vector<BatchOutput> trace_batch(
    const EikonalTracerErased& tracer,
    const std::vector<BatchInput>& inputs,
    double rtol = 1e-10,
    double atol = 1e-12,
    double max_step = 100.0)
{
    std::vector<BatchOutput> outputs(inputs.size());

    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& in = inputs[i];
        EikonalInput inp{in.lat_deg, in.lon_deg, in.h_m,
                         in.elevation_deg, in.azimuth_deg};
        EikonalStopCondition stop{
            .target_travel_time_s = in.travel_time_s,
            .detect_ground_hit = false
        };

        auto result = tracer.trace(inp, stop, rtol, atol, max_step);

        auto& out = outputs[i];
        out.r = result.r.back();
        out.p = result.p.back();
        out.s = result.s.back();
        out.T = result.T.back();
        out.r_prime = result.r_prime.back();
        out.p_prime = result.p_prime.back();
        out.dh_ds = result.dh_ds.back();
        out.dv_ds = result.dv_ds.back();
        out.r_double_prime = result.r_double_prime.back();
        out.curvature = result.curvature.back();
        out.lat_deg = result.lat_deg.back();
        out.lon_deg = result.lon_deg.back();
        out.h_m = result.h_m.back();
        out.v = result.v.back();
        out.theta_deg = result.theta_deg.back();
        out.azimuth_deg = result.azimuth_deg.back();
        out.bending_deg = result.bending_deg;
        out.dr_dT = result.dr_dT.back();
        out.d2r_dT2 = result.d2r_dT2.back();
        out.dp_dT = result.dp_dT.back();
        out.momentum_error = result.momentum_error.back();
    }

    return outputs;
}

// -----------------------------------------------------------------
// Full-path batch trace — returns complete EikonalResult per ray
// -----------------------------------------------------------------

inline std::vector<EikonalResult> trace_batch_full(
    const EikonalTracerErased& tracer,
    const std::vector<BatchInput>& inputs,
    double rtol = 1e-10,
    double atol = 1e-12,
    double max_step = 100.0)
{
    std::vector<EikonalResult> results(inputs.size());

    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& in = inputs[i];
        EikonalInput inp{in.lat_deg, in.lon_deg, in.h_m,
                         in.elevation_deg, in.azimuth_deg};
        EikonalStopCondition stop{
            .target_travel_time_s = in.travel_time_s,
            .detect_ground_hit = false
        };
        results[i] = tracer.trace(inp, stop, rtol, atol, max_step);
    }

    return results;
}

}  // namespace refraction
