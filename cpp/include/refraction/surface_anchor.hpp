#pragma once

// Surface anchoring: tie NRLMSIS 2.1 profiles to surface observations.
//
// Temperature offset (Gaussian taper):
//   dT(h) = dT_0 · exp(-(h - h_s)² / (2·H_T²)),  dT_0 = T_obs - T_model
//
// Density scaling (hydrostatic propagation):
//   S(h) = S_P · exp(I(h)) · T_model(h) / T_anch(h),  T_anch = T_model + dT
//   S_P  = (P_obs - e_obs) / P_model               dry partial-pressure ratio
//   I(h) = (M̄·g0/kB) ∫_{ζ_s}^{ζ(h)} [1/T_model - 1/T_anch] dζ'
//
// I(h) is the exact hydrostatic correction in geopotential height ζ (as
// NRLMSIS integrates it); T_model/T_anch converts pressure ratio to
// number-density ratio.  At the surface S_0 = S_P · T_model/T_obs, so the
// anchored dry air plus water vapor reproduce P_obs and T_obs exactly.
// ln S(h) and its exact derivative are tabulated and evaluated by cubic
// Hermite interpolation (C¹-smooth, for clean dn/dh).

#include <algorithm>
#include <cmath>
#include <array>
#include <stdexcept>
#include <vector>
#include "constants.hpp"
#include "water_vapor.hpp"
#include "atmosphere/nrlmsis21.hpp"

namespace refraction {

struct SurfaceObservation {
    double altitude_km;
    double temperature_C;
    double pressure_kPa;              // Total (moist-air) pressure
    double relative_humidity;         // 0.0 to 1.0
    double temperature_taper_km = 2.0;  // Gaussian taper scale H_T (km)
};

struct AnchorParams {
    double density_scale;      // S_0 = (P_obs - e) / P_model · T_model / T_obs
    double pressure_scale;     // S_P = (P_obs - e) / P_model
    double temperature_offset; // ΔT = T_obs - T_model (K)
    double n_H2O_surface;      // Surface water vapor number density (m^-3)
    double h_surface_km;       // Surface altitude
    double H_w;                // Water vapor scale height (km)
    double H_w_below;          // H2O gradient taper scale below the surface (km)
    double H_T;                // Temperature taper scale (km)
    std::vector<double> table_h_km;       // Uniform altitude nodes (km)
    std::vector<double> table_ln_scale;   // ln S(h) at nodes
    std::vector<double> table_dln_scale;  // d(ln S)/dh at nodes (km^-1)
};

namespace detail {
// Hydrostatic table extent: the temperature offset is below
// exp(-18) ≈ 1.5e-8 of its surface value beyond 6 taper scale heights.
inline constexpr double TABLE_HALF_WIDTH_TAPERS = 6.0;
// NRLMSIS temperatures are unreliable well below sea level, so the table
// does not extend more than this far below min(h_s, 0).
inline constexpr double TABLE_BELOW_SURFACE_KM = 1.0;
// Table node spacing: at most this, and at most H_T / 40.
inline constexpr double TABLE_MAX_STEP_KM = 0.05;

// Cubic Hermite interpolation on a uniform grid with known derivatives.
// Below the grid: linear extrapolation with the end slope.  Above the
// grid: hold the end value (the hydrostatic correction has converged).
inline double hermite_interp(double x, const std::vector<double>& xp,
                             const std::vector<double>& fp,
                             const std::vector<double>& dfp) {
    if (x < xp.front()) return fp.front() + dfp.front() * (x - xp.front());
    if (x >= xp.back()) return fp.back();
    double step = xp[1] - xp[0];
    double u = (x - xp[0]) / step;
    size_t i = std::min(static_cast<size_t>(u), xp.size() - 2);
    double t = std::clamp(u - static_cast<double>(i), 0.0, 1.0);
    double t2 = t * t, t3 = t2 * t;
    return (2.0*t3 - 3.0*t2 + 1.0) * fp[i]
         + (t3 - 2.0*t2 + t) * step * dfp[i]
         + (-2.0*t3 + 3.0*t2) * fp[i + 1]
         + (t3 - t2) * step * dfp[i + 1];
}
}  // namespace detail

// Compute anchor parameters from surface observation and NRLMSIS model.
// NRLMSIS models dry air only, so it is compared with the observed dry
// partial pressure P_obs - e_obs; water vapor is added separately.
inline AnchorParams compute_anchor(const SurfaceObservation& obs,
                                    const atmosphere::NRLMSIS21& msis,
                                    const atmosphere::NRLMSIS21::Input& base_input) {
    using atmosphere::msis21::utils::alt2gph;
    using atmosphere::msis21::utils::dgph_dalt;
    using atmosphere::msis21::constants::MBARG0DIVKB;

    const double H_T = obs.temperature_taper_km;
    if (!(H_T > 0.0))
        throw std::invalid_argument("temperature_taper_km must be positive");

    const double h_s = obs.altitude_km;
    const double T_obs_K = obs.temperature_C + 273.15;
    const double P_obs_Pa = obs.pressure_kPa * 1000.0;

    // Altitude grid with the surface exactly on node i_s
    const double step = std::min(detail::TABLE_MAX_STEP_KM, H_T / 40.0);
    const double h_lo = std::max(h_s - detail::TABLE_HALF_WIDTH_TAPERS * H_T,
                                 std::min(h_s, 0.0) - detail::TABLE_BELOW_SURFACE_KM);
    const double h_hi = h_s + detail::TABLE_HALF_WIDTH_TAPERS * H_T;
    const int n_below = static_cast<int>(std::ceil((h_s - h_lo) / step - 1e-9));
    const int n_above = static_cast<int>(std::ceil((h_hi - h_s) / step - 1e-9));
    const size_t n = static_cast<size_t>(n_below + n_above + 1);
    const size_t i_s = static_cast<size_t>(n_below);

    std::vector<double> h(n), T_m(n), dTm_dzeta(n);
    double P_model_Pa = 0.0;
    atmosphere::NRLMSIS21::Input inp = base_input;
    for (size_t k = 0; k < n; ++k) {
        h[k] = h_s + step * (static_cast<double>(k) - n_below);
        inp.alt = h[k];
        auto full = msis.msiscalc_with_derivative(inp);
        T_m[k] = full.output.tn;
        dTm_dzeta[k] = full.dT_dz;  // K per geopotential km
        if (k == i_s) {
            // Model total number density (sum of species)
            // NRLMSIS output: dn[1]=mass_density, dn[2]=N2, dn[3]=O2, dn[4]=O,
            //   dn[5]=He, dn[6]=H, dn[7]=Ar, dn[8]=N, dn[9]=anomO, dn[10]=NO
            // Skip dn[1] (mass density, not number density)
            double n_model_total = 0.0;
            for (int i = 2; i <= 10; i++) {
                if (full.output.dn[i] > 0.0 && full.output.dn[i] < 1e30)
                    n_model_total += full.output.dn[i];
            }
            P_model_Pa = n_model_total * kB * T_m[k];
        }
    }

    const double T_model_K = T_m[i_s];
    const double n_H2O_surface =
        water_vapor_number_density(T_obs_K, P_obs_Pa, obs.relative_humidity);
    const double e_obs_Pa = n_H2O_surface * kB * T_obs_K;
    const double temperature_offset = T_obs_K - T_model_K;
    const double pressure_scale = (P_obs_Pa - e_obs_Pa) / P_model_Pa;

    // Hydrostatic integrand f = dI/dζ and ln S, d(ln S)/dh at each node
    std::vector<double> zeta(n), f(n), ln_scale(n), dln_scale(n);
    for (size_t k = 0; k < n; ++k) {
        double dh = h[k] - h_s;
        double dT = temperature_offset * std::exp(-dh * dh / (2.0 * H_T * H_T));
        double ddT_dh = -dT * dh / (H_T * H_T);
        double T_a = T_m[k] + dT;
        zeta[k] = alt2gph(base_input.lat, h[k]);
        double dzeta_dh = dgph_dalt(base_input.lat, h[k]);
        f[k] = MBARG0DIVKB * (1.0 / T_m[k] - 1.0 / T_a);
        double dTm_dh = dTm_dzeta[k] * dzeta_dh;
        ln_scale[k] = std::log(T_m[k] / T_a);  // ln S_P + I added below
        dln_scale[k] = f[k] * dzeta_dh + dTm_dh / T_m[k] - (dTm_dh + ddT_dh) / T_a;
    }

    // Trapezoidal integration of f dζ outward from the surface node
    std::vector<double> I(n, 0.0);
    for (size_t k = i_s + 1; k < n; ++k)
        I[k] = I[k - 1] + 0.5 * (f[k] + f[k - 1]) * (zeta[k] - zeta[k - 1]);
    for (size_t k = i_s; k-- > 0;)
        I[k] = I[k + 1] - 0.5 * (f[k] + f[k + 1]) * (zeta[k + 1] - zeta[k]);

    const double ln_SP = std::log(pressure_scale);
    for (size_t k = 0; k < n; ++k) ln_scale[k] += ln_SP + I[k];

    AnchorParams params;
    params.density_scale = pressure_scale * T_model_K / T_obs_K;
    params.pressure_scale = pressure_scale;
    params.temperature_offset = temperature_offset;
    params.n_H2O_surface = n_H2O_surface;
    params.h_surface_km = h_s;
    params.H_w = 2.0;   // km
    params.H_w_below = H2O_BELOW_TAPER_KM;
    params.H_T = H_T;   // km
    params.table_h_km = std::move(h);
    params.table_ln_scale = std::move(ln_scale);
    params.table_dln_scale = std::move(dln_scale);
    return params;
}

// Density scale factor S(h) with hydrostatic propagation
inline double density_scale_at(double h_km, const AnchorParams& p) {
    return std::exp(detail::hermite_interp(
        h_km, p.table_h_km, p.table_ln_scale, p.table_dln_scale));
}

// Apply temperature offset with altitude taper
inline double temperature_offset_at(double h_km, const AnchorParams& p) {
    double dh = h_km - p.h_surface_km;
    return p.temperature_offset * std::exp(-dh * dh / (2.0 * p.H_T * p.H_T));
}

// Water vapor number density at altitude (exponential decay from surface)
inline double anchored_H2O_density(double h_km, const AnchorParams& p) {
    return water_vapor_profile(p.n_H2O_surface, h_km, p.h_surface_km, p.H_w, p.H_w_below);
}

// Altitude derivative of anchored_H2O_density (m^-3 km^-1)
inline double anchored_H2O_gradient(double h_km, const AnchorParams& p) {
    return water_vapor_profile_derivative(p.n_H2O_surface, h_km, p.h_surface_km,
                                          p.H_w, p.H_w_below);
}

}  // namespace refraction
