#pragma once

#include <cmath>
#include <array>
#include "constants.hpp"
#include "water_vapor.hpp"
#include "atmosphere/nrlmsis21.hpp"

namespace refraction {

struct SurfaceObservation {
    double altitude_km;
    double temperature_C;
    double pressure_kPa;
    double relative_humidity;  // 0.0 to 1.0
};

struct AnchorParams {
    double density_scale;     // S = P_obs / P_model
    double temperature_offset; // ΔT = T_obs - T_model (K)
    double n_H2O_surface;     // Surface water vapor number density (m^-3)
    double h_surface_km;      // Surface altitude
    double H_w;               // Water vapor scale height (km)
    double H_T;               // Temperature taper scale (km)
    double hydro_C;            // Hydrostatic propagation constant
};

// Compute anchor parameters from surface observation and NRLMSIS model
inline AnchorParams compute_anchor(const SurfaceObservation& obs,
                                    const atmosphere::NRLMSIS21& msis,
                                    const atmosphere::NRLMSIS21::Input& base_input) {
    // kB from constants.hpp

    atmosphere::NRLMSIS21::Input inp = base_input;
    inp.alt = obs.altitude_km;
    auto out = msis.msiscalc(inp);

    // Model temperature and pressure at surface
    double T_model_K = out.tn;
    double T_obs_K = obs.temperature_C + 273.15;

    // Model total number density (sum of species)
    // NRLMSIS output: dn[1]=mass_density, dn[2]=N2, dn[3]=O2, dn[4]=O,
    //   dn[5]=He, dn[6]=H, dn[7]=Ar, dn[8]=N, dn[9]=anomO, dn[10]=NO
    // Skip dn[1] (mass density, not number density)
    double n_model_total = 0.0;
    for (int i = 2; i <= 10; i++) {
        if (out.dn[i] > 0.0 && out.dn[i] < 1e30) {
            n_model_total += out.dn[i];
        }
    }

    double P_model_Pa = n_model_total * kB * T_model_K;
    double P_obs_Pa = obs.pressure_kPa * 1000.0;

    double temperature_offset = T_obs_K - T_model_K;
    constexpr double H_T = 15.0;  // km

    AnchorParams params;
    params.density_scale = P_obs_Pa / P_model_Pa;
    params.temperature_offset = temperature_offset;
    params.h_surface_km = obs.altitude_km;
    params.H_w = 2.0;   // km
    params.H_T = H_T;   // km
    params.hydro_C = atmosphere::msis21::constants::MBARG0DIVKB * temperature_offset * H_T
                     * std::sqrt(PI / 2.0) / (T_obs_K * T_obs_K);

    // Surface water vapor number density from observations
    params.n_H2O_surface = water_vapor_number_density(T_obs_K, P_obs_Pa, obs.relative_humidity);

    return params;
}

// Apply density scale factor with hydrostatic propagation
inline double density_scale_at(double h_km, const AnchorParams& p) {
    double dh = h_km - p.h_surface_km;
    double erf_arg = dh / (p.H_T * std::sqrt(2.0));
    return p.density_scale * std::exp(p.hydro_C * std::erf(erf_arg));
}

// Apply temperature offset with altitude taper
inline double temperature_offset_at(double h_km, const AnchorParams& p) {
    double dh = h_km - p.h_surface_km;
    return p.temperature_offset * std::exp(-dh * dh / (2.0 * p.H_T * p.H_T));
}

// Water vapor number density at altitude (exponential decay from surface)
inline double anchored_H2O_density(double h_km, const AnchorParams& p) {
    return water_vapor_profile(p.n_H2O_surface, h_km, p.h_surface_km, p.H_w);
}

}  // namespace refraction
