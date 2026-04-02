#pragma once

// Ocean depth-pressure conversion and parametric T(z), S(z) profiles.
// Depth-pressure: Saunders (1981) [1] via Fofonoff & Millard (1983) [2].
// Presets: WOA 2023 climatology [3], CalCOFI [4].
//
// [1] Saunders (1981), J. Phys. Oceanogr., 11, 573-574.
// [2] Fofonoff & Millard (1983), UNESCO Tech. Papers No. 44.
// [3] NOAA NCEI (2023), World Ocean Atlas 2023.
// [4] CalCOFI, California Cooperative Oceanic Fisheries Investigations.

#include <cmath>
#include <vector>
#include "../constants.hpp"
#include "../interpolation.hpp"

namespace refraction {
namespace ocean {

// ---------------------------------------------------------------------------
// Preset profiles
// ---------------------------------------------------------------------------

struct OceanPreset {
    double sst_C;        // sea surface temperature (degC)
    double sss_psu;      // sea surface salinity (PSU)
    double mld_m;        // mixed layer depth (m)
    double t_deep_C;     // deep water temperature (degC)
    double s_deep_psu;   // deep water salinity (PSU)
    double d_thermo_m;   // thermocline thickness scale (m)
};

constexpr OceanPreset FLORIDA_GULF_SUMMER = {30.0, 36.0, 25.0, 4.25, 35.0, 50.0};
constexpr OceanPreset FLORIDA_GULF_WINTER = {23.0, 36.2, 75.0, 4.25, 35.0, 50.0};
constexpr OceanPreset CALIFORNIA_SUMMER   = {19.0, 33.4, 20.0, 3.5,  34.5, 40.0};
constexpr OceanPreset CALIFORNIA_WINTER   = {14.0, 33.4, 45.0, 3.5,  34.5, 40.0};

// Default latitude for depth-pressure conversion (degrees)
inline constexpr double DEFAULT_LATITUDE_DEG = 45.0;

// Finite difference step for ocean gradients (meters)
inline constexpr double FD_STEP_OCEAN_M = 0.1;

// ---------------------------------------------------------------------------
// Ocean conditions (shared by OceanRefractionProfile and SoundSpeedProfile)
// ---------------------------------------------------------------------------

struct OceanConditions {
    double sst_C;                    // sea surface temperature (degC)
    double sss_psu;                  // sea surface salinity (PSU)
    double latitude_deg = DEFAULT_LATITUDE_DEG;  // for depth-pressure conversion
    double mld_m = 50.0;             // mixed layer depth (m)
    double t_deep_C = 4.0;           // deep water temperature (degC)
    double s_deep_psu = 35.0;        // deep water salinity (PSU)
    double d_thermo_m = 50.0;        // thermocline thickness scale (m)
    double d_halo_m = 50.0;          // halocline thickness scale (m)
    double freq_ghz = 10.0;          // microwave frequency (GHz)
    double wavelength_nm = 550.0;    // optical wavelength (nm)

    // Create conditions from a preset (halocline defaults to thermocline scale).
    static OceanConditions from_preset(const OceanPreset& p,
                                       double lat = DEFAULT_LATITUDE_DEG) {
        OceanConditions c;
        c.sst_C = p.sst_C;
        c.sss_psu = p.sss_psu;
        c.latitude_deg = lat;
        c.mld_m = p.mld_m;
        c.t_deep_C = p.t_deep_C;
        c.s_deep_psu = p.s_deep_psu;
        c.d_thermo_m = p.d_thermo_m;
        c.d_halo_m = p.d_thermo_m;
        return c;
    }
};

// ---------------------------------------------------------------------------
// Depth <-> Pressure [1] via [2]
// ---------------------------------------------------------------------------

// [1] Saunders (1981) coefficients: c1 accounts for latitude-dependent
// gravity, c2 for seawater compressibility.
inline std::pair<double, double> saunders_coefficients(double latitude_deg) {
    double sin_lat = std::sin(latitude_deg * DEG_TO_RAD);
    double sin2 = sin_lat * sin_lat;
    return {(5.92 + 5.25 * sin2) * 1.0e-3, 2.21e-6};
}

// [1] Saunders (1981): z = (1-c1)*p - c2*p^2
// UNESCO check value: p=10000 dbar, lat=30 -> z=9712.653 m
inline double pressure_to_depth(double p_dbar, double latitude_deg) {
    auto [c1, c2] = saunders_coefficients(latitude_deg);
    return (1.0 - c1) * p_dbar - c2 * p_dbar * p_dbar;
}

// Iteratively inverts pressure_to_depth via Newton's method
inline double depth_to_pressure(double depth_m, double latitude_deg) {
    if (depth_m <= 0.0) return 0.0;
    auto [c1, c2] = saunders_coefficients(latitude_deg);
    double p = depth_m / (1.0 - c1);  // initial guess
    for (int i = 0; i < 10; ++i) {
        double z = (1.0 - c1) * p - c2 * p * p;
        double err = z - depth_m;
        if (std::abs(err) < 1.0e-6) break;
        double dzdp = (1.0 - c1) - 2.0 * c2 * p;
        p -= err / dzdp;
    }
    return p;
}

// ---------------------------------------------------------------------------
// Parametric T(z), S(z) — tanh thermocline/halocline model
// ---------------------------------------------------------------------------

// T(z) = T_deep + (SST - T_deep) * [1 - tanh((z - MLD)/D)] / 2
inline double temperature_at_depth(double depth_m, double sst_C, double t_deep_C,
                                   double mld_m, double d_thermo_m) {
    return t_deep_C + (sst_C - t_deep_C) * 0.5
         * (1.0 - std::tanh((depth_m - mld_m) / d_thermo_m));
}

inline double salinity_at_depth(double depth_m, double sss_psu, double s_deep_psu,
                                double mld_m, double d_halo_m) {
    return s_deep_psu + (sss_psu - s_deep_psu) * 0.5
         * (1.0 - std::tanh((depth_m - mld_m) / d_halo_m));
}

using refraction::interp;  // from interpolation.hpp

// ---------------------------------------------------------------------------
// CRTP base class for ocean profile computations
// ---------------------------------------------------------------------------

// Derived must provide:
//   double scalar_at_depth(double depth_m, ModeT mode) const;
//   auto   compute_value(double depth_m, ModeT mode) const;
//   ResultT make_result(double depth_m, double p, double T, double S,
//                       auto value, double dv_dz) const;
template <typename Derived, typename ResultT, typename ModeT>
class OceanProfileBase {
public:
    ResultT compute(double depth_m, ModeT mode) const {
        auto [T, S] = ts_at_depth(depth_m);
        double p = depth_to_pressure(depth_m, cond_.latitude_deg);
        auto val = derived().compute_value(depth_m, mode);

        double dz = FD_STEP_OCEAN_M;
        double z_lo = (depth_m > dz) ? depth_m - dz : 0.0;
        double z_hi = depth_m + dz;
        double v_lo = derived().scalar_at_depth(z_lo, mode);
        double v_hi = derived().scalar_at_depth(z_hi, mode);
        double dv_dz = (v_hi - v_lo) / (z_hi - z_lo);

        return derived().make_result(depth_m, p, T, S, val, dv_dz);
    }

    std::vector<ResultT> profile(
        double z_min, double z_max, double dz, ModeT mode) const
    {
        std::vector<ResultT> results;
        for (double z = z_min; z <= z_max + dz * 0.5; z += dz)
            results.push_back(compute(z, mode));
        return results;
    }

protected:
    OceanConditions cond_{};
    std::vector<double> user_depths_;
    std::vector<double> user_temps_;
    std::vector<double> user_sals_;
    bool use_user_profile_ = false;

    void init_from_preset(const OceanPreset& preset, double latitude_deg) {
        cond_ = OceanConditions::from_preset(preset, latitude_deg);
    }

    void init_from_arrays(const std::vector<double>& depths_m,
                          const std::vector<double>& temperatures_C,
                          const std::vector<double>& salinities_psu,
                          double latitude_deg) {
        user_depths_ = depths_m;
        user_temps_ = temperatures_C;
        user_sals_ = salinities_psu;
        use_user_profile_ = true;
        cond_.sst_C = temperatures_C.front();
        cond_.sss_psu = salinities_psu.front();
        cond_.latitude_deg = latitude_deg;
    }

    std::pair<double, double> ts_at_depth(double depth_m) const {
        if (use_user_profile_) {
            double T = interp(depth_m, user_depths_, user_temps_);
            double S = interp(depth_m, user_depths_, user_sals_);
            return {T, S};
        }
        double T = temperature_at_depth(depth_m, cond_.sst_C, cond_.t_deep_C,
                                        cond_.mld_m, cond_.d_thermo_m);
        double S = salinity_at_depth(depth_m, cond_.sss_psu, cond_.s_deep_psu,
                                     cond_.mld_m, cond_.d_halo_m);
        return {T, S};
    }

private:
    const Derived& derived() const { return static_cast<const Derived&>(*this); }
};

}  // namespace ocean
}  // namespace refraction
