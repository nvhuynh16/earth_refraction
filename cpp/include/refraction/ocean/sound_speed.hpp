#pragma once

// Ocean sound speed profile computation.
// Main user-facing class: SoundSpeedProfile with three modes.
//
// Mirrors OceanRefractionProfile (ocean_refraction.hpp) for acoustic
// sound speed instead of electromagnetic refractive index.
//
// Modes:
//   - DelGrosso:   Del Grosso (1974) [1] / Wong & Zhu (1995) ITS-90 [2].
//                  Most accurate at depth.  T=0-30, S=30-40, P=0-9807 dbar.
//   - ChenMillero: Chen & Millero (1977) [3] / Wong & Zhu (1995) ITS-90 [2].
//                  Wider validity range.  T=0-40, S=0-40, P=0-10000 dbar.
//   - Mackenzie:   Mackenzie (1981) [4].  Simple 9-term equation using depth
//                  directly.  T=2-30, S=25-40, Z=0-8000 m.
//
// References:
//   [1] Del Grosso (1974), JASA 56(4), 1084-1091.
//   [2] Wong & Zhu (1995), JASA 97(3), 1732-1736.
//   [3] Chen & Millero (1977), JASA 62(5), 1129-1135.
//   [4] Mackenzie (1981), JASA 70(3), 807-812.

#include <cmath>
#include <vector>
#include <utility>

#include "del_grosso.hpp"
#include "chen_millero.hpp"
#include "mackenzie.hpp"
#include "ocean_profile.hpp"

namespace refraction {
namespace ocean {

enum class SoundMode { DelGrosso, ChenMillero, Mackenzie };

struct SoundSpeedResult {
    double depth_m;
    double pressure_dbar;
    double temperature_C;
    double salinity_psu;
    double sound_speed_m_s;     // m/s
    double dc_dz = 0.0;        // (m/s)/m, z positive downward
};

class SoundSpeedProfile
    : public OceanProfileBase<SoundSpeedProfile, SoundSpeedResult, SoundMode>
{
    using Base = OceanProfileBase<SoundSpeedProfile, SoundSpeedResult, SoundMode>;
    friend Base;

public:
    // Construct from OceanConditions (freq/wavelength fields ignored).
    explicit SoundSpeedProfile(const OceanConditions& cond) {
        cond_ = cond;
    }

    // Construct from preset.
    SoundSpeedProfile(const OceanPreset& preset,
                      double latitude_deg = DEFAULT_LATITUDE_DEG)
    {
        init_from_preset(preset, latitude_deg);
    }

    // Construct from user-supplied T(z), S(z) arrays.
    SoundSpeedProfile(const std::vector<double>& depths_m,
                      const std::vector<double>& temperatures_C,
                      const std::vector<double>& salinities_psu,
                      double latitude_deg = DEFAULT_LATITUDE_DEG)
    {
        init_from_arrays(depths_m, temperatures_C, salinities_psu, latitude_deg);
    }

private:
    double compute_value(double depth_m, SoundMode mode) const {
        return c_at_depth(depth_m, mode);
    }

    double scalar_at_depth(double depth_m, SoundMode mode) const {
        return compute_value(depth_m, mode);
    }

    SoundSpeedResult make_result(double depth_m, double p, double T,
                                  double S, double c, double dc_dz) const {
        return {depth_m, p, T, S, c, dc_dz};
    }

    double c_at_depth(double depth_m, SoundMode mode) const {
        auto [T, S] = ts_at_depth(depth_m);

        if (mode == SoundMode::Mackenzie) {
            return mk::sound_speed(T, S, depth_m);
        }

        // Del Grosso and Chen-Millero need pressure
        double p = depth_to_pressure(depth_m, cond_.latitude_deg);

        if (mode == SoundMode::DelGrosso) {
            return dg::sound_speed(T, S, p);
        } else {  // ChenMillero
            return cm::sound_speed(T, S, p);
        }
    }
};

}  // namespace ocean
}  // namespace refraction
