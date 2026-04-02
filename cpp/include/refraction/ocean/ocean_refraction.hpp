#pragma once

// Ocean refractive index profile computation.
// Main user-facing class: OceanRefractionProfile with four modes.

#include <cmath>
#include <vector>

#include "meissner_wentz.hpp"
#include "millard_seaver.hpp"
#include "quan_fry.hpp"
#include "iapws_r9_97.hpp"
#include "ocean_profile.hpp"

namespace refraction {
namespace ocean {

enum class OceanMode { MeissnerWentz, MillardSeaver, QuanFry, IAPWS };

struct OceanNValue {
    double n_real;
    double n_imag = 0.0;
    double eps_real = 0.0;
    double eps_imag = 0.0;
};

struct OceanRefractivityResult {
    double depth_m;
    double pressure_dbar;
    double temperature_C;
    double salinity_psu;
    double n_real;
    double n_imag = 0.0;     // radio only
    double eps_real = 0.0;   // radio only
    double eps_imag = 0.0;   // radio only
    double dn_dz = 0.0;     // per meter, z positive downward
};

class OceanRefractionProfile
    : public OceanProfileBase<OceanRefractionProfile, OceanRefractivityResult, OceanMode>
{
    using Base = OceanProfileBase<OceanRefractionProfile, OceanRefractivityResult, OceanMode>;
    friend Base;

public:
    explicit OceanRefractionProfile(const OceanConditions& cond) {
        cond_ = cond;
    }

    OceanRefractionProfile(const OceanPreset& preset,
                           double freq_ghz = 10.0,
                           double wavelength_nm = 550.0,
                           double latitude_deg = DEFAULT_LATITUDE_DEG)
    {
        init_from_preset(preset, latitude_deg);
        cond_.freq_ghz = freq_ghz;
        cond_.wavelength_nm = wavelength_nm;
    }

    OceanRefractionProfile(const std::vector<double>& depths_m,
                           const std::vector<double>& temperatures_C,
                           const std::vector<double>& salinities_psu,
                           double freq_ghz = 10.0,
                           double wavelength_nm = 550.0,
                           double latitude_deg = DEFAULT_LATITUDE_DEG)
    {
        init_from_arrays(depths_m, temperatures_C, salinities_psu, latitude_deg);
        cond_.freq_ghz = freq_ghz;
        cond_.wavelength_nm = wavelength_nm;
    }

private:
    OceanNValue compute_value(double depth_m, OceanMode mode) const {
        auto [T, S] = ts_at_depth(depth_m);
        double p = depth_to_pressure(depth_m, cond_.latitude_deg);

        if (mode == OceanMode::MeissnerWentz) {
            auto eps = mw::permittivity(cond_.freq_ghz, T, S);
            auto n = mw::refractive_index(cond_.freq_ghz, T, S);
            return {n.n_real, n.n_imag, eps.real, eps.imag};
        } else if (mode == OceanMode::MillardSeaver) {
            double n = ms::refractive_index(T, S, p, cond_.wavelength_nm);
            return {n};
        } else if (mode == OceanMode::QuanFry) {
            double n = qf::refractive_index(T, S, cond_.wavelength_nm);
            return {n};
        } else {  // IAPWS
            double n = iapws::refractive_index(T, cond_.wavelength_nm, -1.0, p);
            return {n};
        }
    }

    double scalar_at_depth(double depth_m, OceanMode mode) const {
        return compute_value(depth_m, mode).n_real;
    }

    OceanRefractivityResult make_result(double depth_m, double p, double T,
                                        double S, const OceanNValue& val,
                                        double dv_dz) const {
        return {depth_m, p, T, S, val.n_real, val.n_imag, val.eps_real, val.eps_imag, dv_dz};
    }
};

}  // namespace ocean
}  // namespace refraction
