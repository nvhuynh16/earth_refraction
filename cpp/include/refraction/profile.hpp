#pragma once

#include <cmath>
#include <vector>
#include <array>
#include "atmosphere/nrlmsis21.hpp"
#include "ciddor.hpp"
#include "itu_r_p453.hpp"
#include "species.hpp"
#include "water_vapor.hpp"
#include "surface_anchor.hpp"

namespace refraction {

// Finite difference step for atmospheric density derivatives (km)
inline constexpr double FD_STEP_KM = 0.01;

struct AtmosphericConditions {
    double day_of_year;       // 1-366
    double ut_seconds;        // UT seconds of day
    double latitude_deg;      // Geodetic latitude
    double longitude_deg;     // Geodetic longitude
    double f107a = 150.0;     // 81-day avg F10.7
    double f107  = 150.0;     // Daily F10.7
    std::array<double, 7> ap = {4,4,4,4,4,4,4};
};

enum class Mode { Ciddor, ITU_R_P453 };

struct RefractivityResult {
    double h_km;
    double n;               // Refractive index
    double N;               // Refractivity (n-1)×1e6
    double dn_dh;           // dn/dh (km⁻¹)
    double dN_dh;           // dN/dh (N-units/km)
    double temperature_K;
};

class RefractionProfile {
public:
    /// Construct with surface observation (anchored to real weather).
    RefractionProfile(const AtmosphericConditions& atm,
                      const SurfaceObservation& obs)
        : obs_(obs)
    {
        init_base_input(atm);
        anchor_ = compute_anchor(obs_, msis_, base_input_);
    }

    /// Construct without observation (unanchored, raw NRLMSIS prediction).
    explicit RefractionProfile(const AtmosphericConditions& atm)
    {
        init_base_input(atm);
        // Use the model's own surface conditions → zero anchoring correction
        auto out = msis_.msiscalc(base_input_);
        double T_model_K = out.tn;
        double n_total = 0.0;
        for (int i = 2; i <= 10; ++i)
            if (out.dn[i] > 0.0 && out.dn[i] < 1e30)
                n_total += out.dn[i];
        double P_model_Pa = n_total * kB * T_model_K;
        obs_ = SurfaceObservation{0.0, T_model_K - 273.15, P_model_Pa / 1000.0, 0.0};
        anchor_ = compute_anchor(obs_, msis_, base_input_);
    }

private:
    void init_base_input(const AtmosphericConditions& atm) {
        base_input_.day = static_cast<int>(atm.day_of_year);
        base_input_.utsec = atm.ut_seconds;
        base_input_.lat = atm.latitude_deg;
        base_input_.lon = atm.longitude_deg;
        base_input_.f107a = atm.f107a;
        base_input_.f107 = atm.f107;
        base_input_.ap = atm.ap;
    }

public:

    RefractivityResult compute(double h_km, Mode mode,
                               double wavelength_nm = 633.0) const {
        using namespace atmosphere;

        // Get NRLMSIS output with temperature derivative
        NRLMSIS21::Input inp = base_input_;
        inp.alt = h_km;
        auto full = msis_.msiscalc_with_derivative(inp);
        auto& out = full.output;

        // Apply surface anchoring
        double S = density_scale_at(h_km, anchor_);
        double dT = temperature_offset_at(h_km, anchor_);
        double T_K = out.tn + dT;
        double dT_dz = full.dT_dz;  // K/km from NRLMSIS spline derivative

        // Add temperature offset derivative (Gaussian taper)
        double dh = h_km - anchor_.h_surface_km;
        dT_dz += gaussian_taper_dT(dh);

        // Species number densities (m^-3) from NRLMSIS, scaled
        // NRLMSIS output indices: 1=mass_density, 2=N2, 3=O2, 4=O, 5=He, 6=H, 7=Ar, 8=N
        double n_N2 = out.dn[2] * S;
        double n_O2 = out.dn[3] * S;
        double n_O  = out.dn[4] * S;
        double n_He = out.dn[5] * S;
        double n_H  = out.dn[6] * S;
        double n_Ar = out.dn[7] * S;
        double n_N  = out.dn[8] * S;

        // Compute species density derivatives via finite difference on NRLMSIS
        constexpr double fd_dh = FD_STEP_KM;
        auto get_densities = [&](double alt) {
            NRLMSIS21::Input inp2 = base_input_;
            inp2.alt = alt;
            auto out2 = msis_.msiscalc(inp2);
            double S2 = density_scale_at(alt, anchor_);
            struct D { double N2, O2, O, He, H, Ar, N; };
            return D{out2.dn[2]*S2, out2.dn[3]*S2, out2.dn[4]*S2,
                     out2.dn[5]*S2, out2.dn[6]*S2, out2.dn[7]*S2, out2.dn[8]*S2};
        };
        auto dp = get_densities(h_km + fd_dh);
        auto dm = get_densities(h_km - fd_dh);
        double inv_2dh = 1.0 / (2.0 * fd_dh);
        double dn_N2_dh = (dp.N2 - dm.N2) * inv_2dh;
        double dn_O2_dh = (dp.O2 - dm.O2) * inv_2dh;
        double dn_O_dh  = (dp.O  - dm.O)  * inv_2dh;
        double dn_He_dh = (dp.He - dm.He) * inv_2dh;
        double dn_H_dh  = (dp.H  - dm.H)  * inv_2dh;
        double dn_Ar_dh = (dp.Ar - dm.Ar) * inv_2dh;
        double dn_N_dh  = (dp.N  - dm.N)  * inv_2dh;

        // Water vapor from anchored profile
        double n_H2O = anchored_H2O_density(h_km, anchor_);
        double dn_H2O_dh = (h_km >= anchor_.h_surface_km)
                          ? -n_H2O / anchor_.H_w : 0.0;

        RefractivityResult result;
        result.h_km = h_km;
        result.temperature_K = T_K;

        // Pack species densities and derivatives for refractivity helpers
        SpeciesDensities sd{n_N2, n_O2, n_Ar, n_He, n_O, n_N, n_H,
                            dn_N2_dh, dn_O2_dh, dn_Ar_dh, dn_He_dh,
                            dn_O_dh, dn_N_dh, dn_H_dh};

        if (mode == Mode::Ciddor) {
            auto [n_minus_1, dn_dh] = refractivity_optical(
                sd, n_H2O, dn_H2O_dh, wavelength_nm);
            result.n = 1.0 + n_minus_1;
            result.N = n_minus_1 * REFRACTIVITY_SCALE;
            result.dn_dh = dn_dh;
            result.dN_dh = dn_dh * REFRACTIVITY_SCALE;
        } else {
            auto [N_val, dN_dh] = refractivity_radio(
                sd, n_H2O, dn_H2O_dh, T_K, dT_dz);
            result.n = 1.0 + N_val / REFRACTIVITY_SCALE;
            result.N = N_val;
            result.dN_dh = dN_dh;
            result.dn_dh = dN_dh / REFRACTIVITY_SCALE;
        }

        return result;
    }

    std::vector<RefractivityResult> profile(
        double h_min_km, double h_max_km, double dh_km,
        Mode mode, double wavelength_nm = 633.0) const
    {
        std::vector<RefractivityResult> results;
        for (double h = h_min_km; h <= h_max_km + 0.5 * dh_km; h += dh_km) {
            results.push_back(compute(h, mode, wavelength_nm));
        }
        return results;
    }

private:
    struct SpeciesDensities {
        double n_N2, n_O2, n_Ar, n_He, n_O, n_N, n_H;
        double dn_N2, dn_O2, dn_Ar, dn_He, dn_O, dn_N, dn_H;
    };

    static std::pair<double, double> refractivity_optical(
            const SpeciesDensities& sd, double n_H2O, double dn_H2O,
            double wavelength_nm) {
        double um = wavelength_nm * 1e-3;
        double sigma2 = 1.0 / (um * um);

        double K_N2  = species::K_N2_optical(sigma2);
        double K_O2  = species::K_O2_optical(sigma2);
        double K_Ar  = species::K_Ar_optical(sigma2);
        double K_He  = species::K_He_optical(sigma2);
        double K_O   = species::K_O_optical(sigma2);
        double K_N   = species::K_N_optical(sigma2);
        double K_H   = species::K_H_optical(sigma2);
        double K_H2O = species::K_H2O_optical(sigma2);

        double n_minus_1 = sd.n_N2*K_N2 + sd.n_O2*K_O2 + sd.n_Ar*K_Ar
                         + sd.n_He*K_He + sd.n_O*K_O + sd.n_N*K_N + sd.n_H*K_H
                         + n_H2O*K_H2O;

        double dn_dh = K_N2*sd.dn_N2 + K_O2*sd.dn_O2 + K_Ar*sd.dn_Ar
                      + K_He*sd.dn_He + K_O*sd.dn_O + K_N*sd.dn_N
                      + K_H*sd.dn_H + K_H2O*dn_H2O;

        return {n_minus_1, dn_dh};
    }

    static std::pair<double, double> refractivity_radio(
            const SpeciesDensities& sd, double n_H2O, double dn_H2O,
            double T_K, double dT_dz) {
        double K_dry = species::K_radio_dry;
        double n_dry = sd.n_N2 + sd.n_O2 + sd.n_Ar + sd.n_He
                     + sd.n_O + sd.n_N + sd.n_H;
        double K_H2O_total = species::K_H2O_radio(T_K);
        double N_val = n_dry * K_dry + n_H2O * K_H2O_total;

        double dn_dry = sd.dn_N2 + sd.dn_O2 + sd.dn_Ar + sd.dn_He
                       + sd.dn_O + sd.dn_N + sd.dn_H;

        double K_dens = species::K_H2O_radio_density;
        double K_dip = species::K_H2O_radio_dipolar(T_K);
        double dK_dip_dh = -itu::k3 * kB * HPA_TO_PA / (T_K * T_K) * dT_dz;

        double dN_dh = K_dry * dn_dry + K_dens * dn_H2O
                     + K_dip * dn_H2O + n_H2O * dK_dip_dh;

        return {N_val, dN_dh};
    }

    double gaussian_taper_dT(double dh) const {
        return anchor_.temperature_offset
             * (-dh / (anchor_.H_T * anchor_.H_T))
             * std::exp(-dh * dh / (2.0 * anchor_.H_T * anchor_.H_T));
    }

    atmosphere::NRLMSIS21 msis_;
    atmosphere::NRLMSIS21::Input base_input_;
    SurfaceObservation obs_;
    AnchorParams anchor_;
};

}  // namespace refraction
