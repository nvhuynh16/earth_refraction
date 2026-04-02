#pragma once

#include <cmath>
#include "water_vapor.hpp"

namespace refraction {

namespace ciddor {

// Dispersion constants for standard dry air (Ciddor 1996, Table 1)
constexpr double k0 = 238.0185;    // μm^-2
constexpr double k1 = 5792105.0;   // μm^-2
constexpr double k2 = 57.362;      // μm^-2
constexpr double k3 = 167917.0;    // μm^-2

// Water vapor dispersion constants (Ciddor 1996, Table 2)
constexpr double w0 = 295.235;
constexpr double w1 = 2.6422;
constexpr double w2 = -0.032380;
constexpr double w3 = 0.004028;
constexpr double cf = 1.022e-8;

// Compressibility Z coefficients (CIPM-2007 / Ciddor 1996)
constexpr double a0 = 1.58123e-6;   // K/Pa
constexpr double a1 = -2.9331e-8;   // 1/Pa
constexpr double a2 = 1.1043e-10;   // 1/(K*Pa)
constexpr double b0 = 5.707e-6;     // K/Pa
constexpr double b1 = -2.051e-8;    // 1/Pa
constexpr double c0 = 1.9898e-4;    // K/Pa
constexpr double c1 = -2.376e-6;    // 1/Pa
constexpr double d  = 1.83e-11;     // K^2/Pa^2
constexpr double e  = -0.765e-8;    // K^2/Pa^2

// Reference conditions for standard air
constexpr double T_ref = 288.15;      // K (15°C)
constexpr double P_ref = 101325.0;    // Pa
constexpr double T_ref_w = 293.15;    // K (20°C) for water vapor
constexpr double P_ref_w = 1333.0;    // Pa (10 Torr) — Barrell & Sears reference for w0-w3

// Gas constant
constexpr double R = 8.314462618;     // J/(mol·K)

// Molar masses (kg/mol)
constexpr double M_a = 28.9635e-3;    // Dry air (adjusted for CO2)
constexpr double M_w = 18.01528e-3;   // Water

// Refractivity of standard dry air at reference conditions
inline double n_standard_air(double sigma2) {
    // sigma2 = sigma^2 where sigma = 1/lambda in μm^-1
    return 1e-8 * (k1 / (k0 - sigma2) + k3 / (k2 - sigma2));
}

// Refractivity of water vapor at reference conditions
inline double n_water_vapor(double sigma2) {
    return cf * (w0 + w1 * sigma2 + w2 * sigma2 * sigma2 + w3 * sigma2 * sigma2 * sigma2);
}

// Compressibility factor Z
inline double compressibility_Z(double T_K, double P_Pa, double x_w) {
    double t = T_K - 273.15;
    return 1.0 - (P_Pa / T_K) * (a0 + a1 * t + a2 * t * t + (b0 + b1 * t) * x_w
           + (c0 + c1 * t) * x_w * x_w) + (P_Pa / T_K) * (P_Pa / T_K) * (d + e * x_w * x_w);
}

// Full Ciddor refractive index calculation
// T_C: temperature (°C), P_kPa: pressure (kPa), RH: relative humidity (0-1)
// lambda_nm: wavelength (nm), xCO2: CO2 mole fraction (default 450 ppm)
inline double ciddor_n(double T_C, double P_kPa, double RH, double lambda_nm, double xCO2 = 450e-6) {
    double T_K = T_C + 273.15;
    double P_Pa = P_kPa * 1000.0;

    // Wavenumber squared (μm^-2)
    double lambda_um = lambda_nm * 1e-3;
    double sigma = 1.0 / lambda_um;
    double sigma2 = sigma * sigma;

    // Refractivity of standard air (at 15°C, 101325 Pa, dry, 450 ppm CO2)
    double n_as = n_standard_air(sigma2);

    // CO2 correction: adjust for actual CO2 from 450 ppm (Eq. 2)
    double co2_corr = 1.0 + 0.534e-6 * (xCO2 * 1e6 - 450.0);
    double n_axs = n_as * co2_corr;

    // Water vapor refractivity at reference (20°C, 1333 Pa)
    double n_ws = n_water_vapor(sigma2);

    // Mole fraction of water vapor
    double x_w = water_vapor_mole_fraction(T_C, P_Pa, RH);

    // Compressibility factors
    double Z_a = compressibility_Z(T_ref, P_ref, 0.0);       // Standard dry air
    double Z   = compressibility_Z(T_K, P_Pa, x_w);          // Ambient moist air
    double Z_w = compressibility_Z(T_ref_w, P_ref_w, 1.0);   // Water vapor reference

    // Molar mass of dry air (adjusted for CO2)
    double M_a_adj = (28.9635 + 12.011 * (xCO2 - 0.0004)) * 1e-3;

    // Density of standard dry air reference:  ρ_axs = P_ref M_a / (Z_a R T_ref)
    double rho_axs = P_ref * M_a_adj / (Z_a * R * T_ref);

    // Density of dry air component:  ρ_a = P(1-x_w) M_a' / (Z R T)
    double rho_a = P_Pa * (1.0 - x_w) * M_a_adj / (Z * R * T_K);

    // Density of water vapor reference:  ρ_ws = P_ref_w M_w / (Z_w R T_ref_w)
    double rho_ws = P_ref_w * M_w / (Z_w * R * T_ref_w);

    // Density of water vapor component:  ρ_w = P x_w M_w / (Z R T)
    double rho_w = P_Pa * x_w * M_w / (Z * R * T_K);

    // Ciddor 1996 eq.: n - 1 = (ρ_a/ρ_axs)(n_axs-1) + (ρ_w/ρ_ws)(n_ws-1)
    double n_minus_1 = (rho_a / rho_axs) * n_axs + (rho_w / rho_ws) * n_ws;

    return 1.0 + n_minus_1;
}

}  // namespace ciddor

// Convenience wrapper
inline double ciddor_refractive_index(double T_C, double P_kPa, double RH, double lambda_nm, double xCO2 = 450e-6) {
    return ciddor::ciddor_n(T_C, P_kPa, RH, lambda_nm, xCO2);
}

}  // namespace refraction
