#pragma once

#include <cmath>
#include "constants.hpp"
#include "ciddor.hpp"
#include "itu_r_p453.hpp"

namespace refraction {

namespace species {

// Loschmidt number: number density at STP (0°C, 101325 Pa)
constexpr double N_L = 2.6867774e25;  // m^-3

// Number density at Ciddor standard conditions (15°C, 101325 Pa)
// n = P/(kB*T) = 101325 / (1.380649e-23 * 288.15) = 2.5470e25
constexpr double N_std = 2.54708e25;  // m^-3

// Molecular polarizabilities in a0^3 (atomic units)
// 1 a0^3 = 1.6488e-41 m^3 (volume polarizability)
constexpr double a0_cubed = 1.6488e-41;  // m^3

constexpr double alpha_N2  = 11.74;   // a0^3
constexpr double alpha_O2  = 10.67;   // a0^3
constexpr double alpha_Ar  = 11.08;   // a0^3
constexpr double alpha_He  = 1.38;    // a0^3
constexpr double alpha_H   = 4.50;    // a0^3
constexpr double alpha_O   = 5.40;    // a0^3
constexpr double alpha_N   = 7.40;    // a0^3
constexpr double alpha_NO  = 11.50;   // a0^3

// Standard dry air mole fractions
constexpr double f_N2 = 0.78084;
constexpr double f_O2 = 0.20946;
constexpr double f_Ar = 0.00934;

// Weighted average polarizability of dry air
constexpr double alpha_air = f_N2 * alpha_N2 + f_O2 * alpha_O2 + f_Ar * alpha_Ar;

// ---------------------------------------------------------------------------
// Optical specific refractivity K_i(σ²)
// n - 1 = Σ n_i × K_i  where n_i is number density (m^-3)
// ---------------------------------------------------------------------------

// For major dry-air species: decompose Ciddor standard-air refractivity
// n_axs(σ²) = Σ f_j × n_std × K_j(σ²)  =>  K_j(σ²) = n_axs(σ²) × (α_j / α_air) / n_std
inline double K_N2_optical(double sigma2) {
    return ciddor::n_standard_air(sigma2) * (alpha_N2 / alpha_air) / N_std;
}

inline double K_O2_optical(double sigma2) {
    return ciddor::n_standard_air(sigma2) * (alpha_O2 / alpha_air) / N_std;
}

inline double K_Ar_optical(double sigma2) {
    return ciddor::n_standard_air(sigma2) * (alpha_Ar / alpha_air) / N_std;
}

// For minor/atomic species: K_i = (2π/ε₀) × α_i_SI / (4πε₀) simplified to
// K_i = (4π/3) × α_i_SI for Lorentz-Lorenz ≈ 2π α_i_SI for n≈1
// More precisely: K_i = (2π) × α_i_vol where α_vol = α(a0³) × a0_cubed
// But to be consistent with the major species, use the same proportionality:
// K_i = K_N2 × (α_i / α_N2)
inline double K_He_optical(double sigma2) {
    return K_N2_optical(sigma2) * (alpha_He / alpha_N2);
}

inline double K_H_optical(double sigma2) {
    return K_N2_optical(sigma2) * (alpha_H / alpha_N2);
}

inline double K_O_optical(double sigma2) {
    return K_N2_optical(sigma2) * (alpha_O / alpha_N2);
}

inline double K_N_optical(double sigma2) {
    return K_N2_optical(sigma2) * (alpha_N / alpha_N2);
}

inline double K_NO_optical(double sigma2) {
    return K_N2_optical(sigma2) * (alpha_NO / alpha_N2);
}

// Water vapor: from Ciddor water vapor dispersion
// n_ws(σ²) is the refractivity of water vapor at reference density ρ_ws
// Reference: 20°C, 1333 Pa => N_ws = 1333/(kB*293.15) = 3.292e23 m^-3
constexpr double N_ws_ref = 3.2921e23;  // m^-3

inline double K_H2O_optical(double sigma2) {
    return ciddor::n_water_vapor(sigma2) / N_ws_ref;
}

// ---------------------------------------------------------------------------
// Radio specific refractivity K_i
// N = (n-1)×1e6 = Σ n_i × K_i_radio  (K has units that give N-units per m^-3)
// ---------------------------------------------------------------------------

// From ITU-R: N = k1*Pd/T + k2*e/T + k3*e/T²
// For dry species: Pd = n_dry × kB × T / 100 (converting Pa to hPa)
// So N_dry = k1 × (n_dry × kB / 100)  => K_i_radio = k1 × kB / 100
// Factor 1e6 is already in N definition, and k1 is in K/hPa
// kB, HPA_TO_PA from constants.hpp (included via ciddor.hpp -> ... or directly)
constexpr double K_dry_radio = itu::k1 * kB * 10.0;  // 1/m^-3 (×1e6 for N-units)
// k1 in K/hPa, kB in J/K=Pa·m³/K, 10 = 1e6/1e5 (1e6 for N-units, 1e5 Pa/hPa... wait)
// N = k1 * Pd_hPa / T.  Pd_hPa = n*kB*T * 1e-2 (Pa to hPa).
// So N = k1 * n * kB * 1e-2 = n * (k1*kB*1e-2)
// But we want K such that (n-1)*1e6 = Σ n_i*K_i, so K_dry = k1*kB*1e-2  (already gives N-units)

// Let me redo this carefully:
// N = k1*Pd/T where Pd in hPa, T in K
// Pd = Σ n_i * kB * T  [Pa]  = Σ n_i * kB * T * 0.01 [hPa]
// So N_dry = k1 * Σ n_i * kB * T * 0.01 / T = k1 * kB * 0.01 * Σ n_i
// K_i_radio_dry = k1 * kB * 0.01  (same for all dry species)

constexpr double K_radio_dry = itu::k1 * kB * HPA_TO_PA;

// For water vapor density term: N_density = k2*e/T
// e = n_w * kB * T [Pa] = n_w * kB * T * HPA_TO_PA [hPa]
// N_density = k2 * n_w * kB * HPA_TO_PA
constexpr double K_H2O_radio_density = itu::k2 * kB * HPA_TO_PA;

// For water vapor dipolar term: N_dipolar = k3*e/T²
// N_dipolar = k3 * n_w * kB * HPA_TO_PA / T
// This term depends on T, so K_H2O_radio_dipolar(T) = k3 * kB * HPA_TO_PA / T
inline double K_H2O_radio_dipolar(double T_K) {
    return itu::k3 * kB * HPA_TO_PA / T_K;
}

// Total water vapor radio K
inline double K_H2O_radio(double T_K) {
    return K_H2O_radio_density + K_H2O_radio_dipolar(T_K);
}

}  // namespace species

}  // namespace refraction
