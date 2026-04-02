#pragma once

#include <cmath>
#include "water_vapor.hpp"

namespace refraction {

namespace itu {

// Rueger 2002 "best average" coefficients
constexpr double k1 = 77.6890;    // K/hPa
constexpr double k2 = 71.2952;    // K/hPa
constexpr double k3 = 375463.0;   // K²/hPa

// Radio refractivity N = (n-1) × 1e6
// Pd_hPa: dry air partial pressure (hPa), e_hPa: water vapor partial pressure (hPa)
inline double itu_N(double T_K, double Pd_hPa, double e_hPa) {
    return k1 * Pd_hPa / T_K + k2 * e_hPa / T_K + k3 * e_hPa / (T_K * T_K);
}

// Convenience: compute N from surface observations
inline double itu_N_from_surface(double T_C, double P_kPa, double RH) {
    double T_K = T_C + 273.15;
    double P_Pa = P_kPa * 1000.0;
    double P_hPa = P_kPa * 10.0;

    // Water vapor partial pressure
    double x_w = water_vapor_mole_fraction(T_C, P_Pa, RH);
    double e_hPa = x_w * P_hPa;
    double Pd_hPa = P_hPa - e_hPa;

    return itu_N(T_K, Pd_hPa, e_hPa);
}

}  // namespace itu

}  // namespace refraction
