#pragma once

#include <cmath>
#include "constants.hpp"

namespace refraction {

// Saturation vapor pressure (Pa) using Giacomo 1982 / BIPM formula
inline double svp_giacomo(double T_K) {
    constexpr double A = 1.2378847e-5;
    constexpr double B = -1.9121316e-2;
    constexpr double C = 33.93711047;
    constexpr double D = -6.3431645e3;
    return std::exp(A * T_K * T_K + B * T_K + C + D / T_K);
}

// Enhancement factor (Giacomo 1982)
inline double enhancement_factor(double T_C, double P_Pa) {
    constexpr double alpha = 1.00062;
    constexpr double beta = 3.14e-8;
    constexpr double gamma = 5.6e-7;
    return alpha + beta * P_Pa + gamma * T_C * T_C;
}

// Water vapor mole fraction from RH (0-1), temperature (C), pressure (Pa)
inline double water_vapor_mole_fraction(double T_C, double P_Pa, double RH) {
    double T_K = T_C + 273.15;
    double svp = svp_giacomo(T_K);
    double f = enhancement_factor(T_C, P_Pa);
    return f * RH * svp / P_Pa;
}

// Water vapor number density (m^-3) from T(K), P(Pa), RH(0-1)
inline double water_vapor_number_density(double T_K, double P_Pa, double RH) {
    double T_C = T_K - 273.15;
    double x_w = water_vapor_mole_fraction(T_C, P_Pa, RH);
    double e = x_w * P_Pa;  // Partial pressure of water vapor
    return e / (kB * T_K);
}

// Exponential decay profile for water vapor
inline double water_vapor_profile(double n_H2O_surface, double h_km, double h_surface_km, double H_w = 2.0) {
    if (h_km < h_surface_km) return n_H2O_surface;
    return n_H2O_surface * std::exp(-(h_km - h_surface_km) / H_w);
}

}  // namespace refraction
