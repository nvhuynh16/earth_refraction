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

// Gaussian scale (km) over which the water-vapor gradient tapers to zero
// below the surface altitude.
inline constexpr double H2O_BELOW_TAPER_KM = 0.3;

// Water vapor number density vs altitude, anchored at the surface.
// Above the surface (dh = h - h_s >= 0):  n = n0 · exp(-dh/H_w)
// Below: the exponential's gradient continues but fades with a Gaussian of
// scale L, so humidity stays bounded:
//   dn/dh = -(n0/H_w) · exp(-dh/H_w - dh²/(2L²))
//   n     = n0 · [1 + (L/H_w)·√(π/2)·e^{c²}·(erf(c) - erf(dh/(L√2) + c))]
//   c     = L / (H_w·√2)
// n, dn/dh and d²n/dh² are continuous at the surface; far below, n levels
// off at 1.21·n0 for the defaults.
inline double water_vapor_profile(double n_H2O_surface, double h_km, double h_surface_km,
                                  double H_w = 2.0, double L_below = H2O_BELOW_TAPER_KM) {
    double dh = h_km - h_surface_km;
    if (dh >= 0.0) return n_H2O_surface * std::exp(-dh / H_w);
    double c = L_below / (H_w * std::sqrt(2.0));
    double amp = (L_below / H_w) * std::sqrt(PI / 2.0) * std::exp(c * c);
    return n_H2O_surface
         * (1.0 + amp * (std::erf(c) - std::erf(dh / (L_below * std::sqrt(2.0)) + c)));
}

// Altitude derivative dn/dh (m^-3 km^-1) of water_vapor_profile
inline double water_vapor_profile_derivative(double n_H2O_surface, double h_km,
                                             double h_surface_km, double H_w = 2.0,
                                             double L_below = H2O_BELOW_TAPER_KM) {
    double dh = h_km - h_surface_km;
    double taper = (dh < 0.0) ? dh * dh / (2.0 * L_below * L_below) : 0.0;
    return -(n_H2O_surface / H_w) * std::exp(-dh / H_w - taper);
}

}  // namespace refraction
