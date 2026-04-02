#pragma once

// Refractive index of seawater at optical wavelengths (atmospheric pressure).
// Quan & Fry (1995) [1], 10-term empirical equation fitted to Austin & Halikas data.
// Valid: 0-30 C, 0-35 PSU, 400-700 nm.
//
// [1] Quan & Fry (1995), Applied Optics, 34(18), 3477-3480.

#include <cmath>

namespace refraction {
namespace ocean {
namespace qf {

// [1] Eq. 4 coefficients (T in degC, S in PSU, lam in nm)
constexpr double n0 = 1.31405;
constexpr double n1 = 1.779e-4;    // PSU^-1
constexpr double n2 = -1.05e-6;    // PSU^-1 degC^-1
constexpr double n3 = 1.6e-8;      // PSU^-1 degC^-2
constexpr double n4 = -2.02e-6;    // degC^-2
constexpr double n5 = 15.868;      // nm
constexpr double n6 = 0.01155;     // PSU^-1 nm
constexpr double n7 = -0.00423;    // degC^-1 nm
constexpr double n8 = -4382.0;     // nm^2
constexpr double n9 = 1.1455e6;    // nm^3

inline double refractive_index(double T_C, double S_psu, double wavelength_nm) {
    double T = T_C, S = S_psu, lam = wavelength_nm;
    return n0 + (n1 + n2 * T + n3 * T * T) * S + n4 * T * T
         + (n5 + n6 * S + n7 * T) / lam + n8 / (lam * lam) + n9 / (lam * lam * lam);
}

}  // namespace qf
}  // namespace ocean
}  // namespace refraction
