#pragma once

// Refractive index of pure water (IAPWS R9-97).
// Modified Lorentz-Lorenz: n = sqrt((2A+1)/(1-A))
// Based on Schiebener et al. (1990), Harvey et al. (1998).
// Valid: 200-1100 nm, -12 to 500 C.
//
// [1] IAPWS (1997), Release on the Refractive Index of Ordinary Water Substance.
// [2] Kell (1975), J. Chem. Eng. Data, 20(1), 97-105.

#include <cmath>

namespace refraction {
namespace ocean {
namespace iapws {

// [1] Table 1 coefficients (Harvey et al. 1998)
constexpr double a_coeff[] = {
    0.244257733,     // a0
    9.74634476e-3,   // a1 density
    -3.73234996e-3,  // a2 temperature
    2.68678472e-4,   // a3 L^2*theta
    1.58920570e-3,   // a4 1/L^2 (Cauchy)
    2.45934259e-3,   // a5 UV resonance
    0.900704920,     // a6 IR resonance
    -1.66626219e-2,  // a7 density^2
};

constexpr double L_UV   = 0.229202;   // UV resonance (reduced wavelength)
constexpr double L_IR   = 5.432937;   // IR resonance (reduced wavelength)
constexpr double rho_ref = 1000.0;    // kg/m^3
constexpr double T_ref   = 273.15;   // K
constexpr double L_ref   = 0.589;    // um (sodium D)

// Approximate density of pure water [2]
inline double pure_water_density(double T_C, double p_dbar = 0.0) {
    double T = T_C;
    // [2] Eq. 7, valid 0-150 C at 1 atm
    double rho_atm = (999.83952 + 16.945176 * T - 7.9870401e-3 * T * T
                      - 46.170461e-6 * T * T * T + 105.56302e-9 * T * T * T * T
                      - 280.54253e-12 * T * T * T * T * T)
                   / (1.0 + 16.879850e-3 * T);
    return rho_atm * (1.0 + 4.5e-6 * p_dbar);
}

// IAPWS R9-97 refractive index [1] Eqs. 1-2
// Check value: T=25C, rho=997.047, lam=589.3nm -> n=1.33285
inline double refractive_index(double T_C, double wavelength_nm,
                               double rho_kg_m3 = -1.0, double p_dbar = 0.0) {
    if (rho_kg_m3 < 0.0)
        rho_kg_m3 = pure_water_density(T_C, p_dbar);

    double delta = rho_kg_m3 / rho_ref;
    double theta = (T_C + 273.15) / T_ref;
    double L = wavelength_nm * 1.0e-3 / L_ref;
    double L2 = L * L;

    double A = delta * (
        a_coeff[0]
        + a_coeff[1] * delta
        + a_coeff[2] * theta
        + a_coeff[3] * L2 * theta
        + a_coeff[4] / L2
        + a_coeff[5] / (L2 - L_UV * L_UV)
        + a_coeff[6] / (L2 - L_IR * L_IR)
        + a_coeff[7] * delta * delta
    );

    return std::sqrt((2.0 * A + 1.0) / (1.0 - A));
}

}  // namespace iapws
}  // namespace ocean
}  // namespace refraction
