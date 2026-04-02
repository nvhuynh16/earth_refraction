#pragma once

// Refractive index of seawater over temperature, pressure, salinity, and wavelength.
//
// 27-term polynomial algorithm from Millard & Seaver (1990) [1]:
//   N(T,p,S,lam) = N_I(T,lam) + N_II(T,lam,S) + N_III(p,T,lam) + N_IV(S,p,T)
//
// Valid: 0-30 C, 0-43 PSU, 500-700 nm, 0-11000 dbar.
// Accuracy: 0.4 ppm (Region I) to 80 ppm (Region III/IV).
//
// References:
//   [1] Millard & Seaver (1990), Deep-Sea Research, 37(12), 1909-1926.
//
// Coefficients validated against [1] Table 2.  PDF zero-count corrections noted inline.

#include <cmath>

namespace refraction {
namespace ocean {
namespace ms {

// ── Region I — Pure water at atmospheric pressure (12 terms, SD = 0.12 ppm) ──
// N_I = A0 + L2*L^2 + LM2/L^2 + LM4/L^4 + LM6/L^6
//     + T1*T + T2*T^2 + T3*T^3 + T4*T^4 + TL*T*L + T2L*T^2*L + T3L*T^3*L
constexpr double A0  =  1.3280657;
constexpr double L2  = -4.5536802e-3;
constexpr double LM2 =  2.5471707e-3;
constexpr double LM4 =  7.501966e-6;
constexpr double LM6 =  2.802632e-6;
constexpr double T1  = -5.2883907e-6;   // PDF correction: 5 zeros, not 4
constexpr double T2  = -3.0738272e-6;
constexpr double T3  =  3.0124687e-8;
constexpr double T4  = -2.0883178e-10;
constexpr double TL  =  1.0508621e-5;
constexpr double T2L =  2.1282248e-7;   // PDF correction: 6 zeros, not 5
constexpr double T3L = -1.705881e-10;

// ── Region II — Salinity at atmospheric pressure (6 terms, SD = 4.7 ppm) ──
constexpr double S0    =  1.9029121e-4;
constexpr double S1LM2 =  2.4239607e-6;
constexpr double S1T   = -7.3960297e-7;  // PDF correction: 6 zeros, not 5
constexpr double S1T2  =  8.9818478e-9;
constexpr double S1T3  =  1.2078804e-10;
constexpr double STL   = -3.589495e-7;

// ── Region III — Pressure for pure water (6 terms, SD = 26.5 ppm) ──
constexpr double P1   =  1.5868383e-6;
constexpr double P2   = -1.574074e-11;
constexpr double PLM2 =  1.0712063e-8;   // PDF correction: 7 zeros, not 5
constexpr double PT   = -9.4834486e-9;
constexpr double PT2  =  1.0100326e-10;
constexpr double P2T2 =  5.8085198e-15;

// ── Region IV — Pressure x salinity (3 terms, SD = 19.7 ppm) ──
constexpr double P1S  = -1.1177517e-9;
constexpr double PTS  =  5.7311268e-11;
constexpr double PT2S = -1.5460458e-12;

// ── Region evaluation ──

inline double n_region_I(double T, double L) {
    double L2v = L * L;
    return A0 + L2 * L2v + LM2 / L2v + LM4 / (L2v * L2v) + LM6 / (L2v * L2v * L2v)
         + T1 * T + T2 * T * T + T3 * T * T * T + T4 * T * T * T * T
         + TL * T * L + T2L * T * T * L + T3L * T * T * T * L;
}

inline double n_region_II(double T, double L, double S) {
    double L2v = L * L;
    return S0 * S + S1LM2 * S / L2v + S1T * S * T + S1T2 * S * T * T
         + S1T3 * S * T * T * T + STL * S * T * L;
}

inline double n_region_III(double p, double T, double L) {
    double L2v = L * L;
    return P1 * p + P2 * p * p + PLM2 * p / L2v + PT * p * T
         + PT2 * p * T * T + P2T2 * p * p * T * T;
}

inline double n_region_IV(double S, double p, double T) {
    return P1S * p * S + PTS * p * T * S + PT2S * p * T * T * S;
}

// Refractive index of seawater [1] Eq. 5
// T_C: temperature (degC), S_psu: salinity (PSU), p_dbar: pressure (dbar),
// wavelength_nm: vacuum wavelength (nm, converted to um internally)
inline double refractive_index(double T_C, double S_psu, double p_dbar,
                               double wavelength_nm) {
    double L = wavelength_nm * 1.0e-3;  // nm -> um
    return n_region_I(T_C, L) + n_region_II(T_C, L, S_psu)
         + n_region_III(p_dbar, T_C, L) + n_region_IV(S_psu, p_dbar, T_C);
}

}  // namespace ms
}  // namespace ocean
}  // namespace refraction
