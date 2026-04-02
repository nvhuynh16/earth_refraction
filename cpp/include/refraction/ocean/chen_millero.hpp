#pragma once

// Sound speed in seawater — Chen & Millero (1977) [1], Wong & Zhu (1995)
// ITS-90 revision [2].
//
// c(S,T,P) = Cw(T,P) + A(T,P)*S + B(T,P)*S^(3/2) + D(T,P)*S^2   [2] Table III
//
// where Cw is pure-water sound speed, and A, B, D are salinity corrections.
//
// Valid: T = 0-40 degC, S = 0-40 ppt, P = 0-1000 bar (~0-10000 dbar).
// Accuracy: +/- 0.05 m/s (surface), ~0.5 m/s at depth.
//
// This was the UNESCO/IOC standard (Fofonoff & Millard 1983 [3]) until
// superseded by TEOS-10 in 2010.  Known to overestimate sound speed at
// high pressure by ~0.5 m/s due to reliance on Wilson (1959) pure-water data.
//
// Pressure is bar internally; the public function accepts dbar.
//
// References:
//   [1] Chen, C.T. and F.J. Millero (1977). "Speed of sound in seawater at
//       high pressures." J. Acoust. Soc. Am., 62(5), 1129-1135.
//   [2] Wong, G.S.K. and S. Zhu (1995). "Speed of sound in seawater as a
//       function of salinity, temperature and pressure."
//       J. Acoust. Soc. Am., 97(3), 1732-1736.
//   [3] Fofonoff, N.P. and R.C. Millard (1983). "Algorithms for computation
//       of fundamental properties of seawater." UNESCO Tech. Paper 44.

#include <cmath>

namespace refraction {
namespace ocean {
namespace cm {

// ── Unit conversion ─────────────────────────────────────────────────────────
constexpr double DBAR_TO_BAR = 0.1;   // 1 dbar = 0.1 bar

// ── Cw(T,P) — pure water sound speed ────────────────────────────────────────
// Cw = sum_{i=0}^{3} sum_{j=0}^{Ni} Cij * P^i * T^j     [2] Table III
//
// i=0 (atmospheric):  C00..C05   (6 terms)
// i=1 (linear P):     C10..C14   (5 terms)
// i=2 (quadratic P):  C20..C24   (5 terms)
// i=3 (cubic P):      C30..C32   (3 terms)
// Total: 19 coefficients.

// P^0 terms
constexpr double C00 =  1402.388;       // (m/s)                   [2] Table III
constexpr double C01 =  5.03830;        // T    (m/s/degC)         [2] Table III
constexpr double C02 = -5.81090e-2;     // T^2  (m/s/degC^2)       [2] Table III
constexpr double C03 =  3.3432e-4;      // T^3  (m/s/degC^3)       [2] Table III
constexpr double C04 = -1.47797e-6;     // T^4  (m/s/degC^4)       [2] Table III
constexpr double C05 =  3.1419e-9;      // T^5  (m/s/degC^5)       [2] Table III

// P^1 terms
constexpr double C10 =  0.153563;       // P    (m/s/bar)          [2] Table III
constexpr double C11 =  6.8999e-4;      // P*T  (m/s/bar/degC)     [2] Table III
constexpr double C12 = -8.1829e-6;      // P*T^2                   [2] Table III
constexpr double C13 =  1.3632e-7;      // P*T^3                   [2] Table III
constexpr double C14 = -6.1260e-10;     // P*T^4                   [2] Table III

// P^2 terms
constexpr double C20 =  3.1260e-5;      // P^2                     [2] Table III
constexpr double C21 = -1.7111e-6;      // P^2*T                   [2] Table III
constexpr double C22 =  2.5986e-8;      // P^2*T^2                 [2] Table III
constexpr double C23 = -2.5353e-10;     // P^2*T^3                 [2] Table III
constexpr double C24 =  1.0415e-12;     // P^2*T^4                 [2] Table III

// P^3 terms
constexpr double C30 = -9.7729e-9;      // P^3                     [2] Table III
constexpr double C31 =  3.8513e-10;     // P^3*T                   [2] Table III
constexpr double C32 = -2.3654e-12;     // P^3*T^2                 [2] Table III

// ── A(T,P) — salinity linear coefficient ────────────────────────────────────
// A = sum_{i=0}^{3} sum_{j=0}^{Ni} Aij * P^i * T^j     [2] Table III
// Total: 17 coefficients.

// P^0 terms
constexpr double A00 =  1.389;          // (m/s/ppt)               [2] Table III
constexpr double A01 = -1.262e-2;       // T    (m/s/ppt/degC)     [2] Table III
constexpr double A02 =  7.166e-5;       // T^2                     [2] Table III
constexpr double A03 =  2.008e-6;       // T^3                     [2] Table III
constexpr double A04 = -3.21e-8;        // T^4                     [2] Table III

// P^1 terms
constexpr double A10 =  9.4742e-5;      // P    (m/s/ppt/bar)      [2] Table III
constexpr double A11 = -1.2583e-5;      // P*T                     [2] Table III
constexpr double A12 = -6.4928e-8;      // P*T^2                   [2] Table III
constexpr double A13 =  1.0515e-8;      // P*T^3                   [2] Table III
constexpr double A14 = -2.0142e-10;     // P*T^4                   [2] Table III

// P^2 terms
constexpr double A20 = -3.9064e-7;      // P^2                     [2] Table III
constexpr double A21 =  9.1061e-9;      // P^2*T                   [2] Table III
constexpr double A22 = -1.6009e-10;     // P^2*T^2                 [2] Table III
constexpr double A23 =  7.994e-12;      // P^2*T^3                 [2] Table III

// P^3 terms
constexpr double A30 =  1.100e-10;      // P^3                     [2] Table III
constexpr double A31 =  6.651e-12;      // P^3*T                   [2] Table III
constexpr double A32 = -3.391e-13;      // P^3*T^2                 [2] Table III

// ── B(T,P) — salinity S^(3/2) coefficient ───────────────────────────────────
constexpr double B00 = -1.922e-2;       // (m/s/ppt^1.5)           [2] Table III
constexpr double B01 = -4.42e-5;        // T                       [2] Table III
constexpr double B10 =  7.3637e-5;      // P                       [2] Table III
constexpr double B11 =  1.7950e-7;      // P*T                     [2] Table III

// ── D(T,P) — salinity S^2 coefficient ───────────────────────────────────────
constexpr double D00 =  1.727e-3;       // (m/s/ppt^2)             [2] Table III
constexpr double D10 = -7.9836e-6;      // P                       [2] Table III

// ── Helper polynomials ──────────────────────────────────────────────────────

// Pure water sound speed  [2] Table III
inline double c_w(double T, double P) {
    double T2 = T * T, T3 = T2 * T, T4 = T3 * T, T5 = T4 * T;
    double P2 = P * P, P3 = P2 * P;

    return (C00 + C01*T + C02*T2 + C03*T3 + C04*T4 + C05*T5)
         + (C10 + C11*T + C12*T2 + C13*T3 + C14*T4) * P
         + (C20 + C21*T + C22*T2 + C23*T3 + C24*T4) * P2
         + (C30 + C31*T + C32*T2) * P3;
}

// Salinity linear coefficient  [2] Table III
inline double A_coeff(double T, double P) {
    double T2 = T * T, T3 = T2 * T, T4 = T3 * T;
    double P2 = P * P, P3 = P2 * P;

    return (A00 + A01*T + A02*T2 + A03*T3 + A04*T4)
         + (A10 + A11*T + A12*T2 + A13*T3 + A14*T4) * P
         + (A20 + A21*T + A22*T2 + A23*T3) * P2
         + (A30 + A31*T + A32*T2) * P3;
}

// Salinity S^(3/2) coefficient  [2] Table III
inline double B_coeff(double T, double P) {
    return (B00 + B01*T) + (B10 + B11*T) * P;
}

// Salinity S^2 coefficient  [2] Table III
inline double D_coeff(double T, double P) {
    return D00 + D10 * P;
}

// ── Public function ─────────────────────────────────────────────────────────

// Sound speed of seawater [1], [2] Table III.
//   T_C:    temperature (degC, valid 0-40)
//   S_ppt:  practical salinity (ppt/PSU, valid 0-40)
//   p_dbar: sea pressure (dbar, valid 0-10000)
// Returns sound speed in m/s (typical 1400-1600).
inline double sound_speed(double T_C, double S_ppt, double p_dbar) {
    double T = T_C;
    double S = S_ppt;
    double P = p_dbar * DBAR_TO_BAR;   // dbar -> bar

    double S32 = S * std::sqrt(S);     // S^(3/2)

    return c_w(T, P) + A_coeff(T, P) * S + B_coeff(T, P) * S32
         + D_coeff(T, P) * S * S;
}

}  // namespace cm
}  // namespace ocean
}  // namespace refraction
