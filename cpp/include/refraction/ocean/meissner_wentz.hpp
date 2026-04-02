#pragma once

// Complex permittivity and refractive index of seawater at microwave frequencies.
//
// Meissner & Wentz double-Debye relaxation model [1] with 2012 updates [2]:
//   eps(f,T,S) = (eps_s - eps_1)/(1 - j*f/v1) + (eps_1 - eps_inf)/(1 - j*f/v2)
//              + eps_inf + j*sigma*f0/f                              [1] Eq. 1
//
// References:
//   [1] Meissner & Wentz (2004), IEEE TGRS 42(9), 1836-1849.
//   [2] Meissner & Wentz (2012), IEEE TGRS 50(11), 4919-4932.
//   [3] Stogryn (1971), IEEE Trans. MTT, 19(8), 733-736.
//   [4] RSS Fortran (MW2012 errata corrections).

#include <cmath>

namespace refraction {
namespace ocean {
namespace mw {

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

struct ComplexPermittivity {
    double real;  // dielectric constant
    double imag;  // dielectric loss (positive for lossy media)
};

struct ComplexRefractiveIndex {
    double n_real;
    double n_imag;  // extinction coefficient (positive)
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

constexpr double f0 = 17.97510;  // GHz m/S, conductivity-to-loss factor [1] Eq. 1

// Pure-water Debye coefficients [1] Table III
constexpr double a[] = {
    5.7230e+00,   // a0  eps_1 constant
    2.2379e-02,   // a1  eps_1 linear (degC^-1)
    -7.1237e-04,  // a2  eps_1 quadratic (degC^-2)
    5.0478e+00,   // a3  v1 denominator constant
    -7.0315e-02,  // a4  v1 denominator linear
    6.0059e-04,   // a5  v1 denominator quadratic
    3.6143e+00,   // a6  eps_inf constant
    2.8841e-02,   // a7  eps_inf linear (degC^-1)
    1.3652e-01,   // a8  v2 denominator constant
    1.4825e-03,   // a9  v2 denominator linear
    2.4166e-04,   // a10 v2 denominator quadratic
};

// Salinity-dependent coefficients [1] Table VI
constexpr double b_2004[] = {
    -3.56417e-03,  // b0  eps_s (PSU^-1)
    4.74868e-06,   // b1  eps_s (PSU^-2)
    1.15574e-05,   // b2  eps_s T*S
    2.39357e-03,   // b3  v1 (PSU^-1)
    -3.13530e-05,  // b4  v1 S*T
    2.52477e-07,   // b5  v1 S*T^2
    -6.28908e-03,  // b6  eps_1 (PSU^-1)
    1.76032e-04,   // b7  eps_1 (PSU^-2)
    -9.22144e-05,  // b8  eps_1 S*T
    -1.99723e-02,  // b9  v2 (PSU^-1)
    1.81176e-04,   // b10 v2 S*T
    -2.04265e-03,  // b11 eps_inf (PSU^-1)
    1.57883e-04,   // b12 eps_inf S*T
};

// MW2012 updated eps_s coefficient (replaces b[0]) [2] Table 7
constexpr double b0_mw2012 = -0.33330e-02;

// MW2012 v1 5-term polynomial [2] Table 7 (d3 sign corrected per [4] errata)
constexpr double v1_mw2012[] = {
    0.23232e-02, -0.79208e-04, 0.36764e-05, -0.35594e-06, 0.89795e-08
};
constexpr double v1_mw2012_hi_offset = 9.1873715e-04;  // SST > 30 C
constexpr double v1_mw2012_hi_slope  = 1.5012396e-04;

// ---------------------------------------------------------------------------
// Pure-water Debye parameters [1] Eqs. 5-9
// ---------------------------------------------------------------------------

inline double eps_s_pure(double T_C) {
    return (3.70886e4 - 8.2168e1 * T_C) / (4.21854e2 + T_C);  // [1] Eq. 5
}

inline double eps_1_pure(double T_C) {
    return a[0] + a[1] * T_C + a[2] * T_C * T_C;  // [1] Eq. 6
}

inline double v1_pure(double T_C) {
    return (45.00 + T_C) / (a[3] + a[4] * T_C + a[5] * T_C * T_C);  // [1] Eq. 7
}

inline double eps_inf_pure(double T_C) {
    return a[6] + a[7] * T_C;  // [1] Eq. 8
}

inline double v2_pure(double T_C) {
    return (45.00 + T_C) / (a[8] + a[9] * T_C + a[10] * T_C * T_C);  // [1] Eq. 9
}

// ---------------------------------------------------------------------------
// Conductivity [3] via [1] Eqs. 11-16
// ---------------------------------------------------------------------------

inline double sigma35(double T_C) {
    double T = T_C;
    return 2.903602 + 8.60700e-02 * T + 4.738817e-04 * T * T
           - 2.991e-06 * T * T * T + 4.3047e-09 * T * T * T * T;
}

inline double conductivity(double T_C, double S_psu) {
    if (S_psu <= 0.0) return 0.0;
    double S = S_psu;
    double sig35 = sigma35(T_C);
    double r15 = S * (37.5109 + 5.45216 * S + 1.4409e-02 * S * S)
               / (1004.75 + 182.283 * S + S * S);                    // [1] Eq. 14
    double a0 = (6.9431 + 3.2841 * S - 9.9486e-02 * S * S)
              / (84.850 + 69.024 * S + S * S);                       // [1] Eq. 15
    double a1 = 49.843 - 0.2276 * S + 0.198e-02 * S * S;            // [1] Eq. 16
    return sig35 * r15 * (1.0 + (T_C - 15.0) * a0 / (a1 + T_C));   // [1] Eq. 11
}

// ---------------------------------------------------------------------------
// Salinity-dependent parameters [1] Eq. 17 + [2] updates
// ---------------------------------------------------------------------------

inline double eps_s_sea(double T_C, double S_psu) {
    return eps_s_pure(T_C) * std::exp(
        b0_mw2012 * S_psu + b_2004[1] * S_psu * S_psu + b_2004[2] * T_C * S_psu
    );  // [1] Eq. 17a, b[0] from [2]
}

inline double v1_sea(double T_C, double S_psu) {
    double v1_0 = v1_pure(T_C);
    double factor;
    if (T_C <= 30.0) {
        // [2] Table 7, 5-term polynomial
        double T = T_C;
        factor = 1.0 + S_psu * (v1_mw2012[0] + v1_mw2012[1] * T
                 + v1_mw2012[2] * T * T + v1_mw2012[3] * T * T * T
                 + v1_mw2012[4] * T * T * T * T);
    } else {
        factor = 1.0 + S_psu * (v1_mw2012_hi_offset
                 + v1_mw2012_hi_slope * (T_C - 30.0));
    }
    return v1_0 * factor;
}

inline double eps_1_sea(double T_C, double S_psu) {
    return eps_1_pure(T_C) * std::exp(
        b_2004[6] * S_psu + b_2004[7] * S_psu * S_psu + b_2004[8] * S_psu * T_C
    );  // [1] Eq. 17c
}

inline double v2_sea(double T_C, double S_psu) {
    return v2_pure(T_C) * (1.0 + S_psu * (b_2004[9]
           + b_2004[10] * 0.5 * (T_C + 30.0)));  // [1] Eq. 17d, [2] T correction
}

inline double eps_inf_sea(double T_C, double S_psu) {
    return eps_inf_pure(T_C) * (1.0 + S_psu * (b_2004[11] + b_2004[12] * T_C));
}

// ---------------------------------------------------------------------------
// Complex permittivity and refractive index
// ---------------------------------------------------------------------------

// Double-Debye permittivity [1] Eq. 1 with [2] updates
inline ComplexPermittivity permittivity(double freq_ghz, double T_C, double S_psu) {
    double es   = eps_s_sea(T_C, S_psu);
    double e1   = eps_1_sea(T_C, S_psu);
    double einf = eps_inf_sea(T_C, S_psu);
    double nu1  = v1_sea(T_C, S_psu);
    double nu2  = v2_sea(T_C, S_psu);
    double sig  = conductivity(T_C, S_psu);
    double f    = freq_ghz;

    double r1 = f / nu1;
    double d1 = 1.0 + r1 * r1;
    double r2 = f / nu2;
    double d2 = 1.0 + r2 * r2;

    double eps_real = (es - e1) / d1 + (e1 - einf) / d2 + einf;
    double eps_imag = (es - e1) * r1 / d1 + (e1 - einf) * r2 / d2
                    + (f > 0.0 ? sig * f0 / f : 0.0);

    return {eps_real, eps_imag};
}

// Complex refractive index via n = sqrt(eps)
inline ComplexRefractiveIndex refractive_index(double freq_ghz, double T_C, double S_psu) {
    auto eps = permittivity(freq_ghz, T_C, S_psu);
    double mag = std::sqrt(eps.real * eps.real + eps.imag * eps.imag);
    return {std::sqrt((mag + eps.real) / 2.0),
            std::sqrt((mag - eps.real) / 2.0)};
}

}  // namespace mw
}  // namespace ocean
}  // namespace refraction
