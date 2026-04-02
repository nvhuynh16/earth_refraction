#pragma once

// Sound speed in seawater — Del Grosso (1974) [1] with ITS-90 coefficients
// from Wong & Zhu (1995) [2].
//
// c(S,T,P) = C000 + dC_T(T) + dC_S(S) + dC_P(P) + dC_STP(S,T,P)  [2] Table IV
//
// where:
//   dC_T = CT1*T + CT2*T^2 + CT3*T^3
//   dC_S = CS1*S + CS2*S^2
//   dC_P = CP1*P + CP2*P^2 + CP3*P^3
//   dC_STP = CST*S*T + CTP*T*P + CT2P2*T^2*P^2 + CTP2*T*P^2
//          + CTP3*T*P^3 + CT3P*T^3*P + CS2P2*S^2*P^2
//          + CST2*S*T^2 + CS2TP*S^2*T*P + CSTP*S*T*P
//
// Valid: T = 0-30 degC, S = 30-40 ppt, P = 0-1000 kg/cm^2 (~0-9807 dbar).
// Accuracy: +/- 0.05 m/s.
//
// Pressure is kg/cm^2 internally; the public function accepts dbar and
// converts via DBAR_TO_KGCM2.
//
// References:
//   [1] Del Grosso, V.A. (1974). "New equation for the speed of sound in
//       natural waters (with comparisons to other equations)."
//       J. Acoust. Soc. Am., 56(4), 1084-1091.
//   [2] Wong, G.S.K. and S. Zhu (1995). "Speed of sound in seawater as a
//       function of salinity, temperature and pressure."
//       J. Acoust. Soc. Am., 97(3), 1732-1736.

namespace refraction {
namespace ocean {
namespace dg {

// ── Unit conversion ─────────────────────────────────────────────────────────
// 1 dbar = 10^4 Pa; 1 kg/cm^2 = 98066.5 Pa  =>  1 dbar = 10^4/98066.5
constexpr double DBAR_TO_KGCM2 = 0.101972;   // dbar -> kg/cm^2

// ── Coefficients — Wong & Zhu (1995) ITS-90 values [2] Table IV ─────────────
constexpr double C000  = 1402.392;           // base speed (m/s)

// Temperature terms
constexpr double CT1   =  5.012285;          // T     (m/s/degC)           [2] Table IV
constexpr double CT2   = -0.0551184;         // T^2   (m/s/degC^2)         [2] Table IV
constexpr double CT3   =  2.21649e-4;        // T^3   (m/s/degC^3)         [2] Table IV

// Salinity terms
constexpr double CS1   =  1.329530;          // S     (m/s/ppt)            [2] Table IV
constexpr double CS2   =  1.288598e-4;       // S^2   (m/s/ppt^2)          [2] Table IV

// Pressure terms  (P in kg/cm^2)
constexpr double CP1   =  0.1560592;         // P     (m/s/(kg/cm^2))      [2] Table IV
constexpr double CP2   =  2.449993e-5;       // P^2   (m/s/(kg/cm^2)^2)    [2] Table IV
constexpr double CP3   = -8.833959e-9;       // P^3   (m/s/(kg/cm^2)^3)    [2] Table IV

// Cross-product terms
constexpr double CST    = -1.275936e-2;      // S*T                        [2] Table IV
constexpr double CTP    =  6.353509e-3;      // T*P                        [2] Table IV
constexpr double CT2P2  =  2.656174e-8;      // T^2*P^2                    [2] Table IV
constexpr double CTP2   = -1.593895e-6;      // T*P^2                      [2] Table IV
constexpr double CTP3   =  5.222483e-10;     // T*P^3                      [2] Table IV
constexpr double CT3P   = -4.383615e-7;      // T^3*P                      [2] Table IV
constexpr double CS2P2  = -1.616745e-9;      // S^2*P^2                    [2] Table IV
constexpr double CST2   =  9.688441e-5;      // S*T^2                      [2] Table IV
constexpr double CS2TP  =  4.857614e-6;      // S^2*T*P                    [2] Table IV
constexpr double CSTP   = -3.406824e-4;      // S*T*P                      [2] Table IV

// ── Evaluation ──────────────────────────────────────────────────────────────

// Sound speed of seawater [1], [2] Table IV.
//   T_C:    temperature (degC, valid 0-30)
//   S_ppt:  practical salinity (ppt/PSU, valid 30-40)
//   p_dbar: sea pressure (dbar, valid 0-9807)
// Returns sound speed in m/s (typical 1400-1600).
inline double sound_speed(double T_C, double S_ppt, double p_dbar) {
    double T = T_C;
    double S = S_ppt;
    double P = p_dbar * DBAR_TO_KGCM2;   // dbar -> kg/cm^2

    double dC_T = CT1 * T + CT2 * T * T + CT3 * T * T * T;
    double dC_S = CS1 * S + CS2 * S * S;
    double dC_P = CP1 * P + CP2 * P * P + CP3 * P * P * P;

    double dC_STP = CST   * S * T
                  + CTP   * T * P
                  + CT2P2 * T * T * P * P
                  + CTP2  * T * P * P
                  + CTP3  * T * P * P * P
                  + CT3P  * T * T * T * P
                  + CS2P2 * S * S * P * P
                  + CST2  * S * T * T
                  + CS2TP * S * S * T * P
                  + CSTP  * S * T * P;

    return C000 + dC_T + dC_S + dC_P + dC_STP;
}

}  // namespace dg
}  // namespace ocean
}  // namespace refraction
