#pragma once

// Sound speed in seawater — Mackenzie (1981) [1].
//
// Nine-term equation using depth directly (no pressure conversion):
//
//   c = 1448.96 + 4.591*T - 5.304e-2*T^2 + 2.374e-4*T^3          [1] Eq. 1
//     + 1.340*(S - 35) + 1.630e-2*Z + 1.675e-7*Z^2
//     - 1.025e-2*T*(S - 35) - 7.139e-13*T*Z^3
//
// Valid: T = 2-30 degC, S = 25-40 ppt, Z = 0-8000 m.
// Accuracy: +/- 0.070 m/s within validity range.
//
// The equation was fitted to the Del Grosso (1974) equation over the stated
// range.  At depths > 8000 m accuracy degrades: +1.4 m/s at 10000 m.
//
// References:
//   [1] Mackenzie, K.V. (1981). "Nine-term equation for the sound speed in
//       the oceans." J. Acoust. Soc. Am., 70(3), 807-812.

namespace refraction {
namespace ocean {
namespace mk {

// ── Coefficients — Mackenzie (1981) [1] Eq. 1 ──────────────────────────────
constexpr double C0    = 1448.96;        // base speed (m/s)              [1] Eq. 1
constexpr double CT1   =    4.591;       // T     (m/s/degC)             [1] Eq. 1
constexpr double CT2   =   -5.304e-2;    // T^2   (m/s/degC^2)           [1] Eq. 1
constexpr double CT3   =    2.374e-4;    // T^3   (m/s/degC^3)           [1] Eq. 1
constexpr double CS    =    1.340;       // (S-35) (m/s/ppt)             [1] Eq. 1
constexpr double CZ1   =    1.630e-2;    // Z     (m/s/m)               [1] Eq. 1
constexpr double CZ2   =    1.675e-7;    // Z^2   (m/s/m^2)             [1] Eq. 1
constexpr double CTS   =   -1.025e-2;    // T*(S-35) (m/s/degC/ppt)     [1] Eq. 1
constexpr double CTZ3  =   -7.139e-13;   // T*Z^3 (m/s/degC/m^3)        [1] Eq. 1

// Sound speed of seawater [1] Eq. 1.
//   T_C:     temperature (degC, valid 2-30)
//   S_ppt:   practical salinity (ppt/PSU, valid 25-40)
//   depth_m: depth in meters (valid 0-8000)
// Returns sound speed in m/s (typical 1400-1600).
inline double sound_speed(double T_C, double S_ppt, double depth_m) {
    double T = T_C;
    double S = S_ppt;
    double Z = depth_m;

    return C0
         + CT1 * T
         + CT2 * T * T
         + CT3 * T * T * T
         + CS  * (S - 35.0)
         + CZ1 * Z
         + CZ2 * Z * Z
         + CTS * T * (S - 35.0)
         + CTZ3 * T * Z * Z * Z;
}

}  // namespace mk
}  // namespace ocean
}  // namespace refraction
