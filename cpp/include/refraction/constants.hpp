#pragma once

// Shared physical constants for the refraction library.

namespace refraction {

inline constexpr double PI = 3.14159265358979323846;
inline constexpr double DEG_TO_RAD = PI / 180.0;
inline constexpr double kB = 1.380649e-23;        // Boltzmann constant (J/K, exact, SI 2019)
inline constexpr double HPA_TO_PA = 0.01;          // hectopascal to pascal
inline constexpr double REFRACTIVITY_SCALE = 1e6;  // (n-1) * 1e6 = N-units

inline constexpr double EARTH_RADIUS_M = 6'371'000.0;   // Mean Earth radius (m)
inline constexpr double SPEED_OF_LIGHT = 299'792'458.0;  // m/s (exact, SI definition)

// WGS-84 ellipsoid
inline constexpr double WGS84_A  = 6'378'137.0;                          // semi-major axis (m)
inline constexpr double WGS84_F  = 1.0 / 298.257223563;                  // flattening
inline constexpr double WGS84_B  = WGS84_A * (1.0 - WGS84_F);           // semi-minor axis (m)
inline constexpr double WGS84_E2 = 2.0 * WGS84_F - WGS84_F * WGS84_F;   // first eccentricity squared

}  // namespace refraction
