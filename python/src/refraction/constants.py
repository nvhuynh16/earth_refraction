"""Shared physical constants for the refraction package."""

kB = 1.380649e-23       # Boltzmann constant (J/K, exact, SI 2019)
HPA_TO_PA = 0.01        # hectopascal to pascal conversion
REFRACTIVITY_SCALE = 1e6  # (n-1) * 1e6 = N-units
EARTH_RADIUS_M = 6_371_000.0   # Mean Earth radius (m)
SPEED_OF_LIGHT = 299_792_458.0  # m/s (exact, SI definition)

# WGS-84 ellipsoid constants
WGS84_A = 6_378_137.0                          # semi-major axis (m)
WGS84_F = 1.0 / 298.257223563                  # flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F)           # semi-minor axis (m)
WGS84_E2 = 2.0 * WGS84_F - WGS84_F ** 2       # first eccentricity squared
