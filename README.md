# Refraction Index

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)

Refractive index computation and 3D ray tracing for atmospheric and ocean media on the WGS-84 ellipsoid.

Decades of research have produced accurate refraction models for the atmosphere (Ciddor 1996, ITU-R P.453, NRLMSIS 2.1) and ocean (Millard & Seaver 1990, Meissner & Wentz 2004, Del Grosso 1974), but no modern, readily available library brings them together in a form that engineers and scientists can use directly. This project fills that gap: a single library that computes refractive index profiles, traces rays through the real Earth geometry (WGS-84 ellipsoid), and provides the derivatives needed for inversion and uncertainty analysis — all with cross-validated Python and C++ implementations.

## Features

**Atmospheric refraction** (surface to 122 km)
- Optical mode (Ciddor 1996, 300–1700 nm) and radio mode (ITU-R P.453-14)
- Species-sum approach: n − 1 = Σ nᵢ × Kᵢ(λ) using NRLMSIS 2.1
- Surface anchoring to weather observations via temperature offset and density scaling

**Ocean refraction** (surface to 11 km depth)
- Radio: Meissner & Wentz 2004/2012 (double-Debye permittivity)
- Optical with pressure: Millard & Seaver 1990 (500–700 nm)
- Optical surface: Quan & Fry 1995 (400–700 nm)
- Pure water: IAPWS R9-97 (200–1100 nm)

**Ocean sound speed**
- Del Grosso 1974 / Wong & Zhu 1995 (ITS-90, ±0.05 m/s)
- Chen-Millero / UNESCO 1977 / Wong & Zhu 1995 (ITS-90)
- Mackenzie 1981 (9-term, ±0.070 m/s)

**3D ray tracing** (eikonal equation from Fermat's Principle)
- ECEF coordinates on WGS-84 ellipsoid — no flat-Earth approximation
- Slowness momentum formulation — smooth through turning points, no sign tracking
- Travel time as ODE state — stop at measured radio/sonar round-trip time
- First and second derivatives (curvature, acceleration) from solver output
- Travel-time derivatives (dr/dT, d²r/dT², dp/dT) for radar/sonar inversion
- Sensitivity derivatives w.r.t. 5 initial-condition parameters (h₀, θ₀, α₀, φ₀, λ₀)
- Batch tracing — trace thousands of rays; pure Python or C++ via nanobind

**Three implementation tiers** (same interface for Python and nanobind)
- **Pure Python** — easy to use, full features including batch tracing
- **Header-only C++20** — fast, for integration into C++ applications
- **Native Python (nanobind)** — C++ speed from Python, same API as pure Python

## Installation

### Python

```bash
git clone https://github.com/nhuynh/refraction-index.git
cd refraction-index
pip install .
```

This builds and installs both the pure Python package and the C++ native extension.
Requires CMake 3.20+, a C++20 compiler, [scikit-build-core](https://github.com/scikit-build/scikit-build-core), and [nanobind](https://github.com/wjakob/nanobind).

For development (editable install with test dependencies):

```bash
pip install -e ".[dev]"
```

### C++ (header-only, no installation needed)

```cpp
#include <refraction/ray_trace.hpp>   // ray tracing
#include <refraction/profile.hpp>     // atmospheric profiles
#include <refraction/geodetic.hpp>    // WGS-84 geodetic utilities
```

## Quick Start

### Atmospheric refraction

```python
from refraction import AtmosphericConditions, RefractionProfile, Mode

atm = AtmosphericConditions(day_of_year=172, latitude_deg=45.0)

# Use the model directly (NRLMSIS prediction, no anchoring)
profile = RefractionProfile(atm)

# Or anchor to surface weather observations for higher accuracy
from refraction import SurfaceObservation
obs = SurfaceObservation(altitude_km=0.0, temperature_C=15.0,
                         pressure_kPa=101.325, relative_humidity=0.5)
profile = RefractionProfile(atm, obs)

result = profile.compute(10.0, Mode.Ciddor, wavelength_nm=633.0)
print(f"n = {result.n:.8f}, N = {result.N:.2f} N-units")
```

### Ocean sound speed

```python
from refraction.ocean import SoundSpeedProfile, SoundMode, OceanConditions

# From ocean conditions (temperature, salinity)
cond = OceanConditions(sst_C=25.0, sss_psu=35.0)
profile = SoundSpeedProfile(cond)
result = profile.compute(500.0, SoundMode.DelGrosso)  # 500 m depth
print(f"Sound speed = {result.sound_speed_m_s:.1f} m/s")
```

### Ray tracing — atmospheric

```python
from refraction import (
    EikonalTracer, EikonalInput, EikonalStopCondition, Mode,
    RefractionProfile, AtmosphericConditions,
)

profile = RefractionProfile(AtmosphericConditions(day_of_year=172, latitude_deg=45.0))
tracer = EikonalTracer.from_profile(profile, Mode.Ciddor)

inp = EikonalInput(lat_deg=45.0, lon_deg=0.0, h_m=0.0, elevation_deg=10.0, azimuth_deg=0.0)
stop = EikonalStopCondition(target_altitude_m=100_000.0)
result = tracer.trace(inp, stop)

print(f"Arc length: {result.s[-1]:.0f} m")
print(f"Travel time: {result.T[-1]*1e6:.1f} µs")
print(f"Bending: {result.bending_deg*60:.2f} arcmin")
```

### Ray tracing — from measured sound velocity profile

```python
from refraction import EikonalTracer, EikonalInput, EikonalStopCondition

# Measured SVP data (depth positive downward, speed in m/s)
depths = [0, 50, 100, 200, 500, 1000, 2000]
speeds = [1520, 1515, 1490, 1495, 1500, 1510, 1520]
tracer = EikonalTracer.from_depth_speed_table(depths, speeds)

inp = EikonalInput(lat_deg=30.0, lon_deg=-80.0, h_m=-10.0,
                   elevation_deg=-5.0, azimuth_deg=90.0)
stop = EikonalStopCondition(target_travel_time_s=0.5, detect_ground_hit=False)
result = tracer.trace(inp, stop)
```

### Batch tracing

```python
from refraction.native import NativeTracer, TableSpeedProfile
from refraction import BatchInput
import numpy as np

# From atmospheric speed table
h = np.linspace(0, 200_000, 2000)
v = 299_792_458.0 / (1.000315 * np.exp(-39e-9 * h))
tracer = NativeTracer.from_table(TableSpeedProfile(h, v))

# Or from measured SVP (depth positive downward)
# depths = [0, 50, 100, 500, 1000, 2000]
# speeds = [1520, 1515, 1490, 1500, 1510, 1520]
# tracer = NativeTracer.from_table(TableSpeedProfile.from_depth(depths, speeds))

# 1000 rays at different elevations, same travel time
inp = BatchInput(
    lat_deg=np.full(1000, 45.0),
    lon_deg=np.zeros(1000),
    h_m=np.zeros(1000),
    elevation_deg=np.linspace(1, 89, 1000),
    azimuth_deg=np.zeros(1000),
    travel_time_s=np.full(1000, 0.001),
)
result = tracer.trace_batch(inp)  # single C++ call → BatchResult (endpoint only)

print(f"Positions shape: {result.r.shape}")  # (1000, 3)
print(f"Velocities shape: {result.dr_dT.shape}")  # (1000, 3)

# Full ray paths (list of EikonalResult, one per ray)
full = tracer.trace_batch(inp, endpoint=False)
print(f"Ray 0 path length: {len(full[0].s)} steps")
```

> **Note:** `EikonalTracer` also has `trace_batch()` with the same interface,
> just slower (Python loop internally). Use `NativeTracer` for performance.

### C++ ray tracing

```cpp
#include <refraction/ray_trace.hpp>

auto tracer = refraction::EikonalTracer{
    [](double h) { return 1500.0 + 0.017 * h; },  // v(h)
    [](double)   { return 0.017; },                 // v'(h)
};

refraction::EikonalInput inp{30.0, -80.0, -10.0, -5.0, 90.0};
refraction::EikonalStopCondition stop{.target_travel_time_s = 0.5};
auto result = tracer.trace(inp, stop);
```

## Building and Testing

### C++ tests

Requires CMake 3.20+ and a C++20 compiler (GCC 10+, Clang 10+, MSVC 19.29+):

```bash
cd cpp
cmake -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

### Python tests

```bash
pip install -e ".[dev]"
python -m pytest python/tests/ -q
```

## Project Structure

```
cpp/
  include/refraction/          # Header-only C++20 library
    profile.hpp                # Atmospheric refraction profiles
    ray_trace.hpp              # Eikonal ray tracer + sensitivities
    ray_trace_batch.hpp        # Type-erased tracer + batch API
    geodetic.hpp               # WGS-84 coordinate utilities
    constants.hpp              # Physical constants + WGS-84 ellipsoid
    ocean/                     # Ocean refraction + sound speed
    atmosphere/                # NRLMSIS 2.1 model
  tests/                       # 17 test executables

python/
  src/refraction/
    ray_trace.py               # Eikonal tracer, BatchInput/Result, EikonalResult
    geodetic.py                # WGS-84 utilities
    native.py                  # Nanobind wrapper (NativeTracer, TableSpeedProfile)
    profile.py                 # Atmospheric profiles
    ocean/                     # Ocean refraction + sound speed
  nanobind/                    # C++ nanobind extension source
  tests/                       # 698 Python tests

docs/
  Ray Trace/                   # LaTeX derivation from Fermat's Principle
  ocean_model.md               # Ocean model equations and references
```

## Citation

If you use this library in your research, please cite:

```bibtex
@software{huynh2026refraction,
  author = {Huynh, Nen},
  title = {Refraction Index: Atmospheric and Ocean Refractive Index with 3D Ray Tracing},
  year = {2026},
  url = {https://github.com/nhuynh/refraction-index},
  license = {MIT},
}
```

## References

### Atmospheric
- Ciddor, P. E. (1996). "Refractive index of air: new equations for the visible and near infrared." *Applied Optics*, 35(9), 1566–1573.
- ITU-R P.453-14 (2019). "The radio refractive index: its formula and refractivity data."
- Emmert, J. T., et al. (2021). "NRLMSIS 2.0." *Earth and Space Science*, 8(3).

### Ocean
- Meissner, T. & Wentz, F. J. (2004/2012). "The complex dielectric constant of pure and sea water from microwave satellite observations." *IEEE TGRS*.
- Millard, R. C. & Seaver, G. (1990). "An index of refraction algorithm for seawater over temperature, pressure, salinity, density, and wavelength." *Deep-Sea Research*, 37(12), 1909–1926.
- Quan, X. & Fry, E. S. (1995). "Empirical equation for the index of refraction of seawater." *Applied Optics*, 34(18), 3477–3480.
- IAPWS (1997). "Release on the Refractive Index of Ordinary Water Substance as a Function of Wavelength, Temperature and Pressure." R9-97.
- Del Grosso, V. A. (1974). "New equation for the speed of sound in natural waters." *JASA*, 56(4), 1084–1091.
- Chen, C.-T. & Millero, F. J. (1977). "Speed of sound in seawater at high pressures." *JASA*, 62(5), 1129–1135.
- Mackenzie, K. V. (1981). "Nine-term equation for sound speed in the oceans." *JASA*, 70(3), 807–812.

### Ray Tracing
- Huynh, N. (2026). "Deriving Atmospheric and Acoustic Refraction from Fermat's Principle." (included in `docs/Ray Trace/`)

## License

[MIT](LICENSE) © 2026 Nen Huynh
