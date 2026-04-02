#pragma once

// Shared linear interpolation utility for sorted arrays.

#include <algorithm>
#include <cstddef>
#include <vector>

namespace refraction {

/// Linear interpolation in sorted arrays with boundary clamping.
///
/// Returns fp[0] for empty or single-element arrays.
/// Clamps to boundary values outside the range of xp.
inline double interp(double x, const std::vector<double>& xp,
                     const std::vector<double>& fp) {
    if (xp.empty()) return 0.0;
    if (xp.size() == 1) return fp[0];
    if (x <= xp.front()) return fp.front();
    if (x >= xp.back())  return fp.back();
    auto it = std::lower_bound(xp.begin(), xp.end(), x);
    size_t i = static_cast<size_t>(it - xp.begin());
    if (i == 0) i = 1;
    double t = (x - xp[i - 1]) / (xp[i] - xp[i - 1]);
    return fp[i - 1] + t * (fp[i] - fp[i - 1]);
}

}  // namespace refraction
