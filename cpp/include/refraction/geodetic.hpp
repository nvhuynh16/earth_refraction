#pragma once

// WGS-84 geodetic coordinate utilities.
//
// Provides conversions between geodetic (latitude, longitude, height above
// ellipsoid) and ECEF (Earth-Centered Earth-Fixed) Cartesian coordinates,
// plus local ENU frame, principal radii, and normal Jacobian.
//
// All public functions accept angles in degrees and heights in metres.

#include <array>
#include <cmath>
#include <utility>

#include "constants.hpp"

namespace refraction {

// -----------------------------------------------------------------
// 3-vector type and operations
// -----------------------------------------------------------------

using Vec3 = std::array<double, 3>;
using Mat3 = std::array<Vec3, 3>;  // row-major 3x3

namespace vec {

inline double dot(const Vec3& a, const Vec3& b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

inline double norm(const Vec3& a) {
    return std::sqrt(dot(a, a));
}

inline Vec3 scale(double c, const Vec3& a) {
    return {c*a[0], c*a[1], c*a[2]};
}

inline Vec3 add(const Vec3& a, const Vec3& b) {
    return {a[0]+b[0], a[1]+b[1], a[2]+b[2]};
}

inline Vec3 sub(const Vec3& a, const Vec3& b) {
    return {a[0]-b[0], a[1]-b[1], a[2]-b[2]};
}

inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]};
}

// Matrix-vector product (row-major)
inline Vec3 matvec(const Mat3& M, const Vec3& v) {
    return {dot(M[0], v), dot(M[1], v), dot(M[2], v)};
}

// Outer product a * b^T
inline Mat3 outer(const Vec3& a, const Vec3& b) {
    return {Vec3{a[0]*b[0], a[0]*b[1], a[0]*b[2]},
            Vec3{a[1]*b[0], a[1]*b[1], a[1]*b[2]},
            Vec3{a[2]*b[0], a[2]*b[1], a[2]*b[2]}};
}

// Matrix addition
inline Mat3 matadd(const Mat3& A, const Mat3& B) {
    return {Vec3{A[0][0]+B[0][0], A[0][1]+B[0][1], A[0][2]+B[0][2]},
            Vec3{A[1][0]+B[1][0], A[1][1]+B[1][1], A[1][2]+B[1][2]},
            Vec3{A[2][0]+B[2][0], A[2][1]+B[2][1], A[2][2]+B[2][2]}};
}

// Scale matrix
inline Mat3 matscale(double c, const Mat3& A) {
    return {scale(c, A[0]), scale(c, A[1]), scale(c, A[2])};
}

}  // namespace vec

// -----------------------------------------------------------------
// Geodetic structures
// -----------------------------------------------------------------

struct GeodeticCoord {
    double lat_deg;
    double lon_deg;
    double h_m;
};

struct EnuFrame {
    Vec3 E, N, U;
};

struct PrincipalRadii {
    double M;      // meridional
    double N_rad;  // prime-vertical
};

// -----------------------------------------------------------------
// Coordinate conversions
// -----------------------------------------------------------------

inline Vec3 geodetic_to_ecef(double lat_deg, double lon_deg, double h_m) {
    double lat = lat_deg * DEG_TO_RAD;
    double lon = lon_deg * DEG_TO_RAD;
    double sin_lat = std::sin(lat), cos_lat = std::cos(lat);
    double sin_lon = std::sin(lon), cos_lon = std::cos(lon);

    double N = WGS84_A / std::sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat);
    return {(N + h_m) * cos_lat * cos_lon,
            (N + h_m) * cos_lat * sin_lon,
            (N * (1.0 - WGS84_E2) + h_m) * sin_lat};
}

inline GeodeticCoord ecef_to_geodetic(double X, double Y, double Z) {
    double lon = std::atan2(Y, X);
    double p = std::sqrt(X * X + Y * Y);

    // Bowring iteration
    double lat = std::atan2(Z, p * (1.0 - WGS84_E2));
    for (int i = 0; i < 5; ++i) {
        double sin_lat = std::sin(lat);
        double N = WGS84_A / std::sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat);
        lat = std::atan2(Z + WGS84_E2 * N * sin_lat, p);
    }

    double sin_lat = std::sin(lat), cos_lat = std::cos(lat);
    double N = WGS84_A / std::sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat);
    double h;
    if (std::abs(cos_lat) > 1e-10)
        h = p / cos_lat - N;
    else
        h = std::abs(Z) / std::abs(sin_lat) - N * (1.0 - WGS84_E2);

    return {lat / DEG_TO_RAD, lon / DEG_TO_RAD, h};
}

// -----------------------------------------------------------------
// Normal and ENU frame
// -----------------------------------------------------------------

inline Vec3 geodetic_normal(double lat_deg, double lon_deg) {
    double lat = lat_deg * DEG_TO_RAD;
    double lon = lon_deg * DEG_TO_RAD;
    return {std::cos(lat) * std::cos(lon),
            std::cos(lat) * std::sin(lon),
            std::sin(lat)};
}

inline EnuFrame enu_frame(double lat_deg, double lon_deg) {
    double lat = lat_deg * DEG_TO_RAD;
    double lon = lon_deg * DEG_TO_RAD;
    double sl = std::sin(lat), cl = std::cos(lat);
    double sn = std::sin(lon), cn = std::cos(lon);
    return {Vec3{-sn, cn, 0.0},
            Vec3{-sl*cn, -sl*sn, cl},
            Vec3{cl*cn, cl*sn, sl}};
}

// -----------------------------------------------------------------
// Radii of curvature
// -----------------------------------------------------------------

inline PrincipalRadii principal_radii(double lat_deg) {
    double sin_lat = std::sin(lat_deg * DEG_TO_RAD);
    double w = 1.0 - WGS84_E2 * sin_lat * sin_lat;
    double sqrt_w = std::sqrt(w);
    return {WGS84_A * (1.0 - WGS84_E2) / (w * sqrt_w),
            WGS84_A / sqrt_w};
}

// -----------------------------------------------------------------
// Normal Jacobian (for sensitivity equations)
// -----------------------------------------------------------------

inline Mat3 normal_jacobian(double lat_deg, double lon_deg, double h_m) {
    auto [E, N_vec, U] = enu_frame(lat_deg, lon_deg);
    auto [M, N_rad] = principal_radii(lat_deg);
    auto NN = vec::outer(N_vec, N_vec);
    auto EE = vec::outer(E, E);
    return vec::matadd(vec::matscale(1.0 / (M + h_m), NN),
                       vec::matscale(1.0 / (N_rad + h_m), EE));
}

}  // namespace refraction
