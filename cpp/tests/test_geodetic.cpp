#include "test_utils.hpp"
#include <refraction/geodetic.hpp>

using namespace refraction;
using namespace refraction::test;

static TestRunner runner("Geodetic Tests");

TEST_CASE(runner, to_ecef_equator) {
    auto r = geodetic_to_ecef(0.0, 0.0, 0.0);
    TEST_ASSERT_NEAR(r[0], WGS84_A, 1e-6);
    TEST_ASSERT_NEAR(r[1], 0.0, 1e-6);
    TEST_ASSERT_NEAR(r[2], 0.0, 1e-6);
}

TEST_CASE(runner, to_ecef_north_pole) {
    auto r = geodetic_to_ecef(90.0, 0.0, 0.0);
    TEST_ASSERT_NEAR(r[0], 0.0, 1e-6);
    TEST_ASSERT_NEAR(r[1], 0.0, 1e-6);
    TEST_ASSERT_NEAR(r[2], WGS84_B, 1e-6);
}

TEST_CASE(runner, to_ecef_with_height) {
    double h = 10000.0;
    auto r = geodetic_to_ecef(0.0, 0.0, h);
    TEST_ASSERT_NEAR(r[0], WGS84_A + h, 1e-6);
}

TEST_CASE(runner, roundtrip_equator) {
    auto r = geodetic_to_ecef(0.0, 0.0, 0.0);
    auto g = ecef_to_geodetic(r[0], r[1], r[2]);
    TEST_ASSERT_NEAR(g.lat_deg, 0.0, 1e-9);
    TEST_ASSERT_NEAR(g.lon_deg, 0.0, 1e-9);
    TEST_ASSERT_NEAR(g.h_m, 0.0, 1e-3);
}

TEST_CASE(runner, roundtrip_pole) {
    auto r = geodetic_to_ecef(90.0, 0.0, 0.0);
    auto g = ecef_to_geodetic(r[0], r[1], r[2]);
    TEST_ASSERT_NEAR(g.lat_deg, 90.0, 1e-9);
    TEST_ASSERT_NEAR(g.h_m, 0.0, 1e-3);
}

TEST_CASE(runner, roundtrip_midlat) {
    double lat = 45.0, lon = 30.0, h = 5000.0;
    auto r = geodetic_to_ecef(lat, lon, h);
    auto g = ecef_to_geodetic(r[0], r[1], r[2]);
    TEST_ASSERT_NEAR(g.lat_deg, lat, 1e-9);
    TEST_ASSERT_NEAR(g.lon_deg, lon, 1e-9);
    TEST_ASSERT_NEAR(g.h_m, h, 1e-3);
}

TEST_CASE(runner, roundtrip_high_altitude) {
    double lat = -30.0, lon = 120.0, h = 100000.0;
    auto r = geodetic_to_ecef(lat, lon, h);
    auto g = ecef_to_geodetic(r[0], r[1], r[2]);
    TEST_ASSERT_NEAR(g.lat_deg, lat, 1e-9);
    TEST_ASSERT_NEAR(g.lon_deg, lon, 1e-9);
    TEST_ASSERT_NEAR(g.h_m, h, 1e-3);
}

TEST_CASE(runner, normal_equator) {
    auto n = geodetic_normal(0.0, 0.0);
    TEST_ASSERT_NEAR(n[0], 1.0, 1e-15);
    TEST_ASSERT_NEAR(n[1], 0.0, 1e-15);
    TEST_ASSERT_NEAR(n[2], 0.0, 1e-15);
}

TEST_CASE(runner, normal_pole) {
    auto n = geodetic_normal(90.0, 0.0);
    TEST_ASSERT_NEAR(n[0], 0.0, 1e-15);
    TEST_ASSERT_NEAR(n[1], 0.0, 1e-15);
    TEST_ASSERT_NEAR(n[2], 1.0, 1e-15);
}

TEST_CASE(runner, normal_unit_length) {
    double lats[] = {-60, -30, 0, 30, 60, 90};
    double lons[] = {-90, 0, 45, 135};
    for (double lat : lats) {
        for (double lon : lons) {
            auto n = geodetic_normal(lat, lon);
            TEST_ASSERT_NEAR(vec::norm(n), 1.0, 1e-15);
        }
    }
}

TEST_CASE(runner, enu_equator) {
    auto [E, N, U] = enu_frame(0.0, 0.0);
    TEST_ASSERT_NEAR(E[0], 0.0, 1e-15);
    TEST_ASSERT_NEAR(E[1], 1.0, 1e-15);
    TEST_ASSERT_NEAR(E[2], 0.0, 1e-15);
    TEST_ASSERT_NEAR(N[0], 0.0, 1e-15);
    TEST_ASSERT_NEAR(N[1], 0.0, 1e-15);
    TEST_ASSERT_NEAR(N[2], 1.0, 1e-15);
    TEST_ASSERT_NEAR(U[0], 1.0, 1e-15);
    TEST_ASSERT_NEAR(U[1], 0.0, 1e-15);
    TEST_ASSERT_NEAR(U[2], 0.0, 1e-15);
}

TEST_CASE(runner, enu_orthonormal) {
    double lats[] = {-60, 0, 45, 85};
    double lons[] = {-90, 0, 45, 135};
    for (double lat : lats) {
        for (double lon : lons) {
            auto [E, N, U] = enu_frame(lat, lon);
            TEST_ASSERT_NEAR(vec::dot(E, N), 0.0, 1e-14);
            TEST_ASSERT_NEAR(vec::dot(E, U), 0.0, 1e-14);
            TEST_ASSERT_NEAR(vec::dot(N, U), 0.0, 1e-14);
            TEST_ASSERT_NEAR(vec::norm(E), 1.0, 1e-15);
            TEST_ASSERT_NEAR(vec::norm(N), 1.0, 1e-15);
            TEST_ASSERT_NEAR(vec::norm(U), 1.0, 1e-15);
        }
    }
}

TEST_CASE(runner, enu_right_handed) {
    double lats[] = {-45, 0, 45};
    double lons[] = {-90, 0, 90};
    for (double lat : lats) {
        for (double lon : lons) {
            auto [E, N, U] = enu_frame(lat, lon);
            auto c = vec::cross(E, N);
            TEST_ASSERT_NEAR(c[0], U[0], 1e-14);
            TEST_ASSERT_NEAR(c[1], U[1], 1e-14);
            TEST_ASSERT_NEAR(c[2], U[2], 1e-14);
        }
    }
}

TEST_CASE(runner, radii_equator) {
    auto [M, N_rad] = principal_radii(0.0);
    TEST_ASSERT_NEAR_REL(N_rad, WGS84_A, 1e-12);
    TEST_ASSERT_NEAR_REL(M, WGS84_A * (1.0 - WGS84_E2), 1e-12);
}

TEST_CASE(runner, radii_pole) {
    auto [M, N_rad] = principal_radii(90.0);
    double expected = WGS84_A * WGS84_A / WGS84_B;
    TEST_ASSERT_NEAR_REL(M, expected, 1e-10);
    TEST_ASSERT_NEAR_REL(N_rad, expected, 1e-10);
}

TEST_CASE(runner, normal_jacobian_symmetric) {
    auto J = normal_jacobian(45.0, 30.0, 1000.0);
    for (int i = 0; i < 3; ++i)
        for (int j = i + 1; j < 3; ++j)
            TEST_ASSERT_NEAR(J[i][j], J[j][i], 1e-10);
}

TEST_CASE(runner, normal_jacobian_vs_fd) {
    double lat = 45.0, lon = 30.0, h = 5000.0;
    auto J_analytical = normal_jacobian(lat, lon, h);
    auto r0 = geodetic_to_ecef(lat, lon, h);
    double delta = 1.0;

    Mat3 J_fd{};
    for (int j = 0; j < 3; ++j) {
        Vec3 r_p = r0, r_m = r0;
        r_p[j] += delta;
        r_m[j] -= delta;
        auto gp = ecef_to_geodetic(r_p[0], r_p[1], r_p[2]);
        auto gm = ecef_to_geodetic(r_m[0], r_m[1], r_m[2]);
        auto n_p = geodetic_normal(gp.lat_deg, gp.lon_deg);
        auto n_m = geodetic_normal(gm.lat_deg, gm.lon_deg);
        for (int i = 0; i < 3; ++i)
            J_fd[i][j] = (n_p[i] - n_m[i]) / (2.0 * delta);
    }

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            TEST_ASSERT_NEAR(J_analytical[i][j], J_fd[i][j], 1e-8);
}

int main() {
    runner.run();
    return print_final_summary();
}
