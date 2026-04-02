#include "test_utils.hpp"
#include <refraction/ray_trace.hpp>

using namespace refraction;
using namespace refraction::test;

static TestRunner runner("Ray Trace Tests");

// Helpers
static auto constant_tracer(double v0 = 1.0 / 1.000315) {
    return EikonalTracer{[v0](double) { return v0; },
                         [](double) { return 0.0; }};
}

static auto exponential_tracer(double n0 = 1.000315, double a = -39e-9) {
    auto [v, dv] = speed_from_eta(
        [n0, a](double h) { return n0 * std::exp(a * h); },
        [n0, a](double h) { return a * n0 * std::exp(a * h); },
        1.0);
    return EikonalTracer{v, dv};
}

static EikonalInput standard_input(double elev = 30.0, double az = 0.0) {
    return {45.0, 0.0, 0.0, elev, az};
}

// ===== Constant speed tests =====

TEST_CASE(runner, constant_straight_line) {
    auto tracer = constant_tracer();
    auto result = tracer.trace(standard_input(30.0, 45.0),
                               {.max_arc_length_m = 100000.0,
                                .detect_ground_hit = false});

    auto r0 = result.r[0];
    double sigma0 = 1.0 / result.v[0];
    auto t0 = vec::scale(1.0 / sigma0, result.p[0]);

    for (size_t k = 0; k < result.s.size(); ++k) {
        auto r_expected = vec::add(r0, vec::scale(result.s[k], t0));
        for (int i = 0; i < 3; ++i)
            TEST_ASSERT_NEAR(result.r[k][i], r_expected[i], 1.0);
    }
}

TEST_CASE(runner, constant_momentum_preserved) {
    auto tracer = constant_tracer();
    auto result = tracer.trace(standard_input(30.0),
                               {.max_arc_length_m = 50000.0,
                                .detect_ground_hit = false});

    auto p0 = result.p[0];
    for (size_t k = 0; k < result.s.size(); ++k)
        for (int i = 0; i < 3; ++i)
            TEST_ASSERT_NEAR(result.p[k][i], p0[i], 1e-12);
}

TEST_CASE(runner, constant_momentum_magnitude) {
    auto tracer = constant_tracer();
    auto result = tracer.trace(standard_input(45.0),
                               {.max_arc_length_m = 50000.0,
                                .detect_ground_hit = false});

    for (size_t k = 0; k < result.s.size(); ++k)
        TEST_ASSERT_NEAR(result.momentum_error[k], 0.0, 1e-13);
}

// ===== Momentum preservation =====

TEST_CASE(runner, exponential_momentum_preserved) {
    auto tracer = exponential_tracer();
    auto result = tracer.trace(standard_input(30.0),
                               {.target_altitude_m = 100000.0,
                                .max_arc_length_m = 500000.0,
                                .detect_ground_hit = false});

    for (size_t k = 0; k < result.s.size(); ++k)
        TEST_ASSERT_NEAR(result.momentum_error[k], 0.0, 1e-8);
}

// ===== Stop conditions =====

TEST_CASE(runner, stop_altitude) {
    auto tracer = constant_tracer();
    auto result = tracer.trace(standard_input(30.0),
                               {.target_altitude_m = 50000.0,
                                .max_arc_length_m = 500000.0,
                                .detect_ground_hit = false});

    TEST_ASSERT_EQ(result.terminated_by, std::string("altitude"));
    TEST_ASSERT_NEAR_REL(result.h_m.back(), 50000.0, 0.01);
}

TEST_CASE(runner, stop_ground_hit) {
    auto tracer = constant_tracer();
    auto result = tracer.trace({45.0, 0.0, 10000.0, -30.0, 0.0},
                               {.max_arc_length_m = 500000.0,
                                .detect_ground_hit = true});

    TEST_ASSERT_EQ(result.terminated_by, std::string("ground"));
    TEST_ASSERT(result.h_m.back() < 200.0);  // near ground
}

TEST_CASE(runner, stop_arc_length) {
    auto tracer = constant_tracer();
    double s_max = 50000.0;
    auto result = tracer.trace(standard_input(30.0),
                               {.max_arc_length_m = s_max,
                                .detect_ground_hit = false});

    TEST_ASSERT_EQ(result.terminated_by, std::string("arc_length"));
    TEST_ASSERT_NEAR_REL(result.s.back(), s_max, 0.01);
}

// ===== Derivatives =====

TEST_CASE(runner, r_prime_unit_length) {
    auto tracer = exponential_tracer();
    auto result = tracer.trace(standard_input(30.0),
                               {.target_altitude_m = 50000.0,
                                .max_arc_length_m = 500000.0,
                                .detect_ground_hit = false});

    for (size_t k = 0; k < result.s.size(); ++k)
        TEST_ASSERT_NEAR(vec::norm(result.r_prime[k]), 1.0, 1e-10);
}

TEST_CASE(runner, curvature_zero_constant) {
    auto tracer = constant_tracer();
    auto result = tracer.trace(standard_input(30.0),
                               {.max_arc_length_m = 50000.0,
                                .detect_ground_hit = false});

    for (size_t k = 0; k < result.s.size(); ++k) {
        TEST_ASSERT_NEAR(result.curvature[k], 0.0, 1e-13);
        TEST_ASSERT_NEAR(vec::norm(result.r_double_prime[k]), 0.0, 1e-13);
    }
}

TEST_CASE(runner, bending_zero_constant) {
    auto tracer = constant_tracer();
    auto result = tracer.trace(standard_input(30.0),
                               {.max_arc_length_m = 50000.0,
                                .detect_ground_hit = false});

    TEST_ASSERT_NEAR(result.bending_deg, 0.0, 1e-10);
}

TEST_CASE(runner, travel_time_increases) {
    auto tracer = exponential_tracer();
    auto result = tracer.trace(standard_input(30.0),
                               {.target_altitude_m = 50000.0,
                                .max_arc_length_m = 500000.0,
                                .detect_ground_hit = false});

    for (size_t k = 1; k < result.T.size(); ++k)
        TEST_ASSERT(result.T[k] > result.T[k-1]);
}

// ===== Bennett =====

TEST_CASE(runner, bennett_45deg) {
    double N0 = 315e-6, H = 7350.0;
    auto [v, dv] = speed_from_eta(
        [N0, H](double h) { return 1.0 + N0 * std::exp(-h / H); },
        [N0, H](double h) { return -N0/H * std::exp(-h / H); },
        1.0);
    auto tracer = EikonalTracer{v, dv};
    auto result = tracer.trace(standard_input(45.0),
                               {.target_altitude_m = 120000.0,
                                .max_arc_length_m = 1000000.0,
                                .detect_ground_hit = false});

    double bending_arcmin = std::abs(result.bending_deg) * 60.0;
    // Bennett ~1.0' at 45°, allow 25%
    TEST_ASSERT(bending_arcmin > 0.75 && bending_arcmin < 1.25);
}

// ===== North stays on meridian =====

TEST_CASE(runner, north_stays_meridian) {
    auto tracer = constant_tracer();
    auto result = tracer.trace({30.0, 45.0, 0.0, 30.0, 0.0},
                               {.max_arc_length_m = 100000.0,
                                .detect_ground_hit = false});

    for (size_t k = 0; k < result.s.size(); ++k)
        TEST_ASSERT_NEAR(result.lon_deg[k], 45.0, 1e-4);
}

// ===== speed_from_eta =====

TEST_CASE(runner, speed_from_eta_roundtrip) {
    double c_ref = 299792458.0;
    auto eta = [](double h) { return 1.000315 - 39e-9 * h; };
    auto deta = [](double) { return -39e-9; };
    auto [v, dv] = speed_from_eta(eta, deta, c_ref);

    double h = 5000.0;
    TEST_ASSERT_NEAR_REL(v(h), c_ref / eta(h), 1e-12);
    double expected_dv = -c_ref * deta(h) / (eta(h) * eta(h));
    TEST_ASSERT_NEAR_REL(dv(h), expected_dv, 1e-12);
}

// ===== Sensitivity: constant speed, dr/dh0 = n_hat =====

TEST_CASE(runner, sensitivity_dr_dh0_constant) {
    auto tracer = constant_tracer();
    auto [result, sens] = tracer.trace_with_sensitivities(
        standard_input(30.0, 45.0),
        {.max_arc_length_m = 50000.0, .detect_ground_hit = false});

    auto n_hat = geodetic_normal(45.0, 0.0);
    for (size_t k = 0; k < result.s.size(); ++k)
        for (int i = 0; i < 3; ++i)
            TEST_ASSERT_NEAR(sens.dr_dh0[k][i], n_hat[i], 1e-8);
}

// ===== Sensitivity: constant speed, dr/dtheta0 = s * dt/dtheta0 =====

TEST_CASE(runner, sensitivity_dr_dtheta0_constant) {
    auto tracer = constant_tracer();
    auto [result, sens] = tracer.trace_with_sensitivities(
        standard_input(30.0, 45.0),
        {.max_arc_length_m = 50000.0, .detect_ground_hit = false});

    double theta = 30.0 * DEG_TO_RAD;
    double alpha = 45.0 * DEG_TO_RAD;
    auto [E, N, U] = enu_frame(45.0, 0.0);
    Vec3 dt_dth = vec::add(vec::add(
        vec::scale(-std::sin(theta)*std::sin(alpha), E),
        vec::scale(-std::sin(theta)*std::cos(alpha), N)),
        vec::scale(std::cos(theta), U));

    for (size_t k = 1; k < result.s.size(); ++k) {
        auto expected = vec::scale(result.s[k], dt_dth);
        for (int i = 0; i < 3; ++i)
            TEST_ASSERT_NEAR(sens.dr_dtheta0[k][i], expected[i], 1.0);
    }
}

// ===== Sensitivity: momentum preserved with sensitivities enabled =====

TEST_CASE(runner, sensitivity_momentum_preserved) {
    auto tracer = exponential_tracer();
    auto [result, sens] = tracer.trace_with_sensitivities(
        standard_input(30.0),
        {.target_altitude_m = 50000.0, .max_arc_length_m = 500000.0,
         .detect_ground_hit = false});

    for (size_t k = 0; k < result.s.size(); ++k)
        TEST_ASSERT_NEAR(result.momentum_error[k], 0.0, 1e-8);
}

int main() {
    runner.run();
    return print_final_summary();
}
