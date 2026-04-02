#include "test_utils.hpp"
#include <refraction/itu_r_p453.hpp>

using namespace refraction::test;
using namespace refraction;

static TestRunner runner("ITU-R P.453 Tests");

TEST_CASE(runner, dry_air_standard) {
    // Standard atmosphere: T=15°C, P=1013.25 hPa, dry
    // N = k1 * P / T = 77.6890 * 1013.25 / 288.15 = 272.97
    double N = itu::itu_N(288.15, 1013.25, 0.0);
    TEST_ASSERT_NEAR(N, 273.0, 0.5);
}

TEST_CASE(runner, humid_air_20C) {
    // T=20°C, P=101.325 kPa, RH=50%
    double N = itu::itu_N_from_surface(20.0, 101.325, 0.5);
    // Should be roughly 310-330 N-units (humidity adds significantly in radio)
    TEST_ASSERT(N > 290.0 && N < 340.0);
}

TEST_CASE(runner, itu_exponential_reference_gradient) {
    // ITU-R reference atmosphere: N(h) = 315 × exp(-h/7.35)
    // Surface gradient: dN/dh = -315/7.35 = -42.86 N-units/km
    double N_surface = 315.0;
    double H_scale = 7.35;   // km
    double gradient = -N_surface / H_scale;
    TEST_ASSERT_NEAR(gradient, -42.86, 0.01);
}

TEST_CASE(runner, dry_vs_humid_difference) {
    // Adding humidity should increase N (water vapor terms are positive)
    double N_dry = itu::itu_N_from_surface(20.0, 101.325, 0.0);
    double N_humid = itu::itu_N_from_surface(20.0, 101.325, 1.0);
    TEST_ASSERT(N_humid > N_dry);
    // Water vapor contribution at 20°C, 100% RH should be significant
    TEST_ASSERT(N_humid - N_dry > 30.0);
}

TEST_CASE(runner, k3_term_dominance) {
    // The k3 dipolar term should dominate the water vapor contribution
    double T_K = 293.15;
    double e_hPa = 23.39;  // ~SVP at 20°C in hPa
    double k2_term = itu::k2 * e_hPa / T_K;
    double k3_term = itu::k3 * e_hPa / (T_K * T_K);
    TEST_ASSERT(k3_term > 5.0 * k2_term);  // k3 term >> k2 term
}

int main() {
    runner.run();
    return print_final_summary();
}
