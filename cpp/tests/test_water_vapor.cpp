#include "test_utils.hpp"
#include <refraction/water_vapor.hpp>

using namespace refraction::test;
using namespace refraction;

static TestRunner runner("Water Vapor Tests");

TEST_CASE(runner, svp_at_20C) {
    double svp = svp_giacomo(293.15);  // 20°C
    TEST_ASSERT_NEAR(svp, 2339.0, 5.0);  // NIST reference: ~2339 Pa
}

TEST_CASE(runner, svp_at_50C) {
    double svp = svp_giacomo(323.15);  // 50°C
    TEST_ASSERT_NEAR(svp, 12351.0, 50.0);
}

TEST_CASE(runner, svp_at_100C) {
    double svp = svp_giacomo(373.15);  // 100°C
    TEST_ASSERT_NEAR(svp, 101418.0, 500.0);
}

TEST_CASE(runner, enhancement_factor_standard) {
    double f = enhancement_factor(20.0, 101325.0);
    // Should be close to 1.004
    TEST_ASSERT(f > 1.003 && f < 1.005);
}

TEST_CASE(runner, mole_fraction_20C_100RH) {
    double x_w = water_vapor_mole_fraction(20.0, 101325.0, 1.0);
    // At 20°C, 100% RH: x_w ≈ svp/P ≈ 2339/101325 ≈ 0.0231
    TEST_ASSERT_NEAR(x_w, 0.0232, 0.001);
}

TEST_CASE(runner, mole_fraction_dry) {
    double x_w = water_vapor_mole_fraction(20.0, 101325.0, 0.0);
    TEST_ASSERT_NEAR(x_w, 0.0, 1e-15);
}

TEST_CASE(runner, profile_decay) {
    double n0 = 1e23;
    double n_5km = water_vapor_profile(n0, 5.0, 0.0, 2.0);
    double expected = n0 * std::exp(-5.0 / 2.0);
    TEST_ASSERT_NEAR_REL(n_5km, expected, 1e-10);
}

TEST_CASE(runner, profile_negligible_above_15km) {
    double n0 = 1e23;
    double n_15km = water_vapor_profile(n0, 15.0, 0.0, 2.0);
    // exp(-15/2) ≈ 5.5e-4, so n_15km ≈ 5.5e19, negligible compared to n0
    TEST_ASSERT(n_15km < n0 * 1e-3);
}

int main() {
    runner.run();
    return print_final_summary();
}
