#include "test_utils.hpp"
#include <refraction/ocean/ocean_profile.hpp>

using namespace refraction::test;
using namespace refraction::ocean;

static TestRunner runner("Ocean Profile Tests");

// ── Depth <-> Pressure ──

TEST_CASE(runner, surface_pressure) {
    TEST_ASSERT_NEAR(depth_to_pressure(0.0, 45.0), 0.0, 0.01);
}

TEST_CASE(runner, depth_1000m) {
    double p = depth_to_pressure(1000.0, 45.0);
    TEST_ASSERT(p > 1005.0 && p < 1015.0);
}

TEST_CASE(runner, round_trip) {
    double depths[] = {100, 500, 2000, 5000};
    double lats[] = {0, 30, 45, 60, 90};
    for (double z : depths) {
        for (double lat : lats) {
            double p = depth_to_pressure(z, lat);
            double z2 = pressure_to_depth(p, lat);
            std::ostringstream msg;
            msg << "z=" << z << " lat=" << lat << " p=" << p << " z2=" << z2;
            TEST_ASSERT_MSG(std::abs(z2 - z) < 0.5, msg.str());
        }
    }
}

TEST_CASE(runner, unesco_check_value) {
    // UNESCO: p=10000 dbar, lat=30 -> z=9712.653 m
    double z = pressure_to_depth(10000.0, 30.0);
    TEST_ASSERT_NEAR(z, 9712.65, 10.0);
}

TEST_CASE(runner, pressure_monotonic) {
    TEST_ASSERT(depth_to_pressure(200.0, 45.0) > depth_to_pressure(100.0, 45.0));
}

TEST_CASE(runner, latitude_effect) {
    TEST_ASSERT(depth_to_pressure(1000.0, 90.0) > depth_to_pressure(1000.0, 0.0));
}

// ── Parametric T(z), S(z) ──

TEST_CASE(runner, surface_temperature) {
    double T = temperature_at_depth(0.0, 30.0, 4.0, 200.0, 50.0);
    TEST_ASSERT_NEAR(T, 30.0, 0.5);
}

TEST_CASE(runner, deep_temperature) {
    double T = temperature_at_depth(2000.0, 30.0, 4.0, 50.0, 50.0);
    TEST_ASSERT_NEAR(T, 4.0, 0.5);
}

TEST_CASE(runner, thermocline_center) {
    // At z=MLD, T = (SST + T_deep) / 2
    double T = temperature_at_depth(50.0, 30.0, 4.0, 50.0, 50.0);
    TEST_ASSERT_NEAR(T, 17.0, 0.1);
}

TEST_CASE(runner, monotonic_cooling) {
    double prev = temperature_at_depth(0.0, 30.0, 4.0, 50.0, 50.0);
    for (int z = 10; z <= 500; z += 10) {
        double cur = temperature_at_depth(z, 30.0, 4.0, 50.0, 50.0);
        TEST_ASSERT(cur <= prev);
        prev = cur;
    }
}

// ── Presets ──

TEST_CASE(runner, florida_preset) {
    auto p = FLORIDA_GULF_SUMMER;
    double T = temperature_at_depth(0.0, p.sst_C, p.t_deep_C, p.mld_m, p.d_thermo_m);
    TEST_ASSERT(T > 22.0);
}

TEST_CASE(runner, california_preset) {
    auto p = CALIFORNIA_SUMMER;
    double T = temperature_at_depth(0.0, p.sst_C, p.t_deep_C, p.mld_m, p.d_thermo_m);
    TEST_ASSERT(T > 14.0);
}

int main() {
    runner.run();
    return print_final_summary();
}
