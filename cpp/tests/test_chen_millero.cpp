#include "test_utils.hpp"
#include <refraction/ocean/chen_millero.hpp>
#include <refraction/ocean/del_grosso.hpp>
#include <refraction/ocean/ocean_profile.hpp>

using namespace refraction::test;
using namespace refraction::ocean;

static TestRunner runner("Chen-Millero / UNESCO (1977) Sound Speed Tests");

// ── Pond & Pickard (1986) reference values at P=0 ──
// IPTS-68 temperature scale; our ITS-90 coefficients give max ~0.03 m/s drift.

struct PPVector { double T, S, c_ref; };

static const PPVector pond_pickard[] = {
    // S=25
    { 0, 25, 1435.789875}, {10, 25, 1477.681131}, {20, 25, 1510.316386},
    {30, 25, 1535.214379}, {40, 25, 1553.440561},
    // S=35
    { 0, 35, 1449.138828}, {10, 35, 1489.830942}, {20, 35, 1521.475257},
    {30, 35, 1545.609796}, {40, 35, 1563.223222},
};

TEST_CASE(runner, pond_pickard_p0) {
    int n = sizeof(pond_pickard) / sizeof(pond_pickard[0]);
    for (int i = 0; i < n; ++i) {
        const auto& v = pond_pickard[i];
        double c = cm::sound_speed(v.T, v.S, 0);
        std::ostringstream msg;
        msg << "T=" << v.T << " S=" << v.S << " c=" << std::setprecision(7) << c
            << " ref=" << v.c_ref << " diff=" << std::abs(c - v.c_ref);
        TEST_ASSERT_MSG(std::abs(c - v.c_ref) < 0.03, msg.str());
    }
}

// ── UNESCO Technical Paper 44 check value ──
// S=40, T_68=40 C, P=10000 dbar => c=1731.995 m/s (IPTS-68)
// ITS-90 coefficients give ~1732.02; allow 0.05 m/s.

TEST_CASE(runner, unesco_check_value) {
    double c = cm::sound_speed(40, 40, 10000);
    TEST_ASSERT_NEAR(c, 1731.995, 0.05);
}

// ── Base constant check ──

TEST_CASE(runner, c00_at_origin) {
    // C00 = 1402.388 at T=0, S=0, P=0
    TEST_ASSERT_NEAR(cm::sound_speed(0, 0, 0), 1402.388, 0.001);
}

// ── SONAR.m cross-validation at S=30, Z=10 m ──

TEST_CASE(runner, sonar_shallow) {
    struct { double T, c_ref; } rows[] = {
        {0, 1442.62}, {10, 1483.92}, {20, 1516.06}, {30, 1540.59},
    };
    for (const auto& v : rows) {
        double p = depth_to_pressure(10, 45.0);
        double c = cm::sound_speed(v.T, 30, p);
        TEST_ASSERT_NEAR(c, v.c_ref, 0.02);
    }
}

// ── Physics monotonicity ──

TEST_CASE(runner, temperature_increases_speed) {
    TEST_ASSERT(cm::sound_speed(25, 35, 0) > cm::sound_speed(5, 35, 0));
}

TEST_CASE(runner, salinity_increases_speed) {
    TEST_ASSERT(cm::sound_speed(10, 40, 0) > cm::sound_speed(10, 0, 0));
}

TEST_CASE(runner, pressure_increases_speed) {
    TEST_ASSERT(cm::sound_speed(10, 35, 5000) > cm::sound_speed(10, 35, 0));
}

TEST_CASE(runner, fresh_water_valid) {
    double c = cm::sound_speed(20, 0, 0);
    TEST_ASSERT(c > 1480.0 && c < 1485.0);
}

// ── Cross-check with Del Grosso ──

TEST_CASE(runner, agrees_with_del_grosso_at_surface) {
    double c_cm = cm::sound_speed(10, 35, 0);
    double c_dg = dg::sound_speed(10, 35, 0);
    TEST_ASSERT_NEAR(c_cm, c_dg, 0.5);
}

TEST_CASE(runner, diverges_from_del_grosso_at_depth) {
    double c_cm = cm::sound_speed(0, 35, 10000);
    double c_dg = dg::sound_speed(0, 35, 10000);
    double diff = std::abs(c_cm - c_dg);
    TEST_ASSERT(diff > 0.1 && diff < 2.0);
}

int main() {
    runner.run();
    return print_final_summary();
}
