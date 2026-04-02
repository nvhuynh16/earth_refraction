#include "test_utils.hpp"
#include <refraction/ocean/mackenzie.hpp>
#include <refraction/ocean/del_grosso.hpp>

using namespace refraction::test;
namespace mk = refraction::ocean::mk;
namespace dg = refraction::ocean::dg;

static TestRunner runner("Mackenzie (1981) Sound Speed Tests");

// ── SONAR.m reference table at S=30 ──
// Source: gorbatschow.github.io/SonarDocs.  Mackenzie uses depth directly,
// so all entries should match to rounding precision (0.01 m/s).

struct MKVector { double Z, T, c_ref; };

static const MKVector sonar_s30[] = {
    {   10,  0, 1442.42}, {   10, 10, 1483.78}, {   10, 20, 1515.95}, {   10, 30, 1540.36},
    { 1000,  0, 1458.73}, { 1000, 10, 1500.08}, { 1000, 20, 1532.24}, { 1000, 30, 1556.65},
    { 2000,  0, 1475.53}, { 2000, 10, 1516.83}, { 2000, 20, 1548.94}, { 2000, 30, 1573.30},
    { 5000,  0, 1527.95}, { 5000, 10, 1568.41}, { 5000, 20, 1599.69}, { 5000, 30, 1623.21},
};

TEST_CASE(runner, sonar_table) {
    int n = sizeof(sonar_s30) / sizeof(sonar_s30[0]);
    for (int i = 0; i < n; ++i) {
        const auto& v = sonar_s30[i];
        double c = mk::sound_speed(v.T, 30, v.Z);
        std::ostringstream msg;
        msg << "Z=" << v.Z << " T=" << v.T << " c=" << std::setprecision(7) << c
            << " ref=" << v.c_ref << " diff=" << std::abs(c - v.c_ref);
        TEST_ASSERT_MSG(std::abs(c - v.c_ref) < 0.01, msg.str());
    }
}

// ── Base constant check ──
// At T=0, S=35, Z=0: all (S-35), Z, and T terms vanish => c = C0 = 1448.96

TEST_CASE(runner, c0_at_s35_z0_t0) {
    TEST_ASSERT_NEAR(mk::sound_speed(0, 35, 0), 1448.96, 0.001);
}

// ── Physics monotonicity ──

TEST_CASE(runner, temperature_increases_speed) {
    TEST_ASSERT(mk::sound_speed(25, 35, 0) > mk::sound_speed(5, 35, 0));
}

TEST_CASE(runner, salinity_increases_speed) {
    TEST_ASSERT(mk::sound_speed(10, 40, 0) > mk::sound_speed(10, 25, 0));
}

TEST_CASE(runner, depth_increases_speed) {
    TEST_ASSERT(mk::sound_speed(10, 35, 4000) > mk::sound_speed(10, 35, 0));
}

TEST_CASE(runner, depth_monotonic) {
    double prev = mk::sound_speed(10, 35, 0);
    for (double z = 1000; z <= 8000; z += 1000) {
        double c = mk::sound_speed(10, 35, z);
        TEST_ASSERT(c > prev);
        prev = c;
    }
}

// ── Cross-check with Del Grosso (Mackenzie was fitted to Del Grosso) ──

TEST_CASE(runner, agrees_with_del_grosso_at_surface) {
    double c_mk = mk::sound_speed(10, 35, 0);
    double c_dg = dg::sound_speed(10, 35, 0);
    TEST_ASSERT_NEAR(c_mk, c_dg, 0.5);
}

int main() {
    runner.run();
    return print_final_summary();
}
