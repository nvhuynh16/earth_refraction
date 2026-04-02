#include "test_utils.hpp"
#include <refraction/ocean/del_grosso.hpp>
#include <refraction/ocean/ocean_profile.hpp>

using namespace refraction::test;
using namespace refraction::ocean;

static TestRunner runner("Del Grosso (1974) Sound Speed Tests");

// ── SONAR.m reference table at S=30 ──
// Source: gorbatschow.github.io/SonarDocs (S=30, not S=35 as captioned;
// confirmed via exact match with Mackenzie which uses depth directly).
// Del Grosso uses pressure internally, so depth->pressure via Saunders (lat=45).

struct DGVector { double Z, T, c_ref; };

static const DGVector sonar_s30[] = {
    {   10,  0, 1442.55}, {   10, 10, 1483.85}, {   10, 20, 1516.04}, {   10, 30, 1540.44},
    { 1000,  0, 1458.67}, { 1000, 10, 1500.30}, { 1000, 20, 1532.61}, { 1000, 30, 1556.65},
    { 2000,  0, 1475.45}, { 2000, 10, 1517.18}, { 2000, 20, 1549.48}, { 2000, 30, 1573.14},
    { 5000,  0, 1528.32}, { 5000, 10, 1569.16}, { 5000, 20, 1600.96}, { 5000, 30, 1623.67},
};

TEST_CASE(runner, sonar_shallow) {
    // Z=10 m: negligible pressure conversion ambiguity => tight tolerance
    for (int i = 0; i < 4; ++i) {
        const auto& v = sonar_s30[i];
        double p = depth_to_pressure(v.Z, 45.0);
        double c = dg::sound_speed(v.T, 30, p);
        std::ostringstream msg;
        msg << "Z=" << v.Z << " T=" << v.T << " c=" << c << " ref=" << v.c_ref;
        TEST_ASSERT_MSG(std::abs(c - v.c_ref) < 0.02, msg.str());
    }
}

TEST_CASE(runner, sonar_deep) {
    // Deeper points: allow 0.3 m/s for depth-to-pressure formula differences
    int n = sizeof(sonar_s30) / sizeof(sonar_s30[0]);
    for (int i = 4; i < n; ++i) {
        const auto& v = sonar_s30[i];
        double p = depth_to_pressure(v.Z, 45.0);
        double c = dg::sound_speed(v.T, 30, p);
        std::ostringstream msg;
        msg << "Z=" << v.Z << " T=" << v.T << " c=" << c << " ref=" << v.c_ref
            << " diff=" << std::abs(c - v.c_ref);
        TEST_ASSERT_MSG(std::abs(c - v.c_ref) < 0.3, msg.str());
    }
}

// ── Base constant check ──

TEST_CASE(runner, c000_at_origin) {
    // C000 = 1402.392 at T=0, S=0, P=0
    TEST_ASSERT_NEAR(dg::sound_speed(0, 0, 0), 1402.392, 0.001);
}

// ── Physics monotonicity ──

TEST_CASE(runner, temperature_increases_speed) {
    TEST_ASSERT(dg::sound_speed(25, 35, 0) > dg::sound_speed(5, 35, 0));
}

TEST_CASE(runner, salinity_increases_speed) {
    TEST_ASSERT(dg::sound_speed(10, 40, 0) > dg::sound_speed(10, 30, 0));
}

TEST_CASE(runner, pressure_increases_speed) {
    TEST_ASSERT(dg::sound_speed(10, 35, 5000) > dg::sound_speed(10, 35, 0));
}

TEST_CASE(runner, pressure_monotonic) {
    double prev = dg::sound_speed(10, 35, 0);
    for (double p = 1000; p <= 10000; p += 1000) {
        double c = dg::sound_speed(10, 35, p);
        TEST_ASSERT(c > prev);
        prev = c;
    }
}

// ── Physical range check ──

TEST_CASE(runner, output_range) {
    for (double T : {0.0, 10.0, 20.0, 30.0}) {
        for (double S : {30.0, 35.0, 40.0}) {
            double c = dg::sound_speed(T, S, 0);
            TEST_ASSERT(c > 1400.0 && c < 1600.0);
        }
    }
}

int main() {
    runner.run();
    return print_final_summary();
}
