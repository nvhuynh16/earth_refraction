#include "test_utils.hpp"
#include <refraction/ocean/millard_seaver.hpp>

using namespace refraction::test;
using namespace refraction::ocean::ms;

static TestRunner runner("Millard & Seaver 1990 Tests");

// ── Table 2 validation (lambda = 589.26 nm) ──
// Each entry: {T_C, S_psu, p_dbar, expected_n}

struct MSVector {
    double T, S, p, n_expected;
};

static const MSVector table2[] = {
    // S=0, p=0 — Region I only (SD = 0.12 ppm)
    { 0,  0, 0,     1.333949},
    { 5,  0, 0,     1.333884},
    {10,  0, 0,     1.333691},
    {15,  0, 0,     1.333387},
    {20,  0, 0,     1.332988},
    {25,  0, 0,     1.332503},
    // S=35, p=0 — Regions I + II
    { 0, 35, 0,     1.340854},
    { 5, 35, 0,     1.340630},
    {10, 35, 0,     1.340298},
    {15, 35, 0,     1.339878},
    {20, 35, 0,     1.339386},
    {25, 35, 0,     1.338838},
    // S=40, p=0
    { 0, 40, 0,     1.341840},
    { 5, 40, 0,     1.341594},
    {10, 40, 0,     1.341242},
    {15, 40, 0,     1.340805},
    {20, 40, 0,     1.340300},
    {25, 40, 0,     1.339743},
    // S=30, p=0
    { 5, 30, 0,     1.339666},
    {10, 30, 0,     1.339354},
    {15, 30, 0,     1.338950},
    {20, 30, 0,     1.338472},
    {25, 30, 0,     1.337933},
    // S=0, p>0 — Regions I + III
    { 0,  0, 2000,  1.337122},
    { 0,  0, 4000,  1.340168},
    { 0,  0, 6000,  1.343089},
    { 0,  0, 10000, 1.348552},
    {20,  0, 2000,  1.335871},
    {20,  0, 4000,  1.338647},
    // S=35, p>0 — All four regions
    { 0, 35, 2000,  1.343948},
    {20, 35, 2000,  1.342228},
    { 0, 35, 4000,  1.346916},
    {20, 35, 4000,  1.344962},
    { 0, 35, 10000, 1.355065},
    {25, 35, 10000, 1.351836},
};

TEST_CASE(runner, table2_all) {
    int n = sizeof(table2) / sizeof(table2[0]);
    for (int i = 0; i < n; ++i) {
        const auto& v = table2[i];
        double n_calc = refractive_index(v.T, v.S, v.p, 589.26);
        std::ostringstream msg;
        msg << "Table 2 [" << i << "] T=" << v.T << " S=" << v.S << " p=" << v.p
            << ": expected " << std::setprecision(7) << v.n_expected
            << ", got " << n_calc
            << ", diff=" << std::setprecision(2) << 1e6 * std::abs(n_calc - v.n_expected) << " ppm";
        TEST_ASSERT_MSG(std::abs(n_calc - v.n_expected) < 15e-6, msg.str());
    }
}

// ── Physics consistency ──

TEST_CASE(runner, salinity_increases_n) {
    TEST_ASSERT(refractive_index(20, 35, 0, 589.26) > refractive_index(20, 0, 0, 589.26));
}

TEST_CASE(runner, pressure_increases_n) {
    TEST_ASSERT(refractive_index(20, 35, 2000, 589.26) > refractive_index(20, 35, 0, 589.26));
}

TEST_CASE(runner, temperature_decreases_n) {
    TEST_ASSERT(refractive_index(25, 35, 0, 589.26) < refractive_index(10, 35, 0, 589.26));
}

TEST_CASE(runner, normal_dispersion) {
    TEST_ASSERT(refractive_index(20, 35, 0, 500) > refractive_index(20, 35, 0, 700));
}

TEST_CASE(runner, pure_water_sodium_D) {
    TEST_ASSERT_NEAR(refractive_index(20, 0, 0, 589.26), 1.333, 0.001);
}

TEST_CASE(runner, seawater_sodium_D) {
    TEST_ASSERT_NEAR(refractive_index(20, 35, 0, 589.26), 1.339, 0.001);
}

TEST_CASE(runner, deep_ocean) {
    TEST_ASSERT_NEAR(refractive_index(0, 0, 10000, 589.26), 1.349, 0.001);
}

int main() {
    runner.run();
    return print_final_summary();
}
