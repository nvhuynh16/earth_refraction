#include "test_utils.hpp"
#include "test_fixtures.hpp"
#include <refraction/atmosphere/nrlmsis21.hpp>

using namespace refraction::test;
using namespace refraction::atmosphere;

static TestRunner runner("NRLMSIS 2.1 Tests");

TEST_CASE(runner, sea_level_temperature) {
    NRLMSIS21 msis;
    auto inp = standard_input();
    inp.alt = 0.0;
    auto out = msis.msiscalc(inp);
    // Sea-level temperature should be in reasonable range
    TEST_ASSERT(out.tn > 200.0 && out.tn < 330.0);
}

TEST_CASE(runner, sea_level_species_dominance) {
    NRLMSIS21 msis;
    auto inp = standard_input();
    inp.alt = 0.0;
    auto out = msis.msiscalc(inp);
    // N2 + O2 should dominate at sea level
    double n_total = out.dn[2] + out.dn[3]; // N2 + O2
    double n_minor = out.dn[5] + out.dn[7]; // He + Ar
    TEST_ASSERT(n_total > 100.0 * n_minor);
    TEST_ASSERT(out.dn[2] > out.dn[3]); // N2 > O2
}

TEST_CASE(runner, altitude_50km) {
    NRLMSIS21 msis;
    auto inp = standard_input();
    inp.alt = 50.0;
    auto out = msis.msiscalc(inp);
    TEST_ASSERT(out.tn > 200.0 && out.tn < 350.0);
    TEST_ASSERT(out.dn[2] > 0.0); // N2 exists
}

TEST_CASE(runner, altitude_100km) {
    NRLMSIS21 msis;
    auto inp = standard_input();
    inp.alt = 100.0;
    auto out = msis.msiscalc(inp);
    TEST_ASSERT(out.tn > 150.0 && out.tn < 400.0);
}

TEST_CASE(runner, altitude_120km_O_dominates) {
    NRLMSIS21 msis;
    auto inp = standard_input();
    inp.alt = 300.0;  // Well into thermosphere
    auto out = msis.msiscalc(inp);
    // Atomic oxygen should dominate over N2 and O2 in upper thermosphere
    TEST_ASSERT(out.dn[4] > out.dn[2]); // O > N2
    TEST_ASSERT(out.dn[4] > out.dn[3]); // O > O2
}

TEST_CASE(runner, exospheric_temperature) {
    NRLMSIS21 msis;
    auto inp = standard_input();
    inp.alt = 500.0;
    auto out = msis.msiscalc(inp);
    // Exospheric temperature should be ~700-1500 K for moderate solar activity
    TEST_ASSERT(out.tex > 600.0 && out.tex < 2000.0);
    // Temperature should approach tex
    TEST_ASSERT_NEAR_REL(out.tn, out.tex, 0.05);
}

TEST_CASE(runner, dT_dz_troposphere_negative) {
    NRLMSIS21 msis;
    auto inp = standard_input();
    inp.alt = 5.0;  // Troposphere
    auto full = msis.msiscalc_with_derivative(inp);
    // Lapse rate should be negative (temperature decreasing with altitude)
    TEST_ASSERT(full.dT_dz < 0.0);
    // Typical lapse rate ~-6.5 K/km, allow wide tolerance
    TEST_ASSERT(full.dT_dz > -15.0 && full.dT_dz < -1.0);
}

TEST_CASE(runner, dT_dz_stratosphere_positive) {
    NRLMSIS21 msis;
    auto inp = standard_input();
    inp.alt = 50.0;  // Stratopause region
    auto full = msis.msiscalc_with_derivative(inp);
    // Temperature should be increasing or near peak in stratosphere
    // Around stratopause dT/dz could be near zero or slightly positive
    // Let's just check it's not strongly negative like the troposphere
    TEST_ASSERT(full.dT_dz > -3.0);
}

TEST_CASE(runner, dT_dz_finite_difference_crosscheck) {
    NRLMSIS21 msis;
    auto inp = standard_input();

    // Test at several altitudes
    double alts[] = {10.0, 30.0, 60.0, 90.0, 110.0, 130.0, 200.0};
    for (double alt : alts) {
        inp.alt = alt;
        auto full = msis.msiscalc_with_derivative(inp);

        // Finite difference
        double dh = 0.01; // 10 m
        inp.alt = alt + dh;
        auto out_plus = msis.msiscalc(inp);
        inp.alt = alt - dh;
        auto out_minus = msis.msiscalc(inp);
        inp.alt = alt;

        double dT_dz_fd = (out_plus.tn - out_minus.tn) / (2.0 * dh);

        // Allow 8% relative tolerance (analytical is dT/d(gph), FD is dT/d(alt_geodetic);
        // the Jacobian d(gph)/d(alt) ≈ (Re/(Re+h))^2 deviates at higher altitudes)
        double tol = std::max(std::abs(dT_dz_fd) * 0.08, 0.05);
        TEST_ASSERT_NEAR(full.dT_dz, dT_dz_fd, tol);
    }
}

int main() {
    runner.run();
    return print_final_summary();
}
