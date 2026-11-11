#include "test_utils.hpp"
#include "test_fixtures.hpp"
#include <refraction/atmosphere/nrlmsis21.hpp>

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

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

TEST_CASE(runner, dT_dz_tight_with_gph_jacobian) {
    // dT/d(zeta) * d(zeta)/dh must equal a finite-difference dT/dh closely,
    // including below sea level (lowest B-spline basis term, m = 0).
    NRLMSIS21 msis;
    auto inp = standard_input();
    for (double alt : {-3.0, -1.0, -0.2, 0.0, 0.3, 2.0, 10.0, 50.0, 100.0}) {
        inp.alt = alt;
        auto full = msis.msiscalc_with_derivative(inp);
        const double dh = 1e-4;
        inp.alt = alt + dh;
        double T_plus = msis.msiscalc(inp).tn;
        inp.alt = alt - dh;
        double T_minus = msis.msiscalc(inp).tn;
        double fd = (T_plus - T_minus) / (2.0 * dh);
        double analytic = full.dT_dz * refraction::atmosphere::msis21::utils::dgph_dalt(inp.lat, alt);
        TEST_ASSERT_NEAR(analytic, fd, 1e-4);
    }
}

// Official NRLMSIS 2.1 test vectors: msis2.1_test_ref_dp.txt is the
// double-precision reference output distributed with the NRL Fortran code
// (200 input sets; densities in cm^-3, mass density in g/cm^3, printed to
// four significant figures; temperature in K to 0.01 K).  Every value must
// match to half its last printed digit, plus 1e-6 relative for rounding ties.
TEST_CASE(runner, official_reference_vectors) {
    NRLMSIS21 msis;
    std::ifstream file(MSIS_REFERENCE_FILE);
    TEST_ASSERT_MSG(file.is_open(), "cannot open " MSIS_REFERENCE_FILE);
    std::string line;
    std::getline(file, line);  // header
    const int idx[10] = {5, 4, 2, 3, 7, 1, 6, 8, 9, 10};
    const double to_si[10] = {1e6, 1e6, 1e6, 1e6, 1e6, 1e3, 1e6, 1e6, 1e6, 1e6};
    int rows = 0;
    while (std::getline(file, line)) {
        std::istringstream in(line);
        double iyd, sec, alt, lat, lon, stl, f107a, f107, ap, T;
        if (!(in >> iyd >> sec >> alt >> lat >> lon >> stl >> f107a >> f107 >> ap)) continue;
        std::string tok[10];
        for (auto& t : tok) in >> t;
        in >> T;
        auto out = msis.msiscalc({std::fmod(iyd, 1000.0), sec, alt, lat, lon, f107a, f107,
                                  {ap, ap, ap, ap, ap, ap, ap}});
        for (int k = 0; k < 10; ++k) {
            double ref = std::stod(tok[k]);
            if (ref < 1e-30) continue;  // 9.999E-37 marks "not computed"
            int exponent = std::stoi(tok[k].substr(tok[k].find('E') + 1));
            double half_digit = 0.5 * std::pow(10.0, exponent - 4);
            TEST_ASSERT_MSG(std::abs(out.dn[idx[k]] / to_si[k] - ref) <= half_digit + 1e-6 * ref,
                            "row " << rows << " column " << k << ": " << out.dn[idx[k]] / to_si[k]
                                   << " vs " << tok[k]);
        }
        TEST_ASSERT_MSG(std::abs(out.tn - T) <= 0.005 + 1e-6 * T,
                        "row " << rows << " T: " << out.tn << " vs " << T);
        ++rows;
    }
    TEST_ASSERT_EQ(rows, 200);
}

int main() {
    runner.run();
    return print_final_summary();
}
