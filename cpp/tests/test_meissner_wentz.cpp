#include "test_utils.hpp"
#include <refraction/ocean/meissner_wentz.hpp>

using namespace refraction::test;
using namespace refraction::ocean::mw;

static TestRunner runner("Meissner & Wentz 2004/2012 Tests");

// ── Pure-water Debye parameters (vs CRC Handbook) ──

TEST_CASE(runner, eps_s_pure_0C) {
    TEST_ASSERT_NEAR(eps_s_pure(0.0), 87.9, 0.2);
}

TEST_CASE(runner, eps_s_pure_20C) {
    TEST_ASSERT_NEAR(eps_s_pure(20.0), 80.2, 0.2);
}

TEST_CASE(runner, eps_s_pure_25C) {
    TEST_ASSERT_NEAR(eps_s_pure(25.0), 78.4, 0.2);
}

TEST_CASE(runner, eps_1_range) {
    double val = eps_1_pure(20.0);
    TEST_ASSERT(val > 5.0 && val < 7.0);
}

TEST_CASE(runner, eps_inf_range) {
    double val = eps_inf_pure(20.0);
    TEST_ASSERT(val > 3.5 && val < 5.0);
}

TEST_CASE(runner, v1_pure_range) {
    double val = v1_pure(20.0);
    TEST_ASSERT(val > 15.0 && val < 25.0);
}

TEST_CASE(runner, v2_pure_range) {
    double val = v2_pure(20.0);
    TEST_ASSERT(val > 80.0 && val < 400.0);
}

// ── Conductivity (Stogryn) ──

TEST_CASE(runner, sigma35_15C) {
    TEST_ASSERT_NEAR(sigma35(15.0), 4.29, 0.05);
}

TEST_CASE(runner, conductivity_25C_35psu) {
    // MW2004 Eq. 12: sigma35(25) = 5.306 S/m
    TEST_ASSERT_NEAR(conductivity(25.0, 35.0), 5.31, 0.1);
}

TEST_CASE(runner, conductivity_pure_water) {
    TEST_ASSERT_NEAR(conductivity(20.0, 0.0), 0.0, 1e-15);
}

TEST_CASE(runner, conductivity_positive) {
    double temps[] = {0.0, 10.0, 20.0, 29.0};
    double sals[] = {5.0, 20.0, 35.0, 40.0};
    for (double T : temps)
        for (double S : sals)
            TEST_ASSERT(conductivity(T, S) > 0.0);
}

// ── Salinity S=0 reduces to pure water ──

TEST_CASE(runner, eps_s_sea_at_zero_salinity) {
    TEST_ASSERT_NEAR_REL(eps_s_sea(20.0, 0.0), eps_s_pure(20.0), 1e-10);
}

TEST_CASE(runner, v1_sea_at_zero_salinity) {
    TEST_ASSERT_NEAR_REL(v1_sea(20.0, 0.0), v1_pure(20.0), 1e-10);
}

TEST_CASE(runner, eps_1_sea_at_zero_salinity) {
    TEST_ASSERT_NEAR_REL(eps_1_sea(20.0, 0.0), eps_1_pure(20.0), 1e-10);
}

TEST_CASE(runner, salt_reduces_eps_s) {
    TEST_ASSERT(eps_s_sea(20.0, 35.0) < eps_s_pure(20.0));
}

// ── Complex permittivity ──

TEST_CASE(runner, permittivity_pure_water_10ghz) {
    auto eps = permittivity(10.0, 20.0, 0.0);
    TEST_ASSERT(eps.real > 50.0 && eps.real < 65.0);
    TEST_ASSERT(eps.imag > 25.0 && eps.imag < 40.0);
}

TEST_CASE(runner, permittivity_seawater_10ghz) {
    auto eps = permittivity(10.0, 20.0, 35.0);
    TEST_ASSERT(eps.real > 35.0 && eps.real < 60.0);
    TEST_ASSERT(eps.imag > 30.0 && eps.imag < 50.0);
}

TEST_CASE(runner, static_limit) {
    auto eps = permittivity(0.01, 20.0, 35.0);
    double es = eps_s_sea(20.0, 35.0);
    TEST_ASSERT_NEAR_REL(eps.real, es, 0.01);
}

TEST_CASE(runner, no_nan_arctic) {
    auto eps = permittivity(10.0, -2.0, 35.0);
    TEST_ASSERT(std::isfinite(eps.real) && std::isfinite(eps.imag));
}

TEST_CASE(runner, eps_real_positive) {
    double freqs[] = {1.0, 10.0, 37.0, 85.0};
    double temps[] = {-2.0, 15.0, 29.0};
    for (double f : freqs)
        for (double T : temps)
            TEST_ASSERT(permittivity(f, T, 35.0).real > 0.0);
}

// ── Complex refractive index identity: n^2 = eps ──

TEST_CASE(runner, refractive_index_identity) {
    double temps[] = {0.0, 15.0, 25.0};
    double sals[] = {0.0, 35.0};
    double freqs[] = {1.4, 10.0, 37.0};
    for (double T : temps) {
        for (double S : sals) {
            for (double f : freqs) {
                auto eps = permittivity(f, T, S);
                auto n = refractive_index(f, T, S);
                double eps_r = n.n_real * n.n_real - n.n_imag * n.n_imag;
                double eps_i = 2.0 * n.n_real * n.n_imag;
                TEST_ASSERT_NEAR_REL(eps_r, eps.real, 1e-10);
                TEST_ASSERT_NEAR_REL(eps_i, eps.imag, 1e-10);
            }
        }
    }
}

TEST_CASE(runner, n_real_gt_1) {
    auto n = refractive_index(10.0, 20.0, 35.0);
    TEST_ASSERT(n.n_real > 1.0);
}

TEST_CASE(runner, n_imag_positive) {
    auto n = refractive_index(10.0, 20.0, 35.0);
    TEST_ASSERT(n.n_imag > 0.0);
}

int main() {
    runner.run();
    return print_final_summary();
}
