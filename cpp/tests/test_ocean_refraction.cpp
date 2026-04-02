#include "test_utils.hpp"
#include <refraction/ocean/ocean_refraction.hpp>

using namespace refraction::test;
using namespace refraction::ocean;

static TestRunner runner("Ocean Refraction Integration Tests");

// ── Quan & Fry spot checks ──

TEST_CASE(runner, quan_fry_pure_water) {
    TEST_ASSERT_NEAR(qf::refractive_index(20.0, 0.0, 589.0), 1.333, 0.001);
}

TEST_CASE(runner, quan_fry_seawater) {
    TEST_ASSERT_NEAR(qf::refractive_index(20.0, 35.0, 589.0), 1.339, 0.001);
}

TEST_CASE(runner, quan_fry_dispersion) {
    TEST_ASSERT(qf::refractive_index(20, 35, 450) > qf::refractive_index(20, 35, 650));
}

// ── IAPWS spot checks ──

TEST_CASE(runner, iapws_check_value) {
    // IAPWS Table 3: T=25C, rho=997.047, lam=589.3nm -> n=1.33285
    double n = iapws::refractive_index(25.0, 589.3, 997.047);
    TEST_ASSERT_NEAR(n, 1.33285, 0.00002);
}

TEST_CASE(runner, iapws_density_4C) {
    TEST_ASSERT_NEAR(iapws::pure_water_density(4.0), 1000.0, 0.1);
}

TEST_CASE(runner, iapws_density_20C) {
    TEST_ASSERT_NEAR(iapws::pure_water_density(20.0), 998.2, 0.5);
}

// ── OceanRefractionProfile: radio mode ──

TEST_CASE(runner, radio_surface) {
    OceanConditions cond;
    cond.sst_C = 25.0;
    cond.sss_psu = 35.0;
    cond.freq_ghz = 10.0;
    OceanRefractionProfile prof(cond);
    auto r = prof.compute(0.0, OceanMode::MeissnerWentz);

    // Cross-check against standalone
    auto n = mw::refractive_index(10.0, 25.0, 35.0);
    TEST_ASSERT_NEAR_REL(r.n_real, n.n_real, 0.01);
    TEST_ASSERT(r.eps_real > 0.0);
    TEST_ASSERT(r.eps_imag > 0.0);
}

TEST_CASE(runner, radio_dn_dz_finite) {
    OceanConditions cond;
    cond.sst_C = 25.0;
    cond.sss_psu = 35.0;
    OceanRefractionProfile prof(cond);
    auto r = prof.compute(50.0, OceanMode::MeissnerWentz);
    TEST_ASSERT(std::isfinite(r.dn_dz));
}

// ── OceanRefractionProfile: optical modes ──

TEST_CASE(runner, millard_seaver_surface) {
    OceanConditions cond;
    cond.sst_C = 25.0;
    cond.sss_psu = 35.0;
    cond.wavelength_nm = 589.26;
    OceanRefractionProfile prof(cond);
    auto r = prof.compute(0.0, OceanMode::MillardSeaver);

    double n = ms::refractive_index(25.0, 35.0, 0.0, 589.26);
    TEST_ASSERT_NEAR_REL(r.n_real, n, 0.01);
    TEST_ASSERT_NEAR(r.n_imag, 0.0, 1e-15);
}

TEST_CASE(runner, quan_fry_surface) {
    OceanConditions cond;
    cond.sst_C = 20.0;
    cond.sss_psu = 35.0;
    cond.wavelength_nm = 589.0;
    OceanRefractionProfile prof(cond);
    auto r = prof.compute(0.0, OceanMode::QuanFry);

    double n = qf::refractive_index(20.0, 35.0, 589.0);
    TEST_ASSERT_NEAR_REL(r.n_real, n, 0.01);
}

// ── Profile sweep ──

TEST_CASE(runner, profile_length) {
    OceanConditions cond;
    cond.sst_C = 25.0;
    cond.sss_psu = 35.0;
    OceanRefractionProfile prof(cond);
    auto results = prof.profile(0.0, 100.0, 10.0, OceanMode::MeissnerWentz);
    TEST_ASSERT_EQ(static_cast<int>(results.size()), 11);  // 0,10,...,100
}

TEST_CASE(runner, profile_no_nan) {
    OceanConditions cond;
    cond.sst_C = 25.0;
    cond.sss_psu = 35.0;
    OceanRefractionProfile prof(cond);
    auto results = prof.profile(0.0, 200.0, 5.0, OceanMode::MeissnerWentz);
    for (const auto& r : results)
        TEST_ASSERT(std::isfinite(r.n_real));
}

// ── Preset profiles ──

TEST_CASE(runner, florida_preset) {
    OceanRefractionProfile prof(FLORIDA_GULF_SUMMER);
    auto r = prof.compute(0.0, OceanMode::MeissnerWentz);
    TEST_ASSERT(r.temperature_C > 22.0);
}

TEST_CASE(runner, california_preset) {
    OceanRefractionProfile prof(CALIFORNIA_SUMMER);
    auto r = prof.compute(0.0, OceanMode::MeissnerWentz);
    TEST_ASSERT(r.temperature_C > 14.0);
}

// ── User-supplied arrays ──

TEST_CASE(runner, user_arrays) {
    std::vector<double> depths = {0, 50, 100, 200, 500};
    std::vector<double> temps  = {25, 20, 15, 8, 4};
    std::vector<double> sals   = {35, 35, 35, 34.5, 34.5};
    OceanRefractionProfile prof(depths, temps, sals);
    auto r = prof.compute(0.0, OceanMode::MeissnerWentz);
    TEST_ASSERT_NEAR(r.temperature_C, 25.0, 0.1);
}

TEST_CASE(runner, user_arrays_interpolated) {
    std::vector<double> depths = {0, 100};
    std::vector<double> temps  = {25, 5};
    std::vector<double> sals   = {35, 35};
    OceanRefractionProfile prof(depths, temps, sals);
    auto r = prof.compute(50.0, OceanMode::MeissnerWentz);
    TEST_ASSERT_NEAR(r.temperature_C, 15.0, 0.1);
}

int main() {
    runner.run();
    return print_final_summary();
}
