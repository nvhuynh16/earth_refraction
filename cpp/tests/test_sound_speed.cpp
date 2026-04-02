#include "test_utils.hpp"
#include <refraction/ocean/sound_speed.hpp>
#include <refraction/ocean/del_grosso.hpp>
#include <refraction/ocean/chen_millero.hpp>
#include <refraction/ocean/mackenzie.hpp>

using namespace refraction::test;
using namespace refraction::ocean;

static TestRunner runner("SoundSpeedProfile Integration Tests");

// ── Compute basics ──

TEST_CASE(runner, compute_returns_valid_result) {
    OceanConditions cond{};
    cond.sst_C = 25.0;
    cond.sss_psu = 35.0;
    SoundSpeedProfile prof(cond);
    auto r = prof.compute(0.0, SoundMode::DelGrosso);
    TEST_ASSERT(r.sound_speed_m_s > 1400.0 && r.sound_speed_m_s < 1600.0);
    TEST_ASSERT(r.depth_m == 0.0);
    TEST_ASSERT(r.temperature_C > 20.0);
    TEST_ASSERT(r.salinity_psu > 30.0);
}

TEST_CASE(runner, del_grosso_surface_matches_standalone) {
    OceanConditions cond{};
    cond.sst_C = 20.0;
    cond.sss_psu = 35.0;
    SoundSpeedProfile prof(cond);
    auto r = prof.compute(0.0, SoundMode::DelGrosso);
    // Use actual T,S from profile (tanh model shifts T(0) from SST)
    double c_standalone = dg::sound_speed(r.temperature_C, r.salinity_psu, 0.0);
    TEST_ASSERT_NEAR(r.sound_speed_m_s, c_standalone, 0.01);
}

TEST_CASE(runner, chen_millero_surface_matches_standalone) {
    OceanConditions cond{};
    cond.sst_C = 20.0;
    cond.sss_psu = 35.0;
    SoundSpeedProfile prof(cond);
    auto r = prof.compute(0.0, SoundMode::ChenMillero);
    double c_standalone = cm::sound_speed(r.temperature_C, r.salinity_psu, 0.0);
    TEST_ASSERT_NEAR(r.sound_speed_m_s, c_standalone, 0.01);
}

TEST_CASE(runner, mackenzie_surface_matches_standalone) {
    OceanConditions cond{};
    cond.sst_C = 20.0;
    cond.sss_psu = 35.0;
    SoundSpeedProfile prof(cond);
    auto r = prof.compute(0.0, SoundMode::Mackenzie);
    double c_standalone = mk::sound_speed(r.temperature_C, r.salinity_psu, 0.0);
    TEST_ASSERT_NEAR(r.sound_speed_m_s, c_standalone, 0.01);
}

// ── dc/dz gradient ──

TEST_CASE(runner, dc_dz_is_finite) {
    OceanConditions cond{};
    cond.sst_C = 25.0;
    cond.sss_psu = 35.0;
    SoundSpeedProfile prof(cond);
    auto r = prof.compute(50.0, SoundMode::DelGrosso);
    TEST_ASSERT(std::isfinite(r.dc_dz));
    TEST_ASSERT(r.dc_dz != 0.0);
}

TEST_CASE(runner, dc_dz_negative_in_thermocline) {
    // In the thermocline, temperature drops rapidly -> sound speed decreases
    OceanConditions cond{};
    cond.sst_C = 25.0;
    cond.sss_psu = 35.0;
    cond.mld_m = 50.0;
    cond.d_thermo_m = 50.0;
    cond.t_deep_C = 4.0;
    SoundSpeedProfile prof(cond);
    auto r = prof.compute(75.0, SoundMode::DelGrosso);  // center of thermocline
    TEST_ASSERT(r.dc_dz < 0.0);
}

// ── Profile sweep ──

TEST_CASE(runner, profile_length) {
    OceanConditions cond{};
    cond.sst_C = 20.0;
    cond.sss_psu = 35.0;
    SoundSpeedProfile prof(cond);
    auto results = prof.profile(0, 100, 10, SoundMode::DelGrosso);
    TEST_ASSERT_EQ(static_cast<int>(results.size()), 11);  // 0,10,...,100
}

TEST_CASE(runner, profile_no_nan) {
    OceanConditions cond{};
    cond.sst_C = 25.0;
    cond.sss_psu = 35.0;
    SoundSpeedProfile prof(cond);
    auto results = prof.profile(0, 200, 5, SoundMode::DelGrosso);
    for (const auto& r : results) {
        TEST_ASSERT(std::isfinite(r.sound_speed_m_s));
        TEST_ASSERT(std::isfinite(r.dc_dz));
    }
}

// ── Preset profiles ──

TEST_CASE(runner, florida_preset) {
    SoundSpeedProfile prof(FLORIDA_GULF_SUMMER);
    auto r = prof.compute(0.0, SoundMode::DelGrosso);
    // Florida summer SST=30 (tanh-adjusted T(0) ~ 23-28)
    TEST_ASSERT(r.sound_speed_m_s > 1520.0);
}

TEST_CASE(runner, california_preset) {
    SoundSpeedProfile prof(CALIFORNIA_SUMMER);
    auto r = prof.compute(0.0, SoundMode::DelGrosso);
    // California summer SST=19, so c should be moderate
    TEST_ASSERT(r.sound_speed_m_s > 1500.0 && r.sound_speed_m_s < 1540.0);
}

// ── User-supplied arrays ──

TEST_CASE(runner, user_arrays) {
    std::vector<double> z = {0, 100};
    std::vector<double> T = {20, 10};
    std::vector<double> S = {35, 35};
    SoundSpeedProfile prof(z, T, S);
    auto r = prof.compute(0.0, SoundMode::DelGrosso);
    double c_expected = dg::sound_speed(20.0, 35.0, 0.0);
    TEST_ASSERT_NEAR(r.sound_speed_m_s, c_expected, 0.01);
}

TEST_CASE(runner, user_arrays_interpolation) {
    std::vector<double> z = {0, 100};
    std::vector<double> T = {20, 10};
    std::vector<double> S = {35, 35};
    SoundSpeedProfile prof(z, T, S);
    auto r = prof.compute(50.0, SoundMode::DelGrosso);
    // Interpolated T=15 at z=50
    TEST_ASSERT_NEAR(r.temperature_C, 15.0, 0.1);
}

// ── All three modes produce reasonable results ──

TEST_CASE(runner, all_modes_physical_range) {
    OceanConditions cond{};
    cond.sst_C = 15.0;
    cond.sss_psu = 35.0;
    SoundSpeedProfile prof(cond);
    for (auto mode : {SoundMode::DelGrosso, SoundMode::ChenMillero, SoundMode::Mackenzie}) {
        auto r = prof.compute(0.0, mode);
        TEST_ASSERT(r.sound_speed_m_s > 1400.0 && r.sound_speed_m_s < 1600.0);
    }
}

// ── Surface agreement across equations ──

TEST_CASE(runner, three_equations_agree_at_surface) {
    OceanConditions cond{};
    cond.sst_C = 15.0;
    cond.sss_psu = 35.0;
    SoundSpeedProfile prof(cond);
    auto r_dg = prof.compute(0.0, SoundMode::DelGrosso);
    auto r_cm = prof.compute(0.0, SoundMode::ChenMillero);
    auto r_mk = prof.compute(0.0, SoundMode::Mackenzie);
    TEST_ASSERT_NEAR(r_dg.sound_speed_m_s, r_cm.sound_speed_m_s, 0.5);
    TEST_ASSERT_NEAR(r_dg.sound_speed_m_s, r_mk.sound_speed_m_s, 0.5);
}

int main() {
    runner.run();
    return print_final_summary();
}
