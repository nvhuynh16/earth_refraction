#include "test_utils.hpp"
#include <refraction/profile.hpp>

using namespace refraction::test;
using namespace refraction;

static TestRunner runner("Profile Tests");

static RefractionProfile make_profile() {
    AtmosphericConditions atm;
    atm.day_of_year = 172;
    atm.ut_seconds = 29000;
    atm.latitude_deg = 45.0;
    atm.longitude_deg = -75.0;

    SurfaceObservation obs;
    obs.altitude_km = 0.0;
    obs.temperature_C = 20.0;
    obs.pressure_kPa = 101.325;
    obs.relative_humidity = 0.5;

    return RefractionProfile(atm, obs);
}

TEST_CASE(runner, ciddor_surface_reasonable) {
    auto prof = make_profile();
    auto r = prof.compute(0.0, Mode::Ciddor, 633.0);

    // Surface optical N should be ~270-280 N-units
    TEST_ASSERT(r.N > 250.0 && r.N < 300.0);
    TEST_ASSERT(r.n > 1.00025 && r.n < 1.00030);
}

TEST_CASE(runner, itu_surface_reasonable) {
    auto prof = make_profile();
    auto r = prof.compute(0.0, Mode::ITU_R_P453);

    // Surface radio N should be ~300-340 N-units (humidity adds more in radio)
    TEST_ASSERT(r.N > 280.0 && r.N < 360.0);
}

TEST_CASE(runner, profile_monotonically_decreasing) {
    auto prof = make_profile();
    auto results = prof.profile(0.0, 100.0, 5.0, Mode::Ciddor, 633.0);

    for (size_t i = 1; i < results.size(); i++) {
        TEST_ASSERT(results[i].N < results[i-1].N);
    }
}

TEST_CASE(runner, profile_approaches_vacuum) {
    auto prof = make_profile();
    auto r = prof.compute(100.0, Mode::Ciddor, 633.0);

    // At 100 km, N should be very small (< 1 N-unit)
    TEST_ASSERT(r.N < 1.0);
}

TEST_CASE(runner, dn_dh_negative) {
    auto prof = make_profile();
    // dn/dh should be negative throughout (refractivity decreasing with altitude)
    double alts[] = {0.0, 5.0, 10.0, 20.0, 50.0};
    for (double alt : alts) {
        auto r = prof.compute(alt, Mode::Ciddor, 633.0);
        std::ostringstream msg;
        msg << "dn_dh at " << alt << " km = " << r.dn_dh << " should be negative";
        TEST_ASSERT_MSG(r.dn_dh < 0.0, msg.str());
    }
}

TEST_CASE(runner, radio_surface_gradient) {
    auto prof = make_profile();
    auto r = prof.compute(0.0, Mode::ITU_R_P453);

    // Radio surface gradient should be roughly -40 to -50 N/km
    TEST_ASSERT(r.dN_dh < -20.0 && r.dN_dh > -80.0);
}

TEST_CASE(runner, ciddor_dn_dh_finite_difference) {
    auto prof = make_profile();
    double dh = 0.001;  // 1 m

    double alts[] = {1.0, 5.0, 10.0, 30.0, 50.0};
    for (double alt : alts) {
        auto r = prof.compute(alt, Mode::Ciddor, 633.0);
        auto r_plus = prof.compute(alt + dh, Mode::Ciddor, 633.0);
        auto r_minus = prof.compute(alt - dh, Mode::Ciddor, 633.0);

        double dn_dh_fd = (r_plus.n - r_minus.n) / (2.0 * dh);

        double tol = std::max(std::abs(dn_dh_fd) * 0.05, 1e-12);
        std::ostringstream msg;
        msg << std::setprecision(6) << "at " << alt << " km: analytical="
            << r.dn_dh << " FD=" << dn_dh_fd
            << " diff=" << std::abs(r.dn_dh - dn_dh_fd);
        TEST_ASSERT_MSG(std::abs(r.dn_dh - dn_dh_fd) < tol, msg.str());
    }
}

TEST_CASE(runner, itu_dn_dh_finite_difference) {
    auto prof = make_profile();
    double dh = 0.001;

    // Start from 1 km to avoid H2O profile kink at surface (constant below, decay above)
    double alts[] = {1.0, 5.0, 10.0, 30.0, 50.0};
    for (double alt : alts) {
        auto r = prof.compute(alt, Mode::ITU_R_P453);
        auto r_plus = prof.compute(alt + dh, Mode::ITU_R_P453);
        auto r_minus = prof.compute(alt - dh, Mode::ITU_R_P453);

        double dN_dh_fd = (r_plus.N - r_minus.N) / (2.0 * dh);

        double tol = std::max(std::abs(dN_dh_fd) * 0.05, 1e-6);
        std::ostringstream msg;
        msg << std::setprecision(6) << "at " << alt << " km: analytical="
            << r.dN_dh << " FD=" << dN_dh_fd
            << " diff=" << std::abs(r.dN_dh - dN_dh_fd);
        TEST_ASSERT_MSG(std::abs(r.dN_dh - dN_dh_fd) < tol, msg.str());
    }
}

TEST_CASE(runner, ciddor_vs_itu_surface) {
    auto prof = make_profile();
    auto r_opt = prof.compute(0.0, Mode::Ciddor, 550.0);
    auto r_radio = prof.compute(0.0, Mode::ITU_R_P453);

    // Both should give similar surface N (within ~30%)
    TEST_ASSERT_NEAR_REL(r_opt.N, r_radio.N, 0.30);
}

int main() {
    runner.run();
    return print_final_summary();
}
