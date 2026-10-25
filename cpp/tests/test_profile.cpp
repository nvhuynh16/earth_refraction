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

TEST_CASE(runner, anchored_surface_matches_direct_formulas) {
    // At the station, the species-sum profile must reproduce the direct
    // Ciddor and ITU-R values computed from the same observation.
    AtmosphericConditions atm;
    atm.day_of_year = 172;
    atm.ut_seconds = 29000;
    atm.latitude_deg = 45.0;
    atm.longitude_deg = -75.0;

    struct Case { double T_C, P_kPa, RH, h; };
    for (auto c : {Case{15.0, 101.325, 0.0, 0.0}, Case{30.0, 101.325, 0.0, 0.0},
                   Case{30.0, 101.325, 0.8, 0.0}, Case{0.0, 98.0, 0.5, 0.3},
                   Case{20.0, 70.0, 0.4, 3.0}}) {
        RefractionProfile prof(atm, SurfaceObservation{c.h, c.T_C, c.P_kPa, c.RH});
        // Ciddor includes compressibility and CO2 terms the species sum omits
        double N_ciddor = (ciddor::ciddor_n(c.T_C, c.P_kPa, c.RH, 633.0) - 1.0) * 1e6;
        TEST_ASSERT_NEAR(prof.compute(c.h, Mode::Ciddor, 633.0).N, N_ciddor, 0.2);
        double N_itu = itu::itu_N_from_surface(c.T_C, c.P_kPa, c.RH);
        TEST_ASSERT_NEAR(prof.compute(c.h, Mode::ITU_R_P453).N, N_itu, 1e-6);
    }
}

TEST_CASE(runner, itu_dn_dh_tight_humid) {
    // Radio dN/dh must match a numerical derivative of N closely.  The
    // dipolar water-vapor term needs dT/dh; NRLMSIS returns dT/d(zeta)
    // (geopotential), which without conversion gives ~1e-5 relative error.
    AtmosphericConditions atm;
    atm.day_of_year = 172;
    atm.ut_seconds = 29000;
    atm.latitude_deg = 45.0;
    atm.longitude_deg = -75.0;
    RefractionProfile prof(atm, SurfaceObservation{0.0, 30.0, 101.325, 0.8});
    const double dh = 1e-3;
    for (double alt : {1.0, 2.0, 5.0, 10.0}) {
        double fd = (prof.compute(alt + dh, Mode::ITU_R_P453).N
                   - prof.compute(alt - dh, Mode::ITU_R_P453).N) / (2.0 * dh);
        TEST_ASSERT_NEAR_REL(prof.compute(alt, Mode::ITU_R_P453).dN_dh, fd, 3e-6);
    }
}

TEST_CASE(runner, dn_dh_continuous_at_station) {
    // Refractivity gradient has no jump at the station altitude (the
    // water-vapor gradient tapers smoothly below the surface)
    AtmosphericConditions atm;
    atm.day_of_year = 172;
    atm.ut_seconds = 29000;
    atm.latitude_deg = 45.0;
    atm.longitude_deg = -75.0;
    for (double h_s : {0.0, 2.0}) {
        double P = (h_s == 0.0) ? 101.325 : 80.0;
        RefractionProfile prof(atm, SurfaceObservation{h_s, 30.0, P, 0.8});
        for (Mode mode : {Mode::Ciddor, Mode::ITU_R_P453}) {
            TEST_ASSERT_NEAR_REL(prof.compute(h_s - 1e-6, mode).dN_dh,
                                 prof.compute(h_s + 1e-6, mode).dN_dh, 1e-4);
            double h = h_s - 0.2, e = 1e-3;
            double fd = (prof.compute(h + e, mode).N - prof.compute(h - e, mode).N) / (2 * e);
            TEST_ASSERT_NEAR_REL(prof.compute(h, mode).dN_dh, fd, 1e-4);
        }
    }
}

int main() {
    runner.run();
    return print_final_summary();
}
