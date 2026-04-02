#include "test_utils.hpp"
#include "test_fixtures.hpp"
#include <refraction/surface_anchor.hpp>

using namespace refraction::test;
using namespace refraction;
using namespace refraction::atmosphere;

static TestRunner runner("Surface Anchor Tests");

TEST_CASE(runner, density_scale_at_surface) {
    SurfaceObservation obs{0.0, 20.0, 101.325, 0.5};
    NRLMSIS21 msis;
    auto params = compute_anchor(obs, msis, standard_input());

    // At surface, density scale should equal the computed scale factor
    double s = density_scale_at(0.0, params);
    TEST_ASSERT_NEAR(s, params.density_scale, 1e-10);
}

TEST_CASE(runner, density_scale_hydrostatic) {
    SurfaceObservation obs{0.0, 20.0, 101.325, 0.5};
    NRLMSIS21 msis;
    auto params = compute_anchor(obs, msis, standard_input());

    double s_0 = density_scale_at(0.0, params);
    double s_20 = density_scale_at(20.0, params);
    double s_40 = density_scale_at(40.0, params);
    double s_100 = density_scale_at(100.0, params);

    // Monotonic behavior based on sign of hydro_C
    if (params.hydro_C > 0) {
        TEST_ASSERT(s_20 > s_0);
        TEST_ASSERT(s_40 > s_20);
    } else if (params.hydro_C < 0) {
        TEST_ASSERT(s_20 < s_0);
        TEST_ASSERT(s_40 < s_20);
    }

    // Convergence to asymptote
    double asymptote = params.density_scale * std::exp(params.hydro_C);
    TEST_ASSERT_NEAR_REL(s_100, asymptote, 1e-6);
}

TEST_CASE(runner, density_scale_constant_when_no_temp_offset) {
    // Match model temperature so dT=0 => hydro_C=0 => S constant
    NRLMSIS21 msis;
    auto inp = standard_input();
    inp.alt = 0.0;
    auto out = msis.msiscalc(inp);
    double T_model_C = out.tn - 273.15;

    SurfaceObservation obs{0.0, T_model_C, 101.325, 0.0};
    auto params = compute_anchor(obs, msis, standard_input());

    TEST_ASSERT_NEAR(params.hydro_C, 0.0, 1e-15);
    TEST_ASSERT_NEAR_REL(density_scale_at(10.0, params), params.density_scale, 1e-14);
    TEST_ASSERT_NEAR_REL(density_scale_at(60.0, params), params.density_scale, 1e-14);
    TEST_ASSERT_NEAR_REL(density_scale_at(100.0, params), params.density_scale, 1e-14);
}

TEST_CASE(runner, density_scale_hydrostatic_asymptote) {
    SurfaceObservation obs{0.0, 20.0, 101.325, 0.5};
    NRLMSIS21 msis;
    auto params = compute_anchor(obs, msis, standard_input());

    double asymptote = params.density_scale * std::exp(params.hydro_C);
    double s_200 = density_scale_at(200.0, params);
    TEST_ASSERT_NEAR_REL(s_200, asymptote, 1e-10);
}

TEST_CASE(runner, temperature_offset_at_surface) {
    SurfaceObservation obs{0.0, 20.0, 101.325, 0.5};
    NRLMSIS21 msis;
    auto params = compute_anchor(obs, msis, standard_input());

    double dT = temperature_offset_at(0.0, params);
    TEST_ASSERT_NEAR(dT, params.temperature_offset, 1e-10);
}

TEST_CASE(runner, temperature_offset_tapers) {
    SurfaceObservation obs{0.0, 20.0, 101.325, 0.5};
    NRLMSIS21 msis;
    auto params = compute_anchor(obs, msis, standard_input());

    // Temperature offset should taper with H_T=15 km
    double dT_30 = temperature_offset_at(30.0, params);
    // exp(-30²/(2×15²)) = exp(-2) ≈ 0.135
    double expected = params.temperature_offset * std::exp(-30.0 * 30.0 / (2.0 * 15.0 * 15.0));
    TEST_ASSERT_NEAR(dT_30, expected, 1e-10);
}

TEST_CASE(runner, water_vapor_surface_value) {
    SurfaceObservation obs{0.0, 20.0, 101.325, 0.5};
    NRLMSIS21 msis;
    auto params = compute_anchor(obs, msis, standard_input());

    // n_H2O at surface should be positive for non-zero RH
    TEST_ASSERT(params.n_H2O_surface > 0.0);

    double n_H2O_surface = anchored_H2O_density(0.0, params);
    TEST_ASSERT_NEAR(n_H2O_surface, params.n_H2O_surface, 1e-10);
}

TEST_CASE(runner, water_vapor_decay) {
    SurfaceObservation obs{0.0, 20.0, 101.325, 1.0};
    NRLMSIS21 msis;
    auto params = compute_anchor(obs, msis, standard_input());

    // H2O should decay exponentially with H_w = 2 km
    double n_10 = anchored_H2O_density(10.0, params);
    double expected = params.n_H2O_surface * std::exp(-10.0 / 2.0);
    TEST_ASSERT_NEAR_REL(n_10, expected, 1e-10);

    // Above 15 km, should be negligible
    double n_15 = anchored_H2O_density(15.0, params);
    TEST_ASSERT(n_15 < params.n_H2O_surface * 1e-3);
}

TEST_CASE(runner, dry_air_zero_humidity) {
    SurfaceObservation obs{0.0, 20.0, 101.325, 0.0};
    NRLMSIS21 msis;
    auto params = compute_anchor(obs, msis, standard_input());

    TEST_ASSERT_NEAR(params.n_H2O_surface, 0.0, 1e-10);
}

TEST_CASE(runner, scale_factor_reasonable) {
    // Scale factor should be close to 1 for standard conditions
    SurfaceObservation obs{0.0, 15.0, 101.325, 0.0};
    NRLMSIS21 msis;
    auto params = compute_anchor(obs, msis, standard_input());

    // NRLMSIS isn't perfectly standard, but scale should be within ~20%
    TEST_ASSERT(params.density_scale > 0.8 && params.density_scale < 1.2);
}

int main() {
    runner.run();
    return print_final_summary();
}
