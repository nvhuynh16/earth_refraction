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

// Dry pressure of anchored (and unanchored model) species at altitude h (Pa)
static std::pair<double, double> anchored_pressure(const NRLMSIS21& msis,
                                                   const AnchorParams& p, double h) {
    auto inp = standard_input();
    inp.alt = h;
    auto out = msis.msiscalc(inp);
    double n_model = 0.0;
    for (int i = 2; i <= 10; ++i)
        if (out.dn[i] > 0.0 && out.dn[i] < 1e30) n_model += out.dn[i];
    double T_a = out.tn + temperature_offset_at(h, p);
    return {n_model * density_scale_at(h, p) * kB * T_a, n_model * kB * out.tn};
}

TEST_CASE(runner, anchored_surface_reproduces_observation) {
    // Anchored dry air + water vapor must give the observed T and total P
    struct Case { double T_C, P_kPa, RH, h; };
    for (auto c : {Case{20.0, 101.325, 0.5, 0.0}, Case{35.0, 100.0, 0.9, 0.0},
                   Case{-5.0, 102.0, 0.0, 0.0}, Case{10.0, 85.0, 0.3, 1.5}}) {
        NRLMSIS21 msis;
        SurfaceObservation obs{c.h, c.T_C, c.P_kPa, c.RH};
        auto params = compute_anchor(obs, msis, standard_input());
        double T_K = c.T_C + 273.15;
        double e = anchored_H2O_density(c.h, params) * kB * T_K;
        TEST_ASSERT_NEAR_REL(anchored_pressure(msis, params, c.h).first + e,
                             c.P_kPa * 1000.0, 1e-10);
    }
}

TEST_CASE(runner, density_scale_includes_temperature_ratio) {
    // Warmer air at the same pressure must be less dense
    NRLMSIS21 msis;
    auto cold = compute_anchor(SurfaceObservation{0.0, 0.0, 101.325, 0.0}, msis, standard_input());
    auto hot = compute_anchor(SurfaceObservation{0.0, 30.0, 101.325, 0.0}, msis, standard_input());
    TEST_ASSERT_NEAR_REL(hot.density_scale / cold.density_scale, 273.15 / 303.15, 1e-12);
}

TEST_CASE(runner, density_scale_excludes_vapor_pressure) {
    // At equal T and total P, humid air has less dry air (Dalton)
    NRLMSIS21 msis;
    auto dry = compute_anchor(SurfaceObservation{0.0, 25.0, 101.325, 0.0}, msis, standard_input());
    auto wet = compute_anchor(SurfaceObservation{0.0, 25.0, 101.325, 1.0}, msis, standard_input());
    double e = wet.n_H2O_surface * kB * 298.15;
    TEST_ASSERT_NEAR_REL(wet.pressure_scale / dry.pressure_scale,
                         (101325.0 - e) / 101325.0, 1e-12);
}

TEST_CASE(runner, density_scale_hydrostatic) {
    // ln(P_anch/P_model) must match an independent hydrostatic integral
    using atmosphere::msis21::utils::alt2gph;
    using atmosphere::msis21::constants::MBARG0DIVKB;
    for (double H_T : {2.0, 15.0}) {
        NRLMSIS21 msis;
        SurfaceObservation obs{0.0, 35.0, 101.325, 0.0, H_T};
        auto params = compute_anchor(obs, msis, standard_input());
        auto [P_a0, P_m0] = anchored_pressure(msis, params, 0.0);
        auto inp = standard_input();

        const int n = 4000;
        const double h_max = 8.0 * H_T;
        double integral = 0.0, f_prev = 0.0, z_prev = 0.0;
        for (int k = 0; k <= n; ++k) {
            double h = h_max * k / n;
            inp.alt = h;
            double T_m = msis.msiscalc(inp).tn;
            double T_a = T_m + temperature_offset_at(h, params);
            double f = MBARG0DIVKB * (1.0 / T_m - 1.0 / T_a);
            double z = alt2gph(inp.lat, h);
            if (k > 0) integral += 0.5 * (f + f_prev) * (z - z_prev);
            f_prev = f; z_prev = z;
            if (k > 0 && k % (n / 4) == 0) {
                auto [P_a, P_m] = anchored_pressure(msis, params, h);
                double lhs = std::log(P_a / P_m) - std::log(P_a0 / P_m0);
                TEST_ASSERT_NEAR(lhs, integral, 1e-5);
                TEST_ASSERT(lhs > 0.0);  // warm anomaly: slower pressure decay
            }
        }
    }
}

TEST_CASE(runner, density_scale_constant_when_no_temp_offset) {
    // Match model temperature so dT=0 => no hydrostatic correction => S constant
    NRLMSIS21 msis;
    auto inp = standard_input();
    inp.alt = 0.0;
    auto out = msis.msiscalc(inp);
    double T_model_C = out.tn - 273.15;

    SurfaceObservation obs{0.0, T_model_C, 101.325, 0.0};
    auto params = compute_anchor(obs, msis, standard_input());

    TEST_ASSERT_NEAR(params.temperature_offset, 0.0, 1e-12);
    for (double h : {-0.5, 10.0, 60.0, 100.0})
        TEST_ASSERT_NEAR_REL(density_scale_at(h, params), params.density_scale, 1e-12);
}

TEST_CASE(runner, density_scale_converges_above_taper) {
    SurfaceObservation obs{0.0, 20.0, 101.325, 0.5};
    NRLMSIS21 msis;
    auto params = compute_anchor(obs, msis, standard_input());

    double top = params.h_surface_km + 6.0 * params.H_T;
    double s_top = density_scale_at(top, params);
    for (double h : {top + 1.0, 60.0, 100.0, 200.0})
        TEST_ASSERT_NEAR_REL(density_scale_at(h, params), s_top, 1e-7);
}

TEST_CASE(runner, density_scale_smooth) {
    // Hermite table: one-sided slopes on both sides of each node equal the
    // analytic table derivative, so d(ln S)/dh is continuous across nodes.
    SurfaceObservation obs{0.0, 35.0, 101.325, 0.5};
    NRLMSIS21 msis;
    auto params = compute_anchor(obs, msis, standard_input());
    const double eps = 1e-6;  // truncation error eps·|f''|/2 < 1e-7
    for (size_t i = 10; i + 10 < params.table_h_km.size(); i += 20) {
        double h = params.table_h_km[i];
        double ln_0 = std::log(density_scale_at(h, params));
        double right = (std::log(density_scale_at(h + eps, params)) - ln_0) / eps;
        double left = (ln_0 - std::log(density_scale_at(h - eps, params))) / eps;
        TEST_ASSERT_NEAR(right, params.table_dln_scale[i], 1e-7);
        TEST_ASSERT_NEAR(left, params.table_dln_scale[i], 1e-7);
    }
}

TEST_CASE(runner, python_parity_table) {
    // Python and C++ build identical tables (same grid and formulas)
    SurfaceObservation obs{0.0, 20.0, 101.325, 0.5};
    NRLMSIS21 msis;
    auto params = compute_anchor(obs, msis, standard_input());
    TEST_ASSERT_EQ(params.table_h_km.size(), static_cast<size_t>(261));
    TEST_ASSERT_NEAR(params.table_h_km.front(), -1.0, 1e-12);
    TEST_ASSERT_NEAR(params.table_h_km.back(), 12.0, 1e-12);
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

    // Default taper scale H_T = 2 km
    TEST_ASSERT_NEAR(params.H_T, 2.0, 1e-15);
    double dT_3 = temperature_offset_at(3.0, params);
    double expected = params.temperature_offset * std::exp(-3.0 * 3.0 / (2.0 * 2.0 * 2.0));
    TEST_ASSERT_NEAR(dT_3, expected, 1e-10);
}

TEST_CASE(runner, temperature_taper_configurable) {
    SurfaceObservation obs{0.0, 20.0, 101.325, 0.5, 5.0};
    NRLMSIS21 msis;
    auto params = compute_anchor(obs, msis, standard_input());
    TEST_ASSERT_NEAR(params.H_T, 5.0, 1e-15);
    TEST_ASSERT_NEAR_REL(temperature_offset_at(5.0, params),
                         params.temperature_offset * std::exp(-0.5), 1e-12);
}

TEST_CASE(runner, temperature_taper_must_be_positive) {
    NRLMSIS21 msis;
    for (double H_T : {0.0, -1.0}) {
        bool threw = false;
        try {
            compute_anchor(SurfaceObservation{0.0, 20.0, 101.325, 0.5, H_T},
                           msis, standard_input());
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        TEST_ASSERT(threw);
    }
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
