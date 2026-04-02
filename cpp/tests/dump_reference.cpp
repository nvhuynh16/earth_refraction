#include <iostream>
#include <iomanip>
#include <cmath>
#include <refraction/refraction.hpp>

using namespace refraction;
using namespace refraction::atmosphere;

int main() {
    std::cout << std::setprecision(17);

    // === Section 1: Water vapor ===
    std::cout << "=== WATER_VAPOR ===" << std::endl;
    double temps_K[] = {253.15, 273.15, 293.15, 313.15, 323.15, 373.15};
    for (double T_K : temps_K) {
        std::cout << "svp," << T_K << "," << svp_giacomo(T_K) << std::endl;
    }
    std::cout << "ef," << enhancement_factor(20.0, 101325.0) << std::endl;
    std::cout << "ef," << enhancement_factor(-10.0, 80000.0) << std::endl;
    double xw1 = water_vapor_mole_fraction(20.0, 101325.0, 0.5);
    double xw2 = water_vapor_mole_fraction(35.0, 95000.0, 0.8);
    std::cout << "xw," << xw1 << std::endl;
    std::cout << "xw," << xw2 << std::endl;
    std::cout << "ndens," << water_vapor_number_density(293.15, 101325.0, 0.5) << std::endl;
    std::cout << "ndens," << water_vapor_number_density(308.15, 95000.0, 0.8) << std::endl;
    std::cout << "prof," << water_vapor_profile(1e23, 5.0, 0.0, 2.0) << std::endl;
    std::cout << "prof," << water_vapor_profile(1e23, 0.0, 0.0, 2.0) << std::endl;
    std::cout << "prof," << water_vapor_profile(1e23, 10.0, 2.0, 3.0) << std::endl;

    // === Section 2: Ciddor ===
    std::cout << "=== CIDDOR ===" << std::endl;
    struct { double T_C, P_kPa, RH, lam; } cid_cases[] = {
        {20, 101.325, 0, 633},
        {20, 101.325, 0.5, 633},
        {-40, 100, 0, 633},
        {50, 120, 1.0, 633},
        {20, 101.325, 0, 300},
        {20, 101.325, 0, 1700},
        {40, 110, 1.0, 1700},
        {-40, 120, 0, 300},
        {5, 100, 0, 633},
        {0, 80, 0.3, 550},
    };
    for (auto& c : cid_cases) {
        double n = ciddor_refractive_index(c.T_C, c.P_kPa, c.RH, c.lam);
        std::cout << "ciddor," << c.T_C << "," << c.P_kPa << "," << c.RH << "," << c.lam << "," << n << std::endl;
    }

    // Also dump intermediate values for one case
    {
        double sigma2 = 1.0 / (0.633 * 0.633);
        std::cout << "n_std_air," << ciddor::n_standard_air(sigma2) << std::endl;
        std::cout << "n_wv," << ciddor::n_water_vapor(sigma2) << std::endl;
        std::cout << "Z," << ciddor::compressibility_Z(288.15, 101325.0, 0.0) << std::endl;
        std::cout << "Z," << ciddor::compressibility_Z(293.15, 101325.0, 0.012) << std::endl;
    }

    // === Section 3: ITU ===
    std::cout << "=== ITU ===" << std::endl;
    std::cout << "itu_N," << itu::itu_N(288.15, 1013.25, 0.0) << std::endl;
    std::cout << "itu_N," << itu::itu_N(293.15, 990.0, 23.39) << std::endl;
    std::cout << "itu_N," << itu::itu_N(253.15, 800.0, 1.0) << std::endl;
    std::cout << "itu_Ns," << itu::itu_N_from_surface(20.0, 101.325, 0.5) << std::endl;
    std::cout << "itu_Ns," << itu::itu_N_from_surface(-10.0, 95.0, 0.3) << std::endl;

    // === Section 4: Species K values ===
    std::cout << "=== SPECIES ===" << std::endl;
    double sigma2_vals[] = {
        1.0 / (0.633 * 0.633),
        1.0 / (0.300 * 0.300),
        1.0 / (1.700 * 1.700),
    };
    for (double s2 : sigma2_vals) {
        std::cout << "K_N2," << s2 << "," << species::K_N2_optical(s2) << std::endl;
        std::cout << "K_O2," << s2 << "," << species::K_O2_optical(s2) << std::endl;
        std::cout << "K_Ar," << s2 << "," << species::K_Ar_optical(s2) << std::endl;
        std::cout << "K_He," << s2 << "," << species::K_He_optical(s2) << std::endl;
        std::cout << "K_H," << s2 << "," << species::K_H_optical(s2) << std::endl;
        std::cout << "K_O," << s2 << "," << species::K_O_optical(s2) << std::endl;
        std::cout << "K_N," << s2 << "," << species::K_N_optical(s2) << std::endl;
        std::cout << "K_NO," << s2 << "," << species::K_NO_optical(s2) << std::endl;
        std::cout << "K_H2O," << s2 << "," << species::K_H2O_optical(s2) << std::endl;
    }
    std::cout << "K_dry_radio," << species::K_radio_dry << std::endl;
    std::cout << "K_H2O_radio," << 288.15 << "," << species::K_H2O_radio(288.15) << std::endl;
    std::cout << "K_H2O_radio," << 220.0 << "," << species::K_H2O_radio(220.0) << std::endl;

    // === Section 5: NRLMSIS ===
    std::cout << "=== NRLMSIS ===" << std::endl;
    NRLMSIS21 msis;
    NRLMSIS21::Input inp;
    inp.day = 172; inp.utsec = 29000; inp.lat = 45.0; inp.lon = -75.0;
    inp.f107a = 150.0; inp.f107 = 150.0; inp.ap = {4,4,4,4,4,4,4};

    double alts[] = {0, 5, 10, 20, 30, 50, 70, 85, 100, 120, 200, 300, 500};
    for (double alt : alts) {
        inp.alt = alt;
        auto full = msis.msiscalc_with_derivative(inp);
        auto& out = full.output;
        std::cout << "msis," << alt
                  << "," << out.tn << "," << out.tex;
        for (int i = 1; i <= 10; i++) std::cout << "," << out.dn[i];
        std::cout << "," << full.dT_dz << std::endl;
    }

    // Different location/time
    inp.day = 1; inp.utsec = 43200; inp.lat = -30.0; inp.lon = 120.0;
    double alts2[] = {0, 10, 50, 100, 200};
    for (double alt : alts2) {
        inp.alt = alt;
        auto out = msis.msiscalc(inp);
        std::cout << "msis2," << alt << "," << out.tn << "," << out.tex;
        for (int i = 1; i <= 10; i++) std::cout << "," << out.dn[i];
        std::cout << std::endl;
    }

    // === Section 6: Surface Anchor ===
    std::cout << "=== ANCHOR ===" << std::endl;
    {
        SurfaceObservation obs{0.0, 20.0, 101.325, 0.5};
        inp.day = 172; inp.utsec = 29000; inp.lat = 45.0; inp.lon = -75.0;
        inp.f107a = 150.0; inp.f107 = 150.0; inp.ap = {4,4,4,4,4,4,4};
        auto params = compute_anchor(obs, msis, inp);
        std::cout << "anchor," << params.density_scale << "," << params.temperature_offset
                  << "," << params.n_H2O_surface << std::endl;

        double test_alts[] = {0, 5, 10, 20, 40, 60};
        for (double h : test_alts) {
            std::cout << "ds," << h << "," << density_scale_at(h, params) << std::endl;
            std::cout << "to," << h << "," << temperature_offset_at(h, params) << std::endl;
            std::cout << "h2o," << h << "," << anchored_H2O_density(h, params) << std::endl;
        }
    }

    // === Section 7: Profile ===
    std::cout << "=== PROFILE ===" << std::endl;
    {
        AtmosphericConditions atm;
        atm.day_of_year = 172; atm.ut_seconds = 29000;
        atm.latitude_deg = 45.0; atm.longitude_deg = -75.0;

        SurfaceObservation obs{0.0, 20.0, 101.325, 0.5};
        RefractionProfile prof(atm, obs);

        double test_alts[] = {0, 1, 5, 10, 20, 30, 50, 70, 85, 100};
        for (double h : test_alts) {
            auto r = prof.compute(h, Mode::Ciddor, 633.0);
            std::cout << "ciddor_prof," << h << "," << r.n << "," << r.N
                      << "," << r.dn_dh << "," << r.dN_dh
                      << "," << r.temperature_K << std::endl;
        }
        for (double h : test_alts) {
            auto r = prof.compute(h, Mode::ITU_R_P453);
            std::cout << "itu_prof," << h << "," << r.n << "," << r.N
                      << "," << r.dn_dh << "," << r.dN_dh
                      << "," << r.temperature_K << std::endl;
        }
    }

    return 0;
}
