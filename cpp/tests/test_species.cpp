#include "test_utils.hpp"
#include <refraction/species.hpp>
#include <refraction/atmosphere/nrlmsis21.hpp>

using namespace refraction::test;
using namespace refraction;

static TestRunner runner("Species Tests");

TEST_CASE(runner, optical_ciddor_crosscheck) {
    // At standard conditions (15°C, 101325 Pa, dry, 633 nm),
    // Σ K_i × n_i using NRLMSIS densities should reproduce Ciddor n_axs
    double lambda_nm = 633.0;
    double sigma2 = 1.0 / (lambda_nm * 1e-3) / (lambda_nm * 1e-3);

    double n_axs = ciddor::n_standard_air(sigma2);  // Ciddor refractivity

    // Use standard air composition at 15°C, 101325 Pa
    // n_total = P/(kB*T) = 101325/(1.380649e-23 * 288.15)
    constexpr double kB = 1.380649e-23;
    double n_total = 101325.0 / (kB * 288.15);

    double n_N2 = species::f_N2 * n_total;
    double n_O2 = species::f_O2 * n_total;
    double n_Ar = species::f_Ar * n_total;

    double refractivity = n_N2 * species::K_N2_optical(sigma2)
                        + n_O2 * species::K_O2_optical(sigma2)
                        + n_Ar * species::K_Ar_optical(sigma2);

    // Should reproduce Ciddor standard air refractivity
    // Allow 1% tolerance because minor species (CO2, Ne, etc.) are not included
    TEST_ASSERT_NEAR_REL(refractivity, n_axs, 0.01);
}

TEST_CASE(runner, optical_species_sum_reproduces_ciddor_n) {
    // Full test: species-sum approach should reproduce ciddor_refractive_index
    // at sea level for dry air
    double lambda_nm = 633.0;
    double sigma2 = 1.0 / (lambda_nm * 1e-3) / (lambda_nm * 1e-3);

    // Ciddor result at 20°C, 101.325 kPa, dry
    double n_ciddor = ciddor_refractive_index(20.0, 101.325, 0.0, lambda_nm);
    double refractivity_ciddor = n_ciddor - 1.0;

    // Species-sum approach using ideal gas
    constexpr double kB = 1.380649e-23;
    double T_K = 293.15;
    double P_Pa = 101325.0;
    double n_total = P_Pa / (kB * T_K);
    double n_N2 = species::f_N2 * n_total;
    double n_O2 = species::f_O2 * n_total;
    double n_Ar = species::f_Ar * n_total;

    double refractivity = n_N2 * species::K_N2_optical(sigma2)
                        + n_O2 * species::K_O2_optical(sigma2)
                        + n_Ar * species::K_Ar_optical(sigma2);

    // Tolerance 0.5% (ideal gas vs real gas with compressibility)
    TEST_ASSERT_NEAR_REL(refractivity, refractivity_ciddor, 0.005);
}

TEST_CASE(runner, radio_dry_crosscheck) {
    // At standard conditions, species-sum radio should match ITU-R
    double T_K = 288.15;
    double P_hPa = 1013.25;

    double N_itu = itu::itu_N(T_K, P_hPa, 0.0);  // Dry

    // Species sum
    constexpr double kB = 1.380649e-23;
    double n_total = (P_hPa * 100.0) / (kB * T_K);
    double n_N2 = species::f_N2 * n_total;
    double n_O2 = species::f_O2 * n_total;
    double n_Ar = species::f_Ar * n_total;

    double N_species = (n_N2 + n_O2 + n_Ar) * species::K_radio_dry;

    // Should match within 1% (minor species not included)
    TEST_ASSERT_NEAR_REL(N_species, N_itu, 0.01);
}

TEST_CASE(runner, radio_humid_crosscheck) {
    // Check that water vapor radio K gives correct contribution
    double T_K = 293.15;
    double e_hPa = 23.39;  // ~SVP at 20°C

    // ITU water vapor contribution
    double N_wv_itu = itu::k2 * e_hPa / T_K + itu::k3 * e_hPa / (T_K * T_K);

    // Species approach: n_w = e_Pa / (kB * T)
    constexpr double kB = 1.380649e-23;
    double e_Pa = e_hPa * 100.0;
    double n_w = e_Pa / (kB * T_K);

    double N_wv_species = n_w * species::K_H2O_radio(T_K);

    TEST_ASSERT_NEAR_REL(N_wv_species, N_wv_itu, 0.001);
}

TEST_CASE(runner, K_values_positive) {
    double sigma2 = 2.496;  // 633 nm
    TEST_ASSERT(species::K_N2_optical(sigma2) > 0);
    TEST_ASSERT(species::K_O2_optical(sigma2) > 0);
    TEST_ASSERT(species::K_Ar_optical(sigma2) > 0);
    TEST_ASSERT(species::K_H2O_optical(sigma2) > 0);
    TEST_ASSERT(species::K_radio_dry > 0);
    TEST_ASSERT(species::K_H2O_radio(293.15) > 0);
}

int main() {
    runner.run();
    return print_final_summary();
}
