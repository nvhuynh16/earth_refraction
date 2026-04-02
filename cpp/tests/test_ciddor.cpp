#include "test_utils.hpp"
#include <refraction/ciddor.hpp>
#include <refraction/water_vapor.hpp>

using namespace refraction::test;
using namespace refraction;

static TestRunner runner("Ciddor 1996 Tests");

struct CiddorTestVector {
    double T_C;
    double P_kPa;
    double RH;
    double lambda_nm;
    double n_expected;
};

static const CiddorTestVector vectors[] = {
    { 20,  101.325,   0, 633, 1.000271800},
    { 20,   60,       0, 633, 1.000160924},
    { 20,  120,       0, 633, 1.000321916},
    { 50,  100,       0, 633, 1.000243285},
    {  5,  100,       0, 633, 1.000282756},
    {-40,  100,       0, 633, 1.000337580},
    { 50,  120,     100, 633, 1.000287924},
    { 40,  120,      75, 633, 1.000299418},
    { 20,  100,     100, 633, 1.000267394},
    { 40,  110,     100, 1700, 1.000270247},
    { 20,  101.325,   0, 1700, 1.000268479},
    { 40,  110,     100, 300, 1.000289000},
    { 20,  101.325,   0, 300, 1.000286581},
    {-40,  120,       0, 300, 1.000427233},
};

TEST_CASE(runner, nist_test_vectors) {
    for (int i = 0; i < 14; i++) {
        const auto& v = vectors[i];
        double RH_frac = v.RH / 100.0;  // Convert percentage to fraction
        double n = ciddor_refractive_index(v.T_C, v.P_kPa, RH_frac, v.lambda_nm);

        std::ostringstream msg;
        msg << "Vector " << (i+1) << ": T=" << v.T_C << "C, P=" << v.P_kPa
            << "kPa, RH=" << v.RH << "%, lambda=" << v.lambda_nm
            << "nm: expected " << std::setprecision(10) << v.n_expected
            << ", got " << n << ", diff=" << std::abs(n - v.n_expected);
        TEST_ASSERT_MSG(std::abs(n - v.n_expected) < 2e-8, msg.str());
    }
}

// ── Ciddor 1996, Table 1: Phase refractivity of dry air (450 ppm CO2, λ=633 nm) ──
TEST_CASE(runner, ciddor_table1_dry_air) {
    struct V { double T_C; double P_kPa; double expected; };
    static const V vecs[] = {
        {20, 80,  21458.0},
        {20, 100, 26824.4},
        {20, 120, 32191.6},
        {10, 100, 27774.7},
        {30, 100, 25937.2},
    };
    for (int i = 0; i < 5; i++) {
        const auto& v = vecs[i];
        double n = ciddor_refractive_index(v.T_C, v.P_kPa, 0.0, 633.0, 450e-6);
        double refractivity = 1e8 * (n - 1.0);
        double diff = std::abs(refractivity - v.expected);
        std::ostringstream msg;
        msg << "Table1[" << i << "]: T=" << v.T_C << "C, P=" << v.P_kPa
            << "kPa: expected " << v.expected << ", got " << std::setprecision(10)
            << refractivity << ", diff=" << diff;
        TEST_ASSERT_MSG(diff < 3.0, msg.str());  // 3 in units of 1e-8
    }
}

// ── Ciddor 1996, Table 2: Phase refractivity of moist air (λ=633 nm) ──
TEST_CASE(runner, ciddor_table2_moist_air) {
    struct V { double T_C; double P_Pa; double pw_Pa; double xc_ppm; double expected; };
    static const V vecs[] = {
        {19.526, 102094.8, 1065, 510, 27392.9},
        {19.517, 102096.8, 1065, 510, 27394.0},
        {19.173, 102993.0,  641, 450, 27682.4},
        {19.173, 103006.0,  642, 440, 27685.8},
        {19.188, 102918.8,  706, 450, 27658.7},
        {19.189, 102927.8,  708, 440, 27660.8},
        {19.532, 103603.2,  986, 600, 27802.0},
        {19.534, 103596.2,  962, 600, 27800.8},
        {19.534, 103599.2,  951, 610, 27802.2},
    };
    for (int i = 0; i < 9; i++) {
        const auto& v = vecs[i];
        double T_K = v.T_C + 273.15;
        double svp = refraction::svp_giacomo(T_K);
        double RH_frac = v.pw_Pa / svp;
        double P_kPa = v.P_Pa / 1000.0;
        double xCO2 = v.xc_ppm * 1e-6;
        double n = ciddor_refractive_index(v.T_C, P_kPa, RH_frac, 633.0, xCO2);
        double refractivity = 1e8 * (n - 1.0);
        double diff = std::abs(refractivity - v.expected);
        std::ostringstream msg;
        msg << "Table2[" << i << "]: T=" << v.T_C << "C, P=" << v.P_Pa
            << "Pa, pw=" << v.pw_Pa << "Pa, xc=" << v.xc_ppm
            << "ppm: expected " << v.expected << ", got " << std::setprecision(10)
            << refractivity << ", diff=" << diff;
        TEST_ASSERT_MSG(diff < 2.0, msg.str());  // 2 in units of 1e-8
    }
}

// ── Ciddor 1996, Table 3: Extreme atmospheric conditions (450 ppm CO2, λ=633 nm) ──
TEST_CASE(runner, ciddor_table3_extreme) {
    struct V { double T_C; double P_kPa; double RH_pct; double expected; };
    static const V vecs[] = {
        {20,  80,  75, 21394.0},
        {20, 120,  75, 32127.8},
        {40,  80,  75, 19896.5},  // Paper prints 19996.5 — typo: Difference col gives -12.9 = 19883.6 - 19896.5
        {40, 120,  75, 29941.8},
        {50,  80, 100, 19058.4},
        {50, 120, 100, 28792.4},
    };
    for (int i = 0; i < 6; i++) {
        const auto& v = vecs[i];
        double RH_frac = v.RH_pct / 100.0;
        double n = ciddor_refractive_index(v.T_C, v.P_kPa, RH_frac, 633.0, 450e-6);
        double refractivity = 1e8 * (n - 1.0);
        double diff = std::abs(refractivity - v.expected);
        std::ostringstream msg;
        msg << "Table3[" << i << "]: T=" << v.T_C << "C, P=" << v.P_kPa
            << "kPa, RH=" << v.RH_pct << "%: expected " << v.expected
            << ", got " << std::setprecision(10) << refractivity << ", diff=" << diff;
        TEST_ASSERT_MSG(diff < 5.0, msg.str());  // 5 in units of 1e-8
    }
}

int main() {
    runner.run();
    return print_final_summary();
}
