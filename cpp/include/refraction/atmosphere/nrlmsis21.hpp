#pragma once

// =============================================================================
// NRLMSIS 2.1 Atmosphere Model - Stripped C++ Implementation
// =============================================================================
//
// Stripped from ballistic/atmosphere/nrlmsis21_full.hpp for refraction use.
// Removed: density(), full_output(), Epoch/WeatherConditions dependencies.
// Added: msiscalc_with_derivative() exposing dT/dz via B-spline derivative.
//
// Reference: Emmert, J.T., et al. (2022), NRLMSIS 2.1: An Empirical Model of
//            Nitric Oxide Incorporated into NRLMSIS 2.0, JGR Space Physics.
//
// =============================================================================

#include <array>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include "nrlmsis21_parm.hpp"

namespace refraction {
namespace atmosphere {

// =============================================================================
// Constants Module (from msis_constants.F90)
// =============================================================================
namespace msis21 {
namespace constants {

constexpr double DMISSING = 9.999e-38;

constexpr double PI = 3.14159265358979323846;
constexpr double DEG2RAD = PI / 180.0;
constexpr double DOY2RAD = 2.0 * PI / 365.0;
constexpr double LST2RAD = PI / 12.0;
constexpr double TANH1 = 0.7615941559557649;

constexpr double KB = 1.380649e-23;
constexpr double NA = 6.02214076e23;
constexpr double G0 = 9.80665;

constexpr int SPEC_MASS = 1;
constexpr int SPEC_N2 = 2;
constexpr int SPEC_O2 = 3;
constexpr int SPEC_O = 4;
constexpr int SPEC_HE = 5;
constexpr int SPEC_H = 6;
constexpr int SPEC_AR = 7;
constexpr int SPEC_N = 8;
constexpr int SPEC_OA = 9;
constexpr int SPEC_NO = 10;

constexpr std::array<double, 11> SPECMASS = {
    0.0, 0.0,
    28.0134 / (1.0e3 * NA),
    31.9988 / (1.0e3 * NA),
    31.9988 / 2.0 / (1.0e3 * NA),
    4.0 / (1.0e3 * NA),
    1.0 / (1.0e3 * NA),
    39.948 / (1.0e3 * NA),
    28.0134 / 2.0 / (1.0e3 * NA),
    31.9988 / 2.0 / (1.0e3 * NA),
    (28.0134 + 31.9988) / 2.0 / (1.0e3 * NA)
};

constexpr double MBAR = 28.96546 / (1.0e3 * NA);

constexpr std::array<double, 11> LNVMR = {
    0.0, 0.0,
    -0.247359612336553,
    -1.563296139553406,
    0.0,
    -12.16717628831267,
    0.0,
    -4.674383440483631,
    0.0, 0.0, 0.0
};

constexpr double LNP0 = 11.515614;

constexpr double G0DIVKB = G0 / KB * 1.0e3;
constexpr double MBARG0DIVKB = MBAR * G0DIVKB;

constexpr int ND = 27;
constexpr int P_ORDER = 4;
constexpr int NL = ND - P_ORDER;
constexpr int NLS = 9;

constexpr double ZETA_F = 70.0;
constexpr double ZETA_B = 122.5;
constexpr double ZETA_A = 85.0;
constexpr double ZETA_GAMMA = 100.0;
constexpr double H_GAMMA = 1.0 / 30.0;

constexpr double H_OA = 4000.0 * KB / (SPECMASS[SPEC_OA] * G0) * 1.0e-3;

constexpr int IZFMX = 13;
constexpr int IZFX = 14;
constexpr int IZAX = 17;
constexpr int ITEX = NL;
constexpr int ITGB0 = NL - 1;
constexpr int ITB0 = NL - 2;

constexpr std::array<double, 30> NODES_TN = {
    -15., -10., -5., 0., 5., 10., 15., 20., 25., 30.,
    35., 40., 45., 50., 55., 60., 65., 70., 75., 80.,
    85., 92.5, 102.5, 112.5, 122.5, 132.5, 142.5, 152.5, 162.5, 172.5
};

constexpr int MBF = 383;
constexpr int MAXN = 6;
constexpr int MAXL = 3;
constexpr int MAXM = 2;

constexpr int CTIMEIND = 0;
constexpr int CINTANN = 7;
constexpr int CTIDE = 35;
constexpr int CSPW = 185;
constexpr int CSFX = 295;
constexpr int NSFX = 5;
constexpr int CEXTRA = 300;
constexpr int NSFXMOD = 5;
constexpr int NMAG = 54;
constexpr int NUT = 12;
constexpr int CNONLIN = 384;
constexpr int CSFXMOD = 384;
constexpr int CMAG = 389;
constexpr int CUT = 443;

constexpr int NDO1 = 13;
constexpr int NSPLO1 = NDO1 - 5;
constexpr double NODES_O1[14] = {
    35., 40., 45., 50., 55., 60., 65., 70., 75., 80., 85., 92.5, 102.5, 112.5
};
constexpr double ZETAREF_O1 = ZETA_A;

constexpr int NDNO = 13;
constexpr int NSPLNO = NDNO - 5;
constexpr double NODES_NO[14] = {
    47.5, 55., 62.5, 70., 77.5, 85., 92.5, 100., 107.5, 115., 122.5, 130., 137.5, 145.
};
constexpr double ZETAREF_NO = ZETA_B;
constexpr double ZETAREF_OA = ZETA_B;

constexpr double C1O1[2][2] = {
    { 1.75,               -1.624999900076852},
    {-2.916666573405061,  21.458332647194382}
};
constexpr double C1O1ADJ[2] = {0.257142857142857, -0.102857142686844};

constexpr double C1NO[2][2] = {
    { 1.5, 0.0},
    {-3.75, 15.0}
};
constexpr double C1NOADJ[2] = {0.166666666666667, -0.066666666666667};

constexpr std::array<double, 3> WGHTAXDZ = {-0.102857142857, 0.0495238095238, 0.053333333333};

constexpr std::array<double, 4> S5ZETA_B = {0.041666666666667, 0.458333333333333, 0.458333333333333, 0.041666666666667};
constexpr std::array<double, 5> S6ZETA_B = {0.008771929824561, 0.216228070175439, 0.550000000000000, 0.216666666666667, 0.008333333333333};
constexpr std::array<double, 3> S4ZETA_F = {0.166666666666667, 0.666666666666667, 0.166666666666667};
constexpr std::array<double, 4> S5ZETA_F = {0.041666666666667, 0.458333333333333, 0.458333333333333, 0.041666666666667};
constexpr std::array<double, 3> S5ZETA_0 = {0.458333333333333, 0.458333333333333, 0.041666666666667};
constexpr std::array<double, 3> S4ZETA_A = {0.257142857142857, 0.653968253968254, 0.088888888888889};
constexpr std::array<double, 4> S5ZETA_A = {0.085714285714286, 0.587590187590188, 0.313020313020313, 0.013675213675214};
constexpr std::array<double, 5> S6ZETA_A = {0.023376623376623, 0.378732378732379, 0.500743700743701, 0.095538448479625, 0.001608848667672};

constexpr std::array<std::array<double, 3>, 3> C2TN = {{
    {1.0, 1.0, 1.0},
    {-10.0, 0.0, 10.0},
    {33.333333333333336, -16.666666666666668, 33.333333333333336}
}};

}  // namespace constants

// =============================================================================
// Utility Functions Module
// =============================================================================
namespace utils {

using namespace constants;

inline double alt2gph(double lat, double alt) {
    constexpr double a = 6378.1370e3;
    constexpr double finv = 298.257223563;
    constexpr double w = 7.292115e-5;
    constexpr double GM = 398600.4418e9;

    constexpr double asq = a * a;
    constexpr double wsq = w * w;
    constexpr double f = 1.0 / finv;
    constexpr double esq = 2.0 * f - f * f;
    constexpr double e = 0.08181919084262149;
    constexpr double Elin = a * e;
    constexpr double Elinsq = Elin * Elin;
    constexpr double epr = e / (1.0 - f);
    constexpr double q0 = 7.33462578708548e-05;
    constexpr double U0 = -62636851.7149236;
    constexpr double g0 = 9.80665;
    constexpr double GMdivElin = GM / Elin;

    constexpr double x0sq = 2.0e7 * 2.0e7;
    constexpr double Hsq = 1.2e7 * 1.2e7;

    double altm = alt * 1000.0;
    double sinsqlat = std::sin(lat * DEG2RAD);
    sinsqlat = sinsqlat * sinsqlat;

    double v = a / std::sqrt(1.0 - esq * sinsqlat);
    double xsq = (v + altm) * (v + altm) * (1.0 - sinsqlat);
    double zsq = (v * (1.0 - esq) + altm) * (v * (1.0 - esq) + altm) * sinsqlat;

    double rsqminElinsq = xsq + zsq - Elinsq;
    double usq = rsqminElinsq / 2.0 + std::sqrt(rsqminElinsq * rsqminElinsq / 4.0 + Elinsq * zsq);
    double cossqdelta = zsq / usq;

    double epru = Elin / std::sqrt(usq);
    double atanepru = std::atan(epru);
    double q = ((1.0 + 3.0 / (epru * epru)) * atanepru - 3.0 / epru) / 2.0;
    double U = -GMdivElin * atanepru - wsq * (asq * q * (cossqdelta - 1.0/3.0) / q0) / 2.0;

    double Vc;
    if (xsq <= x0sq) {
        Vc = (wsq / 2.0) * xsq;
    } else {
        Vc = (wsq / 2.0) * (Hsq * std::tanh((xsq - x0sq) / Hsq) + x0sq);
    }
    U = U - Vc;

    return (U - U0) / g0 / 1000.0;
}

inline double dilog(double x0) {
    constexpr double pi2_6 = PI * PI / 6.0;
    double x = x0;
    if (x > 0.5) {
        double lnx = std::log(x);
        x = 1.0 - x;
        double xx = x * x;
        double x4 = 4.0 * x;
        return pi2_6 - lnx * std::log(x)
               - (4.0 * xx * (23.0/16.0 + x/36.0 + xx/576.0 + xx*x/3600.0)
                  + x4 + 3.0 * (1.0 - xx) * lnx) / (1.0 + x4 + xx);
    } else {
        double xx = x * x;
        double x4 = 4.0 * x;
        return (4.0 * xx * (23.0/16.0 + x/36.0 + xx/576.0 + xx*x/3600.0)
                + x4 + 3.0 * (1.0 - xx) * std::log(1.0 - x)) / (1.0 + x4 + xx);
    }
}

struct BSplineResult {
    int iz;
    std::array<double, 6> S3;   // Order 3 spline weights (for temperature derivative)
    std::array<double, 6> S4;
    std::array<double, 6> S5;
    std::array<double, 6> S6;
};

inline BSplineResult bspline_eval(double z, const double* nodes, int nd, const std::array<std::array<double, 30>, 5>& eta) {
    BSplineResult result;
    result.S3.fill(0.0);
    result.S4.fill(0.0);
    result.S5.fill(0.0);
    result.S6.fill(0.0);

    if (z >= nodes[nd - 1]) {
        result.iz = nd - 1;
        return result;
    }
    if (z <= nodes[0]) {
        result.iz = -1;
        return result;
    }
    int low = 0, high = nd - 1;
    result.iz = (low + high) / 2;
    while (z < nodes[result.iz] || z >= nodes[result.iz + 1]) {
        if (z < nodes[result.iz]) {
            high = result.iz;
        } else {
            low = result.iz;
        }
        result.iz = (low + high) / 2;
    }

    const int i = result.iz;

    // Order 2
    std::array<double, 2> S2 = {};
    double w0 = (z - nodes[i]) * eta[0][i];
    S2[1] = w0;
    if (i > 0) S2[0] = 1.0 - w0;
    if (i >= nd - 1) S2[1] = 0.0;

    // Order 3
    std::array<double, 3> S3 = {};
    std::array<double, 2> w3 = {};
    w3[1] = (z - nodes[i]) * eta[1][i];
    if (i >= 1) w3[0] = (z - nodes[i-1]) * eta[1][i-1];

    if (i < nd - 2) S3[2] = w3[1] * S2[1];
    if (i >= 1 && i - 1 < nd - 2) S3[1] = w3[0] * S2[0] + (1.0 - w3[1]) * S2[1];
    if (i >= 2) S3[0] = (1.0 - w3[0]) * S2[0];

    // Store S3 in result (shifted for consistency: S3[j] maps to result.S3[j+2] for j in {-2,-1,0})
    // We store so that result.S3[k] corresponds to the weight for cf[iz-2+k] for k=0,1,2
    result.S3[0] = S3[0];
    result.S3[1] = S3[1];
    result.S3[2] = S3[2];

    // Order 4
    std::array<double, 4> S4 = {};
    std::array<double, 3> w4 = {};
    for (int l = 0; l >= -2; l--) {
        int j = i + l;
        if (j >= 0 && j < nd - 2) w4[l + 2] = (z - nodes[j]) * eta[2][j];
    }

    if (i < nd - 3) S4[3] = w4[2] * S3[2];
    for (int l = -1; l >= -2; l--) {
        if (i + l >= 0 && i + l < nd - 3) {
            S4[l + 3] = w4[l + 2] * S3[l + 2] + (1.0 - w4[l + 3]) * S3[l + 3];
        }
    }
    if (i >= 3) S4[0] = (1.0 - w4[0]) * S3[0];

    for (int j = 0; j < 4; j++) result.S4[j] = S4[j];

    // Order 5
    std::array<double, 5> S5 = {};
    std::array<double, 4> w5 = {};
    for (int l = 0; l >= -3; l--) {
        int j = i + l;
        if (j >= 0 && j < nd - 3) w5[l + 3] = (z - nodes[j]) * eta[3][j];
    }

    if (i < nd - 4) S5[4] = w5[3] * S4[3];
    for (int l = -1; l >= -3; l--) {
        if (i + l >= 0 && i + l < nd - 4) {
            S5[l + 4] = w5[l + 3] * S4[l + 3] + (1.0 - w5[l + 4]) * S4[l + 4];
        }
    }
    if (i >= 4) S5[0] = (1.0 - w5[0]) * S4[0];

    for (int j = 0; j < 5; j++) result.S5[j] = S5[j];

    // Order 6
    std::array<double, 5> w6 = {};
    for (int l = 0; l >= -4; l--) {
        int j = i + l;
        if (j >= 0 && j < nd - 4) w6[l + 4] = (z - nodes[j]) * eta[4][j];
    }

    if (i < nd - 5) result.S6[5] = w6[4] * S5[4];
    for (int l = -1; l >= -4; l--) {
        if (i + l >= 0 && i + l < nd - 5) {
            result.S6[l + 5] = w6[l + 4] * S5[l + 4] + (1.0 - w6[l + 5]) * S5[l + 5];
        }
    }
    if (i >= 5) result.S6[0] = (1.0 - w6[0]) * S5[0];

    return result;
}

}  // namespace utils

}  // namespace msis21

// =============================================================================
// Temperature and Density profile parameters
// =============================================================================
struct TnParm {
    std::array<double, 24> cf;
    double tzetaF;
    double tzetaA;
    double dlntdzA;
    double tex;
    double tgb0;
    double tb0;
    double sigma;
    double sigmasq;
    double b;
    std::array<double, 24> beta;
    std::array<double, 24> gamma;
    double cVS, cVB, cWS, cWB;
    double VzetaF, VzetaA, WzetaA, Vzeta0;
    double lndtotF;
};

struct DnParm {
    int ispec;
    double lnPhiF;
    double lndref;
    double zref;
    double zmin;
    double zhyd;

    double zetaM;
    double HML;
    double HMU;
    std::array<double, 5> zetaMi;
    std::array<double, 5> Mi;
    std::array<double, 4> aMi;
    std::array<double, 5> WMi;
    std::array<double, 5> XMi;

    double C;
    double zetaC;
    double HC;
    double R;
    double zetaR;
    double HR;

    std::array<double, 10> cf;

    double Mzref;
    double Izref;
    double Tref;
};

// =============================================================================
// Main NRLMSIS 2.1 Class
// =============================================================================
class NRLMSIS21 {
public:
    struct Input {
        double day;
        double utsec;
        double alt;
        double lat;
        double lon;
        double f107a;
        double f107;
        std::array<double, 7> ap;
    };

    struct Output {
        double tn;
        double tex;
        std::array<double, 11> dn;
    };

    struct FullOutput {
        Output output;
        double dT_dz;  // Temperature derivative (K/km)
    };

    NRLMSIS21();

    Output msiscalc(const Input& input) const;
    FullOutput msiscalc_with_derivative(const Input& input) const;

private:
    std::array<std::array<double, 30>, 5> etaTN_;
    std::array<std::array<double, 30>, 3> etaO1_;
    std::array<std::array<double, 30>, 3> etaNO_;

    std::array<bool, 24> smod_;
    std::array<bool, 384> zsfx_;
    std::array<bool, 384> tsfx_;
    std::array<bool, 384> psfx_;

    double HRfactO1ref_, dHRfactO1ref_;
    double HRfactNOref_, dHRfactNOref_;

    void compute_globe(const Input& input, std::array<double, 512>& bf) const;

    template<std::size_t N>
    double sfluxmod(int iz, const std::array<double, 512>& gf,
                    const std::array<std::array<double, N>, 512>& beta,
                    double dffact) const;
    double sfluxmod_tn(int iz, const std::array<double, 512>& gf, double dffact) const;
    static double solzen(double doy, double lst, double lat, double lon);

    TnParm compute_tfnparm(const std::array<double, 512>& bf) const;
    DnParm compute_dfnparm(int ispec, const std::array<double, 512>& bf, const TnParm& tpro) const;
    double compute_dfnx(double zeta, double tn, double lndtotz, double Vz, double Wz, double HRfact,
                        const TnParm& tpro, const DnParm& dpro) const;
    double pwmp(double z, const DnParm& dpro) const;
    double compute_temperature(double zeta, const TnParm& tpro, int iz, const std::array<double, 6>& S4) const;
    double compute_Wz(double zeta, const TnParm& tpro, int iz, const std::array<double, 6>& S6) const;

    double compute_dT_dz(double zeta, const TnParm& tpro, int iz,
                         const std::array<double, 6>& S3) const;

    struct BSplineResult4 {
        int iz;
        std::array<double, 4> S4;
    };
    BSplineResult4 bspline_eval_o4(double z, const double* nodes, int nd,
                                    const std::array<std::array<double, 30>, 3>& eta) const;
};

// =============================================================================
// Implementation
// =============================================================================

inline NRLMSIS21::NRLMSIS21() {
    using namespace msis21::constants;
    using namespace msis21::parm;

    for (auto& a : etaTN_) a.fill(0.0);
    for (int k = 0; k < 5; k++) {
        for (int i = 0; i <= NL; i++) {
            double denom = NODES_TN[i + k + 1] - NODES_TN[i];
            etaTN_[k][i] = (denom > 0.0) ? 1.0 / denom : 0.0;
        }
    }

    for (auto& a : etaO1_) a.fill(0.0);
    for (int k = 0; k < 3; k++) {
        for (int j = 0; j <= NDO1 - k - 2; j++) {
            double denom = NODES_O1[j + k + 1] - NODES_O1[j];
            etaO1_[k][j] = (denom > 0.0) ? 1.0 / denom : 0.0;
        }
    }

    for (auto& a : etaNO_) a.fill(0.0);
    for (int k = 0; k < 3; k++) {
        for (int j = 0; j <= NDNO - k - 2; j++) {
            double denom = NODES_NO[j + k + 1] - NODES_NO[j];
            etaNO_[k][j] = (denom > 0.0) ? 1.0 / denom : 0.0;
        }
    }

    zsfx_.fill(false);
    tsfx_.fill(false);
    psfx_.fill(false);
    zsfx_[9] = zsfx_[10] = true;
    zsfx_[13] = zsfx_[14] = true;
    zsfx_[17] = zsfx_[18] = true;
    for (int i = CTIDE; i < CSPW && i < 384; i++) tsfx_[i] = true;
    for (int i = CSPW; i < CSPW + 60 && i < 384; i++) psfx_[i] = true;

    smod_.fill(false);
    for (int ix = 0; ix <= NL; ix++) {
        if (TN_BETA[CSFXMOD][ix] != 0.0 ||
            TN_BETA[CSFXMOD + 1][ix] != 0.0 ||
            TN_BETA[CSFXMOD + 2][ix] != 0.0) {
            smod_[ix] = true;
        }
    }

    double gammaterm0 = std::tanh((ZETAREF_O1 - ZETA_GAMMA) * H_GAMMA);
    HRfactO1ref_ = 0.5 * (1.0 + gammaterm0);
    dHRfactO1ref_ = (1.0 - (ZETAREF_O1 - ZETA_GAMMA) * (1.0 - gammaterm0) * H_GAMMA) / HRfactO1ref_;

    gammaterm0 = std::tanh((ZETAREF_NO - ZETA_GAMMA) * H_GAMMA);
    HRfactNOref_ = 0.5 * (1.0 + gammaterm0);
    dHRfactNOref_ = (1.0 - (ZETAREF_NO - ZETA_GAMMA) * (1.0 - gammaterm0) * H_GAMMA) / HRfactNOref_;
}

inline double NRLMSIS21::solzen(double ddd, double lst, double lat, double lon) {
    using namespace msis21::constants;
    constexpr double humr = PI / 12.0;
    constexpr double p[5] = {0.017203534, 0.034407068, 0.051610602, 0.068814136, 0.103221204};

    double teqnx = ddd + 0.9369;
    double dec = 23.256 * std::sin(p[0] * (teqnx - 82.242)) + 0.381 * std::sin(p[1] * (teqnx - 44.855))
               + 0.167 * std::sin(p[2] * (teqnx - 23.355)) - 0.013 * std::sin(p[3] * (teqnx + 11.97))
               + 0.011 * std::sin(p[4] * (teqnx - 10.410)) + 0.339137;
    dec *= DEG2RAD;

    double tf = teqnx - 0.5;
    double teqt = -7.38 * std::sin(p[0] * (tf - 4.0)) - 9.87 * std::sin(p[1] * (tf + 9.0))
                + 0.27 * std::sin(p[2] * (tf - 53.0)) - 0.2 * std::cos(p[3] * (tf - 17.0));

    double phi = humr * (lst - 12.0) + teqt * DEG2RAD / 4.0;
    double rlat = lat * DEG2RAD;
    double cosx = std::sin(rlat) * std::sin(dec) + std::cos(rlat) * std::cos(dec) * std::cos(phi);
    if (std::abs(cosx) > 1.0) cosx = (cosx > 0) ? 1.0 : -1.0;
    return std::acos(cosx) / DEG2RAD;
}

template<std::size_t N>
inline double NRLMSIS21::sfluxmod(int iz, const std::array<double, 512>& gf,
                                   const std::array<std::array<double, N>, 512>& beta,
                                   double dffact) const {
    using namespace msis21::constants;
    double f1 = beta[CSFXMOD][iz] * gf[CSFXMOD]
              + (beta[CSFX + 2][iz] * gf[CSFXMOD + 2] + beta[CSFX + 3][iz] * gf[CSFXMOD + 3]) * dffact;
    double f2 = beta[CSFXMOD + 1][iz] * gf[CSFXMOD]
              + (beta[CSFX + 2][iz] * gf[CSFXMOD + 2] + beta[CSFX + 3][iz] * gf[CSFXMOD + 3]) * dffact;
    double f3 = beta[CSFXMOD + 2][iz] * gf[CSFXMOD];

    double sum = 0.0;
    for (int j = 0; j <= MBF; j++) {
        if (zsfx_[j]) {
            sum += beta[j][iz] * gf[j] * f1;
        } else if (tsfx_[j]) {
            sum += beta[j][iz] * gf[j] * f2;
        } else if (psfx_[j]) {
            sum += beta[j][iz] * gf[j] * f3;
        }
    }
    return sum;
}

inline double NRLMSIS21::sfluxmod_tn(int iz, const std::array<double, 512>& gf, double dffact) const {
    using namespace msis21::constants;
    using namespace msis21::parm;

    double f1 = TN_BETA[CSFXMOD][iz] * gf[CSFXMOD]
              + (TN_BETA[CSFX + 2][iz] * gf[CSFXMOD + 2] + TN_BETA[CSFX + 3][iz] * gf[CSFXMOD + 3]) * dffact;
    double f2 = TN_BETA[CSFXMOD + 1][iz] * gf[CSFXMOD]
              + (TN_BETA[CSFX + 2][iz] * gf[CSFXMOD + 2] + TN_BETA[CSFX + 3][iz] * gf[CSFXMOD + 3]) * dffact;
    double f3 = TN_BETA[CSFXMOD + 2][iz] * gf[CSFXMOD];

    double sum = 0.0;
    for (int j = 0; j <= MBF; j++) {
        if (zsfx_[j]) {
            sum += TN_BETA[j][iz] * gf[j] * f1;
        } else if (tsfx_[j]) {
            sum += TN_BETA[j][iz] * gf[j] * f2;
        } else if (psfx_[j]) {
            sum += TN_BETA[j][iz] * gf[j] * f3;
        }
    }
    return sum;
}

// Geomagnetic activity function
template<std::size_t N>
double geomag_impl(const std::array<std::array<double, N>, 512>& beta, int level,
                   const std::array<double, 512>& bf) {
    using namespace msis21::constants;

    double k00r = beta[CMAG][level];
    double k00s = beta[CMAG + 1][level];
    if (k00s == 0.0) return 0.0;

    auto G0fn = [](double a, double k00r_, double k00s_) -> double {
        return a + (k00r_ - 1.0) * (a + (std::exp(-a * k00s_) - 1.0) / k00s_);
    };

    double delA = G0fn(bf[CMAG], k00r, k00s);
    const double* plg0 = &bf[CMAG + 13];
    const double* plg1 = &bf[CMAG + 20];

    double p2 = beta[CMAG + 2][level];
    double p3 = beta[CMAG + 3][level];
    double p4 = beta[CMAG + 4][level];
    double p5 = beta[CMAG + 5][level];
    double p6 = beta[CMAG + 6][level];
    double p7 = beta[CMAG + 7][level];
    double p8 = beta[CMAG + 8][level];
    double p9 = beta[CMAG + 9][level];
    double p10 = beta[CMAG + 10][level];
    double p11 = beta[CMAG + 11][level];
    double p12 = beta[CMAG + 12][level];
    double p13 = beta[CMAG + 13][level];
    double p14 = beta[CMAG + 14][level];
    double p15 = beta[CMAG + 15][level];
    double p16 = beta[CMAG + 16][level];
    double p17 = beta[CMAG + 17][level];
    double p18 = beta[CMAG + 18][level];
    double p19 = beta[CMAG + 19][level];
    double p20 = beta[CMAG + 20][level];
    double p21 = beta[CMAG + 21][level];
    double p22 = beta[CMAG + 22][level];
    double p23 = beta[CMAG + 23][level];
    double p24 = beta[CMAG + 24][level];
    double p25 = beta[CMAG + 25][level];

    double doy_rad = bf[CMAG + 8];
    double lst_rad = bf[CMAG + 9];
    double lon_rad = bf[CMAG + 10];
    double ut_rad = bf[CMAG + 11];

    double result = (p2 * plg0[0] + p3 * plg0[2] + p4 * plg0[4]
        + (p5 * plg0[1] + p6 * plg0[3] + p7 * plg0[5]) * std::cos(doy_rad - p8)
        + (p9 * plg1[1] + p10 * plg1[3] + p11 * plg1[5]) * std::cos(lst_rad - p12)
        + (1.0 + p13 * plg0[1]) *
          (p14 * plg1[2] + p15 * plg1[4] + p16 * plg1[6]) * std::cos(lon_rad - p17)
        + (p18 * plg1[1] + p19 * plg1[3] + p20 * plg1[5]) * std::cos(lon_rad - p21)
          * std::cos(doy_rad - p8)
        + (p22 * plg0[1] + p23 * plg0[3] + p24 * plg0[5]) * std::cos(ut_rad - p25))
        * delA;

    return result;
}

// UT dependence function
template<std::size_t N>
double utdep_impl(const std::array<std::array<double, N>, 512>& beta, int level,
                  const std::array<double, 512>& bf) {
    using namespace msis21::constants;

    double p0 = beta[CUT][level];
    double p1 = beta[CUT + 1][level];
    double p2 = beta[CUT + 2][level];
    double p3 = beta[CUT + 3][level];
    double p4 = beta[CUT + 4][level];
    double p5 = beta[CUT + 5][level];
    double p6 = beta[CUT + 6][level];
    double p7 = beta[CUT + 7][level];
    double p8 = beta[CUT + 8][level];
    double p9 = beta[CUT + 9][level];
    double p10 = beta[CUT + 10][level];
    double p11 = beta[CUT + 11][level];

    double ut_rad = bf[CUT];
    double doy_rad = bf[CUT + 1];
    double dfa = bf[CUT + 2];
    double lon_rad = bf[CUT + 3];
    double plg10 = bf[CUT + 4];
    double plg30 = bf[CUT + 5];
    double plg50 = bf[CUT + 6];
    double plg32 = bf[CUT + 7];
    double plg52 = bf[CUT + 8];

    return std::cos(ut_rad - p0) *
           (1.0 + p3 * plg10 * std::cos(doy_rad - p1)) *
           (1.0 + p4 * dfa) * (1.0 + p5 * plg10) *
           (p6 * plg10 + p7 * plg30 + p8 * plg50) +
           std::cos(ut_rad - p2 + 2.0 * lon_rad) * (p9 * plg32 + p10 * plg52) * (1.0 + p11 * dfa);
}

inline NRLMSIS21::BSplineResult4 NRLMSIS21::bspline_eval_o4(
    double z, const double* nodes, int nd,
    const std::array<std::array<double, 30>, 3>& eta) const {

    BSplineResult4 result;
    result.S4.fill(0.0);

    result.iz = 0;
    for (int i = 0; i < nd - 1; i++) {
        if (z >= nodes[i]) result.iz = i;
    }
    int iz = result.iz;

    double S2[4] = {};
    double w2 = (z - nodes[iz]) * eta[0][iz];
    S2[2] = w2;
    S2[1] = 1.0 - w2;

    double S3[4] = {};
    for (int l = 0; l >= -1; l--) {
        int j = iz + l;
        if (j >= 0 && j <= nd - 3) {
            double wl = (z - nodes[j]) * eta[1][j];
            int idx = l + 2;
            if (l == 0) {
                S3[idx] = wl * S2[2];
            }
            if (l == -1) {
                S3[idx] = wl * S2[1] + (1.0 - ((z - nodes[iz]) * eta[1][iz])) * S2[2];
            }
        }
    }
    if (iz - 2 >= 0 && iz - 1 <= nd - 3) {
        double wm1 = (z - nodes[iz - 1]) * eta[1][iz - 1];
        S3[0] = (1.0 - wm1) * S2[1];
    }

    result.S4.fill(0.0);
    for (int l = 0; l >= -3; l--) {
        int j = iz + l;
        double wl = 0.0;
        if (j >= 0 && j <= nd - 4) wl = (z - nodes[j]) * eta[2][j];

        double wl1 = 0.0;
        int j1 = iz + l + 1;
        if (j1 >= 0 && j1 <= nd - 4) wl1 = (z - nodes[j1]) * eta[2][j1];

        int s3_l = l + 2;
        int s3_l1 = l + 3;
        double s3_val = (s3_l >= 0 && s3_l < 4) ? S3[s3_l] : 0.0;
        double s3_val1 = (s3_l1 >= 0 && s3_l1 < 4) ? S3[s3_l1] : 0.0;

        int s4_idx = l + 3;
        if (s4_idx >= 0 && s4_idx < 4) {
            result.S4[s4_idx] = wl * s3_val + (1.0 - wl1) * s3_val1;
        }
    }

    return result;
}

inline void NRLMSIS21::compute_globe(const Input& input, std::array<double, 512>& bf) const {
    using namespace msis21::constants;

    bf.fill(0.0);

    double lat_rad = input.lat * DEG2RAD;
    double slat = std::sin(lat_rad);
    double clat = std::cos(lat_rad);

    double clat_leg = slat;
    double slat_leg = clat;
    double clat2 = clat_leg * clat_leg;
    double clat4 = clat2 * clat2;
    double slat2 = slat_leg * slat_leg;

    std::array<std::array<double, 4>, 7> plg = {};
    plg[0][0] = 1.0;
    plg[1][0] = clat_leg;
    plg[2][0] = 0.5 * (3.0 * clat2 - 1.0);
    plg[3][0] = 0.5 * (5.0 * clat_leg * clat2 - 3.0 * clat_leg);
    plg[4][0] = (35.0 * clat4 - 30.0 * clat2 + 3.0) / 8.0;
    plg[5][0] = (63.0 * clat2 * clat2 * clat_leg - 70.0 * clat2 * clat_leg + 15.0 * clat_leg) / 8.0;
    plg[6][0] = (11.0 * clat_leg * plg[5][0] - 5.0 * plg[4][0]) / 6.0;

    plg[1][1] = slat_leg;
    plg[2][1] = 3.0 * clat_leg * slat_leg;
    plg[3][1] = 1.5 * (5.0 * clat2 - 1.0) * slat_leg;
    plg[4][1] = 2.5 * (7.0 * clat2 * clat_leg - 3.0 * clat_leg) * slat_leg;
    plg[5][1] = 1.875 * (21.0 * clat4 - 14.0 * clat2 + 1.0) * slat_leg;
    plg[6][1] = (11.0 * clat_leg * plg[5][1] - 6.0 * plg[4][1]) / 5.0;

    plg[2][2] = 3.0 * slat2;
    plg[3][2] = 15.0 * slat2 * clat_leg;
    plg[4][2] = 7.5 * (7.0 * clat2 - 1.0) * slat2;
    plg[5][2] = 3.0 * clat_leg * plg[4][2] - 2.0 * plg[3][2];
    plg[6][2] = (11.0 * clat_leg * plg[5][2] - 7.0 * plg[4][2]) / 4.0;

    plg[3][3] = 15.0 * slat2 * slat_leg;
    plg[4][3] = 105.0 * slat2 * slat_leg * clat_leg;
    plg[5][3] = (9.0 * clat_leg * plg[4][3] - 7.0 * plg[3][3]) / 2.0;
    plg[6][3] = (11.0 * clat_leg * plg[5][3] - 8.0 * plg[4][3]) / 3.0;

    double lst = input.utsec / 3600.0 + input.lon / 15.0;
    lst = std::fmod(lst, 24.0);
    if (lst < 0.0) lst += 24.0;

    double cdoy1 = std::cos(DOY2RAD * input.day);
    double sdoy1 = std::sin(DOY2RAD * input.day);
    double cdoy2 = std::cos(DOY2RAD * input.day * 2.0);
    double sdoy2 = std::sin(DOY2RAD * input.day * 2.0);

    double clst1 = std::cos(LST2RAD * lst);
    double slst1 = std::sin(LST2RAD * lst);
    double clst2 = std::cos(LST2RAD * lst * 2.0);
    double slst2 = std::sin(LST2RAD * lst * 2.0);
    double clst3 = std::cos(LST2RAD * lst * 3.0);
    double slst3 = std::sin(LST2RAD * lst * 3.0);

    double clon1 = std::cos(DEG2RAD * input.lon);
    double slon1 = std::sin(DEG2RAD * input.lon);
    double clon2 = std::cos(DEG2RAD * input.lon * 2.0);
    double slon2 = std::sin(DEG2RAD * input.lon * 2.0);

    int c = CTIMEIND;
    for (int n = 0; n <= MAXN; n++) {
        bf[c++] = plg[n][0];
    }

    std::array<double, 2> cdoy = {cdoy1, cdoy2};
    std::array<double, 2> sdoy = {sdoy1, sdoy2};
    for (int s = 0; s < 2; s++) {
        for (int n = 0; n <= MAXN; n++) {
            bf[c++] = plg[n][0] * cdoy[s];
            bf[c++] = plg[n][0] * sdoy[s];
        }
    }

    std::array<double, 3> clst = {clst1, clst2, clst3};
    std::array<double, 3> slst = {slst1, slst2, slst3};
    for (int l = 1; l <= MAXL; l++) {
        for (int n = l; n <= MAXN; n++) {
            bf[c++] = plg[n][l] * clst[l-1];
            bf[c++] = plg[n][l] * slst[l-1];
        }
        for (int s = 0; s < 2; s++) {
            for (int n = l; n <= MAXN; n++) {
                bf[c++] = plg[n][l] * clst[l-1] * cdoy[s];
                bf[c++] = plg[n][l] * slst[l-1] * cdoy[s];
                bf[c++] = plg[n][l] * clst[l-1] * sdoy[s];
                bf[c++] = plg[n][l] * slst[l-1] * sdoy[s];
            }
        }
    }

    std::array<double, 2> clon = {clon1, clon2};
    std::array<double, 2> slon = {slon1, slon2};
    for (int m = 1; m <= MAXM; m++) {
        for (int n = m; n <= MAXN; n++) {
            bf[c++] = plg[n][m] * clon[m-1];
            bf[c++] = plg[n][m] * slon[m-1];
        }
        for (int s = 0; s < 2; s++) {
            for (int n = m; n <= MAXN; n++) {
                bf[c++] = plg[n][m] * clon[m-1] * cdoy[s];
                bf[c++] = plg[n][m] * slon[m-1] * cdoy[s];
                bf[c++] = plg[n][m] * clon[m-1] * sdoy[s];
                bf[c++] = plg[n][m] * slon[m-1] * sdoy[s];
            }
        }
    }

    double dfa = input.f107a - 150.0;
    double df = input.f107 - input.f107a;
    bf[CSFX] = dfa;
    bf[CSFX + 1] = dfa * dfa;
    bf[CSFX + 2] = df;
    bf[CSFX + 3] = df * df;
    bf[CSFX + 4] = df * dfa;

    double sza = solzen(input.day, lst, input.lat, input.lon);
    bf[300] = -0.5 * std::tanh((sza - 98.0) / 6.0);
    bf[301] = -0.5 * std::tanh((sza - 101.5) / 20.0);
    bf[302] = dfa * bf[300];
    bf[303] = dfa * bf[301];
    bf[304] = dfa * plg[2][0];
    bf[305] = dfa * plg[4][0];
    bf[306] = dfa * plg[0][0] * cdoy1;
    bf[307] = dfa * plg[0][0] * sdoy1;
    bf[308] = dfa * plg[0][0] * cdoy2;
    bf[309] = dfa * plg[0][0] * sdoy2;
    double sfluxavg_quad_cutoff = 150.0;
    double sfluxavgref = 150.0;
    double dfa_trunc;
    if (input.f107a <= sfluxavg_quad_cutoff) {
        dfa_trunc = dfa * dfa;
    } else {
        dfa_trunc = (sfluxavg_quad_cutoff - sfluxavgref) * (2.0 * dfa - (sfluxavg_quad_cutoff - sfluxavgref));
    }
    bf[310] = dfa_trunc;
    bf[311] = dfa_trunc * plg[2][0];
    bf[312] = dfa_trunc * plg[4][0];
    bf[313] = df * plg[2][0];
    bf[314] = df * plg[4][0];

    bf[CSFXMOD] = dfa;
    bf[CSFXMOD + 1] = dfa * dfa;
    bf[CSFXMOD + 2] = df;
    bf[CSFXMOD + 3] = df * df;
    bf[CSFXMOD + 4] = df * dfa;

    for (int i = 0; i < 7; i++) {
        bf[CMAG + i] = input.ap[i] - 4.0;
    }

    double doy_rad = DOY2RAD * input.day;
    double lon_rad = DEG2RAD * input.lon;
    double ut_rad = LST2RAD * (input.utsec / 3600.0);
    bf[CMAG + 8] = doy_rad;
    bf[CMAG + 9] = LST2RAD * lst;
    bf[CMAG + 10] = lon_rad;
    bf[CMAG + 11] = ut_rad;
    bf[CMAG + 12] = std::abs(input.lat);

    bf[CMAG + 13] = plg[0][0];
    bf[CMAG + 14] = plg[1][0];
    bf[CMAG + 15] = plg[2][0];
    bf[CMAG + 16] = plg[3][0];
    bf[CMAG + 17] = plg[4][0];
    bf[CMAG + 18] = plg[5][0];
    bf[CMAG + 19] = plg[6][0];
    bf[CMAG + 20] = 0.0;
    bf[CMAG + 21] = plg[1][1];
    bf[CMAG + 22] = plg[2][1];
    bf[CMAG + 23] = plg[3][1];
    bf[CMAG + 24] = plg[4][1];
    bf[CMAG + 25] = plg[5][1];
    bf[CMAG + 26] = plg[6][1];

    bf[CUT]     = ut_rad;
    bf[CUT + 1] = doy_rad;
    bf[CUT + 2] = dfa;
    bf[CUT + 3] = lon_rad;
    bf[CUT + 4] = plg[1][0];
    bf[CUT + 5] = plg[3][0];
    bf[CUT + 6] = plg[5][0];
    bf[CUT + 7] = plg[3][2];
    bf[CUT + 8] = plg[5][2];
}

inline TnParm NRLMSIS21::compute_tfnparm(const std::array<double, 512>& bf) const {
    using namespace msis21::constants;
    using namespace msis21::parm;

    TnParm tpro;
    tpro.cf.fill(0.0);
    tpro.beta.fill(0.0);
    tpro.gamma.fill(0.0);

    for (int ix = 0; ix < ITB0; ix++) {
        tpro.cf[ix] = 0.0;
        for (int j = 0; j <= MBF; j++) {
            tpro.cf[ix] += TN_BETA[j][ix] * bf[j];
        }
        if (smod_[ix]) {
            double dffact_ix = 1.0 / TN_BETA[0][ix];
            tpro.cf[ix] += sfluxmod_tn(ix, bf, dffact_ix);
        }
    }

    tpro.tex = 0.0;
    for (int j = 0; j <= MBF; j++) {
        tpro.tex += TN_BETA[j][ITEX] * bf[j];
    }
    tpro.tex += sfluxmod_tn(ITEX, bf, 1.0 / TN_BETA[0][ITEX]);
    tpro.tex += geomag_impl(TN_BETA, ITEX, bf);
    tpro.tex += utdep_impl(TN_BETA, ITEX, bf);

    tpro.tgb0 = 0.0;
    for (int j = 0; j <= MBF; j++) {
        tpro.tgb0 += TN_BETA[j][ITGB0] * bf[j];
    }
    if (smod_[ITGB0]) tpro.tgb0 += sfluxmod_tn(ITGB0, bf, 1.0 / TN_BETA[0][ITGB0]);
    tpro.tgb0 += geomag_impl(TN_BETA, ITGB0, bf);

    tpro.tb0 = 0.0;
    for (int j = 0; j <= MBF; j++) {
        tpro.tb0 += TN_BETA[j][ITB0] * bf[j];
    }
    if (smod_[ITB0]) tpro.tb0 += sfluxmod_tn(ITB0, bf, 1.0 / TN_BETA[0][ITB0]);
    tpro.tb0 += geomag_impl(TN_BETA, ITB0, bf);

    tpro.sigma = tpro.tgb0 / (tpro.tex - tpro.tb0);

    double bc1 = 1.0 / tpro.tb0;
    double bc2 = -tpro.tgb0 / (tpro.tb0 * tpro.tb0);
    double bc3 = -bc2 * (tpro.sigma + 2.0 * tpro.tgb0 / tpro.tb0);

    tpro.cf[ITB0] = bc1 * C2TN[0][0] + bc2 * C2TN[1][0] + bc3 * C2TN[2][0];
    tpro.cf[ITGB0] = bc1 * C2TN[0][1] + bc2 * C2TN[1][1] + bc3 * C2TN[2][1];
    tpro.cf[ITEX] = bc1 * C2TN[0][2] + bc2 * C2TN[1][2] + bc3 * C2TN[2][2];

    double dot_zetaF = tpro.cf[IZFX] * S4ZETA_F[0] +
                       tpro.cf[IZFX + 1] * S4ZETA_F[1] +
                       tpro.cf[IZFX + 2] * S4ZETA_F[2];
    tpro.tzetaF = 1.0 / dot_zetaF;

    double dot_zetaA = tpro.cf[IZAX] * S4ZETA_A[0] +
                       tpro.cf[IZAX + 1] * S4ZETA_A[1] +
                       tpro.cf[IZAX + 2] * S4ZETA_A[2];
    tpro.tzetaA = 1.0 / dot_zetaA;

    double dfdz_A = tpro.cf[IZAX] * WGHTAXDZ[0] +
                    tpro.cf[IZAX + 1] * WGHTAXDZ[1] +
                    tpro.cf[IZAX + 2] * WGHTAXDZ[2];
    tpro.dlntdzA = -tpro.tzetaA * dfdz_A;

    tpro.beta[0] = tpro.cf[0] * (NODES_TN[4] - NODES_TN[0]) / 4.0;
    for (int ix = 1; ix <= NL; ix++) {
        double wbeta_ix = (NODES_TN[ix + 4] - NODES_TN[ix]) / 4.0;
        tpro.beta[ix] = tpro.beta[ix - 1] + tpro.cf[ix] * wbeta_ix;
    }

    tpro.gamma[0] = tpro.beta[0] * (NODES_TN[5] - NODES_TN[0]) / 5.0;
    for (int ix = 1; ix <= NL; ix++) {
        double wgamma_ix = (NODES_TN[ix + 5] - NODES_TN[ix]) / 5.0;
        tpro.gamma[ix] = tpro.gamma[ix - 1] + tpro.beta[ix] * wgamma_ix;
    }

    tpro.b = 1.0 - tpro.tb0 / tpro.tex;
    tpro.sigmasq = tpro.sigma * tpro.sigma;

    tpro.cVS = -(tpro.beta[ITB0 - 1] * S5ZETA_B[0] +
                 tpro.beta[ITB0] * S5ZETA_B[1] +
                 tpro.beta[ITB0 + 1] * S5ZETA_B[2] +
                 tpro.beta[ITB0 + 2] * S5ZETA_B[3]);

    tpro.cWS = -(tpro.gamma[ITB0 - 2] * S6ZETA_B[0] +
                 tpro.gamma[ITB0 - 1] * S6ZETA_B[1] +
                 tpro.gamma[ITB0] * S6ZETA_B[2] +
                 tpro.gamma[ITB0 + 1] * S6ZETA_B[3] +
                 tpro.gamma[ITB0 + 2] * S6ZETA_B[4]);

    tpro.cVB = -std::log(1.0 - tpro.b) / (tpro.sigma * tpro.tex);
    tpro.cWB = -msis21::utils::dilog(tpro.b) / (tpro.sigmasq * tpro.tex);

    tpro.VzetaF = tpro.beta[IZFX - 1] * S5ZETA_F[0] +
                  tpro.beta[IZFX] * S5ZETA_F[1] +
                  tpro.beta[IZFX + 1] * S5ZETA_F[2] +
                  tpro.beta[IZFX + 2] * S5ZETA_F[3] + tpro.cVS;

    tpro.VzetaA = tpro.beta[IZAX - 1] * S5ZETA_A[0] +
                  tpro.beta[IZAX] * S5ZETA_A[1] +
                  tpro.beta[IZAX + 1] * S5ZETA_A[2] +
                  tpro.beta[IZAX + 2] * S5ZETA_A[3] + tpro.cVS;

    double delzA = ZETA_A - ZETA_B;
    tpro.WzetaA = tpro.gamma[IZAX - 2] * S6ZETA_A[0] +
                  tpro.gamma[IZAX - 1] * S6ZETA_A[1] +
                  tpro.gamma[IZAX] * S6ZETA_A[2] +
                  tpro.gamma[IZAX + 1] * S6ZETA_A[3] +
                  tpro.gamma[IZAX + 2] * S6ZETA_A[4] +
                  tpro.cVS * delzA + tpro.cWS;

    tpro.Vzeta0 = tpro.beta[0] * S5ZETA_0[0] +
                  tpro.beta[1] * S5ZETA_0[1] +
                  tpro.beta[2] * S5ZETA_0[2] + tpro.cVS;

    tpro.lndtotF = LNP0 - MBARG0DIVKB * (tpro.VzetaF - tpro.Vzeta0) - std::log(KB * tpro.tzetaF);

    return tpro;
}

inline double NRLMSIS21::pwmp(double z, const DnParm& dpro) const {
    if (z >= dpro.zetaMi[4]) return dpro.Mi[4];
    if (z <= dpro.zetaMi[0]) return dpro.Mi[0];

    for (int i = 0; i < 4; i++) {
        if (z < dpro.zetaMi[i + 1]) {
            return dpro.Mi[i] + dpro.aMi[i] * (z - dpro.zetaMi[i]);
        }
    }
    return dpro.Mi[4];
}

inline DnParm NRLMSIS21::compute_dfnparm(int ispec, const std::array<double, 512>& bf, const TnParm& tpro) const {
    using namespace msis21::constants;
    using namespace msis21::parm;

    DnParm dpro;
    dpro.ispec = ispec;
    dpro.zetaMi.fill(0.0);
    dpro.Mi.fill(0.0);
    dpro.aMi.fill(0.0);
    dpro.WMi.fill(0.0);
    dpro.XMi.fill(0.0);
    dpro.cf.fill(0.0);

    auto dot_bf = [&](const auto& beta, int iz) {
        double sum = 0.0;
        for (int j = 0; j <= MBF; j++) sum += beta[j][iz] * bf[j];
        return sum;
    };

    switch (ispec) {
        case SPEC_N2: {
            dpro.lnPhiF = LNVMR[ispec];
            dpro.lndref = tpro.lndtotF + dpro.lnPhiF;
            dpro.zref = ZETA_F;
            dpro.zmin = -1.0;
            dpro.zhyd = ZETA_F;
            dpro.zetaM = dot_bf(N2_BETA, 1);
            dpro.HML = N2_BETA[0][2];
            dpro.HMU = N2_BETA[0][3];
            dpro.C = 0.0;
            dpro.zetaC = 0.0;
            dpro.HC = 1.0;
            dpro.R = 0.0;
            dpro.zetaR = N2_BETA[0][8];
            dpro.HR = N2_BETA[0][9];
            break;
        }
        case SPEC_O2: {
            dpro.lnPhiF = LNVMR[ispec];
            dpro.lndref = tpro.lndtotF + dpro.lnPhiF;
            dpro.zref = ZETA_F;
            dpro.zmin = -1.0;
            dpro.zhyd = ZETA_F;
            dpro.zetaM = O2_BETA[0][1];
            dpro.HML = O2_BETA[0][2];
            dpro.HMU = O2_BETA[0][3];
            dpro.C = 0.0;
            dpro.zetaC = 0.0;
            dpro.HC = 1.0;
            dpro.R = dot_bf(O2_BETA, 7);
            dpro.R += geomag_impl(O2_BETA, 7, bf);
            dpro.zetaR = O2_BETA[0][8];
            dpro.HR = O2_BETA[0][9];
            break;
        }
        case SPEC_O: {
            dpro.lnPhiF = 0.0;
            dpro.lndref = dot_bf(O1_BETA, 0);
            dpro.zref = ZETAREF_O1;
            dpro.zmin = NODES_O1[3];
            dpro.zhyd = ZETAREF_O1;
            dpro.zetaM = O1_BETA[0][1];
            dpro.HML = O1_BETA[0][2];
            dpro.HMU = O1_BETA[0][3];
            dpro.C = dot_bf(O1_BETA, 4);
            dpro.zetaC = O1_BETA[0][5];
            dpro.HC = O1_BETA[0][6];
            dpro.R = dot_bf(O1_BETA, 7);
            dpro.R += sfluxmod(7, bf, O1_BETA, 0.0);
            dpro.R += geomag_impl(O1_BETA, 7, bf);
            dpro.R += utdep_impl(O1_BETA, 7, bf);
            dpro.zetaR = O1_BETA[0][8];
            dpro.HR = O1_BETA[0][9];

            for (int k = 0; k < NSPLO1; k++) {
                dpro.cf[k] = dot_bf(O1_BETA, 10 + k);
            }
            break;
        }
        case SPEC_HE: {
            dpro.lnPhiF = LNVMR[ispec];
            dpro.lndref = tpro.lndtotF + dpro.lnPhiF;
            dpro.zref = ZETA_F;
            dpro.zmin = -1.0;
            dpro.zhyd = ZETA_F;
            dpro.zetaM = HE_BETA[0][1];
            dpro.HML = HE_BETA[0][2];
            dpro.HMU = HE_BETA[0][3];
            dpro.C = 0.0;
            dpro.zetaC = 0.0;
            dpro.HC = 1.0;
            dpro.R = dot_bf(HE_BETA, 7);
            dpro.R += sfluxmod(7, bf, HE_BETA, 1.0);
            dpro.R += geomag_impl(HE_BETA, 7, bf);
            dpro.R += utdep_impl(HE_BETA, 7, bf);
            dpro.zetaR = HE_BETA[0][8];
            dpro.HR = HE_BETA[0][9];
            break;
        }
        case SPEC_H: {
            dpro.lnPhiF = 0.0;
            dpro.lndref = dot_bf(H1_BETA, 0);
            dpro.zref = ZETA_A;
            dpro.zmin = 75.0;
            dpro.zhyd = ZETA_F;
            dpro.zetaM = H1_BETA[0][1];
            dpro.HML = H1_BETA[0][2];
            dpro.HMU = H1_BETA[0][3];
            dpro.C = dot_bf(H1_BETA, 4);
            dpro.zetaC = dot_bf(H1_BETA, 5);
            dpro.HC = H1_BETA[0][6];
            dpro.R = dot_bf(H1_BETA, 7);
            dpro.R += sfluxmod(7, bf, H1_BETA, 0.0);
            dpro.R += geomag_impl(H1_BETA, 7, bf);
            dpro.R += utdep_impl(H1_BETA, 7, bf);
            dpro.zetaR = H1_BETA[0][8];
            dpro.HR = H1_BETA[0][9];
            break;
        }
        case SPEC_AR: {
            dpro.lnPhiF = LNVMR[ispec];
            dpro.lndref = tpro.lndtotF + dpro.lnPhiF;
            dpro.zref = ZETA_F;
            dpro.zmin = -1.0;
            dpro.zhyd = ZETA_F;
            dpro.zetaM = AR_BETA[0][1];
            dpro.HML = AR_BETA[0][2];
            dpro.HMU = AR_BETA[0][3];
            dpro.C = 0.0;
            dpro.zetaC = 0.0;
            dpro.HC = 1.0;
            dpro.R = dot_bf(AR_BETA, 7);
            dpro.R += geomag_impl(AR_BETA, 7, bf);
            dpro.R += utdep_impl(AR_BETA, 7, bf);
            dpro.zetaR = AR_BETA[0][8];
            dpro.HR = AR_BETA[0][9];
            break;
        }
        case SPEC_N: {
            dpro.lnPhiF = 0.0;
            dpro.lndref = dot_bf(N1_BETA, 0);
            dpro.lndref += sfluxmod(0, bf, N1_BETA, 0.0);
            dpro.lndref += geomag_impl(N1_BETA, 0, bf);
            dpro.lndref += utdep_impl(N1_BETA, 0, bf);
            dpro.zref = ZETA_B;
            dpro.zmin = 90.0;
            dpro.zhyd = ZETA_F;
            dpro.zetaM = N1_BETA[0][1];
            dpro.HML = N1_BETA[0][2];
            dpro.HMU = N1_BETA[0][3];
            dpro.C = N1_BETA[0][4];
            dpro.zetaC = N1_BETA[0][5];
            dpro.HC = N1_BETA[0][6];
            dpro.R = dot_bf(N1_BETA, 7);
            dpro.zetaR = N1_BETA[0][8];
            dpro.HR = N1_BETA[0][9];
            break;
        }
        case SPEC_OA: {
            dpro.lnPhiF = 0.0;
            dpro.lndref = dot_bf(OA_BETA, 0);
            dpro.lndref += geomag_impl(OA_BETA, 0, bf);
            dpro.zref = ZETAREF_OA;
            dpro.zmin = 120.0;
            dpro.zhyd = 0.0;
            dpro.C = OA_BETA[0][4];
            dpro.zetaC = OA_BETA[0][5];
            dpro.HC = OA_BETA[0][6];
            return dpro;
        }
        case SPEC_NO: {
            dpro.lnPhiF = 0.0;
            dpro.lndref = dot_bf(NO_BETA, 0);
            dpro.lndref += geomag_impl(NO_BETA, 0, bf);
            dpro.zref = ZETAREF_NO;
            dpro.zmin = 72.5;
            dpro.zhyd = ZETAREF_NO;
            dpro.zetaM = dot_bf(NO_BETA, 1);
            dpro.HML = dot_bf(NO_BETA, 2);
            dpro.HMU = dot_bf(NO_BETA, 3);
            dpro.C = dot_bf(NO_BETA, 4);
            dpro.C += geomag_impl(NO_BETA, 4, bf);
            dpro.zetaC = dot_bf(NO_BETA, 5);
            dpro.HC = dot_bf(NO_BETA, 6);
            dpro.R = dot_bf(NO_BETA, 7);
            dpro.zetaR = dot_bf(NO_BETA, 8);
            dpro.HR = dot_bf(NO_BETA, 9);

            for (int k = 0; k < NSPLNO; k++) {
                dpro.cf[k] = dot_bf(NO_BETA, 10 + k);
                dpro.cf[k] += geomag_impl(NO_BETA, 10 + k, bf);
            }
            break;
        }
        default:
            return dpro;
    }

    if (ispec != SPEC_OA) {
        dpro.zetaMi[0] = dpro.zetaM - 2.0 * dpro.HML;
        dpro.zetaMi[1] = dpro.zetaM - dpro.HML;
        dpro.zetaMi[2] = dpro.zetaM;
        dpro.zetaMi[3] = dpro.zetaM + dpro.HMU;
        dpro.zetaMi[4] = dpro.zetaM + 2.0 * dpro.HMU;

        dpro.Mi[0] = MBAR;
        dpro.Mi[4] = SPECMASS[ispec];
        dpro.Mi[2] = (dpro.Mi[0] + dpro.Mi[4]) / 2.0;
        double delM = TANH1 * (dpro.Mi[4] - dpro.Mi[0]) / 2.0;
        dpro.Mi[1] = dpro.Mi[2] - delM;
        dpro.Mi[3] = dpro.Mi[2] + delM;

        for (int i = 0; i < 4; i++) {
            dpro.aMi[i] = (dpro.Mi[i + 1] - dpro.Mi[i]) / (dpro.zetaMi[i + 1] - dpro.zetaMi[i]);
        }

        for (int i = 0; i < 5; i++) {
            double delz = dpro.zetaMi[i] - ZETA_B;
            if (dpro.zetaMi[i] < ZETA_B) {
                auto spl = msis21::utils::bspline_eval(dpro.zetaMi[i], NODES_TN.data(), ND + 2, etaTN_);
                double W = 0.0;
                for (int j = 0; j < 6; j++) {
                    int idx = spl.iz - 5 + j;
                    if (idx >= 0 && idx <= NL) {
                        W += tpro.gamma[idx] * spl.S6[j];
                    }
                }
                dpro.WMi[i] = W + tpro.cVS * delz + tpro.cWS;
            } else {
                dpro.WMi[i] = (0.5 * delz * delz +
                    msis21::utils::dilog(tpro.b * std::exp(-tpro.sigma * delz)) / tpro.sigmasq) / tpro.tex
                    + tpro.cVB * delz + tpro.cWB;
            }
        }

        dpro.XMi[0] = -dpro.aMi[0] * dpro.WMi[0];
        for (int i = 1; i < 4; i++) {
            dpro.XMi[i] = dpro.XMi[i - 1] - dpro.WMi[i] * (dpro.aMi[i] - dpro.aMi[i - 1]);
        }
        dpro.XMi[4] = dpro.XMi[3] + dpro.WMi[4] * dpro.aMi[3];

        if (dpro.zref == ZETA_F) {
            dpro.Mzref = MBAR;
            dpro.Tref = tpro.tzetaF;
            dpro.Izref = MBAR * tpro.VzetaF;
        } else if (dpro.zref == ZETA_A) {
            dpro.Mzref = pwmp(ZETA_A, dpro);
            dpro.Tref = tpro.tzetaA;
            double Izref_raw = dpro.Mzref * tpro.VzetaA;
            dpro.Izref = Izref_raw;
            if (ZETA_A > dpro.zetaMi[0] && ZETA_A < dpro.zetaMi[4]) {
                int seg = 0;
                for (int i = 1; i < 4; i++) {
                    if (ZETA_A >= dpro.zetaMi[i]) seg = i;
                }
                double adj = dpro.aMi[seg] * tpro.WzetaA + dpro.XMi[seg];
                dpro.Izref -= adj;
            } else if (ZETA_A >= dpro.zetaMi[4]) {
                dpro.Izref -= dpro.XMi[4];
            }
        } else if (dpro.zref == ZETA_B) {
            dpro.Mzref = pwmp(ZETA_B, dpro);
            dpro.Tref = tpro.tb0;
            dpro.Izref = 0.0;
            if (ZETA_B > dpro.zetaMi[0] && ZETA_B < dpro.zetaMi[4]) {
                int seg = 0;
                for (int i = 1; i < 4; i++) {
                    if (ZETA_B >= dpro.zetaMi[i]) seg = i;
                }
                dpro.Izref -= dpro.XMi[seg];
            } else if (ZETA_B >= dpro.zetaMi[4]) {
                dpro.Izref -= dpro.XMi[4];
            }
        }
    }

    if (ispec == SPEC_O) {
        double Cterm = dpro.C * std::exp(-(dpro.zref - dpro.zetaC) / dpro.HC);
        double Rterm0 = std::tanh((dpro.zref - dpro.zetaR) / (HRfactO1ref_ * dpro.HR));
        double Rterm = dpro.R * (1.0 + Rterm0);

        double bc0 = dpro.lndref - Cterm + Rterm
                   - dpro.cf[NSPLO1 - 1] * C1O1ADJ[0];
        double bc1 = -dpro.Mzref * G0DIVKB / tpro.tzetaA
                   - tpro.dlntdzA
                   + Cterm / dpro.HC
                   + Rterm * (1.0 - Rterm0) / dpro.HR * dHRfactO1ref_
                   - dpro.cf[NSPLO1 - 1] * C1O1ADJ[1];

        dpro.cf[NSPLO1]     = bc0 * C1O1[0][0] + bc1 * C1O1[1][0];
        dpro.cf[NSPLO1 + 1] = bc0 * C1O1[0][1] + bc1 * C1O1[1][1];
    }

    if (ispec == SPEC_NO && dpro.lndref != 0.0) {
        double Cterm = dpro.C * std::exp(-(dpro.zref - dpro.zetaC) / dpro.HC);
        double Rterm0 = std::tanh((dpro.zref - dpro.zetaR) / (HRfactNOref_ * dpro.HR));
        double Rterm = dpro.R * (1.0 + Rterm0);

        double bc0 = dpro.lndref - Cterm + Rterm
                   - dpro.cf[NSPLNO - 1] * C1NOADJ[0];
        double bc1 = -dpro.Mzref * G0DIVKB / tpro.tb0
                   - tpro.tgb0 / tpro.tb0
                   + Cterm / dpro.HC
                   + Rterm * (1.0 - Rterm0) / dpro.HR * dHRfactNOref_
                   - dpro.cf[NSPLNO - 1] * C1NOADJ[1];

        dpro.cf[NSPLNO]     = bc0 * C1NO[0][0] + bc1 * C1NO[1][0];
        dpro.cf[NSPLNO + 1] = bc0 * C1NO[0][1] + bc1 * C1NO[1][1];
    }

    return dpro;
}

inline double NRLMSIS21::compute_dfnx(double zeta, double tn, double lndtotz, double Vz, double Wz, double HRfact,
                                       const TnParm& tpro, const DnParm& dpro) const {
    using namespace msis21::constants;

    if (zeta < dpro.zmin) {
        return DMISSING;
    }

    if (dpro.ispec == SPEC_OA) {
        double lnd = dpro.lndref - (zeta - dpro.zref) / H_OA;
        if (dpro.C != 0.0) {
            lnd -= dpro.C * std::exp(-(zeta - dpro.zetaC) / dpro.HC);
        }
        return std::exp(lnd);
    }

    if (dpro.ispec == SPEC_NO && dpro.lndref == 0.0) {
        return DMISSING;
    }

    double ccor = 0.0;
    switch (dpro.ispec) {
        case SPEC_N2:
        case SPEC_O2:
        case SPEC_HE:
        case SPEC_AR:
            ccor = dpro.R * (1.0 + std::tanh((zeta - dpro.zetaR) / (HRfact * dpro.HR)));
            break;
        case SPEC_O:
        case SPEC_H:
        case SPEC_N:
        case SPEC_NO:
            ccor = -dpro.C * std::exp(-(zeta - dpro.zetaC) / dpro.HC) +
                    dpro.R * (1.0 + std::tanh((zeta - dpro.zetaR) / (HRfact * dpro.HR)));
            break;
    }

    if (zeta < dpro.zhyd) {
        if (dpro.ispec == SPEC_N2 || dpro.ispec == SPEC_O2 ||
            dpro.ispec == SPEC_HE || dpro.ispec == SPEC_AR) {
            return std::exp(lndtotz + dpro.lnPhiF + ccor);
        }
        if (dpro.ispec == SPEC_O) {
            auto spl = bspline_eval_o4(zeta, NODES_O1, NDO1 + 1, etaO1_);
            double lnf = 0.0;
            for (int k = 0; k < 4; k++) {
                int idx = spl.iz - 3 + k;
                if (idx >= 0 && idx < NSPLO1 + 2) {
                    lnf += dpro.cf[idx] * spl.S4[k];
                }
            }
            return std::exp(lnf);
        }
        if (dpro.ispec == SPEC_NO) {
            auto spl = bspline_eval_o4(zeta, NODES_NO, NDNO + 1, etaNO_);
            double lnf = 0.0;
            for (int k = 0; k < 4; k++) {
                int idx = spl.iz - 3 + k;
                if (idx >= 0 && idx < NSPLNO + 2) {
                    lnf += dpro.cf[idx] * spl.S4[k];
                }
            }
            return std::exp(lnf);
        }
        return DMISSING;
    }

    double Mz = pwmp(zeta, dpro);
    double Ihyd = Mz * Vz - dpro.Izref;

    if (zeta > dpro.zetaMi[0] && zeta < dpro.zetaMi[4]) {
        int seg = 0;
        for (int i = 1; i < 4; i++) {
            if (zeta >= dpro.zetaMi[i]) seg = i;
        }
        Ihyd -= (dpro.aMi[seg] * Wz + dpro.XMi[seg]);
    } else if (zeta >= dpro.zetaMi[4]) {
        Ihyd -= dpro.XMi[4];
    }

    double lnd = dpro.lndref - Ihyd * G0DIVKB + ccor;
    return std::exp(lnd) * dpro.Tref / tn;
}

inline double NRLMSIS21::compute_temperature(double zeta, const TnParm& tpro,
                                              int iz, const std::array<double, 6>& S4) const {
    using namespace msis21::constants;

    if (zeta < ZETA_B) {
        int i = std::max(iz - 3, 0);
        int j_start = (iz < 3) ? -iz : -3;

        double dot = 0.0;
        for (int j = j_start; j <= 0; j++) {
            dot += tpro.cf[i + j - j_start] * S4[j + 3];
        }
        return 1.0 / dot;
    } else {
        return tpro.tex - (tpro.tex - tpro.tb0) * std::exp(-tpro.sigma * (zeta - ZETA_B));
    }
}

inline double NRLMSIS21::compute_Wz(double zeta, const TnParm& tpro, int iz, const std::array<double, 6>& S6) const {
    using namespace msis21::constants;

    double delz = zeta - ZETA_B;
    if (zeta < ZETA_B) {
        double W = 0.0;
        for (int j = 0; j < 6; j++) {
            int idx = iz - 5 + j;
            if (idx >= 0 && idx <= NL) {
                W += tpro.gamma[idx] * S6[j];
            }
        }
        return W + tpro.cVS * delz + tpro.cWS;
    } else {
        return (0.5 * delz * delz +
            msis21::utils::dilog(tpro.b * std::exp(-tpro.sigma * delz)) / tpro.sigmasq) / tpro.tex
            + tpro.cVB * delz + tpro.cWB;
    }
}

// dT/dz computation using B-spline derivative
inline double NRLMSIS21::compute_dT_dz(double zeta, const TnParm& tpro, int iz,
                                         const std::array<double, 6>& S3) const {
    using namespace msis21::constants;

    if (zeta >= ZETA_B) {
        // Bates profile: T(z) = tex - (tex - tb0)*exp(-sigma*(z - zetaB))
        // dT/dz = sigma * (tex - tb0) * exp(-sigma*(z - zetaB)) = sigma * (tex - T(z))
        double T = tpro.tex - (tpro.tex - tpro.tb0) * std::exp(-tpro.sigma * (zeta - ZETA_B));
        return tpro.sigma * (tpro.tex - T);
    }

    // Below zetaB: T = 1/f where f = Σ cf[j]*B_{j,4}(z)
    // dT/dz = -T^2 * df/dz
    // df/dz = Σ_m d[m]*N_{m,3}(z) where d[m] = 3*(cf[m]-cf[m-1])/(t[m+3]-t[m])
    // S3 weights: S3[j] = N_{iz-2+j, 3}(z) for j=0,1,2
    double dfdz = 0.0;
    for (int j = 0; j < 3; j++) {
        int m = iz - 2 + j;  // Basis function index for order-3 spline
        if (m >= 1 && m <= NL) {
            double dcf = tpro.cf[m] - tpro.cf[m - 1];
            double dnode = NODES_TN[m + 3] - NODES_TN[m];
            if (dnode > 0.0) {
                dfdz += dcf * 3.0 / dnode * S3[j];
            }
        }
    }

    auto spl = msis21::utils::bspline_eval(zeta, NODES_TN.data(), ND + 2, etaTN_);
    double T = compute_temperature(zeta, tpro, spl.iz, spl.S4);

    return -T * T * dfdz;
}

inline NRLMSIS21::Output NRLMSIS21::msiscalc(const Input& input) const {
    using namespace msis21::constants;
    using namespace msis21::utils;

    Output output;
    output.dn.fill(DMISSING);

    double zeta = alt2gph(input.lat, input.alt);

    std::array<double, 512> bf;
    compute_globe(input, bf);

    TnParm tpro = compute_tfnparm(bf);
    output.tex = tpro.tex;

    BSplineResult spl;
    if (zeta < ZETA_B) {
        spl = bspline_eval(zeta, NODES_TN.data(), ND + 2, etaTN_);
    } else {
        spl.iz = NL;
        spl.S3.fill(0.0);
        spl.S4.fill(0.0);
        spl.S5.fill(0.0);
        spl.S6.fill(0.0);
    }

    output.tn = compute_temperature(zeta, tpro, spl.iz, spl.S4);

    double delz = zeta - ZETA_B;
    double Vz, Wz, lndtotz = 0.0;
    if (zeta < ZETA_F) {
        int i_start = std::max(spl.iz - 4, 0);
        int j_start = (spl.iz < 4) ? -spl.iz : -4;
        Vz = 0.0;
        for (int l = j_start; l <= 0; l++) {
            int beta_idx = i_start + (l - j_start);
            Vz += tpro.beta[beta_idx] * spl.S5[l + 4];
        }
        Vz += tpro.cVS;
        Wz = 0.0;
        double lnPz = LNP0 - MBARG0DIVKB * (Vz - tpro.Vzeta0);
        lndtotz = lnPz - std::log(KB * output.tn);
    } else if (zeta < ZETA_B) {
        Vz = 0.0;
        for (int k = 0; k < 5; k++) {
            int idx = spl.iz - 4 + k;
            if (idx >= 0 && idx <= NL) {
                Vz += tpro.beta[idx] * spl.S5[k];
            }
        }
        Vz += tpro.cVS;
        Wz = compute_Wz(zeta, tpro, spl.iz, spl.S6);
    } else {
        Vz = (delz + std::log(output.tn / tpro.tex) / tpro.sigma) / tpro.tex + tpro.cVB;
        Wz = (0.5 * delz * delz +
            dilog(tpro.b * std::exp(-tpro.sigma * delz)) / tpro.sigmasq) / tpro.tex
            + tpro.cVB * delz + tpro.cWB;
    }

    double HRfact = 0.5 * (1.0 + std::tanh(H_GAMMA * (zeta - ZETA_GAMMA)));

    for (int ispec = 2; ispec <= 10; ispec++) {
        DnParm dpro = compute_dfnparm(ispec, bf, tpro);
        output.dn[ispec] = compute_dfnx(zeta, output.tn, lndtotz, Vz, Wz, HRfact, tpro, dpro);
    }

    output.dn[1] = 0.0;
    for (int i = 2; i <= 9; i++) {
        if (output.dn[i] != DMISSING && output.dn[i] > 0.0) {
            output.dn[1] += output.dn[i] * SPECMASS[i];
        }
    }

    return output;
}

inline NRLMSIS21::FullOutput NRLMSIS21::msiscalc_with_derivative(const Input& input) const {
    using namespace msis21::constants;
    using namespace msis21::utils;

    FullOutput full;
    full.output = msiscalc(input);

    double zeta = alt2gph(input.lat, input.alt);

    std::array<double, 512> bf;
    compute_globe(input, bf);
    TnParm tpro = compute_tfnparm(bf);

    if (zeta < ZETA_B) {
        BSplineResult spl = bspline_eval(zeta, NODES_TN.data(), ND + 2, etaTN_);
        full.dT_dz = compute_dT_dz(zeta, tpro, spl.iz, spl.S3);
    } else {
        full.dT_dz = tpro.sigma * (tpro.tex - full.output.tn);
    }

    return full;
}

}  // namespace atmosphere
}  // namespace refraction
