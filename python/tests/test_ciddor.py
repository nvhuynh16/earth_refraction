"""Tests for Ciddor 1996 model."""

import pytest
from refraction.ciddor import ciddor_n
from refraction.water_vapor import svp_giacomo


VECTORS = [
    (20,  101.325,   0, 633, 1.000271800),
    (20,   60,       0, 633, 1.000160924),
    (20,  120,       0, 633, 1.000321916),
    (50,  100,       0, 633, 1.000243285),
    ( 5,  100,       0, 633, 1.000282756),
    (-40, 100,       0, 633, 1.000337580),
    (50,  120,     100, 633, 1.000287924),
    (40,  120,      75, 633, 1.000299418),
    (20,  100,     100, 633, 1.000267394),
    (40,  110,     100, 1700, 1.000270247),
    (20,  101.325,   0, 1700, 1.000268479),
    (40,  110,     100, 300, 1.000289000),
    (20,  101.325,   0, 300, 1.000286581),
    (-40, 120,       0, 300, 1.000427233),
]


@pytest.mark.parametrize("T_C,P_kPa,RH_pct,lam,n_exp", VECTORS,
                         ids=[f"vec{i+1}" for i in range(len(VECTORS))])
def test_nist_test_vectors(T_C, P_kPa, RH_pct, lam, n_exp):
    RH_frac = RH_pct / 100.0
    n = ciddor_n(T_C, P_kPa, RH_frac, lam)
    assert abs(n - n_exp) < 2e-8, (
        f"T={T_C}C P={P_kPa}kPa RH={RH_pct}% lam={lam}nm: "
        f"expected {n_exp:.10f}, got {n:.10f}, diff={abs(n - n_exp):.2e}"
    )


# ── Ciddor 1996, Table 1: Phase refractivity of dry air (450 ppm CO2, λ=633 nm) ──
TABLE1_VECTORS = [
    # (T_C, P_kPa, expected 1e8*(n-1))
    (20,  80, 21458.0),
    (20, 100, 26824.4),
    (20, 120, 32191.6),
    (10, 100, 27774.7),
    (30, 100, 25937.2),
]


@pytest.mark.parametrize("T_C,P_kPa,expected", TABLE1_VECTORS,
                         ids=[f"T1_{i}" for i in range(len(TABLE1_VECTORS))])
def test_ciddor_table1_dry_air(T_C, P_kPa, expected):
    n = ciddor_n(T_C, P_kPa, 0.0, 633.0, 450e-6)
    refractivity = 1e8 * (n - 1.0)
    diff = abs(refractivity - expected)
    assert diff < 3.0, (
        f"Table1: T={T_C}C P={P_kPa}kPa: "
        f"expected {expected}, got {refractivity:.4f}, diff={diff:.4f}"
    )


# ── Ciddor 1996, Table 2: Phase refractivity of moist air (λ=633 nm) ──
TABLE2_VECTORS = [
    # (T_C, P_Pa, pw_Pa, xc_ppm, expected 1e8*(n-1))
    (19.526, 102094.8, 1065, 510, 27392.9),
    (19.517, 102096.8, 1065, 510, 27394.0),
    (19.173, 102993.0,  641, 450, 27682.4),
    (19.173, 103006.0,  642, 440, 27685.8),
    (19.188, 102918.8,  706, 450, 27658.7),
    (19.189, 102927.8,  708, 440, 27660.8),
    (19.532, 103603.2,  986, 600, 27802.0),
    (19.534, 103596.2,  962, 600, 27800.8),
    (19.534, 103599.2,  951, 610, 27802.2),
]


@pytest.mark.parametrize("T_C,P_Pa,pw_Pa,xc_ppm,expected", TABLE2_VECTORS,
                         ids=[f"T2_{i}" for i in range(len(TABLE2_VECTORS))])
def test_ciddor_table2_moist_air(T_C, P_Pa, pw_Pa, xc_ppm, expected):
    T_K = T_C + 273.15
    svp = svp_giacomo(T_K)
    RH_frac = pw_Pa / svp
    P_kPa = P_Pa / 1000.0
    xCO2 = xc_ppm * 1e-6
    n = ciddor_n(T_C, P_kPa, RH_frac, 633.0, xCO2)
    refractivity = 1e8 * (n - 1.0)
    diff = abs(refractivity - expected)
    assert diff < 2.0, (
        f"Table2: T={T_C}C P={P_Pa}Pa pw={pw_Pa}Pa xc={xc_ppm}ppm: "
        f"expected {expected}, got {refractivity:.4f}, diff={diff:.4f}"
    )


# ── Ciddor 1996, Table 3: Extreme atmospheric conditions (450 ppm CO2, λ=633 nm) ──
TABLE3_VECTORS = [
    # (T_C, P_kPa, RH_pct, expected 1e8*(n-1))
    (20,  80,  75, 21394.0),
    (20, 120,  75, 32127.8),
    (40,  80,  75, 19896.5),  # Paper prints 19996.5 — typo: Difference col gives -12.9 = 19883.6 - 19896.5
    (40, 120,  75, 29941.8),
    (50,  80, 100, 19058.4),
    (50, 120, 100, 28792.4),
]


@pytest.mark.parametrize("T_C,P_kPa,RH_pct,expected", TABLE3_VECTORS,
                         ids=[f"T3_{i}" for i in range(len(TABLE3_VECTORS))])
def test_ciddor_table3_extreme(T_C, P_kPa, RH_pct, expected):
    RH_frac = RH_pct / 100.0
    n = ciddor_n(T_C, P_kPa, RH_frac, 633.0, 450e-6)
    refractivity = 1e8 * (n - 1.0)
    diff = abs(refractivity - expected)
    assert diff < 5.0, (
        f"Table3: T={T_C}C P={P_kPa}kPa RH={RH_pct}%: "
        f"expected {expected}, got {refractivity:.4f}, diff={diff:.4f}"
    )
