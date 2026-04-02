"""
Ciddor 1996 optical refractive index of moist air.

Implements the equations from:

    Ciddor, P.E. (1996). "Refractive index of air: new equations for the
    visible and near infrared." Applied Optics, 35(9), 1566-1573.

The method decomposes the refractivity of moist air into two independently
scaled components -- dry air and water vapor -- each referenced to standard
conditions at which laboratory dispersion measurements were made:

    n_prop - 1 = (rho_a / rho_axs)(n_axs - 1)
              + (rho_w / rho_ws )(n_ws  - 1)        [Eq. 5]

where:
    rho_a   = density of the dry-air component at ambient conditions
    rho_axs = density of standard dry air (15 degC, 101325 Pa, x_c ppm CO2)
    n_axs   = refractivity of standard dry air at x_c ppm CO2
    rho_w   = density of the water-vapor component at ambient conditions
    rho_ws  = density of pure water vapor at reference conditions
              (20 degC, 1333 Pa)
    n_ws    = refractivity of pure water vapor at reference conditions

All densities use the BIPM 1981/91 equation (Giacomo [23], Davis [24]):

    rho = (p * M_a) / (Z * R * T) * [1 - x_w(1 - M_w / M_a)]   [Eq. 4]

Valid ranges (from paper, Section 5):
    Temperature:   -40 to +100 degC
    Pressure:      80 to 120 kPa
    Humidity:      0 to 100%
    Wavelength:    300 to 1690 nm (code accepts up to 1700 nm)
    CO2:           0 to 600 ppm

Stated uncertainty: 2-5 parts in 10^8 in refractive index (1-sigma).

References
----------
.. [1] Peck and Reeder (1972), J. Opt. Soc. Am. 62, 958-962.
       Dry-air dispersion equation (Eq. 1 constants).
.. [2] Erickson (1962), J. Opt. Soc. Am. 52, 777-780.
       Water-vapor dispersion equation (Eq. 3 constants).
.. [3] Barrell and Sears (1939), Phil. Trans. R. Soc. London A 238, 1-64.
       Original water-vapor refractivity measurements at 20 degC, 1333 Pa.
.. [4] Giacomo (1982), Metrologia 18, 33-40.  BIPM density equation,
       SVP formula, and enhancement factor.
.. [5] Davis (1992), Metrologia 29, 67-70.  Revised BIPM density equation.
"""

from .water_vapor import water_vapor_mole_fraction

# ---------------------------------------------------------------------------
# Dispersion constants for standard dry air
# ---------------------------------------------------------------------------
# From Peck and Reeder [1], adjusted by Ciddor for ITS-90 temperature scale
# (+9 mK at 15 degC) and CO2 content change (300 -> 450 ppm, net +50 to
# numerators).  See Appendix A and the discussion following Eq. (1).
#
#   10^8 (n_as - 1) = k1 / (k0 - sigma^2) + k3 / (k2 - sigma^2)   [Eq. 1]
#
# where sigma = 1/lambda in um^-1.  All k_i have units of um^-2.
k0 = 238.0185
k1 = 5792105.0
k2 = 57.362
k3 = 167917.0

# ---------------------------------------------------------------------------
# Water vapor dispersion constants
# ---------------------------------------------------------------------------
# From Erickson [2] via Owens [7], scaled to SI and fitted to modern absolute
# refractivity from Barrell and Sears [3].  Reference conditions: 20 degC,
# 1333 Pa (10 Torr).  See Appendix A.
#
#   10^8 (n_ws - 1) = cf * (w0 + w1*sigma^2 + w2*sigma^4 + w3*sigma^6)
#                                                                    [Eq. 3]
#
# The code absorbs the 10^-8 factor into cf (= 1.022e-8) so that
# n_water_vapor() returns (n_ws - 1) directly, consistent with
# n_standard_air() which applies its own 1e-8 multiplier.
#
# cf = 1.022 is a correction factor to reconcile calculated refractivity
# with Barrell & Sears measurements, compensating for an apparent ~2.2%
# error in the absolute refractivity of water vapor at the standard
# conditions (see paper Section 2, paragraph beginning "However...").
w0 = 295.235
w1 = 2.6422
w2 = -0.032380
w3 = 0.004028
cf = 1.022e-8

# ---------------------------------------------------------------------------
# Compressibility factor Z coefficients
# ---------------------------------------------------------------------------
# From the BIPM 1981/91 density equation (Giacomo [4], Davis [5]).
# See Appendix A, Eq. (12).
#
#   Z = 1 - (p/T)(a0 + a1*t + a2*t^2 + (b0+b1*t)*x_w
#           + (c0+c1*t)*x_w^2) + (p/T)^2 * (d + e*x_w^2)
#
# where t = T - 273.15 (degC), p in Pa, T in K, x_w = water vapor mole
# fraction.
a0 = 1.58123e-6   # K Pa^-1
a1 = -2.9331e-8   # Pa^-1
a2 = 1.1043e-10   # K^-1 Pa^-1
b0 = 5.707e-6     # K Pa^-1
b1 = -2.051e-8    # Pa^-1
c0 = 1.9898e-4    # K Pa^-1
c1 = -2.376e-6    # Pa^-1
d = 1.83e-11       # K^2 Pa^-2
e = -0.765e-8      # K^2 Pa^-2

# ---------------------------------------------------------------------------
# Reference conditions
# ---------------------------------------------------------------------------
# Standard dry air: 15 degC, 101325 Pa, 450 ppm CO2, 0% humidity.
# These are the conditions at which n_axs (Eqs. 1-2) is defined.
T_ref = 288.15     # K  (15 degC)
P_ref = 101325.0   # Pa (1 atm)

# Standard water vapor: 20 degC, 1333 Pa (10 Torr).
# These are the conditions assumed by Barrell and Sears [3] for their
# water-vapor refractivity measurements from which Eq. (3) is derived.
# NOTE: 1333 Pa = 10 Torr, NOT the SVP at 20 degC (~2339 Pa).  Using SVP
# here is a known error source that produces a factor ~1.756 error in the
# water-vapor refractivity contribution.
T_ref_w = 293.15   # K  (20 degC)
P_ref_w = 1333.0   # Pa (10 Torr)

# Physical constants
R = 8.314462618    # J mol^-1 K^-1, gas constant (CODATA 2018)

# Molar masses
M_a = 28.9635e-3   # kg/mol, dry air at 400 ppm CO2 (base value before
                    # CO2 adjustment; see M_a_adj in ciddor_n)
M_w = 18.01528e-3  # kg/mol, water


def n_standard_air(sigma2):
    """
    Refractivity of standard dry air at 15 degC, 101325 Pa, 450 ppm CO2.

    Implements Eq. (1) from Ciddor 1996, using the Peck and Reeder [1]
    dispersion equation adjusted for ITS-90 and 450 ppm CO2.

    Parameters
    ----------
    sigma2 : float
        Square of the vacuum wavenumber, sigma^2 = (1/lambda_um)^2,
        in um^-2.

    Returns
    -------
    float
        Refractivity (n_as - 1), a dimensionless quantity typically
        of order 2.7e-4 in the visible.

    Notes
    -----
    The function returns refractivity (n - 1), not the refractive index n.
    The 10^-8 scaling from Eq. (1) is applied internally.
    """
    return 1e-8 * (k1 / (k0 - sigma2) + k3 / (k2 - sigma2))


def n_water_vapor(sigma2):
    """
    Refractivity of pure water vapor at 20 degC, 1333 Pa.

    Implements Eq. (3) from Ciddor 1996, using Erickson's [2] dispersion
    equation with the cf = 1.022 correction factor to match Barrell and
    Sears [3] absolute refractivity measurements.

    Parameters
    ----------
    sigma2 : float
        Square of the vacuum wavenumber, sigma^2 = (1/lambda_um)^2,
        in um^-2.

    Returns
    -------
    float
        Refractivity (n_ws - 1), a dimensionless quantity typically
        of order 3e-6 at visible wavelengths and 1333 Pa.

    Notes
    -----
    The 10^-8 factor from Eq. (3) is absorbed into the ``cf`` constant
    (stored as 1.022e-8 rather than 1.022), so this function directly
    returns (n_ws - 1) without an additional scaling step.
    """
    return cf * (w0 + w1 * sigma2 + w2 * sigma2**2 + w3 * sigma2**3)


def compressibility_Z(T_K, P_Pa, x_w):
    """
    Compressibility factor Z of moist air.

    Implements Eq. (12) from Ciddor 1996 / BIPM 1981/91 (Giacomo [4],
    Davis [5]).  Z accounts for the non-ideal-gas behavior of moist air
    and appears in the denominator of the BIPM density equation (Eq. 4).

    Parameters
    ----------
    T_K : float
        Temperature in kelvin.
    P_Pa : float
        Total pressure in pascals.
    x_w : float
        Mole fraction of water vapor (0 to 1).  Use 0.0 for dry air
        and 1.0 for pure water vapor.

    Returns
    -------
    float
        Compressibility factor Z (dimensionless, close to 1.0).
        For standard air at 15 degC, 101325 Pa, Z ~ 0.99960.
    """
    t = T_K - 273.15
    return (1.0 - (P_Pa / T_K) * (a0 + a1 * t + a2 * t * t
            + (b0 + b1 * t) * x_w + (c0 + c1 * t) * x_w * x_w)
            + (P_Pa / T_K) * (P_Pa / T_K) * (d + e * x_w * x_w))


def ciddor_n(T_C, P_kPa, RH, lambda_nm, xCO2=450e-6):
    """
    Refractive index of moist air (Ciddor 1996).

    Computes the phase refractive index of moist air using the two-component
    (dry air + water vapor) density-scaling method of Ciddor [Eq. 5]:

        n - 1 = (rho_a / rho_axs)(n_axs - 1)
              + (rho_w / rho_ws )(n_ws  - 1)

    This follows Appendix B steps 1-10 of the paper.

    Parameters
    ----------
    T_C : float
        Temperature in degrees Celsius.
    P_kPa : float
        Total atmospheric pressure in kilopascals.
    RH : float
        Fractional relative humidity, 0.0 (dry) to 1.0 (saturated).
    lambda_nm : float
        Vacuum wavelength in nanometers.  Valid range: 300-1700 nm.
    xCO2 : float, optional
        CO2 mole fraction.  Default is 450e-6 (450 ppm).

    Returns
    -------
    float
        Phase refractive index n (typically ~1.000270 at sea level, visible).

    Notes
    -----
    The calculation follows these steps (Appendix B):

    1. Compute vacuum wavenumber sigma = 1/lambda (um^-1) and sigma^2.
    2. Compute refractivity of standard dry air n_as from Eq. (1).
    3. Apply CO2 correction to get n_axs from Eq. (2).
    4. Compute refractivity of standard water vapor n_ws from Eq. (3).
    5. Compute water vapor mole fraction x_w from SVP, enhancement factor,
       and relative humidity (Giacomo [4]).
    6. Compute compressibility factors Z_a (dry ref), Z (ambient), Z_w
       (water vapor ref) from Eq. (12).
    7. Compute CO2-adjusted molar mass M_a' from the text following Eq. (4).
    8. Compute four densities from Eq. (4):
       - rho_axs: standard dry air (x_w=0, T_ref, P_ref)
       - rho_a:   dry-air component of ambient moist air
       - rho_ws:  pure water vapor at reference conditions (x_w=1)
       - rho_w:   water-vapor component of ambient moist air
    9. Combine via Eq. (5) and return n = 1 + n_minus_1.

    **Density equation (Eq. 4) simplifications:**

    For rho_axs (standard dry air, x_w = 0):
        rho = p*M_a / (Z*R*T) * [1 - 0*(1 - M_w/M_a)]
            = p*M_a / (Z*R*T)

    For rho_a (dry component of moist air, partial pressure p*(1-x_w)):
        rho_a = p*(1-x_w)*M_a / (Z*R*T)
        This uses the total-pressure form with the dry fraction (1-x_w)
        extracted, since Eq. (4) for the full mixture gives the dry-air
        density contribution as p*M_a*(1-x_w) / (Z*R*T).

    For rho_ws (pure water vapor, x_w = 1 by definition):
        rho = p*M_a / (Z*R*T) * [1 - 1*(1 - M_w/M_a)]
            = p*M_a / (Z*R*T) * (M_w/M_a)
            = p*M_w / (Z*R*T)
        The M_a cancels algebraically.  This is exact, not an approximation:
        rho_ws IS the density of pure water vapor at 20 degC, 1333 Pa.
        x_w = 1 because rho_ws is a fixed reference density matching
        Barrell & Sears' lab conditions, not the actual atmosphere.
        The actual x_w enters through rho_w (see below), and the ratio
        rho_w / rho_ws scales the reference refractivity to actual conditions.

    For rho_w (water vapor component of moist air):
        rho_w = p*x_w*M_w / (Z*R*T)

    Examples
    --------
    >>> ciddor_n(20.0, 101.325, 0.0, 633.0, 450e-6)
    1.0002714...

    Dry air at 20 degC, 1 atm, 633 nm matches Ciddor Table 1 within
    the stated uncertainty.
    """
    T_K = T_C + 273.15
    P_Pa = P_kPa * 1000.0

    # --- Step 1: Vacuum wavenumber ---
    # Convert wavelength to um, then sigma = 1/lambda_um (um^-1).
    lambda_um = lambda_nm * 1e-3
    sigma = 1.0 / lambda_um
    sigma2 = sigma * sigma

    # --- Steps 2-3: Dry-air refractivity with CO2 correction ---
    # n_as: refractivity at 450 ppm CO2 [Eq. 1]
    # n_axs: refractivity at xCO2 ppm CO2 [Eq. 2]
    n_as = n_standard_air(sigma2)
    co2_corr = 1.0 + 0.534e-6 * (xCO2 * 1e6 - 450.0)
    n_axs = n_as * co2_corr

    # --- Step 4: Water vapor refractivity [Eq. 3] ---
    n_ws = n_water_vapor(sigma2)

    # --- Step 5: Water vapor mole fraction ---
    # x_w = f * h * svp / p, where f is the enhancement factor (Giacomo),
    # h is fractional RH, and svp is the saturation vapor pressure.
    x_w = water_vapor_mole_fraction(T_C, P_Pa, RH)

    # --- Step 6: Compressibility factors [Eq. 12] ---
    Z_a = compressibility_Z(T_ref, P_ref, 0.0)       # dry standard air
    Z = compressibility_Z(T_K, P_Pa, x_w)             # ambient moist air
    Z_w = compressibility_Z(T_ref_w, P_ref_w, 1.0)    # pure water vapor ref

    # --- Step 7: CO2-adjusted molar mass of dry air ---
    # M_a = 10^-3 [28.9635 + 12.011e-6 * (xc - 400)]  (text after Eq. 4)
    # Here xCO2 is a mole fraction (e.g. 450e-6), so (xCO2 - 0.0004)
    # replaces 10^-6 * (xc_ppm - 400).
    M_a_adj = (28.9635 + 12.011 * (xCO2 - 0.0004)) * 1e-3

    # --- Steps 8-9: Densities from Eq. (4) ---
    # See docstring "Density equation simplifications" for derivations.
    rho_axs = P_ref * M_a_adj / (Z_a * R * T_ref)         # standard dry air
    rho_a = P_Pa * (1.0 - x_w) * M_a_adj / (Z * R * T_K)  # dry component
    rho_ws = P_ref_w * M_w / (Z_w * R * T_ref_w)          # pure water vapor ref
    rho_w = P_Pa * x_w * M_w / (Z * R * T_K)              # water vapor component

    # --- Step 10: Combine via Eq. (5) ---
    n_minus_1 = (rho_a / rho_axs) * n_axs + (rho_w / rho_ws) * n_ws
    return 1.0 + n_minus_1
