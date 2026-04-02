"""
Refractive index of pure water (IAPWS R9-97).

Implements the IAPWS 1997 release [1] on the refractive index of ordinary
water substance, based on the modified Lorentz-Lorenz relation of
Schiebener et al. (1990) [2] with revised coefficients from Harvey et al.
(1998) [3]:

    n^2 = (2*A + 1) / (1 - A)                                     [Eq. 1]

    A = delta * (a0 + a1*delta + a2*theta + a3*L^2*theta
               + a4/L^2 + a5/(L^2 - L_UV^2) + a6/(L^2 - L_IR^2)
               + a7*delta^2)                                       [Eq. 2]

where delta = rho/rho*, theta = T/T*, L = lambda/lambda* are reduced
density, temperature, and wavelength, with reference values rho* = 1000 kg/m^3,
T* = 273.15 K, lambda* = 0.589 um (sodium D line).  L_UV and L_IR are UV and
IR resonance parameters (already in reduced units).

Valid ranges (from [1]):
    Temperature:   -12 to 500 deg C (261.15 to 773.15 K)
    Density:       0 to 1060 kg/m^3
    Wavelength:    0.2 to 1.1 um (200 to 1100 nm)

IAPWS check value: n = 1.33285 at T = 298.15 K, rho = 997.047 kg/m^3,
lambda = 0.5893 um.

References
----------
.. [1] IAPWS (1997). "Release on the Refractive Index of Ordinary Water
       Substance as a Function of Wavelength, Temperature, and Pressure."
       Adopted September 1997.
.. [2] Schiebener, P. et al. (1990). "Refractive index of water and steam
       as function of wavelength, temperature and density."
       J. Phys. Chem. Ref. Data, 19(3), 677-717.
.. [3] Harvey, A.H. et al. (1998). "Revised formulation for the refractive
       index of water and steam as a function of wavelength, temperature and
       density." J. Phys. Chem. Ref. Data, 27(4), 761-774.
.. [4] Kell, G.S. (1975). "Density, thermal expansivity, and compressibility
       of liquid water from 0 to 150 C." J. Chem. Eng. Data, 20(1), 97-105.

Parameters
----------
Temperatures in deg C, pressures in dbar, wavelengths in nm, densities
in kg/m^3.
"""

import math

# ---------------------------------------------------------------------------
# IAPWS R9-97 coefficients  [3] Table 1 (= [1] Table 1)
# ---------------------------------------------------------------------------
# The a_i coefficients were determined by least-squares fit to experimental
# refractive index data for water spanning 0-60 C, 0-1100 bar, 226-1014 nm.
_A_COEFF = (
    0.244257733,     # a0  constant term
    9.74634476e-3,   # a1  density (delta) term
    -3.73234996e-3,  # a2  temperature (theta) term
    2.68678472e-4,   # a3  L^2 * theta cross-term
    1.58920570e-3,   # a4  1/L^2 (Cauchy-type) term
    2.45934259e-3,   # a5  UV resonance term
    0.900704920,     # a6  IR resonance term
    -1.66626219e-2,  # a7  density^2 (delta^2) term
)

# UV and IR resonance parameters [3] Table 1, in reduced wavelength units
# (i.e., lambda_UV / lambda* and lambda_IR / lambda*, where lambda* = 0.589 um)
_L_UV = 0.229202    # ~135 nm / 589 nm (UV electronic absorption)
_L_IR = 5.432937    # ~3200 nm / 589 nm (IR vibrational absorption)

# Reference values for reduced variables  [1]
_RHO_REF = 1000.0   # rho* (kg/m^3)
_T_REF = 273.15     # T* (K)
_L_REF = 0.589      # lambda* (um), sodium D line


def pure_water_density(T_C: float, p_dbar: float = 0.0) -> float:
    """Approximate density of pure water.

    Uses the Kell (1975) [4] 5th-order rational polynomial for atmospheric
    pressure, with a linear compressibility correction for elevated pressure.

    Parameters
    ----------
    T_C : float
        Temperature in degrees Celsius (valid 0-150).
    p_dbar : float
        Gauge sea pressure in dbar (0 = atmospheric, 1 dbar = 10^4 Pa).

    Returns
    -------
    float
        Density in kg/m^3.
        ~999.84 at 0 deg C, ~999.97 at 4 deg C (maximum), ~998.2 at 20 deg C.
    """
    T = T_C
    # [4] Eq. 7, valid 0-150 C at 1 atm
    rho_atm = (999.83952
               + 16.945176 * T
               - 7.9870401e-3 * T * T
               - 46.170461e-6 * T * T * T
               + 105.56302e-9 * T * T * T * T
               - 280.54253e-12 * T * T * T * T * T
               ) / (1.0 + 16.879850e-3 * T)
    # Pressure correction: rho(p) ~ rho_atm * (1 + kappa * dp)
    # kappa ~ 4.5e-10 Pa^-1 for liquid water, 1 dbar = 1e4 Pa
    rho = rho_atm * (1.0 + 4.5e-6 * p_dbar)
    return rho


def refractive_index(T_C: float, wavelength_nm: float,
                     rho_kg_m3: float = None,
                     p_dbar: float = 0.0) -> float:
    """Refractive index of pure water.

    Evaluates the IAPWS R9-97 [1] modified Lorentz-Lorenz formula (Eqs. 1-2).

    Parameters
    ----------
    T_C : float
        Temperature in degrees Celsius.
    wavelength_nm : float
        Vacuum wavelength in nm (valid 200-1100).
    rho_kg_m3 : float, optional
        Water density in kg/m^3.  If None, computed from T_C and p_dbar
        using the Kell (1975) approximation.
    p_dbar : float
        Gauge sea pressure in dbar (used only if rho_kg_m3 is None).

    Returns
    -------
    float
        Refractive index (dimensionless).

    Notes
    -----
    IAPWS check value: at T = 25 deg C, rho = 997.047 kg/m^3,
    lambda = 589.3 nm, the function returns n = 1.33285 +/- 0.00002.
    """
    if rho_kg_m3 is None:
        rho_kg_m3 = pure_water_density(T_C, p_dbar)

    # Reduced variables  [1]
    delta = rho_kg_m3 / _RHO_REF          # reduced density
    theta = (T_C + 273.15) / _T_REF       # reduced temperature
    L = wavelength_nm * 1.0e-3 / _L_REF   # reduced wavelength (nm -> um -> L/L*)
    L2 = L * L
    a = _A_COEFF

    # [1] Eq. 2 (modified Lorentz-Lorenz polarizability)
    A = delta * (
        a[0]
        + a[1] * delta
        + a[2] * theta
        + a[3] * L2 * theta
        + a[4] / L2
        + a[5] / (L2 - _L_UV * _L_UV)   # UV resonance
        + a[6] / (L2 - _L_IR * _L_IR)   # IR resonance
        + a[7] * delta * delta
    )

    # [1] Eq. 1
    return math.sqrt((2.0 * A + 1.0) / (1.0 - A))
