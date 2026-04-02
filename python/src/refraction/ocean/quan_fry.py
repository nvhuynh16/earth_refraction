"""
Refractive index of seawater at optical wavelengths (atmospheric pressure).

Implements the 10-term empirical equation of Quan & Fry (1995) [1], fitted to
the Austin & Halikas (1976) [2] experimental dataset:

    n(S, T, lam) = n0 + (n1 + n2*T + n3*T^2)*S + n4*T^2
                 + (n5 + n6*S + n7*T)/lam
                 + n8/lam^2 + n9/lam^3                             [Eq. 4]

where T is temperature in deg C, S is salinity in PSU, and lam is wavelength
in nm.  The equation reproduces the Austin & Halikas data with an average
error of ~1.5e-5 (comparable to experimental uncertainty).

Valid ranges (from [2]):
    Temperature:   0-30 deg C
    Salinity:      0-35 PSU
    Wavelength:    400-700 nm (extends to 200-1100 nm per [3])
    Pressure:      atmospheric only

References
----------
.. [1] Quan, X. and E. Fry (1995). "Empirical equation for the index of
       refraction of seawater." Applied Optics, 34(18), 3477-3480.
.. [2] Austin, R. and G. Halikas (1976). "The index of refraction of seawater."
       SIO Ref. 76-1, Scripps Institution of Oceanography.
.. [3] Zhang, X. and L. Hu (2009). "Scattering by pure seawater: effect of
       salinity." Optics Express, 17(7), 5698-5710.

Parameters
----------
Temperatures in deg C, salinities in PSU, wavelengths in nm.
"""

# ---------------------------------------------------------------------------
# Coefficients from [1] Eq. 4
# ---------------------------------------------------------------------------
_N0 = 1.31405       # constant (dimensionless)
_N1 = 1.779e-4      # S coefficient (PSU^-1)
_N2 = -1.05e-6      # S*T cross-term (PSU^-1 degC^-1)
_N3 = 1.6e-8        # S*T^2 cross-term (PSU^-1 degC^-2)
_N4 = -2.02e-6      # T^2 coefficient (degC^-2)
_N5 = 15.868        # 1/lam coefficient (nm)
_N6 = 0.01155       # S/lam cross-term (PSU^-1 nm)
_N7 = -0.00423      # T/lam cross-term (degC^-1 nm)
_N8 = -4382.0       # 1/lam^2 coefficient (nm^2)
_N9 = 1.1455e6      # 1/lam^3 coefficient (nm^3)


def refractive_index(T_C: float, S_psu: float, wavelength_nm: float) -> float:
    """Refractive index of seawater at atmospheric pressure.

    Evaluates [1] Eq. 4.

    Parameters
    ----------
    T_C : float
        Temperature in degrees Celsius (valid 0-30).
    S_psu : float
        Practical salinity in PSU (valid 0-35).
    wavelength_nm : float
        Vacuum wavelength in nm (valid 400-700, extends to 200-1100).

    Returns
    -------
    float
        Refractive index (dimensionless).
        ~1.333 for pure water at 589 nm, ~1.339 for seawater (35 PSU).
    """
    T = T_C
    S = S_psu
    lam = wavelength_nm
    # [1] Eq. 4
    return (_N0
            + (_N1 + _N2 * T + _N3 * T * T) * S
            + _N4 * T * T
            + (_N5 + _N6 * S + _N7 * T) / lam
            + _N8 / (lam * lam)
            + _N9 / (lam * lam * lam))
