"""
Atmosphere sub-package: NRLMSIS 2.1 empirical atmosphere model.

Provides temperature and species number densities from the surface to
the exobase (~500 km), though this project only uses the 0-122 km range.

NRLMSIS 2.1 is an empirical model of the Earth's atmosphere based on
satellite drag, mass spectrometer, incoherent scatter radar, and rocket
data.  It provides climatological averages parameterized by date, time,
location, solar activity (F10.7), and geomagnetic activity (Ap).

The implementation is a direct Python port of the original Fortran code
(Emmert et al., 2022) using B-spline basis functions and pre-computed
parameter arrays.

References
----------
.. [1] Emmert, J.T., et al. (2022). "NRLMSIS 2.1: An Empirical Model of
       Nitric Oxide Incorporated into MSIS." Journal of Geophysical
       Research: Space Physics, 127, e2022JA030896.
"""

from .nrlmsis21 import NRLMSIS21, MSISInput, MSISOutput, MSISFullOutput, MSISProfileOutput

__all__ = ["NRLMSIS21", "MSISInput", "MSISOutput", "MSISFullOutput", "MSISProfileOutput"]
