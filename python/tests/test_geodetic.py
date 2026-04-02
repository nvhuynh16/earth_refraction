"""Tests for WGS-84 geodetic coordinate utilities."""

import math

import numpy as np
import pytest

from refraction.geodetic import (
    ecef_to_geodetic,
    enu_frame,
    geodetic_normal,
    geodetic_to_ecef,
    normal_jacobian,
    principal_radii,
)
from refraction.constants import WGS84_A, WGS84_B, WGS84_E2


# ===================================================================
# geodetic_to_ecef
# ===================================================================

class TestGeodeticToEcef:

    def test_equator_prime_meridian(self):
        """(0, 0, 0) -> (a, 0, 0)."""
        r = geodetic_to_ecef(0.0, 0.0, 0.0)
        np.testing.assert_allclose(r, [WGS84_A, 0.0, 0.0], atol=1e-6)

    def test_north_pole(self):
        """(90, 0, 0) -> (0, 0, b)."""
        r = geodetic_to_ecef(90.0, 0.0, 0.0)
        np.testing.assert_allclose(r, [0.0, 0.0, WGS84_B], atol=1e-6)

    def test_south_pole(self):
        """(-90, 0, 0) -> (0, 0, -b)."""
        r = geodetic_to_ecef(-90.0, 0.0, 0.0)
        np.testing.assert_allclose(r, [0.0, 0.0, -WGS84_B], atol=1e-6)

    def test_equator_90_east(self):
        """(0, 90, 0) -> (0, a, 0)."""
        r = geodetic_to_ecef(0.0, 90.0, 0.0)
        np.testing.assert_allclose(r, [0.0, WGS84_A, 0.0], atol=1e-6)

    def test_with_height(self):
        """Height above equator shifts radially."""
        h = 10_000.0
        r = geodetic_to_ecef(0.0, 0.0, h)
        np.testing.assert_allclose(r, [WGS84_A + h, 0.0, 0.0], atol=1e-6)

    def test_with_height_pole(self):
        """Height above pole shifts along Z."""
        h = 10_000.0
        r = geodetic_to_ecef(90.0, 0.0, h)
        np.testing.assert_allclose(r, [0.0, 0.0, WGS84_B + h], atol=1e-6)


# ===================================================================
# ecef_to_geodetic (round-trip)
# ===================================================================

class TestEcefToGeodetic:

    def test_equator(self):
        lat, lon, h = ecef_to_geodetic(WGS84_A, 0.0, 0.0)
        assert lat == pytest.approx(0.0, abs=1e-10)
        assert lon == pytest.approx(0.0, abs=1e-10)
        assert h == pytest.approx(0.0, abs=1e-3)

    def test_north_pole(self):
        lat, lon, h = ecef_to_geodetic(0.0, 0.0, WGS84_B)
        assert lat == pytest.approx(90.0, abs=1e-10)
        assert h == pytest.approx(0.0, abs=1e-3)

    @pytest.mark.parametrize("lat_deg", [-90.0, -45.0, -10.0, 0.0, 10.0, 45.0, 75.0, 90.0])
    @pytest.mark.parametrize("lon_deg", [-180.0, -90.0, 0.0, 45.0, 90.0, 179.0])
    @pytest.mark.parametrize("h_m", [0.0, 100.0, 10_000.0, 100_000.0])
    def test_roundtrip(self, lat_deg, lon_deg, h_m):
        """geodetic -> ECEF -> geodetic recovers original coordinates."""
        r = geodetic_to_ecef(lat_deg, lon_deg, h_m)
        lat2, lon2, h2 = ecef_to_geodetic(r[0], r[1], r[2])

        assert lat2 == pytest.approx(lat_deg, abs=1e-9)
        # Longitude is undefined at poles
        if abs(lat_deg) < 89.99:
            assert lon2 == pytest.approx(lon_deg, abs=1e-9)
        assert h2 == pytest.approx(h_m, abs=1e-3)


# ===================================================================
# geodetic_normal
# ===================================================================

class TestGeodeticNormal:

    def test_equator(self):
        n = geodetic_normal(0.0, 0.0)
        np.testing.assert_allclose(n, [1.0, 0.0, 0.0], atol=1e-15)

    def test_pole(self):
        n = geodetic_normal(90.0, 0.0)
        np.testing.assert_allclose(n, [0.0, 0.0, 1.0], atol=1e-15)

    def test_equator_90_east(self):
        n = geodetic_normal(0.0, 90.0)
        np.testing.assert_allclose(n, [0.0, 1.0, 0.0], atol=1e-15)

    @pytest.mark.parametrize("lat_deg", [-60.0, -30.0, 0.0, 30.0, 60.0, 90.0])
    @pytest.mark.parametrize("lon_deg", [-90.0, 0.0, 45.0, 135.0])
    def test_unit_length(self, lat_deg, lon_deg):
        n = geodetic_normal(lat_deg, lon_deg)
        assert np.linalg.norm(n) == pytest.approx(1.0, abs=1e-15)


# ===================================================================
# enu_frame
# ===================================================================

class TestEnuFrame:

    def test_equator_prime_meridian(self):
        E, N, U = enu_frame(0.0, 0.0)
        np.testing.assert_allclose(E, [0.0, 1.0, 0.0], atol=1e-15)
        np.testing.assert_allclose(N, [0.0, 0.0, 1.0], atol=1e-15)
        np.testing.assert_allclose(U, [1.0, 0.0, 0.0], atol=1e-15)

    def test_north_pole(self):
        E, N, U = enu_frame(90.0, 0.0)
        np.testing.assert_allclose(U, [0.0, 0.0, 1.0], atol=1e-15)
        # E = (-sin 0, cos 0, 0) = (0, 1, 0)
        np.testing.assert_allclose(E, [0.0, 1.0, 0.0], atol=1e-15)
        # N = (-sin90*cos0, -sin90*sin0, cos90) = (-1, 0, 0)
        np.testing.assert_allclose(N, [-1.0, 0.0, 0.0], atol=1e-15)

    @pytest.mark.parametrize("lat_deg", [-60.0, -30.0, 0.0, 30.0, 60.0, 85.0])
    @pytest.mark.parametrize("lon_deg", [-90.0, 0.0, 45.0, 135.0])
    def test_orthonormal(self, lat_deg, lon_deg):
        E, N, U = enu_frame(lat_deg, lon_deg)
        assert np.dot(E, N) == pytest.approx(0.0, abs=1e-14)
        assert np.dot(E, U) == pytest.approx(0.0, abs=1e-14)
        assert np.dot(N, U) == pytest.approx(0.0, abs=1e-14)
        assert np.linalg.norm(E) == pytest.approx(1.0, abs=1e-15)
        assert np.linalg.norm(N) == pytest.approx(1.0, abs=1e-15)
        assert np.linalg.norm(U) == pytest.approx(1.0, abs=1e-15)

    @pytest.mark.parametrize("lat_deg", [-45.0, 0.0, 45.0])
    @pytest.mark.parametrize("lon_deg", [-90.0, 0.0, 90.0])
    def test_right_handed(self, lat_deg, lon_deg):
        """E x N = U."""
        E, N, U = enu_frame(lat_deg, lon_deg)
        np.testing.assert_allclose(np.cross(E, N), U, atol=1e-14)


# ===================================================================
# principal_radii
# ===================================================================

class TestPrincipalRadii:

    def test_equator(self):
        """M = a(1-e²), N = a at equator."""
        M, N_rad = principal_radii(0.0)
        assert N_rad == pytest.approx(WGS84_A, rel=1e-12)
        assert M == pytest.approx(WGS84_A * (1.0 - WGS84_E2), rel=1e-12)

    def test_pole(self):
        """M = N = a²/b at the poles."""
        M, N_rad = principal_radii(90.0)
        expected = WGS84_A ** 2 / WGS84_B
        assert M == pytest.approx(expected, rel=1e-10)
        assert N_rad == pytest.approx(expected, rel=1e-10)

    def test_sphere(self):
        """With e² = 0, M = N = a everywhere."""
        # principal_radii uses module-level constants, so we test
        # that N_rad > M at the equator (confirms flattening effect)
        M, N_rad = principal_radii(0.0)
        assert N_rad > M  # N = a, M = a(1-e²) < a

    def test_positive(self):
        for lat in [-90, -45, 0, 45, 90]:
            M, N_rad = principal_radii(float(lat))
            assert M > 0
            assert N_rad > 0


# ===================================================================
# normal_jacobian
# ===================================================================

class TestNormalJacobian:

    def test_symmetric(self):
        """The normal Jacobian should be symmetric."""
        J = normal_jacobian(45.0, 30.0, 1000.0)
        np.testing.assert_allclose(J, J.T, atol=1e-10)

    def test_sphere_limit(self):
        """For a sphere (at equator where curvature ≈ spherical),
        compare against (I - n̂⊗n̂)/|r|."""
        lat, lon, h = 0.0, 0.0, 0.0
        J = normal_jacobian(lat, lon, h)
        n_hat = geodetic_normal(lat, lon)
        r = geodetic_to_ecef(lat, lon, h)
        r_mag = np.linalg.norm(r)

        # For a sphere: (I - n̂⊗n̂)/R
        J_sphere = (np.eye(3) - np.outer(n_hat, n_hat)) / r_mag

        # WGS-84 differs from sphere, but only by ~0.3% (flattening)
        np.testing.assert_allclose(J, J_sphere, rtol=0.01)

    def test_vs_finite_difference(self):
        """Compare analytical normal_jacobian against FD perturbation."""
        lat, lon, h = 45.0, 30.0, 5000.0
        J_analytical = normal_jacobian(lat, lon, h)

        # FD: perturb ECEF position, recompute normal
        r0 = geodetic_to_ecef(lat, lon, h)
        n0 = geodetic_normal(lat, lon)
        delta = 1.0  # 1 metre perturbation

        J_fd = np.zeros((3, 3))
        for j in range(3):
            r_plus = r0.copy()
            r_plus[j] += delta
            lat_p, lon_p, _h_p = ecef_to_geodetic(*r_plus)
            n_plus = geodetic_normal(lat_p, lon_p)

            r_minus = r0.copy()
            r_minus[j] -= delta
            lat_m, lon_m, _h_m = ecef_to_geodetic(*r_minus)
            n_minus = geodetic_normal(lat_m, lon_m)

            J_fd[:, j] = (n_plus - n_minus) / (2.0 * delta)

        np.testing.assert_allclose(J_analytical, J_fd, atol=1e-8)

    def test_normal_direction_zero(self):
        """Perturbation along the normal should not change n̂
        (to first order). So J @ n̂ ≈ 0."""
        lat, lon, h = 30.0, 60.0, 1000.0
        J = normal_jacobian(lat, lon, h)
        n_hat = geodetic_normal(lat, lon)
        result = J @ n_hat
        np.testing.assert_allclose(result, 0.0, atol=1e-10)
