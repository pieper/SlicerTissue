"""FEBio Tier 1 validation tests: analytical constitutive-model benchmarks.

These tests verify the Neo-Hookean stress formula used in the MPM simulator
against closed-form analytical solutions drawn from the FEBio verification
suite (Maas et al. 2012, J. Biomech. Eng.).

All tests are pure-Python (numpy only, no GPU/Warp) and run in <1 s each.
They validate the constitutive law in isolation — not the MPM grid dynamics.

FEBio Neo-Hookean (coupled logarithmic form, Simo & Hughes 1998):
    W    = (mu/2)(I1 - 3) - mu*ln(J) + (lam/2)(ln J)^2
    tau  = mu*(B - I) + lam*ln(J)*I          (Kirchhoff stress, tau = J*sigma)
    sigma = tau / J                            (Cauchy stress)

This matches exactly the _p2g kernel in mpm.py:
    tau = mu * (FFt - I3) + lam * log(J_safe) * I3

FEBio HGO fiber model (Holzapfel-Gasser-Ogden, uncoupled isochoric part):
    W_f  = k1/(2*k2) * [exp(k2 * <I4-1>^2) - 1]
    S_f  = 2*k1*(I4-1)*exp(k2*(I4-1)^2) * a0 otimes a0   (2nd PK)
    sigma_f = (1/J) * F * S_f * F^T

where I4 = a0 . (C . a0) = |F a0|^2 is the squared fiber stretch,
<.> = Macaulay bracket = max(., 0) (fibers buckle under compression).
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Constitutive formulae (mirror mpm.py _p2g and AnisotropicMaterial)
# ---------------------------------------------------------------------------

def nh_kirchhoff(F: np.ndarray, mu: float, lam: float) -> np.ndarray:
    """Neo-Hookean Kirchhoff stress tau = J * sigma.

    tau = mu*(B - I) + lam*ln(J)*I
    """
    J = np.linalg.det(F)
    B = F @ F.T
    I = np.eye(3)
    return mu * (B - I) + lam * np.log(max(J, 1e-10)) * I


def nh_cauchy(F: np.ndarray, mu: float, lam: float) -> np.ndarray:
    """Neo-Hookean Cauchy stress sigma = tau / J."""
    J = np.linalg.det(F)
    return nh_kirchhoff(F, mu, lam) / max(J, 1e-10)


def hgo_fiber_cauchy(
    F: np.ndarray,
    mu: float,
    lam: float,
    k1: float,
    k2: float,
    a0: np.ndarray,
) -> np.ndarray:
    """Full Cauchy stress for Neo-Hookean + single HGO fiber family.

    sigma = sigma_iso + sigma_fiber
    sigma_iso = (mu/J)*(B - I) + (lam/J)*ln(J)*I
    sigma_fiber = (1/J) * F * S_fiber * F^T
    S_fiber = 2*k1*(I4-1)*exp(k2*(I4-1)^2) * a0 x a0   [only if I4 > 1]
    """
    J = np.linalg.det(F)
    B = F @ F.T
    I = np.eye(3)
    sigma_iso = (mu * (B - I) + lam * np.log(max(J, 1e-10)) * I) / max(J, 1e-10)

    # Fiber invariant I4 = |F a0|^2
    Fa0 = F @ a0
    I4 = float(Fa0 @ Fa0)

    I4m1 = max(I4 - 1.0, 0.0)   # Macaulay bracket
    if I4m1 == 0.0:
        return sigma_iso          # fibers buckled — no fiber stress

    S_scalar = 2.0 * k1 * I4m1 * np.exp(k2 * I4m1**2)
    S_fiber = S_scalar * np.outer(a0, a0)
    sigma_fiber = (F @ S_fiber @ F.T) / max(J, 1e-10)
    return sigma_iso + sigma_fiber


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniaxial_lateral_stretch(
    lam_axial: float, mu: float, lam_param: float
) -> float:
    """Newton solve for lateral stretch lambda_T s.t. sigma_22 = 0.

    For F = diag(lam_axial, lam_T, lam_T) the lateral Kirchhoff stress is:
        tau_22 = mu*(lam_T^2 - 1) + lam_param * ln(lam_axial * lam_T^2) = 0

    Uses Newton's method with linear-elastic initial guess.
    """
    # Initial guess from linear Poisson's ratio
    nu_approx = lam_param / (2.0 * (lam_param + mu))
    lam_T = max(1.0 - nu_approx * (lam_axial - 1.0), 1e-4)

    for _ in range(60):
        J = lam_axial * lam_T**2
        f  = mu * (lam_T**2 - 1.0) + lam_param * np.log(J)
        df = 2.0 * mu * lam_T + 2.0 * lam_param / lam_T
        delta = -f / df
        lam_T += delta
        lam_T = max(lam_T, 1e-8)
        if abs(delta) < 1e-12:
            break
    return lam_T


def _biaxial_transverse_stretch(
    lam_b: float, mu: float, lam_param: float
) -> float:
    """Newton solve for out-of-plane stretch lambda_Z s.t. sigma_33 = 0.

    For F = diag(lam_b, lam_b, lam_Z) the out-of-plane Kirchhoff stress is:
        tau_33 = mu*(lam_Z^2 - 1) + lam_param * ln(lam_b^2 * lam_Z) = 0
    """
    lam_Z = 1.0 / lam_b  # initial guess: approximate incompressibility
    lam_Z = max(lam_Z, 1e-4)

    for _ in range(60):
        J = lam_b**2 * lam_Z
        f  = mu * (lam_Z**2 - 1.0) + lam_param * np.log(J)
        df = 2.0 * mu * lam_Z + lam_param / lam_Z
        delta = -f / df
        lam_Z += delta
        lam_Z = max(lam_Z, 1e-8)
        if abs(delta) < 1e-12:
            break
    return lam_Z


def _lame(E: float, nu: float) -> tuple[float, float]:
    mu  = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


# ---------------------------------------------------------------------------
# Test 1 — Uniaxial tension (FEBio Problem 1, tension variant)
# ---------------------------------------------------------------------------

class TestUniaxialTension:
    """Neo-Hookean uniaxial tension — FEBio Problem 1 (tension).

    Setup:
      F = diag(lambda, lambda_T, lambda_T), lambda_T from sigma_22 = 0.
      Reference: Maas et al. 2012, Fig. 3.
    """

    E  = 1_000.0  # Pa  (representative kPa-scale soft tissue)
    nu = 0.3

    @property
    def mu(self):  return self.E / (2.0 * (1.0 + self.nu))

    @property
    def lam(self): return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

    def _sigma11(self, lam_axial: float) -> float:
        lam_T = _uniaxial_lateral_stretch(lam_axial, self.mu, self.lam)
        F = np.diag([lam_axial, lam_T, lam_T])
        return float(nh_cauchy(F, self.mu, self.lam)[0, 0])

    def test_small_strain_hookes_law(self):
        """At small strain, sigma_11 → E * epsilon (linear elasticity limit)."""
        eps = 0.001
        sigma = self._sigma11(1.0 + eps)
        sigma_linear = self.E * eps
        rel_err = abs(sigma - sigma_linear) / sigma_linear
        assert rel_err < 0.01, (
            f"sigma={sigma:.4f} Pa, Hooke={sigma_linear:.4f} Pa, rel_err={rel_err:.4f}"
        )

    def test_lateral_stress_free(self):
        """Lateral Cauchy stress must be zero (uniaxial stress state)."""
        for lam in [1.1, 1.5, 2.0, 3.0]:
            lam_T = _uniaxial_lateral_stretch(lam, self.mu, self.lam)
            F = np.diag([lam, lam_T, lam_T])
            sigma = nh_cauchy(F, self.mu, self.lam)
            assert abs(sigma[1, 1]) < 1e-6, (
                f"sigma_22={sigma[1,1]:.2e} Pa at lambda={lam} (should be 0)"
            )
            assert abs(sigma[2, 2]) < 1e-6, (
                f"sigma_33={sigma[2,2]:.2e} Pa at lambda={lam} (should be 0)"
            )

    def test_monotonic_stress_stretch(self):
        """sigma_11 must be strictly increasing with stretch."""
        stretches = np.linspace(1.01, 4.0, 30)
        stresses  = [self._sigma11(l) for l in stretches]
        diffs     = np.diff(stresses)
        assert np.all(diffs > 0), (
            "sigma_11 not monotonically increasing; "
            f"first non-monotonic at lambda~{stretches[np.argmax(diffs <= 0)]:.3f}"
        )

    def test_kirchhoff_formula_matches_cauchy(self):
        """tau / J must equal sigma computed directly."""
        for lam in [1.1, 2.0, 3.0]:
            lam_T = _uniaxial_lateral_stretch(lam, self.mu, self.lam)
            F = np.diag([lam, lam_T, lam_T])
            J = np.linalg.det(F)
            tau   = nh_kirchhoff(F, self.mu, self.lam)
            sigma = nh_cauchy(F, self.mu, self.lam)
            np.testing.assert_allclose(tau / J, sigma, rtol=1e-10,
                                       err_msg=f"tau/J != sigma at lambda={lam}")

    def test_stress_table(self, capsys):
        """Print sigma vs lambda table comparable to FEBio Problem 1 output."""
        print(
            "\n\nNeo-Hookean Uniaxial Tension — FEBio Problem 1 equivalent\n"
            f"  E={self.E:.0f} Pa, nu={self.nu:.2f}, "
            f"mu={self.mu:.2f} Pa, lam={self.lam:.2f} Pa\n"
            f"  {'lambda':>8}  {'sigma_11 [Pa]':>14}  {'lambda_T':>10}  {'J':>8}"
        )
        for lam in [1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 4.0]:
            lam_T = _uniaxial_lateral_stretch(lam, self.mu, self.lam)
            J     = lam * lam_T**2
            s11   = self._sigma11(lam)
            print(f"  {lam:>8.3f}  {s11:>14.4f}  {lam_T:>10.6f}  {J:>8.5f}")


# ---------------------------------------------------------------------------
# Test 2 — Uniaxial compression (FEBio Problem 1, compression variant)
# ---------------------------------------------------------------------------

class TestUniaxialCompression:
    """Neo-Hookean uniaxial compression — FEBio Problem 1 (compression).

    Same setup as tension but lambda < 1.
    """

    E  = 1_000.0
    nu = 0.3

    @property
    def mu(self):  return self.E / (2.0 * (1.0 + self.nu))

    @property
    def lam(self): return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

    def test_negative_axial_stress(self):
        """Compressive stretch (lambda < 1) must give sigma_11 < 0."""
        for lam in [0.9, 0.7, 0.5]:
            lam_T = _uniaxial_lateral_stretch(lam, self.mu, self.lam)
            F = np.diag([lam, lam_T, lam_T])
            sigma = nh_cauchy(F, self.mu, self.lam)
            assert sigma[0, 0] < 0, (
                f"sigma_11={sigma[0,0]:.4f} Pa at lambda={lam}, expected < 0"
            )

    def test_lateral_expansion(self):
        """Under axial compression, lateral stretch lambda_T must be > 1."""
        for lam in [0.9, 0.7, 0.5]:
            lam_T = _uniaxial_lateral_stretch(lam, self.mu, self.lam)
            assert lam_T > 1.0, (
                f"lam_T={lam_T:.4f} at lambda={lam}, expected lateral expansion"
            )

    def test_small_compression_hookes_law(self):
        """Small compressive strain recovers sigma_11 = E * epsilon."""
        eps = -0.001
        lam = 1.0 + eps
        lam_T = _uniaxial_lateral_stretch(lam, self.mu, self.lam)
        F = np.diag([lam, lam_T, lam_T])
        sigma = nh_cauchy(F, self.mu, self.lam)
        sigma_linear = self.E * eps
        rel_err = abs(sigma[0, 0] - sigma_linear) / abs(sigma_linear)
        assert rel_err < 0.01, (
            f"sigma_11={sigma[0,0]:.6f}, Hooke={sigma_linear:.6f}, "
            f"rel_err={rel_err:.4f}"
        )

    def test_lateral_stress_free(self):
        """Lateral Cauchy stress must remain zero in uniaxial compression."""
        for lam in [0.9, 0.75, 0.5]:
            lam_T = _uniaxial_lateral_stretch(lam, self.mu, self.lam)
            F = np.diag([lam, lam_T, lam_T])
            sigma = nh_cauchy(F, self.mu, self.lam)
            assert abs(sigma[1, 1]) < 1e-6, (
                f"sigma_22={sigma[1,1]:.2e} Pa at lambda={lam}"
            )


# ---------------------------------------------------------------------------
# Test 3 — Simple shear
# ---------------------------------------------------------------------------

class TestSimpleShear:
    """Neo-Hookean simple shear — analytical solution.

    F = [[1, gamma, 0], [0, 1, 0], [0, 0, 1]], det F = 1 (isochoric).

    Exact analytical results (J = 1, ln J = 0):
        sigma_12 = mu * gamma              (shear stress, linear in gamma)
        sigma_11 = mu * gamma^2            (Poynting normal stress effect)
        sigma_22 = 0
        sigma_33 = 0

    The Poynting/normal-stress effect (sigma_11 ≠ 0) is a hallmark of
    nonlinear hyperelastic behavior — absent in linear elasticity.
    FEBio uses this mode to validate hyperelastic material implementations.
    """

    mu  = 500.0   # Pa  (shear modulus)
    lam = 1000.0  # Pa  (second Lamé parameter)

    @staticmethod
    def _F(gamma: float) -> np.ndarray:
        return np.array([[1.0, gamma, 0.0],
                         [0.0, 1.0,  0.0],
                         [0.0, 0.0,  1.0]])

    def test_isochoric(self):
        """Simple shear is volume-preserving: det F = 1 for all gamma."""
        for gamma in [0.1, 0.5, 1.0, 2.0, 5.0]:
            assert abs(np.linalg.det(self._F(gamma)) - 1.0) < 1e-14, \
                f"det F != 1 at gamma={gamma}"

    def test_shear_stress_linear(self):
        """sigma_12 = mu * gamma (exact, all gamma)."""
        for gamma in [0.1, 0.5, 1.0, 2.0]:
            sigma = nh_cauchy(self._F(gamma), self.mu, self.lam)
            expected = self.mu * gamma
            np.testing.assert_allclose(
                sigma[0, 1], expected, rtol=1e-10,
                err_msg=f"sigma_12 != mu*gamma at gamma={gamma}"
            )

    def test_poynting_normal_stress(self):
        """sigma_11 = mu * gamma^2 (Poynting effect, nonlinear signature)."""
        for gamma in [0.5, 1.0, 2.0]:
            sigma = nh_cauchy(self._F(gamma), self.mu, self.lam)
            expected = self.mu * gamma**2
            np.testing.assert_allclose(
                sigma[0, 0], expected, rtol=1e-10,
                err_msg=f"sigma_11 != mu*gamma^2 at gamma={gamma}"
            )

    def test_sigma22_zero(self):
        """sigma_22 = 0 for all gamma (exact result)."""
        for gamma in [0.1, 0.5, 1.0, 2.0]:
            sigma = nh_cauchy(self._F(gamma), self.mu, self.lam)
            assert abs(sigma[1, 1]) < 1e-12, \
                f"sigma_22={sigma[1,1]:.2e} at gamma={gamma}, expected 0"

    def test_sigma33_zero(self):
        """sigma_33 = 0 for all gamma (exact result)."""
        for gamma in [0.1, 0.5, 1.0, 2.0]:
            sigma = nh_cauchy(self._F(gamma), self.mu, self.lam)
            assert abs(sigma[2, 2]) < 1e-12, \
                f"sigma_33={sigma[2,2]:.2e} at gamma={gamma}, expected 0"

    def test_symmetry_sigma12_equals_sigma21(self):
        """Cauchy stress must be symmetric."""
        for gamma in [0.1, 1.0, 2.0]:
            sigma = nh_cauchy(self._F(gamma), self.mu, self.lam)
            np.testing.assert_allclose(sigma, sigma.T, atol=1e-12,
                                       err_msg=f"Cauchy stress not symmetric at gamma={gamma}")

    def test_small_gamma_linear_limit(self):
        """At small gamma, Poynting term mu*gamma^2 is negligible vs shear."""
        gamma = 1e-4
        sigma = nh_cauchy(self._F(gamma), self.mu, self.lam)
        ratio = abs(sigma[0, 0]) / abs(sigma[0, 1])   # normal / shear
        assert ratio < gamma, (
            f"Poynting/shear ratio = {ratio:.2e} at gamma={gamma:.0e}, "
            f"expected < {gamma:.0e} in linear limit"
        )

    def test_stress_table(self, capsys):
        """Print shear-stress table for comparison with FEBio output."""
        print(
            f"\n\nNeo-Hookean Simple Shear  (mu={self.mu:.0f} Pa)\n"
            f"  {'gamma':>8}  {'sigma_12 [Pa]':>14}  {'sigma_11 [Pa]':>14}  "
            f"{'sigma_22 [Pa]':>14}"
        )
        for gamma in [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]:
            s = nh_cauchy(self._F(gamma), self.mu, self.lam)
            print(f"  {gamma:>8.3f}  {s[0,1]:>14.4f}  {s[0,0]:>14.4f}  {s[1,1]:>14.4f}")


# ---------------------------------------------------------------------------
# Test 4 — Equibiaxial stretch
# ---------------------------------------------------------------------------

class TestEquibiaxialStretch:
    """Neo-Hookean equibiaxial stretch — analytical solution.

    F = diag(lambda, lambda, lambda_Z), lambda_Z from sigma_33 = 0.
    By symmetry: sigma_11 = sigma_22 at all lambda.
    In biaxial tension (lambda > 1): lambda_Z < 1 (thinning).

    At the same axial stretch, equibiaxial sigma_11 must exceed uniaxial
    sigma_11 because both lateral directions are constrained simultaneously.
    """

    E  = 1_000.0
    nu = 0.3

    @property
    def mu(self):  return self.E / (2.0 * (1.0 + self.nu))

    @property
    def lam(self): return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

    def test_sigma11_equals_sigma22(self):
        """By symmetry, sigma_11 must equal sigma_22."""
        for lam_b in [1.1, 1.5, 2.0, 3.0]:
            lam_Z = _biaxial_transverse_stretch(lam_b, self.mu, self.lam)
            F = np.diag([lam_b, lam_b, lam_Z])
            sigma = nh_cauchy(F, self.mu, self.lam)
            np.testing.assert_allclose(
                sigma[0, 0], sigma[1, 1], rtol=1e-10,
                err_msg=f"sigma_11 != sigma_22 at lambda={lam_b}"
            )

    def test_sigma33_zero(self):
        """Out-of-plane stress must be zero (plane-stress condition)."""
        for lam_b in [1.1, 1.5, 2.0, 3.0]:
            lam_Z = _biaxial_transverse_stretch(lam_b, self.mu, self.lam)
            F = np.diag([lam_b, lam_b, lam_Z])
            sigma = nh_cauchy(F, self.mu, self.lam)
            assert abs(sigma[2, 2]) < 1e-6, (
                f"sigma_33={sigma[2,2]:.2e} Pa at lambda={lam_b}"
            )

    def test_thinning_in_biaxial_tension(self):
        """Biaxial tension must produce out-of-plane thinning (lambda_Z < 1)."""
        for lam_b in [1.1, 1.5, 2.0]:
            lam_Z = _biaxial_transverse_stretch(lam_b, self.mu, self.lam)
            assert lam_Z < 1.0, (
                f"lam_Z={lam_Z:.4f} at lambda={lam_b}, expected < 1"
            )

    def test_biaxial_stiffer_than_uniaxial_near_incompressible(self):
        """For near-incompressible material (nu=0.49), equibiaxial sigma_11 > uniaxial.

        This holds because both lateral directions are simultaneously stretched,
        creating a larger hydrostatic pressure.  For compressible materials (small nu)
        the relationship can reverse, so we use nu=0.49 (soft tissue regime).
        """
        E_soft, nu_soft = 1_000.0, 0.49
        mu_s, lam_s = _lame(E_soft, nu_soft)
        for lam in [1.2, 1.5, 2.0]:
            # Biaxial
            lam_Z = _biaxial_transverse_stretch(lam, mu_s, lam_s)
            sigma_b = float(nh_cauchy(np.diag([lam, lam, lam_Z]), mu_s, lam_s)[0, 0])
            # Uniaxial
            lam_T = _uniaxial_lateral_stretch(lam, mu_s, lam_s)
            sigma_u = float(nh_cauchy(np.diag([lam, lam_T, lam_T]), mu_s, lam_s)[0, 0])
            assert sigma_b > sigma_u, (
                f"Equibiaxial sigma={sigma_b:.4f} Pa not > uniaxial sigma={sigma_u:.4f} Pa "
                f"at lambda={lam} (nu={nu_soft})"
            )

    def test_small_strain_hookes_law_biaxial(self):
        """Small equibiaxial strain: sigma_11 → E/(1-nu) * epsilon (plane-stress biaxial)."""
        eps = 0.001
        lam_b = 1.0 + eps
        lam_Z = _biaxial_transverse_stretch(lam_b, self.mu, self.lam)
        F = np.diag([lam_b, lam_b, lam_Z])
        sigma = nh_cauchy(F, self.mu, self.lam)
        sigma_linear = self.E / (1.0 - self.nu) * eps   # plane-stress biaxial
        rel_err = abs(sigma[0, 0] - sigma_linear) / sigma_linear
        assert rel_err < 0.01, (
            f"sigma_11={sigma[0,0]:.4f}, plane-stress Hooke={sigma_linear:.4f}, "
            f"rel_err={rel_err:.4f}"
        )

    def test_stress_table(self, capsys):
        """Print sigma vs lambda table for comparison with FEBio output."""
        print(
            f"\n\nNeo-Hookean Equibiaxial Stretch  "
            f"(E={self.E:.0f} Pa, nu={self.nu:.2f})\n"
            f"  {'lambda_b':>10}  {'sigma_11 [Pa]':>14}  {'lambda_Z':>10}  {'J':>8}"
        )
        for lam_b in [1.0, 1.1, 1.25, 1.5, 2.0, 3.0]:
            lam_Z = _biaxial_transverse_stretch(lam_b, self.mu, self.lam)
            J = lam_b**2 * lam_Z
            sigma = nh_cauchy(np.diag([lam_b, lam_b, lam_Z]), self.mu, self.lam)
            print(f"  {lam_b:>10.3f}  {sigma[0,0]:>14.4f}  {lam_Z:>10.6f}  {J:>8.5f}")


# ---------------------------------------------------------------------------
# Test 5 — HGO fiber model: collagen crimp and activation
# ---------------------------------------------------------------------------

class TestHGOFiberTension:
    """Holzapfel-Gasser-Ogden (HGO) fiber model — analytical benchmark.

    Tests the AnisotropicMaterial / hgo_fiber_cauchy() formula against
    closed-form results for uniaxial tension aligned with the fiber axis.

    Setup:
      - Fiber direction a0 = [1, 0, 0]
      - F = diag(lambda, lambda_T, lambda_T), lambda_T from sigma_22=0
      - I4 = lambda^2 (squared fiber stretch along fiber axis)

    Fiber Cauchy stress (along a0):
      sigma_11_fiber = k1*(I4-1)*exp(k2*(I4-1)^2) * lambda^2 / J

    Collagen crimp (MPMMaterial model):
      Fiber activates when engineering strain along fiber > collagen_crimp
      i.e., (lambda - 1) > crimp_threshold
      Equivalent in HGO terms: I4 > (1 + crimp)^2
    """

    # Matrix (isotropic Neo-Hookean background)
    E_matrix = 1_000.0   # Pa
    nu_matrix = 0.3

    # HGO fiber parameters (Holzapfel 2000 Table 1 order-of-magnitude)
    k1 = 500.0    # Pa   fiber stiffness
    k2 = 1.0      # [-]  fiber nonlinearity (dimensionless)

    a0 = np.array([1.0, 0.0, 0.0])   # fiber along x-axis

    @property
    def mu(self):  return self.E_matrix / (2.0 * (1.0 + self.nu_matrix))

    @property
    def lam(self): return (self.E_matrix * self.nu_matrix /
                           ((1.0 + self.nu_matrix) * (1.0 - 2.0 * self.nu_matrix)))

    def _sigma_total(self, lam_axial: float) -> np.ndarray:
        """Full Cauchy stress for fiber-aligned uniaxial tension."""
        lam_T = _uniaxial_lateral_stretch(lam_axial, self.mu, self.lam)
        F = np.diag([lam_axial, lam_T, lam_T])
        return hgo_fiber_cauchy(F, self.mu, self.lam, self.k1, self.k2, self.a0)

    def _sigma_fiber_analytical(self, lam_axial: float) -> float:
        """Analytical fiber contribution to sigma_11 at lambda."""
        lam_T = _uniaxial_lateral_stretch(lam_axial, self.mu, self.lam)
        J = lam_axial * lam_T**2
        I4 = lam_axial**2
        I4m1 = max(I4 - 1.0, 0.0)
        # sigma_11_fiber = (1/J) * F S_fiber F^T [0,0]
        # = (1/J) * 2*k1*I4m1*exp(k2*I4m1^2) * (F a0)^2[0]
        # = (1/J) * 2*k1*I4m1*exp(k2*I4m1^2) * lam_axial^2
        return (2.0 * self.k1 * I4m1 * np.exp(self.k2 * I4m1**2)
                * lam_axial**2 / J)

    def test_no_fiber_stress_before_activation(self):
        """For lambda ≤ 1 (compression along fiber), fiber contributes no stress."""
        for lam in [0.5, 0.8, 0.99]:
            lam_T = _uniaxial_lateral_stretch(lam, self.mu, self.lam)
            F = np.diag([lam, lam_T, lam_T])
            sigma = hgo_fiber_cauchy(F, self.mu, self.lam, self.k1, self.k2, self.a0)
            sigma_iso = nh_cauchy(F, self.mu, self.lam)
            np.testing.assert_allclose(
                sigma, sigma_iso, atol=1e-12,
                err_msg=f"Fiber stress present at lambda={lam} (compressed fiber)"
            )

    def test_fiber_activates_in_tension(self):
        """At lambda > 1, total stress must exceed isotropic Neo-Hookean alone."""
        for lam in [1.1, 1.5, 2.0]:
            lam_T = _uniaxial_lateral_stretch(lam, self.mu, self.lam)
            F = np.diag([lam, lam_T, lam_T])
            sigma_total = hgo_fiber_cauchy(F, self.mu, self.lam,
                                           self.k1, self.k2, self.a0)
            sigma_iso = nh_cauchy(F, self.mu, self.lam)
            assert sigma_total[0, 0] > sigma_iso[0, 0], (
                f"sigma_total={sigma_total[0,0]:.4f} Pa not > "
                f"sigma_iso={sigma_iso[0,0]:.4f} Pa at lambda={lam}"
            )

    def test_fiber_stress_formula_matches_analytical(self):
        """Numerical hgo_fiber_cauchy must match closed-form sigma_11_fiber."""
        for lam in [1.1, 1.5, 2.0, 3.0]:
            lam_T = _uniaxial_lateral_stretch(lam, self.mu, self.lam)
            F = np.diag([lam, lam_T, lam_T])
            sigma_total = hgo_fiber_cauchy(F, self.mu, self.lam,
                                           self.k1, self.k2, self.a0)
            sigma_iso   = nh_cauchy(F, self.mu, self.lam)

            # Difference = fiber contribution
            sigma_fiber_11 = sigma_total[0, 0] - sigma_iso[0, 0]
            expected = self._sigma_fiber_analytical(lam)
            np.testing.assert_allclose(
                sigma_fiber_11, expected, rtol=1e-6,
                err_msg=f"fiber sigma_11 mismatch at lambda={lam}"
            )

    def test_fiber_stress_exponential_stiffening(self):
        """HGO fiber stress must increase super-linearly (toe region → stiff)."""
        lambdas = np.linspace(1.1, 2.5, 20)
        fiber_stresses = []
        for lam in lambdas:
            lam_T = _uniaxial_lateral_stretch(lam, self.mu, self.lam)
            F = np.diag([lam, lam_T, lam_T])
            sig_tot = hgo_fiber_cauchy(F, self.mu, self.lam, self.k1, self.k2, self.a0)
            sig_iso = nh_cauchy(F, self.mu, self.lam)
            fiber_stresses.append(sig_tot[0, 0] - sig_iso[0, 0])

        # Tangent (d sigma_fiber / d lambda) must be increasing — super-linear
        tangents = np.diff(fiber_stresses) / np.diff(lambdas)
        assert np.all(tangents > 0), "HGO fiber stress must be monotonically increasing"
        assert tangents[-1] > tangents[0], (
            "HGO fiber tangent stiffness must increase (exponential stiffening)"
        )

    def test_off_axis_fiber_no_effect_on_transverse_tension(self):
        """Fiber aligned with x — transverse tension (F_22 > 1) should NOT activate fiber.

        I4 = lambda_x^2 = 1 (fiber not stretched in y-direction pull),
        so fiber stress contribution must be zero.
        """
        # Pure y-direction stretch: F = diag(lam_x, lam_y, lam_z)
        # lam_y > 1, solve for lam_x = lam_z from sigma_11 = sigma_33 = 0
        # For simplicity: just use F with lam_y=1.5, lam_x=lam_z=1 (biaxial free)
        lam_y = 1.5
        # Fiber axis is x; I4 = (F a0).(F a0) = F[0,0]^2 = 1 when F[0,0]=1
        F = np.diag([1.0, lam_y, 1.0 / lam_y])   # volume-preserving, fiber unchanged
        sigma_total = hgo_fiber_cauchy(F, self.mu, self.lam, self.k1, self.k2, self.a0)
        sigma_iso   = nh_cauchy(F, self.mu, self.lam)
        np.testing.assert_allclose(
            sigma_total, sigma_iso, atol=1e-12,
            err_msg="Fiber incorrectly activated by transverse stretch"
        )

    def test_collagen_crimp_threshold(self):
        """Below crimp threshold, HGO fiber with Macaulay bracket is inactive.

        Using k2=0 (linear fiber) and comparing threshold behavior.
        Standard HGO Macaulay bracket: I4 > 1 activates fiber.
        The MPMMaterial crimp uses (lambda - 1) > crimp_threshold,
        equivalent to I4 > (1 + crimp)^2.
        Here we verify the I4-based Macaulay bracket directly.
        """
        k1_test = 1_000.0
        k2_test = 0.0001   # near-linear: exp(k2*(I4-1)^2) ≈ 1
        # lambda = 0.99: fiber compressed, I4 < 1 → inactive
        lam = 0.99
        lam_T = _uniaxial_lateral_stretch(lam, self.mu, self.lam)
        F = np.diag([lam, lam_T, lam_T])
        sigma_total = hgo_fiber_cauchy(F, self.mu, self.lam, k1_test, k2_test, self.a0)
        sigma_iso   = nh_cauchy(F, self.mu, self.lam)
        np.testing.assert_allclose(
            sigma_total, sigma_iso, atol=1e-12,
            err_msg="Fiber should be inactive at lambda=0.99 (compressed)"
        )

        # lambda = 1.01: fiber slightly in tension, I4 > 1 → active
        lam = 1.01
        lam_T = _uniaxial_lateral_stretch(lam, self.mu, self.lam)
        F = np.diag([lam, lam_T, lam_T])
        sigma_total = hgo_fiber_cauchy(F, self.mu, self.lam, k1_test, k2_test, self.a0)
        sigma_iso   = nh_cauchy(F, self.mu, self.lam)
        assert sigma_total[0, 0] > sigma_iso[0, 0], (
            "Fiber should be active at lambda=1.01 (tension)"
        )

    def test_stress_table(self, capsys):
        """Print sigma vs lambda table with fiber contribution breakdown."""
        print(
            f"\n\nHGO Fiber Tension (a0=[1,0,0])  "
            f"E_matrix={self.E_matrix:.0f} Pa, k1={self.k1:.0f} Pa, k2={self.k2:.2f}\n"
            f"  {'lambda':>8}  {'sigma_total':>13}  {'sigma_iso':>11}  "
            f"{'sigma_fiber':>13}  {'I4':>8}"
        )
        for lam in [0.9, 1.0, 1.05, 1.1, 1.25, 1.5, 2.0]:
            lam_T = _uniaxial_lateral_stretch(lam, self.mu, self.lam)
            F = np.diag([lam, lam_T, lam_T])
            s_tot = hgo_fiber_cauchy(F, self.mu, self.lam, self.k1, self.k2, self.a0)
            s_iso = nh_cauchy(F, self.mu, self.lam)
            I4 = lam**2
            print(
                f"  {lam:>8.3f}  {s_tot[0,0]:>13.4f}  {s_iso[0,0]:>11.4f}  "
                f"{s_tot[0,0]-s_iso[0,0]:>13.4f}  {I4:>8.4f}"
            )


# ---------------------------------------------------------------------------
# Test 6 — Volumetric / pressure-volume response
# ---------------------------------------------------------------------------

class TestVolumetricResponse:
    """Neo-Hookean volumetric (hydrostatic) response.

    F = lambda * I (isotropic dilation/compression), J = lambda^3.
    Cauchy stress = p * I where:
        p = (lam/J) * ln(J) + (mu/J) * (lambda^2 - 1)
    Reduces to (bulk modulus) * volumetric strain at small strain.
    """

    E  = 10_000.0
    nu = 0.45     # nearly incompressible (typical soft tissue)

    @property
    def mu(self):  return self.E / (2.0 * (1.0 + self.nu))

    @property
    def lam(self): return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

    @property
    def K(self):   return self.E / (3.0 * (1.0 - 2.0 * self.nu))  # bulk modulus

    def test_hydrostatic_stress_isotropic(self):
        """Hydrostatic deformation gives isotropic Cauchy stress (sigma = p*I)."""
        for lam_v in [0.9, 1.0, 1.1, 1.5]:
            F = lam_v * np.eye(3)
            sigma = nh_cauchy(F, self.mu, self.lam)
            # Off-diagonal must be zero
            off_diag = sigma - np.diag(np.diag(sigma))
            assert np.allclose(off_diag, 0, atol=1e-10), \
                f"sigma not isotropic at lam_v={lam_v}: {sigma}"
            # All diagonal entries equal
            diags = np.diag(sigma)
            assert np.allclose(diags, diags[0], rtol=1e-10), \
                f"sigma diagonal not uniform at lam_v={lam_v}: {diags}"

    def test_small_strain_bulk_modulus(self):
        """Volumetric pressure recovers K * e_vol at small strain."""
        eps_vol = 0.001   # 0.1% volumetric strain
        lam_v = (1.0 + eps_vol) ** (1.0 / 3.0)   # isotropic stretch for given J
        F = lam_v * np.eye(3)
        sigma = nh_cauchy(F, self.mu, self.lam)
        p_numerical = sigma[0, 0]   # isotropic stress = pressure
        p_linear    = self.K * eps_vol
        rel_err = abs(p_numerical - p_linear) / abs(p_linear)
        assert rel_err < 0.01, (
            f"p_numerical={p_numerical:.4f}, K*e_vol={p_linear:.4f}, "
            f"rel_err={rel_err:.4f}"
        )

    def test_pressure_negative_in_compression(self):
        """Hydrostatic compression (J < 1) gives compressive (negative) pressure."""
        for lam_v in [0.99, 0.9, 0.8]:
            F = lam_v * np.eye(3)
            sigma = nh_cauchy(F, self.mu, self.lam)
            assert sigma[0, 0] < 0, \
                f"sigma_11={sigma[0,0]:.4f} > 0 in hydrostatic compression at lam_v={lam_v}"

    def test_pressure_positive_in_dilation(self):
        """Hydrostatic dilation (J > 1) gives tensile (positive) pressure."""
        for lam_v in [1.01, 1.1, 1.2]:
            F = lam_v * np.eye(3)
            sigma = nh_cauchy(F, self.mu, self.lam)
            assert sigma[0, 0] > 0, \
                f"sigma_11={sigma[0,0]:.4f} < 0 in hydrostatic dilation at lam_v={lam_v}"

    def test_nearly_incompressible_large_nu(self):
        """At nu→0.5 (soft tissue), bulk modulus K >> shear modulus mu.

        For nu=0.45: K/mu = 2*(1+nu)/(3*(1-2*nu)) = 2.9/0.3 ≈ 9.7 (large but < 10).
        Use threshold of 5×mu which is comfortably exceeded.
        """
        assert self.K > 5.0 * self.mu, (
            f"K={self.K:.1f}, mu={self.mu:.1f}; expected K >> mu for nu={self.nu}"
        )
