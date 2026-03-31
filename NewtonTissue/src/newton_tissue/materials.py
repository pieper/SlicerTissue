"""Material property definitions for soft tissue FEM simulation.

Supports specification via Young's modulus + Poisson's ratio or Lamé parameters.
Internally everything is stored as Lamé parameters (mu, lambda) since that is
what Newton's TetMesh expects (k_mu, k_lambda).

Conversion formulas:
    mu  = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2*nu))
"""

from __future__ import annotations

import abc

import numpy as np


class Material(abc.ABC):
    """Abstract base class for material definitions."""

    @abc.abstractmethod
    def get_mu(self) -> float | np.ndarray:
        """First Lamé parameter (shear modulus) [Pa]."""

    @abc.abstractmethod
    def get_lambda(self) -> float | np.ndarray:
        """Second Lamé parameter [Pa]."""

    @abc.abstractmethod
    def get_density(self) -> float | np.ndarray:
        """Mass density [kg/m^3]."""

    @abc.abstractmethod
    def to_lame_arrays(
        self, num_elements: int, elements: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return per-element (k_mu, k_lambda) arrays of shape (num_elements,).

        Args:
            num_elements: Expected number of elements.
            elements: Element connectivity array, shape (M, nodes_per_elem).
                Required by NodalMaterial to average node values per element.
        """


def _young_poisson_to_lame(E: float, nu: float) -> tuple[float, float]:
    """Convert Young's modulus and Poisson's ratio to Lamé parameters."""
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


def _lame_to_young_poisson(mu: float, lam: float) -> tuple[float, float]:
    """Convert Lamé parameters to Young's modulus and Poisson's ratio."""
    E = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
    nu = lam / (2.0 * (lam + mu))
    return E, nu


class IsotropicMaterial(Material):
    """Homogeneous isotropic linear elastic material.

    Create from either (E, nu) or (mu, lam). Exactly one pair must be provided.

    Args:
        E: Young's modulus [Pa]. Must be > 0.
        nu: Poisson's ratio (dimensionless). Must be in (-1, 0.5).
        mu: First Lamé parameter (shear modulus) [Pa]. Must be > 0.
        lam: Second Lamé parameter [Pa].
        density: Mass density [kg/m^3]. Default 1000.0 (water).
    """

    def __init__(
        self,
        E: float | None = None,
        nu: float | None = None,
        mu: float | None = None,
        lam: float | None = None,
        density: float = 1000.0,
    ):
        have_young = E is not None and nu is not None
        have_lame = mu is not None and lam is not None

        if have_young == have_lame:
            raise ValueError(
                "Specify exactly one of (E, nu) or (mu, lam), not both or neither."
            )

        if have_young:
            if E <= 0:
                raise ValueError(f"Young's modulus must be positive, got {E}")
            if nu <= -1.0 or nu >= 0.5:
                raise ValueError(
                    f"Poisson's ratio must be in (-1, 0.5), got {nu}"
                )
            self._mu, self._lam = _young_poisson_to_lame(E, nu)
        else:
            if mu <= 0:
                raise ValueError(f"Shear modulus (mu) must be positive, got {mu}")
            self._mu = mu
            self._lam = lam

        if density <= 0:
            raise ValueError(f"Density must be positive, got {density}")
        self._density = density

    @classmethod
    def from_young_poisson(
        cls, E: float, nu: float, density: float = 1000.0
    ) -> IsotropicMaterial:
        return cls(E=E, nu=nu, density=density)

    @classmethod
    def from_lame(
        cls, mu: float, lam: float, density: float = 1000.0
    ) -> IsotropicMaterial:
        return cls(mu=mu, lam=lam, density=density)

    @property
    def mu(self) -> float:
        return self._mu

    @property
    def lam(self) -> float:
        return self._lam

    @property
    def E(self) -> float:
        E, _ = _lame_to_young_poisson(self._mu, self._lam)
        return E

    @property
    def nu(self) -> float:
        _, nu = _lame_to_young_poisson(self._mu, self._lam)
        return nu

    @property
    def density(self) -> float:
        return self._density

    def get_mu(self) -> float:
        return self._mu

    def get_lambda(self) -> float:
        return self._lam

    def get_density(self) -> float:
        return self._density

    def to_lame_arrays(
        self, num_elements: int, elements: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.full(num_elements, self._mu, dtype=np.float32),
            np.full(num_elements, self._lam, dtype=np.float32),
        )

    def __repr__(self) -> str:
        return (
            f"IsotropicMaterial(E={self.E:.1f}, nu={self.nu:.4f}, "
            f"mu={self._mu:.1f}, lam={self._lam:.1f}, "
            f"density={self._density:.1f})"
        )


class HeterogeneousMaterial(Material):
    """Per-element material properties.

    Args:
        k_mu: Per-element shear modulus array, shape (num_elements,) [Pa].
        k_lambda: Per-element second Lamé parameter array, shape (num_elements,) [Pa].
        density: Scalar or per-element density [kg/m^3].
    """

    def __init__(
        self,
        k_mu: np.ndarray,
        k_lambda: np.ndarray,
        density: float | np.ndarray = 1000.0,
    ):
        k_mu = np.asarray(k_mu, dtype=np.float32)
        k_lambda = np.asarray(k_lambda, dtype=np.float32)
        if k_mu.ndim != 1 or k_lambda.ndim != 1:
            raise ValueError("k_mu and k_lambda must be 1D arrays")
        if k_mu.shape != k_lambda.shape:
            raise ValueError(
                f"k_mu and k_lambda must have same shape, got {k_mu.shape} and {k_lambda.shape}"
            )
        if np.any(k_mu <= 0):
            raise ValueError("All k_mu values must be positive")

        self._k_mu = k_mu
        self._k_lambda = k_lambda
        self._density = (
            np.asarray(density, dtype=np.float32)
            if isinstance(density, np.ndarray)
            else float(density)
        )

    @property
    def num_elements(self) -> int:
        return len(self._k_mu)

    def get_mu(self) -> np.ndarray:
        return self._k_mu

    def get_lambda(self) -> np.ndarray:
        return self._k_lambda

    def get_density(self) -> float | np.ndarray:
        return self._density

    def to_lame_arrays(
        self, num_elements: int, elements: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if num_elements != len(self._k_mu):
            raise ValueError(
                f"Expected {len(self._k_mu)} elements, got {num_elements}"
            )
        return self._k_mu.copy(), self._k_lambda.copy()

    def __repr__(self) -> str:
        return (
            f"HeterogeneousMaterial(num_elements={self.num_elements}, "
            f"mu=[{self._k_mu.min():.1f}, {self._k_mu.max():.1f}], "
            f"lam=[{self._k_lambda.min():.1f}, {self._k_lambda.max():.1f}])"
        )


class NodalMaterial(Material):
    """Per-node material properties with smooth interpolation across elements.

    Material properties (μ, λ) are defined at every mesh node and interpolated
    to quadrature points using the element shape functions. This gives a smooth,
    continuously varying material field — useful for gradual tissue transitions
    (e.g., liver → muscle → fat).

    For the Newton backend (which requires per-element constants), node values
    are averaged over each element's vertices. For backends that support field
    interpolation (e.g., Warp.fem), use ``get_nodal_mu()`` / ``get_nodal_lambda()``
    directly and let the quadrature integration handle the spatial variation.

    Note:
        Sharp material interfaces should be handled by mesh conformance, not
        per-node interpolation, which will smear step changes across elements.

    Args:
        k_mu: Per-node shear modulus array, shape (num_nodes,) [Pa]. Must be > 0.
        k_lambda: Per-node second Lamé parameter array, shape (num_nodes,) [Pa].
        density: Scalar or per-node density [kg/m^3].
    """

    def __init__(
        self,
        k_mu: np.ndarray,
        k_lambda: np.ndarray,
        density: float | np.ndarray = 1000.0,
    ):
        k_mu = np.asarray(k_mu, dtype=np.float32)
        k_lambda = np.asarray(k_lambda, dtype=np.float32)
        if k_mu.ndim != 1 or k_lambda.ndim != 1:
            raise ValueError("k_mu and k_lambda must be 1D arrays")
        if k_mu.shape != k_lambda.shape:
            raise ValueError(
                f"k_mu and k_lambda must have same shape, got {k_mu.shape} and {k_lambda.shape}"
            )
        if np.any(k_mu <= 0):
            raise ValueError("All k_mu values must be positive")

        self._k_mu = k_mu
        self._k_lambda = k_lambda
        self._density = (
            np.asarray(density, dtype=np.float32)
            if isinstance(density, np.ndarray)
            else float(density)
        )

    @property
    def num_nodes(self) -> int:
        return len(self._k_mu)

    def get_mu(self) -> np.ndarray:
        """Per-node shear modulus array [Pa]."""
        return self._k_mu

    def get_lambda(self) -> np.ndarray:
        """Per-node second Lamé parameter array [Pa]."""
        return self._k_lambda

    def get_density(self) -> float | np.ndarray:
        return self._density

    def get_nodal_mu(self) -> np.ndarray:
        """Per-node shear modulus, shape (num_nodes,) [Pa].

        Use this with backends that support continuous field interpolation
        (e.g., Warp.fem degree-1 field) to preserve the smooth spatial variation.
        """
        return self._k_mu

    def get_nodal_lambda(self) -> np.ndarray:
        """Per-node second Lamé parameter, shape (num_nodes,) [Pa].

        Use this with backends that support continuous field interpolation.
        """
        return self._k_lambda

    def to_lame_arrays(
        self, num_elements: int, elements: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return per-element Lamé arrays by averaging nodal values.

        Each element's μ and λ are computed as the mean of its nodes' values.
        This is the correct approach for the Newton backend which requires
        piecewise-constant material fields.

        Args:
            num_elements: Expected number of elements (used for validation).
            elements: Element connectivity, shape (M, nodes_per_elem). Required.

        Raises:
            ValueError: If ``elements`` is None or has unexpected element count.
        """
        if elements is None:
            raise ValueError(
                "NodalMaterial.to_lame_arrays requires the 'elements' array "
                "so node values can be averaged per element. Pass the mesh "
                "connectivity, e.g. mat.to_lame_arrays(num_elements, model.elements)."
            )
        elements = np.asarray(elements)
        if elements.shape[0] != num_elements:
            raise ValueError(
                f"Expected {num_elements} elements, got {elements.shape[0]}"
            )
        # Average nodal values over each element's vertices
        k_mu_elem = self._k_mu[elements].mean(axis=1).astype(np.float32)
        k_lam_elem = self._k_lambda[elements].mean(axis=1).astype(np.float32)
        return k_mu_elem, k_lam_elem

    def __repr__(self) -> str:
        return (
            f"NodalMaterial(num_nodes={self.num_nodes}, "
            f"mu=[{self._k_mu.min():.1f}, {self._k_mu.max():.1f}], "
            f"lam=[{self._k_lambda.min():.1f}, {self._k_lambda.max():.1f}])"
        )


class AnisotropicMaterial(Material):
    """Transversely isotropic Neo-Hookean with one HGO fiber family.

    Strain energy density:
        W = W_iso(mu, lam) + k1/(2*k2) * [exp(k2 * <I4-1>^2) - 1]

    where:
        W_iso  = standard stable Neo-Hookean (mu, lam)
        I4     = a0 . (C . a0)  — squared fiber stretch along unit vector a0
        <.>    = Macaulay bracket: max(., 0)  — fibers buckle in compression
        k1 [Pa] — fiber stiffness (> 0 for active fiber; 0 = isotropic)
        k2 [-]  — fiber nonlinearity (dimensionless, > 0)

    All parameters are per-element arrays of shape (num_elements,) except
    fiber_dirs which is (num_elements, 3).

    Example — fat layer allowing through-thickness sliding:
        fiber_dirs = [[1, 0, 0], ...] (in-plane)
        k1 = 1000 Pa, k2 = 2, mu = 1000 Pa
        → stiff in-plane, soft in through-thickness shear

    Args:
        k_mu:       Per-element shear modulus [Pa], shape (N,). Must be > 0.
        k_lambda:   Per-element second Lamé parameter [Pa], shape (N,).
        k1:         Per-element fiber stiffness [Pa], shape (N,). Must be ≥ 0.
        k2:         Per-element fiber nonlinearity, shape (N,). Must be > 0.
        fiber_dirs: Per-element fiber direction, shape (N, 3).
                    Normalised internally; warns if not already unit vectors.
        density:    Scalar or per-element density [kg/m³].
    """

    def __init__(
        self,
        k_mu: np.ndarray,
        k_lambda: np.ndarray,
        k1: np.ndarray,
        k2: np.ndarray,
        fiber_dirs: np.ndarray,
        density: float | np.ndarray = 1000.0,
    ):
        k_mu       = np.asarray(k_mu,       dtype=np.float32)
        k_lambda   = np.asarray(k_lambda,   dtype=np.float32)
        k1         = np.asarray(k1,         dtype=np.float32)
        k2         = np.asarray(k2,         dtype=np.float32)
        fiber_dirs = np.asarray(fiber_dirs, dtype=np.float32)

        if k_mu.ndim != 1:
            raise ValueError("k_mu must be 1D")
        n = k_mu.shape[0]
        for name, arr in [("k_lambda", k_lambda), ("k1", k1), ("k2", k2)]:
            if arr.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")
        if fiber_dirs.shape != (n, 3):
            raise ValueError(f"fiber_dirs must have shape ({n}, 3), got {fiber_dirs.shape}")
        if np.any(k_mu <= 0):
            raise ValueError("All k_mu values must be positive")
        if np.any(k1 < 0):
            raise ValueError("All k1 values must be >= 0")
        if np.any(k2 <= 0):
            raise ValueError("All k2 values must be > 0")

        # Normalise fiber directions; warn if significantly non-unit
        norms = np.linalg.norm(fiber_dirs, axis=1, keepdims=True)
        if np.any(np.abs(norms - 1.0) > 1e-4):
            import warnings
            warnings.warn(
                "AnisotropicMaterial: fiber_dirs are not unit vectors; "
                "normalising automatically.",
                UserWarning, stacklevel=2,
            )
        fiber_dirs = fiber_dirs / np.maximum(norms, 1e-12)

        self._k_mu       = k_mu
        self._k_lambda   = k_lambda
        self._k1         = k1
        self._k2         = k2
        self._fiber_dirs = fiber_dirs
        self._density    = (
            np.asarray(density, dtype=np.float32)
            if isinstance(density, np.ndarray)
            else float(density)
        )

    @classmethod
    def uniform(
        cls,
        num_elements: int,
        E: float,
        nu: float,
        k1: float,
        k2: float,
        fiber_dir: tuple | list | np.ndarray,
        density: float = 1000.0,
    ) -> "AnisotropicMaterial":
        """Broadcast scalar parameters to all elements.

        Args:
            num_elements: Number of elements.
            E:   Young's modulus [Pa].
            nu:  Poisson's ratio.
            k1:  Fiber stiffness [Pa].
            k2:  Fiber nonlinearity (dimensionless).
            fiber_dir: Unit fiber direction (3,) — same for all elements.
            density: Mass density [kg/m³].
        """
        mu  = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        fiber_dir_arr = np.asarray(fiber_dir, dtype=np.float32)
        if fiber_dir_arr.shape != (3,):
            raise ValueError("fiber_dir must have shape (3,)")
        return cls(
            k_mu       = np.full(num_elements, mu,  dtype=np.float32),
            k_lambda   = np.full(num_elements, lam, dtype=np.float32),
            k1         = np.full(num_elements, k1,  dtype=np.float32),
            k2         = np.full(num_elements, k2,  dtype=np.float32),
            fiber_dirs = np.tile(fiber_dir_arr, (num_elements, 1)),
            density    = density,
        )

    @property
    def num_elements(self) -> int:
        return len(self._k_mu)

    def get_mu(self) -> np.ndarray:
        return self._k_mu

    def get_lambda(self) -> np.ndarray:
        return self._k_lambda

    def get_density(self) -> float | np.ndarray:
        return self._density

    def get_k1(self) -> np.ndarray:
        """Per-element fiber stiffness [Pa], shape (N,)."""
        return self._k1

    def get_k2(self) -> np.ndarray:
        """Per-element fiber nonlinearity, shape (N,)."""
        return self._k2

    def get_fiber_dirs(self) -> np.ndarray:
        """Per-element unit fiber directions, shape (N, 3)."""
        return self._fiber_dirs

    def to_lame_arrays(
        self, num_elements: int, elements: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return isotropic Lamé arrays (k_mu, k_lambda).

        The fiber parameters (k1, k2, fiber_dirs) are accessed separately
        via get_k1(), get_k2(), get_fiber_dirs() for use in anisotropic solvers.
        """
        if num_elements != self.num_elements:
            raise ValueError(
                f"Expected {self.num_elements} elements, got {num_elements}"
            )
        return self._k_mu.copy(), self._k_lambda.copy()

    def __repr__(self) -> str:
        return (
            f"AnisotropicMaterial(num_elements={self.num_elements}, "
            f"mu=[{self._k_mu.min():.1f}, {self._k_mu.max():.1f}], "
            f"k1=[{self._k1.min():.1f}, {self._k1.max():.1f}], "
            f"k2=[{self._k2.min():.2f}, {self._k2.max():.2f}])"
        )


# ── Tissue presets ──────────────────────────────────────────────────────────

PROSTATE_PERIPHERAL = IsotropicMaterial(E=20.0e3, nu=0.48, density=1040.0)
"""Prostate peripheral zone: E ≈ 20 kPa, ν ≈ 0.48, ρ = 1040 kg/m³."""

PROSTATE_TRANSITION = IsotropicMaterial(E=40.0e3, nu=0.45, density=1040.0)
"""Prostate transition zone: E ≈ 40 kPa, ν ≈ 0.45, ρ = 1040 kg/m³."""
