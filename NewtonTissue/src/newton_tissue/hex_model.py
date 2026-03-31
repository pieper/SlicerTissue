"""HexTissueModel: 20-node serendipity hexahedral mesh for soft tissue FEM.

Supports structured Grid3D geometry with per-element or per-node material
properties, boundary conditions, and loading conditions.
"""

from __future__ import annotations

import numpy as np

from .boundary import BoundaryCondition, FixedByBox
from .loading import LoadingCondition, Gravity
from .materials import Material, IsotropicMaterial, HeterogeneousMaterial, NodalMaterial, AnisotropicMaterial


class HexTissueModel:
    """Finite element model using 20-node serendipity hexahedral elements.

    Uses a structured Grid3D geometry (warp.fem) with degree-2 serendipity
    basis functions — 20 nodes per element. Suitable for GPU-accelerated
    materially nonlinear large-deformation simulation via Neo-Hookean
    hyperelasticity.

    For materially heterogeneous tissue, material properties can be:
      - ``IsotropicMaterial``: uniform across all elements
      - ``HeterogeneousMaterial``: one value per hex element
      - ``NodalMaterial``: one value per DOF node — interpolated at every
        quadrature point via the serendipity shape functions (smooth spatial
        variation, correct for 27-point Gauss quadrature)

    Args:
        res: Grid resolution ``(nx, ny, nz)`` — number of hex elements
            along each axis.
        bounds_lo: Physical lower corner ``(x, y, z)`` [m].
        bounds_hi: Physical upper corner ``(x, y, z)`` [m].
        material: Material definition.
        boundary_conditions: List of boundary conditions.
        loading_conditions: List of loading conditions.
    """

    def __init__(
        self,
        res: tuple[int, int, int],
        bounds_lo: tuple[float, float, float] = (0.0, 0.0, 0.0),
        bounds_hi: tuple[float, float, float] = (1.0, 1.0, 1.0),
        material: Material | None = None,
        boundary_conditions: list[BoundaryCondition] | None = None,
        loading_conditions: list[LoadingCondition] | None = None,
    ):
        self._res = tuple(int(r) for r in res)
        self._bounds_lo = tuple(float(b) for b in bounds_lo)
        self._bounds_hi = tuple(float(b) for b in bounds_hi)
        self._material = material or IsotropicMaterial(E=10e3, nu=0.45)
        self._boundary_conditions = list(boundary_conditions or [])
        self._loading_conditions = list(loading_conditions or [])

    # ── Geometry ──────────────────────────────────────────────────────────

    @property
    def res(self) -> tuple[int, int, int]:
        """Grid resolution (nx, ny, nz)."""
        return self._res

    @property
    def bounds_lo(self) -> tuple[float, float, float]:
        return self._bounds_lo

    @property
    def bounds_hi(self) -> tuple[float, float, float]:
        return self._bounds_hi

    @property
    def num_elements(self) -> int:
        nx, ny, nz = self._res
        return nx * ny * nz

    @property
    def element_size(self) -> tuple[float, float, float]:
        """Physical size of one hex element (dx, dy, dz) [m]."""
        nx, ny, nz = self._res
        lo, hi = self._bounds_lo, self._bounds_hi
        return (
            (hi[0] - lo[0]) / nx,
            (hi[1] - lo[1]) / ny,
            (hi[2] - lo[2]) / nz,
        )

    # ── Material ──────────────────────────────────────────────────────────

    @property
    def material(self) -> Material:
        return self._material

    # ── Boundary / loading ────────────────────────────────────────────────

    @property
    def boundary_conditions(self) -> list[BoundaryCondition]:
        return self._boundary_conditions

    @property
    def loading_conditions(self) -> list[LoadingCondition]:
        return self._loading_conditions

    def add_boundary_condition(self, bc: BoundaryCondition) -> None:
        self._boundary_conditions.append(bc)

    def add_loading_condition(self, lc: LoadingCondition) -> None:
        self._loading_conditions.append(lc)

    # ── Material array builders ───────────────────────────────────────────

    def build_element_lame_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return per-element (k_mu, k_lambda) arrays, shape (num_elements,).

        For ``NodalMaterial``, element values are the mean of the 8 corner
        node values of each hex (conservative fallback; the solver uses the
        full nodal field for quadrature-point interpolation).
        For ``IsotropicMaterial`` and ``HeterogeneousMaterial``, returns
        the standard ``to_lame_arrays`` result.
        """
        mat = self._material
        n = self.num_elements

        if isinstance(mat, NodalMaterial):
            # Fallback: broadcast per-element mean from corner nodes.
            # The solver builds a proper continuous warp.fem field instead.
            corner_indices = self._corner_node_indices_per_element()
            k_mu = mat.get_nodal_mu()[corner_indices].mean(axis=1)
            k_lam = mat.get_nodal_lambda()[corner_indices].mean(axis=1)
            return k_mu.astype(np.float32), k_lam.astype(np.float32)

        return mat.to_lame_arrays(n)

    def _corner_node_indices_per_element(self) -> np.ndarray:
        """Return (num_elements, 8) array of Grid3D corner node indices.

        Grid3D vertex ordering: index = i*(ny+1)*(nz+1) + j*(nz+1) + k
        """
        nx, ny, nz = self._res
        indices = np.empty((nx * ny * nz, 8), dtype=np.int32)
        eidx = 0
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    c = 0
                    for di in range(2):
                        for dj in range(2):
                            for dk in range(2):
                                indices[eidx, c] = (
                                    (i + di) * (ny + 1) * (nz + 1)
                                    + (j + dj) * (nz + 1)
                                    + (k + dk)
                                )
                                c += 1
                    eidx += 1
        return indices

    # ── Convenience constructors ──────────────────────────────────────────

    @classmethod
    def layered_block(
        cls,
        res: tuple[int, int, int],
        size: float,
        layers: list[tuple[float, float, Material]],
        fixed_bottom: bool = True,
    ) -> HexTissueModel:
        """Create a tissue block with horizontal material layers.

        Args:
            res: Grid resolution (nx, ny, nz).
            size: Cube side length [m].
            layers: List of (y_lo, y_hi, material) tuples defining layers
                along the y-axis. The last matching layer wins.
            fixed_bottom: If True, add a FixedByBox BC on the bottom face.

        Returns:
            A ``HexTissueModel`` with per-element ``HeterogeneousMaterial``.
        """
        nx, ny, nz = res
        dy = size / ny
        k_mu = np.empty(nx * ny * nz, dtype=np.float32)
        k_lam = np.empty(nx * ny * nz, dtype=np.float32)

        for i in range(nx):
            for j in range(ny):
                cy = (j + 0.5) * dy
                for k in range(nz):
                    eidx = i * ny * nz + j * nz + k
                    mat = layers[0][2]
                    for y_lo, y_hi, layer_mat in layers:
                        if y_lo <= cy < y_hi:
                            mat = layer_mat
                    mu, lam = mat.to_lame_arrays(1)
                    k_mu[eidx] = mu[0]
                    k_lam[eidx] = lam[0]

        het_mat = HeterogeneousMaterial(k_mu=k_mu, k_lambda=k_lam)
        bcs = []
        if fixed_bottom:
            eps = min(size / max(res), 1e-6)
            bcs.append(FixedByBox(
                [-eps, -eps, -eps],
                [size + eps, eps, size + eps],
            ))

        return cls(
            res=res,
            bounds_lo=(0.0, 0.0, 0.0),
            bounds_hi=(size, size, size),
            material=het_mat,
            boundary_conditions=bcs,
        )

    def build_element_aniso_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return per-element (k1, k2, fiber_dirs) for anisotropic material.

        Returns:
            k1:         shape (num_elements,) [Pa]
            k2:         shape (num_elements,) [dimensionless]
            fiber_dirs: shape (num_elements, 3) unit fiber directions

        Raises:
            TypeError: If the material is not AnisotropicMaterial.
        """
        mat = self._material
        if not isinstance(mat, AnisotropicMaterial):
            raise TypeError(
                "build_element_aniso_arrays() requires AnisotropicMaterial, "
                f"got {type(mat).__name__}"
            )
        return mat.get_k1().copy(), mat.get_k2().copy(), mat.get_fiber_dirs().copy()

    @classmethod
    def anisotropic_layered_block(
        cls,
        res: tuple[int, int, int],
        size: float,
        aniso_layers: list[tuple],
        fixed_bottom: bool = True,
    ) -> "HexTissueModel":
        """Create a tissue block with anisotropic horizontal material layers.

        Args:
            res:          Grid resolution (nx, ny, nz).
            size:         Cube side length [m].
            aniso_layers: List of tuples:
                (y_lo, y_hi, E, nu, k1, k2, fiber_dir, density)
                where fiber_dir is a (3,) array or list.
                The last matching layer (y_lo <= cy < y_hi) is used.
            fixed_bottom: If True, add a FixedByBox BC on the bottom face.

        Returns:
            A HexTissueModel with AnisotropicMaterial.
        """
        nx, ny, nz = res
        n = nx * ny * nz
        dy = size / ny

        k_mu       = np.empty(n, dtype=np.float32)
        k_lam      = np.empty(n, dtype=np.float32)
        k1_arr     = np.empty(n, dtype=np.float32)
        k2_arr     = np.empty(n, dtype=np.float32)
        fiber_arr  = np.empty((n, 3), dtype=np.float32)

        for i in range(nx):
            for j in range(ny):
                cy = (j + 0.5) * dy
                for k in range(nz):
                    eidx = i * ny * nz + j * nz + k
                    # Default to first layer
                    y_lo0, y_hi0, E0, nu0, k1_0, k2_0, fd0, rho0 = aniso_layers[0]
                    E_e, nu_e, k1_e, k2_e, fd_e = E0, nu0, k1_0, k2_0, fd0
                    for (y_lo, y_hi, E_l, nu_l, k1_l, k2_l, fd_l, rho_l) in aniso_layers:
                        if y_lo <= cy < y_hi:
                            E_e, nu_e, k1_e, k2_e, fd_e = E_l, nu_l, k1_l, k2_l, fd_l
                    mu  = E_e / (2.0 * (1.0 + nu_e))
                    lam = E_e * nu_e / ((1.0 + nu_e) * (1.0 - 2.0 * nu_e))
                    k_mu[eidx]      = mu
                    k_lam[eidx]     = lam
                    k1_arr[eidx]    = k1_e
                    k2_arr[eidx]    = k2_e
                    fd = np.asarray(fd_e, dtype=np.float32)
                    fiber_arr[eidx] = fd / max(np.linalg.norm(fd), 1e-12)

        aniso_mat = AnisotropicMaterial(
            k_mu=k_mu, k_lambda=k_lam,
            k1=k1_arr, k2=k2_arr, fiber_dirs=fiber_arr,
        )
        bcs = []
        if fixed_bottom:
            eps = min(size / max(res), 1e-6)
            bcs.append(FixedByBox(
                [-eps, -eps, -eps],
                [size + eps, eps, size + eps],
            ))
        return cls(
            res=res,
            bounds_lo=(0.0, 0.0, 0.0),
            bounds_hi=(size, size, size),
            material=aniso_mat,
            boundary_conditions=bcs,
        )

    def validate(self) -> list[str]:
        """Check model for common issues."""
        warnings = []
        nx, ny, nz = self._res
        if any(r < 1 for r in [nx, ny, nz]):
            warnings.append("All resolution values must be >= 1.")
        lo, hi = self._bounds_lo, self._bounds_hi
        if any(hi[i] <= lo[i] for i in range(3)):
            warnings.append("bounds_hi must be strictly greater than bounds_lo.")
        mat = self._material
        if isinstance(mat, (HeterogeneousMaterial, AnisotropicMaterial)):
            if mat.num_elements != self.num_elements:
                warnings.append(
                    f"{type(mat).__name__} has {mat.num_elements} elements "
                    f"but mesh has {self.num_elements}."
                )
        return warnings

    def __repr__(self) -> str:
        nx, ny, nz = self._res
        return (
            f"HexTissueModel(res={self._res}, "
            f"elements={nx*ny*nz}, "
            f"material={type(self._material).__name__})"
        )
