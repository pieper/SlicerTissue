"""TissueModel: central assembly of mesh, materials, BCs, and loads."""

from __future__ import annotations

import numpy as np

from .boundary import BoundaryCondition
from .loading import LoadingCondition
from .materials import Material


class TissueModel:
    """Finite element model for soft tissue simulation.

    Assembles a tetrahedral mesh with material properties, boundary conditions,
    and loading conditions into a complete model ready for solving.

    All internal storage uses SI units (meters, Pascals, kg/m^3).

    Args:
        nodes: Vertex positions, shape (N, 3).
        elements: Tetrahedral connectivity, shape (M, 4). Each row contains
            four node indices defining one tetrahedron.
        material: Material property definition.
        boundary_conditions: List of boundary conditions (fixed nodes).
        loading_conditions: List of loading conditions (forces).
        unit_scale: Multiply input node coordinates by this factor to get
            meters. Use 0.001 if input is in millimeters (Slicer convention).
    """

    def __init__(
        self,
        nodes: np.ndarray,
        elements: np.ndarray,
        material: Material,
        boundary_conditions: list[BoundaryCondition] | None = None,
        loading_conditions: list[LoadingCondition] | None = None,
        unit_scale: float = 1.0,
    ):
        self._nodes = np.asarray(nodes, dtype=np.float64) * unit_scale
        self._elements = np.asarray(elements, dtype=np.int32)
        self._material = material
        self._boundary_conditions = list(boundary_conditions or [])
        self._loading_conditions = list(loading_conditions or [])

        if self._nodes.ndim != 2 or self._nodes.shape[1] != 3:
            raise ValueError(
                f"nodes must have shape (N, 3), got {self._nodes.shape}"
            )
        if self._elements.ndim != 2 or self._elements.shape[1] != 4:
            raise ValueError(
                f"elements must have shape (M, 4), got {self._elements.shape}"
            )

    @property
    def num_nodes(self) -> int:
        return self._nodes.shape[0]

    @property
    def num_elements(self) -> int:
        return self._elements.shape[0]

    @property
    def nodes(self) -> np.ndarray:
        """Node positions in meters, shape (N, 3)."""
        return self._nodes

    @property
    def elements(self) -> np.ndarray:
        """Tetrahedral connectivity, shape (M, 4)."""
        return self._elements

    @property
    def material(self) -> Material:
        return self._material

    @property
    def boundary_conditions(self) -> list[BoundaryCondition]:
        return self._boundary_conditions

    @property
    def loading_conditions(self) -> list[LoadingCondition]:
        return self._loading_conditions

    @property
    def fixed_node_indices(self) -> np.ndarray:
        """Union of all fixed node indices from all boundary conditions."""
        if not self._boundary_conditions:
            return np.array([], dtype=np.intp)
        indices = np.concatenate(
            [bc.get_fixed_node_indices(self._nodes) for bc in self._boundary_conditions]
        )
        return np.unique(indices)

    @property
    def free_node_indices(self) -> np.ndarray:
        """Indices of nodes that are not fixed."""
        fixed = set(self.fixed_node_indices)
        return np.array([i for i in range(self.num_nodes) if i not in fixed], dtype=np.intp)

    def add_boundary_condition(self, bc: BoundaryCondition) -> None:
        self._boundary_conditions.append(bc)

    def add_loading_condition(self, lc: LoadingCondition) -> None:
        self._loading_conditions.append(lc)

    def compute_element_volumes(self) -> np.ndarray:
        """Compute volume of each tetrahedron.

        Returns:
            Array of element volumes, shape (M,) [m^3].
            Uses the formula: V = |det([v1-v0, v2-v0, v3-v0])| / 6.
        """
        v0 = self._nodes[self._elements[:, 0]]
        v1 = self._nodes[self._elements[:, 1]]
        v2 = self._nodes[self._elements[:, 2]]
        v3 = self._nodes[self._elements[:, 3]]

        d1 = v1 - v0
        d2 = v2 - v0
        d3 = v3 - v0

        # det of 3x3 matrix [d1, d2, d3] per element
        det = (
            d1[:, 0] * (d2[:, 1] * d3[:, 2] - d2[:, 2] * d3[:, 1])
            - d1[:, 1] * (d2[:, 0] * d3[:, 2] - d2[:, 2] * d3[:, 0])
            + d1[:, 2] * (d2[:, 0] * d3[:, 1] - d2[:, 1] * d3[:, 0])
        )
        return np.abs(det) / 6.0

    def compute_lumped_masses(self) -> np.ndarray:
        """Compute lumped nodal masses by distributing element mass equally.

        Each tet contributes density * volume / 4 to each of its 4 nodes.

        Returns:
            Lumped mass array, shape (N,) [kg].
        """
        volumes = self.compute_element_volumes()
        density = self._material.get_density()

        if isinstance(density, np.ndarray):
            element_mass = density * volumes
        else:
            element_mass = float(density) * volumes

        masses = np.zeros(self.num_nodes, dtype=np.float64)
        for k in range(4):
            np.add.at(masses, self._elements[:, k], element_mass / 4.0)
        return masses

    def assemble_forces(self) -> np.ndarray:
        """Assemble external force vector from all loading conditions.

        Returns:
            Force array, shape (N, 3) [N].
        """
        forces = np.zeros((self.num_nodes, 3), dtype=np.float64)
        masses = self.compute_lumped_masses()
        for lc in self._loading_conditions:
            lc.apply(forces, self._nodes, masses)
        return forces

    def validate(self) -> list[str]:
        """Check the model for common issues.

        Returns:
            List of warning/error messages. Empty list means no issues found.
        """
        warnings = []

        # Check element indices are in range
        if np.any(self._elements < 0) or np.any(self._elements >= self.num_nodes):
            warnings.append(
                "Element connectivity contains out-of-range node indices."
            )

        # Check for degenerate tets (zero or near-zero volume)
        volumes = self.compute_element_volumes()
        degenerate = np.sum(volumes < 1e-20)
        if degenerate > 0:
            warnings.append(
                f"{degenerate} degenerate tetrahedra with near-zero volume."
            )

        # Check for orphan nodes (not referenced by any element)
        used_nodes = np.unique(self._elements.ravel())
        orphan_count = self.num_nodes - len(used_nodes)
        if orphan_count > 0:
            warnings.append(f"{orphan_count} orphan nodes not used by any element.")

        # Warn about near-incompressibility with linear tets
        mat = self._material
        if hasattr(mat, "nu"):
            nu = mat.nu
            if nu > 0.45:
                warnings.append(
                    f"Poisson's ratio {nu:.3f} > 0.45 with linear tetrahedra. "
                    "Volumetric locking may occur. Consider a mixed u-p formulation."
                )

        # Check heterogeneous material element count
        from .materials import HeterogeneousMaterial, NodalMaterial

        if isinstance(mat, HeterogeneousMaterial):
            if mat.num_elements != self.num_elements:
                warnings.append(
                    f"HeterogeneousMaterial has {mat.num_elements} elements "
                    f"but mesh has {self.num_elements}."
                )

        if isinstance(mat, NodalMaterial):
            if mat.num_nodes != self.num_nodes:
                warnings.append(
                    f"NodalMaterial has {mat.num_nodes} nodes "
                    f"but mesh has {self.num_nodes}."
                )

        return warnings
