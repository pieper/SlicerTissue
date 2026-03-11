"""Shared pytest fixtures for NewtonTissue tests."""

import numpy as np
import pytest

from newton_tissue import (
    FixedByBox,
    Gravity,
    IsotropicMaterial,
    TissueModel,
    PROSTATE_PERIPHERAL,
)


def make_single_tet():
    """A single regular tetrahedron with unit-ish dimensions."""
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    elements = np.array([[0, 1, 2, 3]], dtype=np.int32)
    return nodes, elements


def make_cantilever_mesh(nx=10, ny=2, nz=2, Lx=1.0, Ly=0.1, Lz=0.1):
    """Generate a structured tetrahedral mesh for a cantilever beam.

    Creates a regular hexahedral grid and splits each hex into 5 tetrahedra.

    Args:
        nx, ny, nz: Number of hex cells in each direction.
        Lx, Ly, Lz: Beam dimensions [m].

    Returns:
        nodes: (N, 3) float64 array of vertex positions.
        elements: (M, 4) int32 array of tet connectivity.
    """
    # Node grid
    xs = np.linspace(0, Lx, nx + 1)
    ys = np.linspace(0, Ly, ny + 1)
    zs = np.linspace(0, Lz, nz + 1)

    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)
    nodes = grid.reshape(-1, 3)

    def node_idx(i, j, k):
        return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k

    # Split each hex into 5 tets using a consistent decomposition
    elements = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # 8 corners of the hex
                n = [
                    node_idx(i, j, k),
                    node_idx(i + 1, j, k),
                    node_idx(i + 1, j + 1, k),
                    node_idx(i, j + 1, k),
                    node_idx(i, j, k + 1),
                    node_idx(i + 1, j, k + 1),
                    node_idx(i + 1, j + 1, k + 1),
                    node_idx(i, j + 1, k + 1),
                ]
                # 5-tet decomposition (alternating parity for mesh conformity)
                parity = (i + j + k) % 2
                if parity == 0:
                    elements.append([n[0], n[1], n[3], n[4]])
                    elements.append([n[1], n[2], n[3], n[6]])
                    elements.append([n[4], n[6], n[5], n[1]])
                    elements.append([n[3], n[4], n[6], n[7]])
                    elements.append([n[1], n[3], n[4], n[6]])
                else:
                    elements.append([n[0], n[1], n[2], n[5]])
                    elements.append([n[0], n[2], n[3], n[7]])
                    elements.append([n[0], n[4], n[5], n[7]])
                    elements.append([n[2], n[5], n[6], n[7]])
                    elements.append([n[0], n[2], n[5], n[7]])

    elements = np.array(elements, dtype=np.int32)
    return nodes, elements


@pytest.fixture
def single_tet_model():
    """A single-tet model with prostate material."""
    nodes, elements = make_single_tet()
    return TissueModel(
        nodes=nodes,
        elements=elements,
        material=PROSTATE_PERIPHERAL,
    )


@pytest.fixture
def cantilever_model():
    """Cantilever beam model: steel, fixed at x=0, gravity in -y."""
    nodes, elements = make_cantilever_mesh()
    steel = IsotropicMaterial(E=200e9, nu=0.3, density=7800.0)
    return TissueModel(
        nodes=nodes,
        elements=elements,
        material=steel,
        boundary_conditions=[FixedByBox([-0.001, -0.001, -0.001], [0.001, 0.101, 0.101])],
        loading_conditions=[Gravity()],
    )


@pytest.fixture
def prostate_material():
    return PROSTATE_PERIPHERAL
