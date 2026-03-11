"""Tests for boundary condition definitions."""

import numpy as np

from newton_tissue import FixedBC, FixedByPredicate, FixedByBox


# Simple node grid for testing
GRID_NODES = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [1.0, 0.5, 0.0],
    ],
    dtype=np.float64,
)


class TestFixedBC:
    def test_explicit_indices(self):
        bc = FixedBC([0, 2, 4])
        indices = bc.get_fixed_node_indices(GRID_NODES)
        np.testing.assert_array_equal(indices, [0, 2, 4])

    def test_numpy_indices(self):
        bc = FixedBC(np.array([1, 3]))
        indices = bc.get_fixed_node_indices(GRID_NODES)
        np.testing.assert_array_equal(indices, [1, 3])


class TestFixedByPredicate:
    def test_x_less_than(self):
        bc = FixedByPredicate(lambda p: p[0] < 0.25)
        indices = bc.get_fixed_node_indices(GRID_NODES)
        # Nodes 0 and 3 have x=0.0
        np.testing.assert_array_equal(indices, [0, 3])

    def test_all_fixed(self):
        bc = FixedByPredicate(lambda p: True)
        indices = bc.get_fixed_node_indices(GRID_NODES)
        assert len(indices) == len(GRID_NODES)


class TestFixedByBox:
    def test_box_selection(self):
        bc = FixedByBox(lower=[-0.1, -0.1, -0.1], upper=[0.25, 1.0, 0.1])
        indices = bc.get_fixed_node_indices(GRID_NODES)
        # Nodes 0 (0,0,0) and 3 (0,0.5,0) are inside
        np.testing.assert_array_equal(indices, [0, 3])

    def test_empty_box(self):
        bc = FixedByBox(lower=[10, 10, 10], upper=[20, 20, 20])
        indices = bc.get_fixed_node_indices(GRID_NODES)
        assert len(indices) == 0


class TestMultipleBCs:
    def test_union_no_duplicates(self):
        """Model.fixed_node_indices should be the unique union of all BCs."""
        from newton_tissue import TissueModel, IsotropicMaterial

        nodes = GRID_NODES
        # Minimal 1-tet using first 4 nodes (just for model construction)
        elements = np.array([[0, 1, 3, 4]], dtype=np.int32)
        mat = IsotropicMaterial(E=1e3, nu=0.3)

        bc1 = FixedBC([0, 1])
        bc2 = FixedBC([1, 3])  # overlaps on node 1
        model = TissueModel(
            nodes=nodes, elements=elements, material=mat,
            boundary_conditions=[bc1, bc2],
        )
        fixed = model.fixed_node_indices
        np.testing.assert_array_equal(fixed, [0, 1, 3])
