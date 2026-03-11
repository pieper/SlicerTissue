"""Compare festiv and warp.fem 3-element stacks side-by-side in Slicer.

Creates two 3-element vertical stacks of 20-node hexahedral elements:
  - Left:  festiv (linear elastic, small-strain)
  - Right: warp.fem (Neo-Hookean, large deformation)

Both use the same boundary conditions:
  - Bottom face (z = -3*SCALE) fixed
  - Top corner displaced by [10, 10, 10] mm

Dragging the top corner on either side mirrors the BC to the other,
so both solve with identical boundary conditions for direct comparison.

Usage (from Slicer Python console):
  exec(open('/Users/pieper/slicer/latest/SlicerTissue/NewtonTissue/examples/stack3_compare.py').read())
"""

import sys, os, numpy, vtk, time

TISSUE_DIR   = '/Users/pieper/slicer/latest/SlicerTissue/NewtonTissue'
FESTIV_DIR   = '/Users/pieper/slicer/latest/SlicerTissue/TissueSimulation'
EXAMPLES_DIR = os.path.join(TISSUE_DIR, 'examples')

for p in [FESTIV_DIR, EXAMPLES_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import festiv.structure, festiv.element, festiv.node, festiv.isomap, festiv.el_grid
import importlib
for m in [festiv.structure, festiv.element, festiv.node, festiv.isomap, festiv.el_grid]:
    importlib.reload(m)

# -------------------------------------------------------------------------
ELEMENT_SCALE = 20.0   # half-width in mm (each element is 40 mm cube)
SEPARATION    = 120.0  # center-to-center distance along x
NUM_ELEMENTS  = 3      # elements stacked in z
# -------------------------------------------------------------------------


def build_festiv_stack(x_offset, scale=ELEMENT_SCALE, num_elements=NUM_ELEMENTS):
    """Build a festiv structure with num_elements stacked in z.

    Uses position-based node sharing: nodes at coincident positions
    are the same Python object, guaranteeing mesh compatibility.

    Returns (structure, corner_node) where corner_node is the top
    corner at (+x, +y, +z_max).
    """
    s = festiv.structure.structure()
    iso20 = festiv.isomap.iso20()
    node_dict = {}  # (round(x), round(y), round(z)) -> node

    for k in range(num_elements):
        center_z = (k - (num_elements - 1) / 2.0) * 2 * scale
        element = festiv.element.element20()
        s._elements.append(element)

        for i in range(20):
            ux, uy, uz = iso20.__unit_nodes__[i]
            px = ux * scale + x_offset
            py = uy * scale
            pz = uz * scale + center_z
            key = (round(px, 6), round(py, 6), round(pz, 6))

            if key not in node_dict:
                node = festiv.node.node()
                node._p = numpy.array([px, py, pz])
                s._nodes.append(node)
                node_dict[key] = node

            element._nodes[i] = node_dict[key]

    # Mark shared faces
    _mark_shared_faces(s)

    return s


def _mark_shared_faces(structure):
    """Mark faces shared between adjacent elements."""
    el_node_sets = []
    for el in structure._elements:
        el_node_sets.append(set(id(n) for n in el._nodes if n))

    for ei, el in enumerate(structure._elements):
        for face_idx in range(6):
            face_node_ids = set(
                id(el._nodes[ni])
                for ni in el.__faces__[face_idx][:8]
                if el._nodes[ni]
            )
            for ej, other_set in enumerate(el_node_sets):
                if ei != ej and face_node_ids.issubset(other_set):
                    el._shared_faces[face_idx] = 1
                    break


# ======================================================================
#  FESTIV stack  (left side)
# ======================================================================
class FestivStack:
    """3-element festiv stack with interactive control points."""

    def __init__(self, x_offset, coordinator=None):
        self.x_offset = x_offset
        self.coordinator = coordinator

        self.structure = build_festiv_stack(x_offset)

        # Identify bottom face nodes and top corner
        z_min = min(n._p[2] for n in self.structure._nodes)
        z_max = max(n._p[2] for n in self.structure._nodes)
        tol = ELEMENT_SCALE * 0.1

        # Fix bottom face
        self.bottom_nodes = []
        for node in self.structure._nodes:
            if node._p[2] < z_min + tol:
                node._fixed.fill(1)
                self.bottom_nodes.append(node)

        # Top corner: closest to (+SCALE+x_offset, +SCALE, z_max)
        target = numpy.array([ELEMENT_SCALE + x_offset, ELEMENT_SCALE, z_max])
        dists = [numpy.linalg.norm(n._p - target) for n in self.structure._nodes]
        ci = int(numpy.argmin(dists))
        self.corner_node = self.structure._nodes[ci]
        self.corner_node._u = numpy.array([10.0, 10.0, 10.0])
        self.corner_node._fixed.fill(1)
        self.corner_node_idx = ci

        # Solve
        self.structure.make_K()
        self.structure.apply_bc()
        self.structure.solve()

        # Visualization
        self.gridder = festiv.el_grid.gridder(self.structure)
        self.model = None
        self.fiducialList = None
        self._updatingNodeControlPoints = False

    def set_corner_displacement(self, disp):
        """Set corner displacement and re-solve."""
        self.corner_node._u = numpy.array(disp)
        self.structure.apply_bc()
        self.structure.solve()
        self.structure.updateNodes()

    def solve_and_update(self):
        """Re-solve and update visualization."""
        self.structure.apply_bc()
        self.structure.solve()
        self.structure.updateNodes()
        self.updateModel()
        self._sync_control_points()

    def createModel(self):
        self.gridder._steps = (4,) * 6
        self.gridder.surface_grid()
        path = os.path.join(slicer.app.temporaryPath, 'festiv_stack3.vtk')
        self.gridder.write_grid(path)
        _, self.model = slicer.util.loadModel(path, returnNode=True)
        self.model.SetName('festiv_stack')
        dn = self.model.GetDisplayNode()
        dn.SetBackfaceCulling(0)
        dn.SetEdgeVisibility(1)
        dn.SetColor(0.8, 0.8, 0.95)

    def updateModel(self):
        pts = slicer.util.array(self.model.GetID())
        self.gridder.surface_grid()
        pts[:] = numpy.array(self.gridder._points)
        self.model.GetPolyData().GetPoints().GetData().Modified()
        self.model.GetPolyData().GetPoints().Modified()

    def createControlPoints(self):
        ml = slicer.modules.markups.logic()
        slicer.mrmlScene.StartState(slicer.mrmlScene.BatchProcessState)
        self.fiducialList = ml.AddNewMarkupsNode(
            'vtkMRMLMarkupsFiducialNode', 'festiv_stack')
        dn = self.fiducialList.GetDisplayNode()
        dn.SetTextScale(2.0)
        dn.SetGlyphScale(4.0)
        dn.SetGlyphTypeFromString('Sphere3D')
        dn.SetColor(0.1, 0.1, 0.5)           # dark blue for fixed
        dn.SetSelectedColor(0.4, 0.4, 1.0)    # bright blue for free
        dn.SetActiveColor(0.6, 0.6, 1.0)
        unconstrained = slicer.vtkMRMLMarkupsFiducialDisplayNode.SnapModeUnconstrained
        dn.SetSnapMode(unconstrained)
        dn.SetVisibility(True)
        ml.SetActiveListID(self.fiducialList)

        for i, node in enumerate(self.structure._nodes):
            pu = node.pu()
            self.fiducialList.AddControlPoint(*pu)
            idx = self.fiducialList.GetNumberOfControlPoints() - 1
            is_corner = (i == self.corner_node_idx)
            is_fixed_bc = node._fixed.max() > 0 and not is_corner
            self.fiducialList.SetNthControlPointLabel(idx, '')
            # Only the corner point is draggable
            self.fiducialList.SetNthControlPointSelected(idx, is_corner)
            self.fiducialList.SetNthControlPointLocked(idx, not is_corner)

        self.fiducialList.AddObserver(
            self.fiducialList.PointModifiedEvent,
            lambda c, e: self.onControlPointMoved())
        slicer.mrmlScene.EndState(slicer.mrmlScene.BatchProcessState)

    def onControlPointMoved(self):
        if self._updatingNodeControlPoints:
            return
        slicer.mrmlScene.StartState(slicer.mrmlScene.BatchProcessState)

        # Read corner position, compute displacement
        pt = [0, 0, 0]
        self.fiducialList.GetNthControlPointPosition(self.corner_node_idx, pt)
        disp = numpy.array(pt) - self.corner_node._p

        self.corner_node._u = disp
        self.structure.apply_bc()
        self.structure.solve()
        self.structure.updateNodes()
        self.updateModel()
        self._sync_control_points()

        # Mirror to other side
        if self.coordinator:
            self.coordinator.on_festiv_corner_moved(disp)

        slicer.mrmlScene.EndState(slicer.mrmlScene.BatchProcessState)

    def _sync_control_points(self):
        self._updatingNodeControlPoints = True
        for i, node in enumerate(self.structure._nodes):
            pu = node.pu()
            self.fiducialList.SetNthControlPointPosition(i, *pu)
        self._updatingNodeControlPoints = False


# ======================================================================
#  WARP.FEM stack  (right side)
# ======================================================================
import warp as wp
import warp.fem as fem
import warp.examples.fem.utils as fem_example_utils

if 'warpfem_integrands' in sys.modules:
    del sys.modules['warpfem_integrands']
import warpfem_integrands as wfi


class WarpFemStack:
    """3-element warp.fem stack with Neo-Hookean and interactive control points."""

    def __init__(self, x_offset, coordinator=None):
        self.x_offset = x_offset
        self.coordinator = coordinator

        z_half = NUM_ELEMENTS * ELEMENT_SCALE
        lo = wp.vec3(-ELEMENT_SCALE + x_offset, -ELEMENT_SCALE, -z_half)
        hi = wp.vec3( ELEMENT_SCALE + x_offset,  ELEMENT_SCALE,  z_half)
        self.geo = fem.Grid3D(res=wp.vec3i(1, 1, NUM_ELEMENTS),
                              bounds_lo=lo, bounds_hi=hi)

        E, nu = 1e4, 0.3
        mu  = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        mat_space = fem.make_polynomial_space(
            self.geo, degree=0, discontinuous=True, dtype=float)
        self.mu_field  = mat_space.make_field()
        self.lam_field = mat_space.make_field()
        self.mu_field.dof_values.assign(
            wp.array([mu] * NUM_ELEMENTS, dtype=float))
        self.lam_field.dof_values.assign(
            wp.array([lam] * NUM_ELEMENTS, dtype=float))

        self.u_space = fem.make_polynomial_space(
            self.geo, degree=2, dtype=wp.vec3,
            element_basis=fem.ElementBasis.SERENDIPITY)
        self.u_field = self.u_space.make_field()
        self.n_dof = self.u_space.node_count()

        self.domain = fem.Cells(geometry=self.geo)
        self.test  = fem.make_test(space=self.u_space, domain=self.domain)
        self.trial = fem.make_trial(space=self.u_space, domain=self.domain)

        # DOF rest positions via L2 projection
        M = fem.integrate(wfi.pos_mass_form,
                          fields={'u': self.trial, 'v': self.test})
        b = fem.integrate(wfi.pos_rhs_form,
                          fields={'v': self.test}, output_dtype=wp.vec3d)
        pos = wp.zeros(self.n_dof, dtype=wp.vec3d)
        fem_example_utils.bsr_cg(M, b=b, x=pos, quiet=True, tol=1e-10)
        self.dof_positions = pos.numpy().astype(numpy.float64)

        # Bottom face DOFs
        tol_pos = ELEMENT_SCALE * 0.1
        self.bottom_dofs = [i for i in range(self.n_dof)
                            if self.dof_positions[i][2] < -z_half + tol_pos]

        # Top corner DOF
        target = numpy.array([ELEMENT_SCALE + x_offset, ELEMENT_SCALE, z_half])
        dists = [numpy.linalg.norm(self.dof_positions[i] - target)
                 for i in range(self.n_dof)]
        self.top_corner_dof = int(numpy.argmin(dists))

        self.bc_dofs = set(self.bottom_dofs)
        self.bc_dofs.add(self.top_corner_dof)

        # Bottom face BC projector
        boundary = fem.BoundarySides(self.geo)
        bd_test  = fem.make_test(space=self.u_space, domain=boundary)
        bd_trial = fem.make_trial(space=self.u_space, domain=boundary)
        self.u_bd_matrix = fem.integrate(
            wfi.bottom_projector,
            fields={'u': bd_trial, 'v': bd_test}, assembly='nodal')

        # Initial solve with prescribed displacement
        u_vals = self.u_field.dof_values.numpy()
        u_vals[self.top_corner_dof] = numpy.array([10, 10, 10], dtype=numpy.float32)
        self.u_field.dof_values.assign(wp.array(u_vals, dtype=wp.vec3))
        self._newton_solve()

        # Save converged state for interactive solver
        self._last_converged_u = self.u_field.dof_values.numpy().copy()
        self._last_converged_corner = self.u_field.dof_values.numpy()[self.top_corner_dof].copy()

        # Mirror festiv structure for visualization
        self.mirror_structure = build_festiv_stack(x_offset)
        _mark_shared_faces(self.mirror_structure)
        self._build_dof_mapping()
        self._sync_mirror()

        # Visualization
        self.gridder = festiv.el_grid.gridder(self.mirror_structure)
        self.model = None
        self.fiducialList = None
        self._updatingNodeControlPoints = False

    def _build_dof_mapping(self):
        """Map each festiv mirror node to closest warp.fem DOF by position."""
        self.mirror_to_warp = []
        for node in self.mirror_structure._nodes:
            dists = [numpy.linalg.norm(node._p - self.dof_positions[j])
                     for j in range(self.n_dof)]
            self.mirror_to_warp.append(int(numpy.argmin(dists)))

        # Also find which mirror node corresponds to the corner
        self.mirror_corner_idx = self.mirror_to_warp.index(self.top_corner_dof)

    def _sync_mirror(self):
        """Copy warp.fem displacements to mirror festiv structure."""
        u_vals = self.u_field.dof_values.numpy()
        for i, node in enumerate(self.mirror_structure._nodes):
            wi = self.mirror_to_warp[i]
            node._u = u_vals[wi].astype(numpy.float64)

    # -- Newton-Raphson solver -----------------------------------------
    def _project_dof_bc(self, K, rhs, dof_idx):
        """Zero row/col of BSR matrix for a single DOF, set diagonal to I."""
        offsets = K.offsets.numpy()
        columns = K.columns.numpy()
        values  = K.values.numpy()
        rs, re = offsets[dof_idx], offsets[dof_idx + 1]
        for k in range(rs, re):
            values[k] = numpy.eye(3, dtype=values.dtype) if columns[k] == dof_idx \
                        else numpy.zeros((3, 3), dtype=values.dtype)
        n_rows = len(offsets) - 1
        for row in range(n_rows):
            if row == dof_idx:
                continue
            for k in range(offsets[row], offsets[row + 1]):
                if columns[k] == dof_idx:
                    values[k] = numpy.zeros((3, 3), dtype=values.dtype)
                    break
        K.values.assign(wp.array(values, dtype=K.values.dtype))
        rhs_np = rhs.numpy()
        rhs_np[dof_idx] = [0., 0., 0.]
        rhs.assign(wp.array(rhs_np, dtype=rhs.dtype))

    def _newton_solve(self, n_load_steps=10, max_newton=15, tol=1e-3):
        """Incremental Newton-Raphson with full tangent."""
        u_target = self.u_field.dof_values.numpy().copy()
        # Reset to zero for load stepping
        self.u_field.dof_values.fill_(wp.vec3(0, 0, 0))

        for step in range(n_load_steps):
            frac = (step + 1) / n_load_steps
            # Carry forward converged free DOFs, only update prescribed BCs
            u_vals = self.u_field.dof_values.numpy()
            u_vals[self.top_corner_dof] = (u_target[self.top_corner_dof] * frac)
            for di in self.bottom_dofs:
                u_vals[di] = [0, 0, 0]
            self.u_field.dof_values.assign(wp.array(u_vals, dtype=wp.vec3))

            for it in range(max_newton):
                K = fem.integrate(wfi.tangent_stiffness,
                    fields={'u': self.trial, 'v': self.test,
                            'u_cur': self.u_field,
                            'mu_f': self.mu_field, 'lam_f': self.lam_field})
                f_int = fem.integrate(wfi.internal_force,
                    fields={'v': self.test, 'u_cur': self.u_field,
                            'mu_f': self.mu_field, 'lam_f': self.lam_field},
                    output_dtype=wp.vec3d)

                rhs_np = -f_int.numpy()
                for di in self.bc_dofs:
                    rhs_np[di] = [0., 0., 0.]
                rhs = wp.array(rhs_np, dtype=wp.vec3d)

                fem.project_linear_system(K, rhs, self.u_bd_matrix)
                self._project_dof_bc(K, rhs, self.top_corner_dof)

                du = wp.zeros(self.n_dof, dtype=wp.vec3d)
                fem_example_utils.bsr_cg(K, b=rhs, x=du,
                                         quiet=True, tol=1e-10, max_iters=500)
                du_np = du.numpy()
                du_norm = numpy.linalg.norm(du_np)
                if numpy.isnan(du_norm):
                    break

                u_vals = self.u_field.dof_values.numpy()
                u_vals += du_np.astype(numpy.float32)
                u_vals[self.top_corner_dof] = (u_target[self.top_corner_dof] * frac)
                for di in self.bottom_dofs:
                    u_vals[di] = [0, 0, 0]
                self.u_field.dof_values.assign(wp.array(u_vals, dtype=wp.vec3))

                if du_norm < tol:
                    break

    def _newton_solve_from_current(self, n_sub_steps=5, max_newton=10, tol=1e-2):
        """Newton solve from current state with mini load stepping.

        Interpolates the corner BC from the last converged state to the
        new target over n_sub_steps to avoid divergence on large jumps.
        Reverts to last good state on failure. Uses backtracking line search.
        """
        target_corner = self.u_field.dof_values.numpy()[self.top_corner_dof].copy()
        if not hasattr(self, '_last_converged_corner'):
            self._last_converged_corner = numpy.zeros(3, dtype=numpy.float32)
        start_corner = self._last_converged_corner.copy()

        # Restore to last converged state
        if hasattr(self, '_last_converged_u'):
            self.u_field.dof_values.assign(
                wp.array(self._last_converged_u, dtype=wp.vec3))

        # Displacement magnitude limit per element size
        max_disp_per_step = ELEMENT_SCALE * 0.5  # 10mm per sub-step
        disp_change = numpy.linalg.norm(target_corner - start_corner)
        if disp_change > 0:
            n_sub_steps = max(n_sub_steps,
                              int(numpy.ceil(disp_change / max_disp_per_step)))

        failed = False
        last_good_sub = -1

        for sub in range(n_sub_steps):
            frac = (sub + 1) / n_sub_steps
            corner = start_corner + (target_corner - start_corner) * frac

            # Save state before this sub-step
            u_backup = self.u_field.dof_values.numpy().copy()

            u_vals = self.u_field.dof_values.numpy()
            u_vals[self.top_corner_dof] = corner
            for di in self.bottom_dofs:
                u_vals[di] = [0, 0, 0]
            self.u_field.dof_values.assign(wp.array(u_vals, dtype=wp.vec3))

            converged = False
            for it in range(max_newton):
                K = fem.integrate(wfi.tangent_stiffness,
                    fields={'u': self.trial, 'v': self.test,
                            'u_cur': self.u_field,
                            'mu_f': self.mu_field, 'lam_f': self.lam_field})
                f_int = fem.integrate(wfi.internal_force,
                    fields={'v': self.test, 'u_cur': self.u_field,
                            'mu_f': self.mu_field, 'lam_f': self.lam_field},
                    output_dtype=wp.vec3d)

                rhs_np = -f_int.numpy()
                for di in self.bc_dofs:
                    rhs_np[di] = [0., 0., 0.]
                rhs = wp.array(rhs_np, dtype=wp.vec3d)

                fem.project_linear_system(K, rhs, self.u_bd_matrix)
                self._project_dof_bc(K, rhs, self.top_corner_dof)

                du = wp.zeros(self.n_dof, dtype=wp.vec3d)
                fem_example_utils.bsr_cg(K, b=rhs, x=du,
                                         quiet=True, tol=1e-8, max_iters=500)
                du_np = du.numpy()
                du_norm = numpy.linalg.norm(du_np)

                if numpy.isnan(du_norm) or du_norm > 1e6:
                    break

                # Backtracking line search: halve step if update is too large
                alpha = 1.0
                while alpha > 0.05 and du_norm * alpha > ELEMENT_SCALE * 2:
                    alpha *= 0.5

                u_vals = self.u_field.dof_values.numpy()
                u_vals += (du_np * alpha).astype(numpy.float32)
                u_vals[self.top_corner_dof] = corner
                for di in self.bottom_dofs:
                    u_vals[di] = [0, 0, 0]
                self.u_field.dof_values.assign(wp.array(u_vals, dtype=wp.vec3))

                if du_norm < tol:
                    converged = True
                    break

            if not converged:
                # Revert to state before this sub-step failed
                self.u_field.dof_values.assign(wp.array(u_backup, dtype=wp.vec3))
                failed = True
                break
            last_good_sub = sub

        # Only save if we made progress
        if last_good_sub >= 0:
            self._last_converged_u = self.u_field.dof_values.numpy().copy()
            if failed:
                # Partially converged: save the corner we actually reached
                achieved_frac = (last_good_sub + 1) / n_sub_steps
                self._last_converged_corner = (
                    start_corner + (target_corner - start_corner) * achieved_frac
                ).copy()
            else:
                self._last_converged_corner = target_corner.copy()

    def set_corner_displacement(self, disp):
        """Set corner displacement and re-solve."""
        u_vals = self.u_field.dof_values.numpy()
        u_vals[self.top_corner_dof] = numpy.array(disp, dtype=numpy.float32)
        self.u_field.dof_values.assign(wp.array(u_vals, dtype=wp.vec3))
        self._newton_solve_from_current()
        self._sync_mirror()

    def solve_and_update(self):
        """Re-solve and update visualization."""
        self._newton_solve_from_current()
        self._sync_mirror()
        self.updateModel()
        self._sync_control_points()

    # -- Slicer visualization ------------------------------------------
    def createModel(self):
        self.gridder._steps = (4,) * 6
        self._sync_mirror()
        self.gridder.surface_grid()
        path = os.path.join(slicer.app.temporaryPath, 'warpfem_stack3.vtk')
        self.gridder.write_grid(path)
        _, self.model = slicer.util.loadModel(path, returnNode=True)
        self.model.SetName('warpfem_stack')
        dn = self.model.GetDisplayNode()
        dn.SetBackfaceCulling(0)
        dn.SetEdgeVisibility(1)
        dn.SetColor(0.95, 0.85, 0.85)

    def updateModel(self):
        self._sync_mirror()
        pts = slicer.util.array(self.model.GetID())
        self.gridder.surface_grid()
        pts[:] = numpy.array(self.gridder._points)
        self.model.GetPolyData().GetPoints().GetData().Modified()
        self.model.GetPolyData().GetPoints().Modified()

    def createControlPoints(self):
        ml = slicer.modules.markups.logic()
        slicer.mrmlScene.StartState(slicer.mrmlScene.BatchProcessState)
        self.fiducialList = ml.AddNewMarkupsNode(
            'vtkMRMLMarkupsFiducialNode', 'warpfem_stack')
        dn = self.fiducialList.GetDisplayNode()
        dn.SetTextScale(2.0)
        dn.SetGlyphScale(4.0)
        dn.SetGlyphTypeFromString('Sphere3D')
        dn.SetColor(0.5, 0.1, 0.1)           # dark red for fixed
        dn.SetSelectedColor(1.0, 0.4, 0.4)    # bright red for free
        dn.SetActiveColor(1.0, 0.6, 0.6)
        unconstrained = slicer.vtkMRMLMarkupsFiducialDisplayNode.SnapModeUnconstrained
        dn.SetSnapMode(unconstrained)
        dn.SetVisibility(True)
        ml.SetActiveListID(self.fiducialList)

        for i, node in enumerate(self.mirror_structure._nodes):
            pu = node.pu()
            self.fiducialList.AddControlPoint(*pu)
            idx = self.fiducialList.GetNumberOfControlPoints() - 1
            is_corner = (i == self.mirror_corner_idx)
            self.fiducialList.SetNthControlPointLabel(idx, '')
            self.fiducialList.SetNthControlPointSelected(idx, is_corner)
            self.fiducialList.SetNthControlPointLocked(idx, not is_corner)

        self.fiducialList.AddObserver(
            self.fiducialList.PointModifiedEvent,
            lambda c, e: self.onControlPointMoved())
        slicer.mrmlScene.EndState(slicer.mrmlScene.BatchProcessState)

    def onControlPointMoved(self):
        if self._updatingNodeControlPoints:
            return
        slicer.mrmlScene.StartState(slicer.mrmlScene.BatchProcessState)

        # Read corner position, compute displacement
        pt = [0, 0, 0]
        self.fiducialList.GetNthControlPointPosition(self.mirror_corner_idx, pt)
        corner_rest = self.mirror_structure._nodes[self.mirror_corner_idx]._p
        disp = numpy.array(pt) - corner_rest

        self.set_corner_displacement(disp)
        self.updateModel()
        self._sync_control_points()

        # Mirror to other side
        if self.coordinator:
            self.coordinator.on_warp_corner_moved(disp)

        slicer.mrmlScene.EndState(slicer.mrmlScene.BatchProcessState)

    def _sync_control_points(self):
        self._updatingNodeControlPoints = True
        for i, node in enumerate(self.mirror_structure._nodes):
            pu = node.pu()
            self.fiducialList.SetNthControlPointPosition(i, *pu)
        self._updatingNodeControlPoints = False


# ======================================================================
#  Coordinator for mirrored BCs
# ======================================================================
class StackComparison:
    """Coordinates festiv and warp.fem stacks with mirrored boundary conditions."""

    def __init__(self):
        self.festiv = FestivStack(x_offset=-SEPARATION / 2, coordinator=self)
        self.warp   = WarpFemStack(x_offset=+SEPARATION / 2, coordinator=self)

    def on_festiv_corner_moved(self, disp):
        """Festiv corner was dragged — mirror displacement to warp.fem."""
        self.warp.set_corner_displacement(disp)
        self.warp.updateModel()
        self.warp._sync_control_points()

    def on_warp_corner_moved(self, disp):
        """Warp.fem corner was dragged — mirror displacement to festiv."""
        self.festiv.set_corner_displacement(disp)
        self.festiv.updateModel()
        self.festiv._sync_control_points()

    def create_views(self):
        self.festiv.createModel()
        self.festiv.updateModel()  # overwrite LPS→RAS transform from VTK load
        self.festiv.createControlPoints()
        self.warp.createModel()
        self.warp.updateModel()    # overwrite LPS→RAS transform from VTK load
        self.warp.createControlPoints()


# ======================================================================
#  Main
# ======================================================================
import time as _time

print('=' * 60)
print('Compare 3-element stacks: festiv (linear) vs warp.fem (NH)')
print('=' * 60)

slicer.mrmlScene.Clear(0)

_t0 = _time.time()
comp = StackComparison()

# Verify festiv mesh
_n_nodes = len(comp.festiv.structure._nodes)
_n_expected = NUM_ELEMENTS * 20 - (NUM_ELEMENTS - 1) * 8
print(f'\n  Festiv nodes: {_n_nodes} (expected {_n_expected})')

# Check Jacobian signs
for _ei, _el in enumerate(comp.festiv.structure._elements):
    _x = numpy.matrix(numpy.zeros([20, 1]))
    _y = numpy.matrix(numpy.zeros([20, 1]))
    _z = numpy.matrix(numpy.zeros([20, 1]))
    _el.load_xyz_arrays(_x, _y, _z)
    _jac = numpy.matrix(numpy.zeros([3, 3]))
    _jinv = numpy.matrix(numpy.zeros([3, 3]))
    _detj = _el.calculate_J(_x, _y, _z, 0, 0, 0, _jac, _jinv)
    print(f'  Element {_ei} Jacobian det at center: {_detj:.1f}')

# Check shared faces
for _ei, _el in enumerate(comp.festiv.structure._elements):
    _shared = [i for i in range(6) if _el._shared_faces[i]]
    print(f'  Element {_ei} shared faces: {_shared}')

comp.create_views()

# Comparison
_festiv_disp = numpy.array([n._u for n in comp.festiv.structure._nodes])
_warp_disp = comp.warp.u_field.dof_values.numpy()
_festiv_free = [i for i, n in enumerate(comp.festiv.structure._nodes)
               if n._fixed.max() == 0]
_warp_free = [i for i in range(comp.warp.n_dof) if i not in comp.warp.bc_dofs]
_festiv_max = max(numpy.linalg.norm(_festiv_disp[i]) for i in _festiv_free)
_warp_max = max(numpy.linalg.norm(_warp_disp[i]) for i in _warp_free)

print(f'\n  Festiv max free-node disp:  {_festiv_max:.3f} mm')
print(f'  Warp.fem max free-node disp: {_warp_max:.3f} mm')
if _festiv_max > 0:
    print(f'  NH strain stiffening: {(1 - _warp_max/_festiv_max)*100:.1f}% stiffer')

print(f'\n  Setup time: {_time.time()-_t0:.2f}s')

slicer.stack_comp = comp

slicer.app.layoutManager().setLayout(
    slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
_v = slicer.app.layoutManager().threeDWidget(0).threeDView()
_v.resetFocalPoint()

print('\n  Drag blue corner (festiv) or red corner (warp.fem)')
print('  Both stacks update with same BC for direct comparison')
print('  Object: slicer.stack_comp')
print('=' * 60)
