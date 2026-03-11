"""Compare festiv and warp.fem single-element solutions side-by-side in Slicer.

Creates two 20-node hexahedral elements in the Slicer 3D view:
  - Left:  festiv (linear elastic, small-strain)
  - Right: warp.fem (Neo-Hookean, large deformation)

Both use the same boundary conditions:
  - Bottom face (z = -20) fixed
  - Top corner displaced by [10, 10, 10] mm

Interactive: drag any unlocked control point to re-solve and update.

Usage (from Slicer Python console):
  exec(open('/Users/pieper/slicer/latest/SlicerTissue/NewtonTissue/examples/compare_elements_slicer.py').read())
  -- or --
  Run via the Slicer MCP server
"""

# This file is designed to be exec()'d from Slicer.
# Warp.fem integrands and solver live in importable .py files
# to satisfy inspect.getsource() requirements.

import sys, os, numpy, vtk, time

TISSUE_DIR  = '/Users/pieper/slicer/latest/SlicerTissue/NewtonTissue'
FESTIV_DIR  = '/Users/pieper/slicer/latest/SlicerTissue/TissueSimulation'
EXAMPLES_DIR = os.path.join(TISSUE_DIR, 'examples')

for p in [FESTIV_DIR, EXAMPLES_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import festiv.structure, festiv.element, festiv.node, festiv.isomap, festiv.el_grid
import importlib
for m in [festiv.structure, festiv.element, festiv.node, festiv.el_grid]:
    importlib.reload(m)

# -------------------------------------------------------------------------
ELEMENT_SCALE = 20.0   # half-width in mm  (element is 40 mm on a side)
SEPARATION    = 80.0   # center-to-center distance along x
# -------------------------------------------------------------------------


# ======================================================================
#  FESTIV element  (left side, x_offset = -SEPARATION/2)
# ======================================================================
class FestivElement:
    """Thin wrapper around TissueSimulationLogic's one-element pattern."""

    def __init__(self, x_offset):
        self.x_offset = x_offset

        # build structure
        self.structure = festiv.structure.structure()
        iso20 = festiv.isomap.iso20()
        element = festiv.element.element20()
        self.structure._elements.append(element)

        for i in range(20):
            node = festiv.node.node()
            pos = numpy.array(iso20.__unit_nodes__[i], dtype=float) * ELEMENT_SCALE
            pos[0] += x_offset
            node._p = pos.copy()
            self.structure._nodes.append(node)
            element._nodes[i] = node

        # bottom face fixed (face 1 = z-min in festiv convention)
        for node in element.face_nodes(1):
            node._fixed.fill(1)

        # top corner displaced
        corner = self.structure._elements[0]._nodes[0]
        corner._u = numpy.array([10.0, 10.0, 10.0])
        corner._fixed.fill(1)

        # solve
        self.structure.make_K()
        self.structure.apply_bc()
        self.structure.solve()

        # visualization helpers
        self.gridder = festiv.el_grid.gridder(self.structure)
        self.model = None
        self.fiducialList = None
        self._updatingNodeControlPoints = False

    # -- Slicer visualization ------------------------------------------
    def createModel(self):
        self.gridder._steps = (4,) * 6
        self.gridder.surface_grid()
        path = os.path.join(slicer.app.temporaryPath, 'festiv_element.vtk')
        self.gridder.write_grid(path)
        _, self.model = slicer.util.loadModel(path, returnNode=True)
        self.model.SetName('festiv_model')
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
            'vtkMRMLMarkupsFiducialNode', 'festiv')
        dn = self.fiducialList.GetDisplayNode()
        dn.SetTextScale(2.0)
        dn.SetGlyphScale(5.0)
        dn.SetGlyphTypeFromString('Sphere3D')
        dn.SetColor(0.1, 0.1, 0.5)           # dark blue for fixed
        dn.SetSelectedColor(0.4, 0.4, 1.0)    # bright blue for movable
        dn.SetActiveColor(0.6, 0.6, 1.0)
        unconstrained = slicer.vtkMRMLMarkupsFiducialDisplayNode.SnapModeUnconstrained
        dn.SetSnapMode(unconstrained)
        dn.SetVisibility(True)
        ml.SetActiveListID(self.fiducialList)

        for node in self.structure._nodes:
            pu = node.pu()
            self.fiducialList.AddControlPoint(*pu)
            idx = self.fiducialList.GetNumberOfControlPoints() - 1
            fixed = node._fixed.max() > 0
            self.fiducialList.SetNthControlPointLabel(idx, '')
            self.fiducialList.SetNthControlPointSelected(idx, not fixed)
            self.fiducialList.SetNthControlPointLocked(idx, not fixed)

        self.fiducialList.AddObserver(
            self.fiducialList.PointModifiedEvent,
            lambda c, e: self.onControlPointMoved())
        slicer.mrmlScene.EndState(slicer.mrmlScene.BatchProcessState)

    def onControlPointMoved(self):
        if self._updatingNodeControlPoints:
            return
        slicer.mrmlScene.StartState(slicer.mrmlScene.BatchProcessState)
        for i, node in enumerate(self.structure._nodes):
            pt = [0, 0, 0]
            self.fiducialList.GetNthControlPointPosition(i, pt)
            node._u = numpy.array(pt) - node._p
        self.structure.apply_bc()
        self.structure.solve()
        self.structure.updateNodes()
        self.updateModel()

        self._updatingNodeControlPoints = True
        for i, node in enumerate(self.structure._nodes):
            pu = node.pu()
            self.fiducialList.SetNthControlPointPosition(i, *pu)
        self._updatingNodeControlPoints = False
        slicer.mrmlScene.EndState(slicer.mrmlScene.BatchProcessState)


# ======================================================================
#  WARP.FEM element  (right side, x_offset = +SEPARATION/2)
# ======================================================================
import warp as wp
import warp.fem as fem
import warp.examples.fem.utils as fem_example_utils

if 'warpfem_integrands' in sys.modules:
    del sys.modules['warpfem_integrands']
import warpfem_integrands as wfi


class WarpFemElement:
    """20-node serendipity hex with Neo-Hookean via warp.fem."""

    def __init__(self, x_offset):
        self.x_offset = x_offset

        lo = wp.vec3(-ELEMENT_SCALE + x_offset, -ELEMENT_SCALE, -ELEMENT_SCALE)
        hi = wp.vec3( ELEMENT_SCALE + x_offset,  ELEMENT_SCALE,  ELEMENT_SCALE)
        self.geo = fem.Grid3D(res=wp.vec3i(1, 1, 1), bounds_lo=lo, bounds_hi=hi)

        E, nu = 1e4, 0.3
        mu  = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        mat_space = fem.make_polynomial_space(
            self.geo, degree=0, discontinuous=True, dtype=float)
        self.mu_field  = mat_space.make_field()
        self.lam_field = mat_space.make_field()
        self.mu_field.dof_values.assign(wp.array([mu],  dtype=float))
        self.lam_field.dof_values.assign(wp.array([lam], dtype=float))

        self.u_space = fem.make_polynomial_space(
            self.geo, degree=2, dtype=wp.vec3,
            element_basis=fem.ElementBasis.SERENDIPITY)
        self.u_field = self.u_space.make_field()
        self.n_dof = self.u_space.node_count()

        self.domain = fem.Cells(geometry=self.geo)
        self.test  = fem.make_test(space=self.u_space, domain=self.domain)
        self.trial = fem.make_trial(space=self.u_space, domain=self.domain)

        # DOF rest positions (via L2 projection of geometry position)
        M = fem.integrate(wfi.pos_mass_form,
                          fields={'u': self.trial, 'v': self.test})
        b = fem.integrate(wfi.pos_rhs_form,
                          fields={'v': self.test}, output_dtype=wp.vec3d)
        pos = wp.zeros(self.n_dof, dtype=wp.vec3d)
        fem_example_utils.bsr_cg(M, b=b, x=pos, quiet=True, tol=1e-10)
        self.dof_positions = pos.numpy().astype(numpy.float64)

        # Identify z-min face DOFs (bottom, matching festiv face 1)
        tol = ELEMENT_SCALE * 0.1
        self.bottom_dofs = [i for i in range(self.n_dof)
                            if self.dof_positions[i][2] < -ELEMENT_SCALE + tol]

        # Top corner: closest to (+x_max, +y_max, +z_max)
        target = numpy.array([ELEMENT_SCALE + x_offset, ELEMENT_SCALE, ELEMENT_SCALE])
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

        # Apply initial prescribed displacement and solve
        u_vals = self.u_field.dof_values.numpy()
        u_vals[self.top_corner_dof] = numpy.array([10, 10, 10], dtype=numpy.float32)
        self.u_field.dof_values.assign(wp.array(u_vals, dtype=wp.vec3))
        self._newton_solve()

        # visualization
        self.model = None
        self.fiducialList = None
        self._updatingNodeControlPoints = False

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

    def _newton_solve(self, n_load_steps=5, max_newton=15, tol=1e-3):
        """Incremental Newton-Raphson with full tangent."""
        # Target is whatever is currently stored at the BC DOFs
        u_target = self.u_field.dof_values.numpy().copy()
        # Reset field to previous converged state minus the corner
        # (for load-stepping, interpolate from zero to target)
        u_start = self.u_field.dof_values.numpy().copy()
        for di in self.bc_dofs:
            if di == self.top_corner_dof:
                u_start[di] = [0, 0, 0]
            # bottom dofs stay at zero

        for step in range(n_load_steps):
            frac = (step + 1) / n_load_steps
            # Interpolate prescribed displacements
            u_vals = u_start.copy()
            u_vals[self.top_corner_dof] = (u_target[self.top_corner_dof] * frac)
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

    # -- Subdivided surface for visualization --------------------------
    def _serendipity_basis(self, r, s, t):
        """Evaluate 20-node serendipity shape functions at (r,s,t) in [-1,1]^3.

        Returns array of 20 values. Uses the same basis as festiv/isomap iso20.
        """
        # Corner nodes (8): vertices of [-1,1]^3
        # Mid-edge nodes (12): midpoints of edges
        # Standard ordering matches Grid3D serendipity DOF layout
        #
        # We use the festiv iso20 interpolator since it implements exactly
        # the same 20-node serendipity shape functions.
        iso = festiv.isomap.iso20()
        N = numpy.zeros(20)
        for i in range(20):
            N[i] = float(iso.h(r, s, t, i))
        return N

    def _build_subdivided_surface(self):
        """Build a subdivided surface like festiv's gridder, using shape function interpolation."""
        steps = 4  # subdivision level per face
        deformed = self.dof_positions + self.u_field.dof_values.numpy()

        # We need to map warp.fem DOF ordering to festiv node ordering
        # so we can use the iso20 shape functions.
        # Festiv iso20 unit nodes are at specific (r,s,t) positions.
        # We find the mapping by matching rest positions.
        iso20 = festiv.isomap.iso20()
        festiv_unit = numpy.array(iso20.__unit_nodes__)  # (20, 3) in [-1,1]^3

        # Warp DOF rest positions relative to element center, normalized to [-1,1]
        center = numpy.array([self.x_offset, 0, 0])
        normalized = (self.dof_positions - center) / ELEMENT_SCALE  # in [-1,1]^3

        # Build mapping: for each festiv node i, find closest warp DOF
        warp_to_festiv = numpy.zeros(20, dtype=int)
        festiv_to_warp = numpy.zeros(20, dtype=int)
        for fi in range(20):
            dists = [numpy.linalg.norm(normalized[wi] - festiv_unit[fi])
                     for wi in range(20)]
            festiv_to_warp[fi] = int(numpy.argmin(dists))
        # Inverse
        for fi in range(20):
            warp_to_festiv[festiv_to_warp[fi]] = fi

        # Build deformed positions in festiv ordering
        deformed_festiv = deformed[festiv_to_warp]  # (20, 3) in festiv order

        # Now use the gridder's face_increments to generate subdivided faces
        face_increments = festiv.el_grid.gridder.__face_increments__
        points = []
        polys  = []

        for face in range(6):
            fi = face_increments[face]
            r_start, r_end, r_dir = fi[0], fi[1], fi[2]
            s_start, s_end, s_dir = fi[3], fi[4], fi[5]
            t_start, t_end, t_dir = fi[6], fi[7], fi[8]

            r_inc = r_dir * (r_end - r_start) / steps
            s_inc = s_dir * (s_end - s_start) / steps
            t_inc = t_dir * (t_end - t_start) / steps

            if r_inc == 0.:
                r_inc = 1.
                rstep = steps + 1
            else:
                rstep = 1
            if s_inc == 0.:
                s_inc = 1.
                sstep = steps + 1
            else:
                sstep = 1
            if t_inc == 0.:
                t_inc = 1.
                tstep = steps + 1
            else:
                tstep = 1

            pis = []
            r = r_start
            for _ in range(0, steps + 1, rstep):
                s = s_start
                for _ in range(0, steps + 1, sstep):
                    t = t_start
                    for _ in range(0, steps + 1, tstep):
                        N = self._serendipity_basis(r, s, t)
                        pt = N @ deformed_festiv  # (3,) interpolated position
                        pis.append(len(points))
                        points.append(pt.tolist())
                        t += t_inc
                    s += s_inc
                r += r_inc

            for col in range(steps):
                for row in range(steps):
                    ll = col * (steps + 1) + row
                    ul = col * (steps + 1) + row + 1
                    ur = (col + 1) * (steps + 1) + row + 1
                    lr = (col + 1) * (steps + 1) + row
                    polys.append([pis[ll], pis[ul], pis[ur], pis[lr]])

        return points, polys

    def _build_polydata(self, points, polys):
        """Convert points/polys to VTK polydata."""
        pd = vtk.vtkPolyData()
        vpts = vtk.vtkPoints()
        for p in points:
            vpts.InsertNextPoint(*p)
        pd.SetPoints(vpts)
        cells = vtk.vtkCellArray()
        for poly in polys:
            q = vtk.vtkQuad()
            for j, vi in enumerate(poly):
                q.GetPointIds().SetId(j, vi)
            cells.InsertNextCell(q)
        pd.SetPolys(cells)
        return pd

    # -- Slicer visualization ------------------------------------------
    def createModel(self):
        pts, polys = self._build_subdivided_surface()
        pd = self._build_polydata(pts, polys)
        self.model = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLModelNode', 'warpfem_model')
        self.model.SetAndObservePolyData(pd)
        self.model.CreateDefaultDisplayNodes()
        dn = self.model.GetDisplayNode()
        dn.SetBackfaceCulling(0)
        dn.SetEdgeVisibility(1)
        dn.SetColor(0.95, 0.85, 0.85)

    def updateModel(self):
        pts, polys = self._build_subdivided_surface()
        vpts = self.model.GetPolyData().GetPoints()
        for i, p in enumerate(pts):
            vpts.SetPoint(i, *p)
        vpts.GetData().Modified()
        vpts.Modified()

    def createControlPoints(self):
        ml = slicer.modules.markups.logic()
        slicer.mrmlScene.StartState(slicer.mrmlScene.BatchProcessState)
        self.fiducialList = ml.AddNewMarkupsNode(
            'vtkMRMLMarkupsFiducialNode', 'warp_fem')
        dn = self.fiducialList.GetDisplayNode()
        dn.SetTextScale(2.0)
        dn.SetGlyphScale(5.0)
        dn.SetGlyphTypeFromString('Sphere3D')
        dn.SetColor(0.5, 0.1, 0.1)           # dark red for fixed
        dn.SetSelectedColor(1.0, 0.4, 0.4)    # bright red for movable
        dn.SetActiveColor(1.0, 0.6, 0.6)
        unconstrained = slicer.vtkMRMLMarkupsFiducialDisplayNode.SnapModeUnconstrained
        dn.SetSnapMode(unconstrained)
        dn.SetVisibility(True)
        ml.SetActiveListID(self.fiducialList)

        deformed = self.dof_positions + self.u_field.dof_values.numpy()
        for i in range(self.n_dof):
            self.fiducialList.AddControlPoint(*deformed[i])
            idx = self.fiducialList.GetNumberOfControlPoints() - 1
            fixed = i in self.bc_dofs
            self.fiducialList.SetNthControlPointLabel(idx, '')
            self.fiducialList.SetNthControlPointSelected(idx, not fixed)
            self.fiducialList.SetNthControlPointLocked(idx, not fixed)

        self.fiducialList.AddObserver(
            self.fiducialList.PointModifiedEvent,
            lambda c, e: self.onControlPointMoved())
        slicer.mrmlScene.EndState(slicer.mrmlScene.BatchProcessState)

    def onControlPointMoved(self):
        if self._updatingNodeControlPoints:
            return
        slicer.mrmlScene.StartState(slicer.mrmlScene.BatchProcessState)

        # Read ALL fiducial positions into displacement field
        # (matching festiv pattern: node._u = pt - node._p)
        u_vals = self.u_field.dof_values.numpy()
        for i in range(self.n_dof):
            pt = [0, 0, 0]
            self.fiducialList.GetNthControlPointPosition(i, pt)
            u_vals[i] = (numpy.array(pt) - self.dof_positions[i]).astype(numpy.float32)
        self.u_field.dof_values.assign(wp.array(u_vals, dtype=wp.vec3))

        # Re-solve from current state
        self._newton_solve_from_current()

        # Update visualization
        self.updateModel()

        # Update fiducial positions to match solved state
        self._updatingNodeControlPoints = True
        deformed = self.dof_positions + self.u_field.dof_values.numpy()
        for i in range(self.n_dof):
            self.fiducialList.SetNthControlPointPosition(i, *deformed[i])
        self._updatingNodeControlPoints = False
        slicer.mrmlScene.EndState(slicer.mrmlScene.BatchProcessState)

    def _newton_solve_from_current(self, max_newton=10, tol=1e-2):
        """Newton solve from current state (no load stepping, for interactive use).

        Preserves the current prescribed displacement at the top corner
        and zero displacement at bottom face DOFs.
        """
        # Save the prescribed corner displacement
        corner_disp = self.u_field.dof_values.numpy()[self.top_corner_dof].copy()

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
            if numpy.isnan(du_norm):
                break

            u_vals = self.u_field.dof_values.numpy()
            u_vals += du_np.astype(numpy.float32)
            # Re-enforce BCs
            u_vals[self.top_corner_dof] = corner_disp
            for di in self.bottom_dofs:
                u_vals[di] = [0, 0, 0]
            self.u_field.dof_values.assign(wp.array(u_vals, dtype=wp.vec3))

            if du_norm < tol:
                break


# ======================================================================
#  Main
# ======================================================================
def main():
    print('=' * 60)
    print('Compare: festiv (linear) vs warp.fem (Neo-Hookean)')
    print('=' * 60)

    slicer.mrmlScene.Clear(0)

    # Festiv on the left
    print('\n--- Festiv element (left) ---')
    t0 = time.time()
    festiv_elem = FestivElement(x_offset=-SEPARATION / 2)
    festiv_elem.createModel()
    festiv_elem.createControlPoints()
    print(f'  Created in {time.time()-t0:.2f}s')

    # Warp.fem on the right
    print('\n--- Warp.fem element (right) ---')
    t0 = time.time()
    warp_elem = WarpFemElement(x_offset=+SEPARATION / 2)
    warp_elem.createModel()
    warp_elem.createControlPoints()
    print(f'  Created in {time.time()-t0:.2f}s')

    # Comparison
    festiv_disp = numpy.array([n._u for n in festiv_elem.structure._nodes])
    warp_disp = warp_elem.u_field.dof_values.numpy()
    festiv_max = max(numpy.linalg.norm(d) for i, d in enumerate(festiv_disp)
                     if festiv_elem.structure._nodes[i]._fixed.max() == 0)
    warp_max = max(numpy.linalg.norm(d) for i, d in enumerate(warp_disp)
                   if i not in warp_elem.bc_dofs)

    print(f'\n  Festiv max free-node disp:  {festiv_max:.3f} mm')
    print(f'  Warp.fem max free-node disp: {warp_max:.3f} mm')
    print(f'  NH strain stiffening: {(1 - warp_max/festiv_max)*100:.1f}% stiffer')

    # Store references for console access
    slicer.festiv_elem = festiv_elem
    slicer.warp_elem   = warp_elem

    # Set up view
    slicer.app.layoutManager().setLayout(
        slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
    v = slicer.app.layoutManager().threeDWidget(0).threeDView()
    v.resetFocalPoint()

    print('\n  Drag blue points (festiv) or red points (warp.fem) to interact')
    print('  Objects: slicer.festiv_elem, slicer.warp_elem')
    print('=' * 60)


main()
