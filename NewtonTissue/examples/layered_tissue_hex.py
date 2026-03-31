"""Interactive multi-layer tissue simulation using 20-node serendipity hex elements.

Creates a 4-layer tissue block (skin/fat/muscle/liver) using HexTissueModel
with a draggable markup fiducial on the top surface. The bottom face is fixed.
Dragging the top fiducial applies a prescribed displacement BC and re-solves
using Neo-Hookean hyperelasticity via warp.fem.

Usage (Slicer Python console):
  exec(open('.../NewtonTissue/examples/layered_tissue_hex.py').read())
  # Then drag the yellow sphere on the top surface
  # Access via: slicer.layered_hex
"""

from __future__ import annotations
import os, sys, numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
_NEWTON_DIR = os.path.dirname(_SCRIPT_DIR) if os.path.basename(_SCRIPT_DIR) == 'examples' else _SCRIPT_DIR
_SRC_DIR = os.path.join(_NEWTON_DIR, 'src')
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import warp as wp
import warp.fem as fem
import warp.examples.fem.utils as fem_utils

from newton_tissue import (
    HexTissueModel, IsotropicMaterial, HeterogeneousMaterial, NodalMaterial,
)

# ── Tissue layer definitions ──────────────────────────────────────────────

BLOCK_SIZE = 0.08   # 8 cm cube [m]  (~fist sized)
RES        = (3, 6, 3)   # nx, ny, nz
MM_PER_M   = 1000.0      # Slicer uses mm; FEM uses m

LAYERS = [
    (0.000, 0.040, IsotropicMaterial(E=10_000.0, nu=0.45, density=1060.0)),  # liver
    (0.040, 0.055, IsotropicMaterial(E=60_000.0, nu=0.40, density=1050.0)),  # muscle
    (0.055, 0.070, IsotropicMaterial(E= 3_000.0, nu=0.49, density= 900.0)),  # fat
    (0.070, 0.080, IsotropicMaterial(E=100_000.0, nu=0.40, density=1100.0)), # skin
]

# ── warp.fem integrands ───────────────────────────────────────────────────

@fem.integrand
def _internal_force(s: fem.Sample, v: fem.Field, u_cur: fem.Field,
                    mu_f: fem.Field, lam_f: fem.Field):
    F   = wp.identity(n=3, dtype=float) + fem.grad(u_cur, s)
    J   = wp.determinant(F)
    mu  = mu_f(s)
    lam = lam_f(s)
    cof = J * wp.transpose(wp.inverse(F))
    P   = mu * F + (lam * (J - 1.0) - mu) * cof
    return wp.ddot(P, fem.grad(v, s))

@fem.integrand
def _tangent_stiffness(s: fem.Sample, u: fem.Field, v: fem.Field,
                       u_cur: fem.Field, mu_f: fem.Field, lam_f: fem.Field):
    gv  = fem.grad(v, s)
    gdu = fem.grad(u, s)
    F   = wp.identity(n=3, dtype=float) + fem.grad(u_cur, s)
    J   = wp.determinant(F)
    mu  = mu_f(s)
    lam = lam_f(s)
    cof = J * wp.transpose(wp.inverse(F))
    return mu * wp.ddot(gv, gdu) + lam * wp.ddot(cof, gv) * wp.ddot(cof, gdu)

@fem.integrand
def _bottom_projector(s: fem.Sample, domain: fem.Domain,
                      u: fem.Field, v: fem.Field):
    nor = fem.normal(domain, s)
    w   = wp.max(0.0, -nor[1])   # bottom face: outward normal is -y
    return w * wp.dot(u(s), v(s))

@fem.integrand
def _pos_mass(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(u(s), v(s))

@fem.integrand
def _pos_rhs(s: fem.Sample, domain: fem.Domain, v: fem.Field):
    return wp.dot(fem.position(domain, s), v(s))


# ── Main class ────────────────────────────────────────────────────────────

class LayeredTissueHex:
    """Interactive Neo-Hookean layered tissue block with draggable palpation point."""

    def __init__(self, device="cpu"):
        self.device = device
        wp.init()

        # Build HexTissueModel
        self.tissue_model = HexTissueModel.layered_block(
            res=RES, size=BLOCK_SIZE, layers=LAYERS, fixed_bottom=True)
        print(repr(self.tissue_model))

        nx, ny, nz = RES

        # warp.fem geometry
        self.geo = fem.Grid3D(
            res=wp.vec3i(nx, ny, nz),
            bounds_lo=wp.vec3(0.0, 0.0, 0.0),
            bounds_hi=wp.vec3(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE),
        )

        # Displacement space: degree-2 serendipity (20-node hex)
        self.u_space = fem.make_polynomial_space(
            self.geo, degree=2, dtype=wp.vec3,
            element_basis=fem.ElementBasis.SERENDIPITY)
        self.u_field = self.u_space.make_field()
        self.n_dof   = self.u_space.node_count()

        # Volume domain, test, and trial
        domain     = fem.Cells(geometry=self.geo)
        self.test  = fem.make_test(space=self.u_space, domain=domain)
        self.trial = fem.make_trial(space=self.u_space, domain=domain)

        # Material fields from HexTissueModel
        mat = self.tissue_model.material
        mat_space = fem.make_polynomial_space(
            self.geo, degree=0, discontinuous=True, dtype=float)
        self.mu_field  = mat_space.make_field()
        self.lam_field = mat_space.make_field()
        k_mu, k_lam = mat.to_lame_arrays(self.tissue_model.num_elements)
        self.mu_field.dof_values.assign(
            wp.array(k_mu.astype(np.float64), dtype=float))
        self.lam_field.dof_values.assign(
            wp.array(k_lam.astype(np.float64), dtype=float))

        # DOF rest positions via L2 projection (vec3 space, all axes at once)
        M = fem.integrate(_pos_mass, fields={'u': self.trial, 'v': self.test})
        b = fem.integrate(_pos_rhs, fields={'v': self.test}, output_dtype=wp.vec3d)
        pos = wp.zeros(self.n_dof, dtype=wp.vec3d)
        fem_utils.bsr_cg(M, b=b, x=pos, quiet=True, tol=1e-10)
        self.dof_positions = pos.numpy().astype(np.float64)

        # Identify boundary DOFs
        tol = BLOCK_SIZE / max(RES) * 0.1
        self.bottom_dofs = [i for i in range(self.n_dof)
                            if self.dof_positions[i][1] < tol]

        # Palpation DOF: top-surface corner nearest (+x, top, +z) — easy to grab
        top = [i for i in range(self.n_dof)
               if self.dof_positions[i][1] > BLOCK_SIZE - tol]
        self.palp_dof = top[int(np.argmin(
            [np.linalg.norm(self.dof_positions[i] -
                            np.array([BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE]))
             for i in top]))]
        self.bc_dofs = set(self.bottom_dofs) | {self.palp_dof}

        # Bottom BC projector
        boundary = fem.BoundarySides(self.geo)
        bd_test  = fem.make_test(space=self.u_space, domain=boundary)
        bd_trial = fem.make_trial(space=self.u_space, domain=boundary)
        self.bc_matrix = fem.integrate(
            _bottom_projector,
            fields={'u': bd_trial, 'v': bd_test}, assembly='nodal')

        # Build VTK surface connectivity (3×3 serendipity subdivision)
        self._build_subdivided_surface(n_sub=3)

        # Initial solve
        u_vals = self.u_field.dof_values.numpy()
        u_vals[self.palp_dof] = np.array([0.0, -0.005, 0.0], dtype=np.float32)
        self.u_field.dof_values.assign(wp.array(u_vals, dtype=wp.vec3))
        self._newton_solve(n_load_steps=4)

        self._last_converged_u      = self.u_field.dof_values.numpy().copy()
        self._last_converged_corner = self.u_field.dof_values.numpy()[self.palp_dof].copy()

        self.vtk_model     = None
        self.fiducial_list = None
        self._updating     = False

    # ── Subdivided surface (3×3 per element face, serendipity interpolation) ──

    def _build_subdivided_surface(self, n_sub=3):
        """Build visualization mesh with n_sub×n_sub cells per hex face.

        Uses 8-node serendipity shape functions to sample the deformation at
        (n_sub+1)² points per face, capturing the quadratic displacement field.
        """
        nx, ny, nz = RES
        dx, dy, dz = BLOCK_SIZE/nx, BLOCK_SIZE/ny, BLOCK_SIZE/nz

        from scipy.spatial import cKDTree
        tree = cKDTree(self.dof_positions)

        def N8(r, s):
            """Serendipity 8-node 2D shape functions in [-1,1]²."""
            return np.array([
                (1-r)*(1-s)*(-1-r-s)/4,  # corner (-1,-1)
                (1+r)*(1-s)*(-1+r-s)/4,  # corner (+1,-1)
                (1+r)*(1+s)*(-1+r+s)/4,  # corner (+1,+1)
                (1-r)*(1+s)*(-1-r+s)/4,  # corner (-1,+1)
                (1-r**2)*(1-s)/2,          # mid (0,-1)
                (1+r)*(1-s**2)/2,          # mid (+1, 0)
                (1-r**2)*(1+s)/2,          # mid (0,+1)
                (1-r)*(1-s**2)/2,          # mid (-1, 0)
            ])

        # Precompute weights on the (n_sub+1)² parametric grid
        t = np.linspace(-1, 1, n_sub + 1)
        # wts[ri, si] = N8(t[ri], t[si]), shape (n_sub+1, n_sub+1, 8)
        wts = np.array([[N8(r, s) for s in t] for r in t])

        all_rest = []   # (N_total, 3)
        all_dofs = []   # (N_total, 8) — DOF indices
        all_wts  = []   # (N_total, 8) — serendipity weights
        surface_quads = []

        def add_face(c4):
            """c4: (4,3) corners in CCW order (consistent outward normal)."""
            # 8 serendipity nodes: 4 corners + 4 edge midpoints
            nodes = np.array([c4[0], c4[1], c4[2], c4[3],
                               (c4[0]+c4[1])/2, (c4[1]+c4[2])/2,
                               (c4[2]+c4[3])/2, (c4[3]+c4[0])/2])
            _, dof_idx = tree.query(nodes)

            start = len(all_rest)
            for ri in range(n_sub + 1):
                for si in range(n_sub + 1):
                    w = wts[ri, si]          # (8,)
                    all_rest.append(w @ nodes)  # (3,) interpolated rest pos
                    all_dofs.append(dof_idx)
                    all_wts.append(w)

            for ri in range(n_sub):
                for si in range(n_sub):
                    p00 = start +  ri   *(n_sub+1) + si
                    p10 = start +  ri   *(n_sub+1) + si + 1
                    p11 = start + (ri+1)*(n_sub+1) + si + 1
                    p01 = start + (ri+1)*(n_sub+1) + si
                    surface_quads.append((p00, p10, p11, p01))

        # ── 6 outer faces ──────────────────────────────────────────────────
        for i in range(nx):
            for k in range(nz):
                # Bottom (Y=0)
                add_face(np.array([[i*dx,0,k*dz],[( i+1)*dx,0,k*dz],
                                   [(i+1)*dx,0,(k+1)*dz],[i*dx,0,(k+1)*dz]]))
                # Top (Y=ny*dy)
                add_face(np.array([[i*dx,ny*dy,(k+1)*dz],[(i+1)*dx,ny*dy,(k+1)*dz],
                                   [(i+1)*dx,ny*dy,k*dz],[i*dx,ny*dy,k*dz]]))
        for i in range(nx):
            for j in range(ny):
                # Front (Z=0)
                add_face(np.array([[i*dx,j*dy,0],[i*dx,(j+1)*dy,0],
                                   [(i+1)*dx,(j+1)*dy,0],[(i+1)*dx,j*dy,0]]))
                # Back (Z=nz*dz)
                add_face(np.array([[(i+1)*dx,j*dy,nz*dz],[(i+1)*dx,(j+1)*dy,nz*dz],
                                   [i*dx,(j+1)*dy,nz*dz],[i*dx,j*dy,nz*dz]]))
        for j in range(ny):
            for k in range(nz):
                # Left (X=0)
                add_face(np.array([[0,j*dy,(k+1)*dz],[0,(j+1)*dy,(k+1)*dz],
                                   [0,(j+1)*dy,k*dz],[0,j*dy,k*dz]]))
                # Right (X=nx*dx)
                add_face(np.array([[nx*dx,j*dy,k*dz],[nx*dx,(j+1)*dy,k*dz],
                                   [nx*dx,(j+1)*dy,(k+1)*dz],[nx*dx,j*dy,(k+1)*dz]]))

        self.sample_rest = np.array(all_rest, dtype=np.float64)  # (N, 3)
        self.sample_dofs = np.array(all_dofs, dtype=np.int32)    # (N, 8)
        self.sample_wts  = np.array(all_wts,  dtype=np.float64)  # (N, 8)
        self.surface_quads = surface_quads
        n_faces = 2*(nx*nz + nx*ny + ny*nz)
        print(f"  Surface: {n_faces} faces × {n_sub}×{n_sub} = "
              f"{len(surface_quads)//2} quads, {len(surface_quads)} tris, "
              f"{len(all_rest)} sample pts")

    def _deformed_samples_mm(self):
        """Deformed positions of all surface sample points in mm.

        Uses serendipity interpolation: pos = rest + Σ_i N_i · u[DOF_i]
        Fully vectorised via einsum — (N_samples, 8) × (N_samples, 8, 3).
        """
        u_vals = self.u_field.dof_values.numpy()                      # (n_dof, 3)
        u_at   = u_vals[self.sample_dofs].astype(np.float64)          # (N, 8, 3)
        disp   = np.einsum('ni,nij->nj', self.sample_wts, u_at)       # (N, 3)
        return (self.sample_rest + disp) * MM_PER_M

    # ── Newton-Raphson solvers ────────────────────────────────────────────

    def _project_dof_bc(self, K, rhs, dof_idx):
        """Zero row/col of BSR matrix for prescribed DOF, set diagonal to I."""
        offsets = K.offsets.numpy()
        columns = K.columns.numpy()
        values  = K.values.numpy()
        rs, re  = offsets[dof_idx], offsets[dof_idx + 1]
        for k in range(rs, re):
            values[k] = (np.eye(3, dtype=values.dtype)
                         if columns[k] == dof_idx
                         else np.zeros((3, 3), dtype=values.dtype))
        n_rows = len(offsets) - 1
        for row in range(n_rows):
            if row == dof_idx:
                continue
            for k in range(offsets[row], offsets[row + 1]):
                if columns[k] == dof_idx:
                    values[k] = np.zeros((3, 3), dtype=values.dtype)
                    break
        K.values.assign(wp.array(values, dtype=K.values.dtype))
        rhs_np = rhs.numpy()
        rhs_np[dof_idx] = [0., 0., 0.]
        rhs.assign(wp.array(rhs_np, dtype=rhs.dtype))

    def _newton_solve(self, n_load_steps=6, max_newton=12, tol=1e-3):
        """Full incremental Newton-Raphson from zero."""
        u_target = self.u_field.dof_values.numpy().copy()
        self.u_field.dof_values.fill_(wp.vec3(0, 0, 0))
        for step in range(n_load_steps):
            frac   = (step + 1) / n_load_steps
            u_vals = self.u_field.dof_values.numpy()
            u_vals[self.palp_dof] = u_target[self.palp_dof] * frac
            for di in self.bottom_dofs:
                u_vals[di] = [0, 0, 0]
            self.u_field.dof_values.assign(wp.array(u_vals, dtype=wp.vec3))
            for _ in range(max_newton):
                K = fem.integrate(_tangent_stiffness,
                    fields={'u': self.trial, 'v': self.test, 'u_cur': self.u_field,
                            'mu_f': self.mu_field, 'lam_f': self.lam_field})
                f_int = fem.integrate(_internal_force,
                    fields={'v': self.test, 'u_cur': self.u_field,
                            'mu_f': self.mu_field, 'lam_f': self.lam_field},
                    output_dtype=wp.vec3d)
                rhs_np = -f_int.numpy()
                for di in self.bc_dofs:
                    rhs_np[di] = [0., 0., 0.]
                rhs = wp.array(rhs_np, dtype=wp.vec3d)
                fem.project_linear_system(K, rhs, self.bc_matrix)
                self._project_dof_bc(K, rhs, self.palp_dof)
                du = wp.zeros(self.n_dof, dtype=wp.vec3d)
                fem_utils.bsr_cg(K, b=rhs, x=du, quiet=True, tol=1e-10, max_iters=500)
                du_np = du.numpy()
                if np.isnan(np.linalg.norm(du_np)):
                    break
                u_vals = self.u_field.dof_values.numpy()
                u_vals += du_np.astype(np.float32)
                u_vals[self.palp_dof] = u_target[self.palp_dof] * frac
                for di in self.bottom_dofs:
                    u_vals[di] = [0, 0, 0]
                self.u_field.dof_values.assign(wp.array(u_vals, dtype=wp.vec3))
                if np.linalg.norm(du_np) < tol:
                    break

    def _newton_solve_interactive(self, n_sub=6, max_newton=15, tol=1e-3):
        """Incremental Newton from last converged state to new target BC.

        Interpolates from last converged corner to target over n_sub sub-steps.
        """
        target = self.u_field.dof_values.numpy()[self.palp_dof].copy()
        start  = self._last_converged_corner.copy()

        # Determine sub-steps based on displacement magnitude
        disp_change = np.linalg.norm(target - start)
        step_limit  = BLOCK_SIZE / max(RES)   # one element width per step
        n_sub = max(n_sub, int(np.ceil(disp_change / step_limit)) if disp_change > 0 else n_sub)

        last_good = -1
        for sub in range(n_sub):
            frac   = (sub + 1) / n_sub
            corner = start + (target - start) * frac
            backup = self.u_field.dof_values.numpy().copy()

            u_vals = self.u_field.dof_values.numpy()
            u_vals[self.palp_dof] = corner
            for di in self.bottom_dofs:
                u_vals[di] = [0, 0, 0]
            self.u_field.dof_values.assign(wp.array(u_vals, dtype=wp.vec3))

            converged = False
            for _ in range(max_newton):
                K = fem.integrate(_tangent_stiffness,
                    fields={'u': self.trial, 'v': self.test, 'u_cur': self.u_field,
                            'mu_f': self.mu_field, 'lam_f': self.lam_field})
                f_int = fem.integrate(_internal_force,
                    fields={'v': self.test, 'u_cur': self.u_field,
                            'mu_f': self.mu_field, 'lam_f': self.lam_field},
                    output_dtype=wp.vec3d)

                rhs_np = -f_int.numpy()
                for di in self.bc_dofs:
                    rhs_np[di] = [0., 0., 0.]
                rhs = wp.array(rhs_np, dtype=wp.vec3d)

                fem.project_linear_system(K, rhs, self.bc_matrix)
                self._project_dof_bc(K, rhs, self.palp_dof)

                du = wp.zeros(self.n_dof, dtype=wp.vec3d)
                fem_utils.bsr_cg(K, b=rhs, x=du, quiet=True, tol=1e-8, max_iters=1000)
                du_np   = du.numpy()
                du_norm = np.linalg.norm(du_np)

                if np.isnan(du_norm) or du_norm > 1e6:
                    break

                u_vals = self.u_field.dof_values.numpy()
                u_vals += du_np.astype(np.float32)
                u_vals[self.palp_dof] = corner
                for di in self.bottom_dofs:
                    u_vals[di] = [0, 0, 0]
                self.u_field.dof_values.assign(wp.array(u_vals, dtype=wp.vec3))

                if du_norm < tol:
                    converged = True
                    break

            if not converged:
                self.u_field.dof_values.assign(wp.array(backup, dtype=wp.vec3))
                break
            last_good = sub

        if last_good >= 0:
            self._last_converged_u = self.u_field.dof_values.numpy().copy()
            frac = (last_good + 1) / n_sub
            self._last_converged_corner = (start + (target - start) * frac).copy()

    # ── Slicer visualization ──────────────────────────────────────────────

    def createModel(self):
        import vtk, slicer
        pts_np = self._deformed_samples_mm()
        vtk_pts = vtk.vtkPoints()
        for p in pts_np:
            vtk_pts.InsertNextPoint(*p)
        cells = vtk.vtkCellArray()
        for q in self.surface_quads:
            cells.InsertNextCell(4)
            for vi in q: cells.InsertCellPoint(vi)
        self._raw_poly = vtk.vtkPolyData()
        self._raw_poly.SetPoints(vtk_pts)
        self._raw_poly.SetPolys(cells)
        # Smooth normals within each element face; sharp at element boundaries
        nf = vtk.vtkPolyDataNormals()
        nf.SetInputData(self._raw_poly)
        nf.SetFeatureAngle(30.0)   # keep sharp edges at element boundaries
        nf.SplittingOn()
        nf.ComputePointNormalsOn()
        nf.Update()
        self._normals_filter = nf
        node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', 'LayeredTissueHex')
        node.SetAndObservePolyData(nf.GetOutput())
        node.CreateDefaultDisplayNodes()
        dn = node.GetDisplayNode()
        dn.SetColor(0.85, 0.75, 0.65)
        dn.SetOpacity(0.85)
        dn.SetEdgeVisibility(1)
        dn.SetEdgeColor(0.25, 0.25, 0.25)
        dn.SetBackfaceCulling(0)
        self.vtk_model = node

    def updateModel(self):
        pts_np  = self._deformed_samples_mm()
        vtk_pts = self._raw_poly.GetPoints()
        for i, p in enumerate(pts_np):
            vtk_pts.SetPoint(i, *p)
        self._raw_poly.GetPoints().Modified()
        self._raw_poly.Modified()
        self._normals_filter.Update()
        self.vtk_model.GetPolyData().Modified()

    def createControlPoints(self):
        import slicer
        ml = slicer.modules.markups.logic()
        slicer.mrmlScene.StartState(slicer.mrmlScene.BatchProcessState)
        self.fiducial_list = ml.AddNewMarkupsNode(
            'vtkMRMLMarkupsFiducialNode', 'TissueHex_controls')
        dn = self.fiducial_list.GetDisplayNode()
        dn.SetTextScale(0.0)
        dn.SetGlyphScale(5.0)
        dn.SetGlyphTypeFromString('Sphere3D')
        dn.SetColor(0.2, 0.2, 0.6)
        dn.SetSelectedColor(1.0, 0.9, 0.1)   # yellow = draggable
        dn.SetActiveColor(1.0, 0.6, 0.0)
        dn.SetSnapMode(
            slicer.vtkMRMLMarkupsFiducialDisplayNode.SnapModeUnconstrained)
        dn.SetVisibility(True)
        u_vals = self.u_field.dof_values.numpy()
        # Show 4 corner bottom fixed markers (positions in mm)
        self._bottom_fid_indices = []
        for di in sorted(self.bottom_dofs)[:4]:
            pos_mm = (self.dof_positions[di] + u_vals[di].astype(np.float64)) * MM_PER_M
            self.fiducial_list.AddControlPoint(*pos_mm.tolist())
            idx = self.fiducial_list.GetNumberOfControlPoints() - 1
            self.fiducial_list.SetNthControlPointLabel(idx, '')
            self.fiducial_list.SetNthControlPointLocked(idx, True)
            self.fiducial_list.SetNthControlPointSelected(idx, False)
            self._bottom_fid_indices.append(idx)
        # Palpation corner — draggable (yellow)
        palp_pos_mm = (self.dof_positions[self.palp_dof] +
                       u_vals[self.palp_dof].astype(np.float64)) * MM_PER_M
        self.fiducial_list.AddControlPoint(*palp_pos_mm.tolist())
        self._palp_fid_idx = self.fiducial_list.GetNumberOfControlPoints() - 1
        self.fiducial_list.SetNthControlPointLabel(self._palp_fid_idx, 'drag me')
        self.fiducial_list.SetNthControlPointLocked(self._palp_fid_idx, False)
        self.fiducial_list.SetNthControlPointSelected(self._palp_fid_idx, True)
        self.fiducial_list.AddObserver(
            self.fiducial_list.PointEndInteractionEvent,
            lambda c, e: self.onControlPointMoved())
        slicer.mrmlScene.EndState(slicer.mrmlScene.BatchProcessState)

    def onControlPointMoved(self):
        """Re-solve when palpation fiducial is released after dragging."""
        if self._updating:
            return
        self._updating = True
        try:
            pt_mm = [0.0, 0.0, 0.0]
            self.fiducial_list.GetNthControlPointPosition(self._palp_fid_idx, pt_mm)
            # Convert mm -> m, compute displacement from rest position
            pt_m   = np.array(pt_mm) / MM_PER_M
            rest_m = self.dof_positions[self.palp_dof]
            disp_m = pt_m - rest_m
            # Full incremental solve from zero (reliable for any displacement)
            u_vals = np.zeros((self.n_dof, 3), dtype=np.float32)
            u_vals[self.palp_dof] = disp_m.astype(np.float32)
            self.u_field.dof_values.assign(wp.array(u_vals, dtype=wp.vec3))
            self._newton_solve(n_load_steps=10)
            self._last_converged_u      = self.u_field.dof_values.numpy().copy()
            self._last_converged_corner = self.u_field.dof_values.numpy()[self.palp_dof].copy()
            self.updateModel()
            self._sync_fiducials()
        except Exception as e:
            import traceback
            print(f"onControlPointMoved error: {e}")
            traceback.print_exc()
        finally:
            self._updating = False

    def _sync_fiducials(self):
        """Write solved deformed positions (in mm) back to fiducials."""
        self._updating = True
        u_vals = self.u_field.dof_values.numpy()
        for fi, di in zip(self._bottom_fid_indices, sorted(self.bottom_dofs)[:4]):
            pos_mm = (self.dof_positions[di] + u_vals[di].astype(np.float64)) * MM_PER_M
            self.fiducial_list.SetNthControlPointPosition(fi, *pos_mm.tolist())
        palp_pos_mm = ((self.dof_positions[self.palp_dof]
                        + u_vals[self.palp_dof].astype(np.float64)) * MM_PER_M)
        self.fiducial_list.SetNthControlPointPosition(
            self._palp_fid_idx, *palp_pos_mm.tolist())
        self._updating = False

    def _apply_tissue_colors(self):
        """Color the mesh faces by tissue layer based on rest Y position."""
        import vtk, slicer

        layer_defs = [
            (0.000, 0.040, 0, "Liver",  (139/255, 69/255,  19/255)),
            (0.040, 0.055, 1, "Muscle", (210/255, 60/255,  60/255)),
            (0.055, 0.070, 2, "Fat",    (250/255,220/255, 120/255)),
            (0.070, 0.080, 3, "Skin",   (240/255,195/255, 160/255)),
        ]

        def layer_idx(y_m):
            # Clamp to block range to handle small DOF position errors
            y_c = max(0.0, min(float(y_m), BLOCK_SIZE))
            for y_lo, y_hi, idx, name, rgb in layer_defs[:-1]:
                if y_c < y_hi:
                    return idx
            return layer_defs[-1][2]  # skin catches everything above

        y_rest = self.sample_rest[:, 1]
        scalar_arr = vtk.vtkFloatArray()
        scalar_arr.SetName("TissueLayer")
        scalar_arr.SetNumberOfTuples(len(y_rest))
        for i, y in enumerate(y_rest):
            scalar_arr.SetValue(i, layer_idx(float(y)))

        self._raw_poly.GetPointData().SetScalars(scalar_arr)
        self._raw_poly.Modified()
        self._normals_filter.Update()
        self.vtk_model.GetPolyData().Modified()

        ct_name = 'TissueLayerColors'
        ct = slicer.mrmlScene.GetFirstNodeByName(ct_name)
        if ct is None:
            ct = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLColorTableNode', ct_name)
        ct.SetTypeToUser()
        ct.SetNumberOfColors(4)
        lut = ct.GetLookupTable()
        lut.SetNumberOfTableValues(4)
        for y_lo, y_hi, idx, name, rgb in layer_defs:
            ct.SetColor(idx, name, rgb[0], rgb[1], rgb[2], 1.0)
            lut.SetTableValue(idx, rgb[0], rgb[1], rgb[2], 1.0)
        lut.Build()

        dn = self.vtk_model.GetDisplayNode()
        dn.SetAndObserveColorNodeID(ct.GetID())
        dn.SetScalarVisibility(1)
        dn.SetActiveScalarName("TissueLayer")
        dn.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseManualScalarRange)
        dn.SetScalarRange(0, 3)

    def run(self):
        import slicer
        self.createModel()
        self.updateModel()
        self._apply_tissue_colors()
        self.createControlPoints()
        slicer.layered_hex = self
        slicer.app.layoutManager().setLayout(
            slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
        # Position camera to clearly show the palpation corner (upper-right)
        import vtk as _vtk
        v = slicer.app.layoutManager().threeDWidget(0).threeDView()
        renderer = v.renderWindow().GetRenderers().GetFirstRenderer()
        camera = renderer.GetActiveCamera()
        camera.SetPosition(200, -200, 200)
        camera.SetFocalPoint(40, 40, 40)
        camera.SetViewUp(0, 0, 1)
        renderer.ResetCamera()
        v.renderWindow().Render()
        print("Layered tissue hex simulation ready.")
        print(f"  DOFs: {self.n_dof}  Elements: {self.tissue_model.num_elements}")
        print(f"  Palpation DOF: {self.palp_dof}  "
              f"rest pos: {self.dof_positions[self.palp_dof]}")
        print("  Drag the yellow fiducial to palpate.")
        print("  Access via: slicer.layered_hex")


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == '__main__' or ('slicer' in dir() and slicer is not None):
    import slicer
    slicer.mrmlScene.Clear(0)
    sim = LayeredTissueHex(device="cpu")
    sim.run()
