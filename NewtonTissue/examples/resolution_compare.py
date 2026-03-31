"""Compare three mesh resolutions of anisotropic layered tissue (20-node hex).

Three identical tissue cubes side-by-side along X, with the same material
(anisotropic HGO layers + u-p formulation) but different element counts:
  LEFT   — Low    resolution
  CENTRE — Medium resolution
  RIGHT  — High   resolution

A single shared fiducial controls the palpation boundary condition for all
three models.  After dragging the yellow sphere, the models are solved
*sequentially* (Low → Medium → High) so the visual effect of mesh density is
obvious.

If a model fails to converge the markup is *not* snapped back; the same
user-specified displacement is forwarded to the remaining models.

Usage (Slicer Python console):
  exec(open('.../NewtonTissue/examples/resolution_compare.py').read())

Access afterwards:  slicer.res_compare
"""
from __future__ import annotations
import os, sys
import numpy as np

_SCRIPT_DIR = (os.path.dirname(os.path.abspath(__file__))
               if '__file__' in dir() else os.getcwd())
_NEWTON_DIR = (os.path.dirname(_SCRIPT_DIR)
               if os.path.basename(_SCRIPT_DIR) == 'examples' else _SCRIPT_DIR)
_SRC_DIR = os.path.join(_NEWTON_DIR, 'src')
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import warp as wp
import warp.fem as fem
import warp.examples.fem.utils as fem_utils
from newton_tissue import HexTissueModel, AnisotropicMaterial

# ── Configuration ────────────────────────────────────────────────────────────
BLOCK_SIZE   = 0.08        # cube side length [m]  (= 80 mm)
MM_PER_M     = 1000.0

# Low / Medium / High element resolutions  (nx, ny, nz)
CONFIGS = [
    ('Low',    (1, 2, 1),  (0.85, 0.60, 0.60)),   # warm red
    ('Medium', (2, 4, 2),  (0.60, 0.82, 0.60)),   # warm green
    ('High',   (4, 8, 4),  (0.60, 0.60, 0.85)),   # warm blue
]

SEPARATION_M = BLOCK_SIZE * 1.5   # centre-to-centre x gap [m]

# Same anisotropic layers as layered_aniso_hex.py
ANISO_LAYERS = [
    (0.000, 0.040, 10_000., 0.45, 0.,     1., [1., 0., 0.], 1060.),
    (0.040, 0.055, 60_000., 0.40, 5_000., 3., [1., 0., 0.], 1050.),
    (0.055, 0.070,  3_000., 0.49, 1_000., 2., [1., 0., 0.],  900.),
    (0.070, 0.080,100_000., 0.40, 6_000., 3., [0.707, 0., 0.707], 1100.),
]

# ── Shared warp.fem integrands (u-p formulation, HGO fibre term) ─────────────
def _project_dof_bc(K, rhs, dof_idx):
    offsets = K.offsets.numpy(); columns = K.columns.numpy(); values = K.values.numpy()
    rs, re = offsets[dof_idx], offsets[dof_idx + 1]
    for k in range(rs, re):
        values[k] = (np.eye(3, dtype=values.dtype) if columns[k] == dof_idx
                     else np.zeros((3, 3), dtype=values.dtype))
    for row in range(len(offsets) - 1):
        if row == dof_idx:
            continue
        for k in range(offsets[row], offsets[row + 1]):
            if columns[k] == dof_idx:
                values[k] = np.zeros((3, 3), dtype=values.dtype)
                break
    K.values.assign(wp.array(values, dtype=K.values.dtype, device=K.values.device))
    rhs_np = rhs.numpy(); rhs_np[dof_idx] = [0., 0., 0.]
    rhs.assign(wp.array(rhs_np, dtype=rhs.dtype, device=rhs.device))


@fem.integrand
def _pos_mass(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(u(s), v(s))


@fem.integrand
def _pos_rhs(s: fem.Sample, domain: fem.Domain, v: fem.Field):
    return wp.dot(fem.position(domain, s), v(s))


@fem.integrand
def _bottom_proj(s: fem.Sample, domain: fem.Domain, u: fem.Field, v: fem.Field):
    """Select the y=0 (bottom) face via its outward normal y-component."""
    nor = fem.normal(domain, s)
    return wp.max(0.0, -nor[1]) * wp.dot(u(s), v(s))


@fem.integrand
def _up_force(s: fem.Sample, v: fem.Field,
              u_cur: fem.Field, p_cur: fem.Field,
              mu_f: fem.Field, r_f: fem.Field,
              k1_f: fem.Field, k2_f: fem.Field, fiber_f: fem.Field):
    gv      = fem.grad(v, s)
    F       = wp.identity(n=3, dtype=float) + fem.grad(u_cur, s)
    J       = wp.determinant(F)
    J_s     = wp.max(J, 1e-4)
    F_inv_T = wp.transpose(wp.inverse(F))
    cof     = J * F_inv_T
    mu = mu_f(s); r = r_f(s); p = p_cur(s)
    I1    = wp.ddot(F, F)
    J_m23 = wp.exp(-2.0 / 3.0 * wp.log(J_s))
    P_dev = J_m23 * (mu * F - (mu * I1 / 3.0) * F_inv_T)
    P_vol = (p + r * (J - 1.0)) * cof
    a0 = fiber_f(s); k1 = k1_f(s); k2 = k2_f(s)
    C  = wp.transpose(F) @ F
    I4 = wp.dot(a0, C @ a0)
    E4 = wp.min(wp.max(I4 - 1.0, 0.0), wp.sqrt(20.0 / wp.max(k2, 0.001)))
    P_fib = 2.0 * k1 * E4 * wp.exp(k2 * E4 * E4) * (F @ wp.outer(a0, a0))
    return wp.ddot(P_dev + P_vol + P_fib, gv)


@fem.integrand
def _up_Kuu(s: fem.Sample, u: fem.Field, v: fem.Field,
            u_cur: fem.Field, p_cur: fem.Field,
            mu_f: fem.Field, r_f: fem.Field,
            k1_f: fem.Field, k2_f: fem.Field, fiber_f: fem.Field):
    gdu = fem.grad(u, s); gv = fem.grad(v, s)
    F   = wp.identity(n=3, dtype=float) + fem.grad(u_cur, s)
    J   = wp.determinant(F)
    J_s = wp.max(J, 1e-4)
    F_inv_T = wp.transpose(wp.inverse(F))
    mu = mu_f(s); r = r_f(s); p = p_cur(s)
    p_r    = p + r * (J - 1.0)
    J_m23  = wp.exp(-2.0 / 3.0 * wp.log(J_s))
    K_dev  = mu * J_m23 * wp.ddot(gdu, gv)
    FiTgdu = wp.ddot(F_inv_T, gdu)
    FiTgv  = wp.ddot(F_inv_T, gv)
    K_vol  = (p_r * J + r * J * J) * FiTgdu * FiTgv
    a0 = fiber_f(s); k1 = k1_f(s); k2 = k2_f(s)
    C  = wp.transpose(F) @ F
    I4 = wp.dot(a0, C @ a0)
    E4 = wp.min(wp.max(I4 - 1.0, 0.0), wp.sqrt(20.0 / wp.max(k2, 0.001)))
    d2W = k1 * (1.0 + 2.0 * k2 * E4 * E4) * wp.exp(k2 * E4 * E4)
    A   = wp.outer(a0, a0)
    return K_dev + K_vol + 4.0 * d2W * wp.ddot(A, gdu) * wp.ddot(A, gv)


@fem.integrand
def _J_m1_integral(s: fem.Sample, q: fem.Field, u_cur: fem.Field):
    F = wp.identity(n=3, dtype=float) + fem.grad(u_cur, s)
    return (wp.determinant(F) - 1.0) * q(s)


# ── Per-resolution model ──────────────────────────────────────────────────────
class ResolutionModel:
    """One anisotropic hex model at a given resolution and x position."""

    def __init__(self, name: str, res: tuple, x_offset_m: float, color: tuple, device: str):
        self.name       = name
        self.res        = res
        self.x_offset_m = x_offset_m
        self.color      = color
        self.device     = device

        nx, ny, nz = res
        tissue = HexTissueModel.anisotropic_layered_block(
            res=res, size=BLOCK_SIZE, aniso_layers=ANISO_LAYERS, fixed_bottom=True)

        with wp.ScopedDevice(device):
            self.geo = fem.Grid3D(
                res=wp.vec3i(nx, ny, nz),
                bounds_lo=wp.vec3(x_offset_m, 0., 0.),
                bounds_hi=wp.vec3(x_offset_m + BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

            self.u_space = fem.make_polynomial_space(
                self.geo, degree=2, dtype=wp.vec3,
                element_basis=fem.ElementBasis.SERENDIPITY)
            self.u_field = self.u_space.make_field()
            self.n_dof   = self.u_space.node_count()

            domain     = fem.Cells(geometry=self.geo)
            self.test  = fem.make_test(space=self.u_space, domain=domain)
            self.trial = fem.make_trial(space=self.u_space, domain=domain)

            mat = tissue.material; n_elem = tissue.num_elements
            sc  = fem.make_polynomial_space(self.geo, degree=0, discontinuous=True, dtype=float)
            fs  = fem.make_polynomial_space(self.geo, degree=0, discontinuous=True, dtype=wp.vec3)
            self.mu_field    = sc.make_field()
            self.lam_field   = sc.make_field()
            self.k1_field    = sc.make_field()
            self.k2_field    = sc.make_field()
            self.fiber_field = fs.make_field()
            k_mu, k_lam = mat.to_lame_arrays(n_elem)
            self.mu_field.dof_values.assign(   wp.array(k_mu.astype(np.float64),                 dtype=float))
            self.lam_field.dof_values.assign(  wp.array(k_lam.astype(np.float64),                dtype=float))
            self.k1_field.dof_values.assign(   wp.array(mat.get_k1().astype(np.float64),         dtype=float))
            self.k2_field.dof_values.assign(   wp.array(mat.get_k2().astype(np.float64),         dtype=float))
            self.fiber_field.dof_values.assign(wp.array(mat.get_fiber_dirs().astype(np.float32), dtype=wp.vec3))

            p_space        = fem.make_polynomial_space(self.geo, degree=0, discontinuous=True, dtype=float)
            self.p_field   = p_space.make_field()
            self.p_test    = fem.make_test(space=p_space, domain=domain)
            kappa_vals     = k_lam + (2.0 / 3.0) * k_mu
            self.kappa_field = sc.make_field()
            self.kappa_field.dof_values.assign(wp.array(kappa_vals.astype(np.float64), dtype=float))
            self.kappa_vals = kappa_vals.astype(np.float64)
            self.elem_vol   = float((BLOCK_SIZE / nx) * (BLOCK_SIZE / ny) * (BLOCK_SIZE / nz))

            # DOF rest positions via L2 projection
            M   = fem.integrate(_pos_mass, fields={'u': self.trial, 'v': self.test})
            b   = fem.integrate(_pos_rhs,  fields={'v': self.test}, output_dtype=wp.vec3d)
            pos = wp.zeros(self.n_dof, dtype=wp.vec3d)
            fem_utils.bsr_cg(M, b=b, x=pos, quiet=True, tol=1e-10)
            self.dof_positions = pos.numpy().astype(np.float64)

            tol_bc          = BLOCK_SIZE / max(res) * 0.1
            self.bottom_dofs = [i for i in range(self.n_dof)
                                if self.dof_positions[i][1] < tol_bc]
            top_dofs        = [i for i in range(self.n_dof)
                               if self.dof_positions[i][1] > BLOCK_SIZE - tol_bc]
            target_corner   = np.array([x_offset_m + BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE])
            self.palp_dof   = top_dofs[int(np.argmin(
                [np.linalg.norm(self.dof_positions[i] - target_corner) for i in top_dofs]))]
            self.bc_dofs    = set(self.bottom_dofs) | {self.palp_dof}

            boundary  = fem.BoundarySides(self.geo)
            bd_test   = fem.make_test(space=self.u_space, domain=boundary)
            bd_trial  = fem.make_trial(space=self.u_space, domain=boundary)
            self.bc_matrix = fem.integrate(
                _bottom_proj, fields={'u': bd_trial, 'v': bd_test}, assembly='nodal')

        # Subdivided surface for smooth rendering
        self._build_subdivided_surface(n_sub=3)

        # Initial small solve
        self._newton_solve_up(target_palp_disp=[0., -0.005, 0.], n_load_steps=4)
        self._last_good_u = self.u_field.dof_values.numpy().copy()
        self._last_good_p = self.p_field.dof_values.numpy().copy()

        self.vtk_model = None

    # ── Warm-start from a coarser converged model ─────────────────────────
    def _warm_start_from(self, donor: 'ResolutionModel'):
        """Initialise u and p fields by interpolating from a converged donor model.

        Both models share the same physical cube geometry; we work in the
        common local frame (subtract each model's x_offset from the x
        coordinate) so the DOF positions overlap correctly.
        """
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

        # Local-frame DOF positions (cube origin at 0,0,0 for both)
        donor_pts = donor.dof_positions.copy()
        donor_pts[:, 0] -= donor.x_offset_m
        self_pts = self.dof_positions.copy()
        self_pts[:, 0] -= self.x_offset_m

        donor_u = donor._last_good_u          # (n_donor, 3) float32

        # Interpolate each displacement component independently
        u_interp = np.zeros((self.n_dof, 3), dtype=np.float32)
        for comp in range(3):
            vals_d = donor_u[:, comp].astype(np.float64)
            lin  = LinearNDInterpolator(donor_pts, vals_d)
            vals = lin(self_pts)
            # Nearest-neighbour fill for any points outside the convex hull
            nan_mask = np.isnan(vals)
            if nan_mask.any():
                nn = NearestNDInterpolator(donor_pts, vals_d)
                vals[nan_mask] = nn(self_pts[nan_mask])
            u_interp[:, comp] = vals.astype(np.float32)

        # Enforce BCs: bottom fixed, palp free (will be corrected by solver)
        for di in self.bottom_dofs:
            u_interp[di] = [0., 0., 0.]

        # Transfer pressure via nearest element centre
        dnx, dny, dnz = donor.res
        snx, sny, snz = self.res
        d_centers = np.array([
            [(i + 0.5) * BLOCK_SIZE / dnx,
             (j + 0.5) * BLOCK_SIZE / dny,
             (k + 0.5) * BLOCK_SIZE / dnz]
            for i in range(dnx) for j in range(dny) for k in range(dnz)
        ])
        s_centers = np.array([
            [(i + 0.5) * BLOCK_SIZE / snx,
             (j + 0.5) * BLOCK_SIZE / sny,
             (k + 0.5) * BLOCK_SIZE / snz]
            for i in range(snx) for j in range(sny) for k in range(snz)
        ])
        nn_p    = NearestNDInterpolator(d_centers, donor._last_good_p)
        p_interp = nn_p(s_centers).astype(np.float64)

        # Apply to warp fields and save as the new last-good state
        wp.set_device(self.device)
        self.u_field.dof_values.assign(wp.array(u_interp, dtype=wp.vec3))
        self.p_field.dof_values.assign(wp.array(p_interp, dtype=float))
        self._last_good_u = u_interp.copy()
        self._last_good_p = p_interp.copy()

        palp_mm = float(np.linalg.norm(u_interp[self.palp_dof])) * MM_PER_M
        print(f'    warm-started {self.name} from {donor.name} '
              f'(palp ≈ {palp_mm:.2f} mm)')

    # ── Material field dict ───────────────────────────────────────────────
    def _up_fields(self):
        return {'mu_f': self.mu_field, 'r_f': self.kappa_field,
                'k1_f': self.k1_field, 'k2_f': self.k2_field, 'fiber_f': self.fiber_field}

    # ── Batch Newton solve (used for initialisation) ──────────────────────
    def _newton_solve_up(self, target_palp_disp, n_load_steps=4, max_newton=12, tol=5e-3):
        wp.set_device(self.device)
        u_current    = self.u_field.dof_values.numpy().copy()
        current_palp = u_current[self.palp_dof].copy()
        target_palp  = np.asarray(target_palp_disp, dtype=np.float32)
        delta        = target_palp - current_palp
        if float(np.linalg.norm(delta)) < 1e-7:
            return
        last_good_u = u_current.copy()
        last_good_p = self.p_field.dof_values.numpy().copy()
        for step in range(n_load_steps):
            frac   = (step + 1) / n_load_steps
            corner = current_palp + delta * frac
            u_vals = last_good_u.copy()
            u_vals[self.palp_dof] = corner
            for di in self.bottom_dofs: u_vals[di] = [0, 0, 0]
            self.u_field.dof_values.assign(wp.array(u_vals, dtype=wp.vec3))
            self.p_field.dof_values.assign(wp.array(last_good_p, dtype=float))
            prev_du = float('inf'); converged = False
            for _ in range(max_newton):
                K  = fem.integrate(_up_Kuu,   fields={'u': self.trial, 'v': self.test,
                                   'u_cur': self.u_field, 'p_cur': self.p_field, **self._up_fields()})
                fi = fem.integrate(_up_force, fields={'v': self.test,
                                   'u_cur': self.u_field, 'p_cur': self.p_field,
                                   **self._up_fields()}, output_dtype=wp.vec3d)
                rhs_np = -fi.numpy()
                for di in self.bc_dofs: rhs_np[di] = [0., 0., 0.]
                rhs = wp.array(rhs_np, dtype=wp.vec3d)
                fem.project_linear_system(K, rhs, self.bc_matrix)
                _project_dof_bc(K, rhs, self.palp_dof)
                du = wp.zeros(self.n_dof, dtype=wp.vec3d)
                fem_utils.bsr_cg(K, b=rhs, x=du, quiet=True, tol=1e-10, max_iters=500)
                du_np = du.numpy(); dn = float(np.linalg.norm(du_np))
                if np.isnan(dn) or dn > 1e4:
                    self.u_field.dof_values.assign(wp.array(last_good_u, dtype=wp.vec3))
                    self.p_field.dof_values.assign(wp.array(last_good_p, dtype=float)); return
                alpha = 1.0
                if dn > prev_du * 1.5: alpha = 0.5
                if dn > prev_du * 3.0: alpha = 0.25
                u_vals = self.u_field.dof_values.numpy()
                u_vals += (du_np * alpha).astype(np.float32)
                u_vals[self.palp_dof] = corner
                for di in self.bottom_dofs: u_vals[di] = [0, 0, 0]
                self.u_field.dof_values.assign(wp.array(u_vals, dtype=wp.vec3))
                prev_du = dn
                if dn < tol: converged = True; break
            if converged:
                Jm1 = fem.integrate(_J_m1_integral,
                    fields={'q': self.p_test, 'u_cur': self.u_field},
                    output_dtype=float).numpy() / self.elem_vol
                pv = self.p_field.dof_values.numpy(); pv += self.kappa_vals * Jm1
                self.p_field.dof_values.assign(wp.array(pv, dtype=float))
                last_good_u = self.u_field.dof_values.numpy().copy()
                last_good_p = self.p_field.dof_values.numpy().copy()
            else:
                self.u_field.dof_values.assign(wp.array(last_good_u, dtype=wp.vec3))
                self.p_field.dof_values.assign(wp.array(last_good_p, dtype=float)); break

    # ── Subdivided surface geometry ───────────────────────────────────────
    def _build_subdivided_surface(self, n_sub=3):
        from scipy.spatial import cKDTree
        nx, ny, nz = self.res
        x0 = self.x_offset_m
        dx, dy, dz = BLOCK_SIZE / nx, BLOCK_SIZE / ny, BLOCK_SIZE / nz
        tree = cKDTree(self.dof_positions)

        def N8(r, s):
            return np.array([
                (1-r)*(1-s)*(-1-r-s)/4, (1+r)*(1-s)*(-1+r-s)/4,
                (1+r)*(1+s)*(-1+r+s)/4, (1-r)*(1+s)*(-1-r+s)/4,
                (1-r**2)*(1-s)/2,        (1+r)*(1-s**2)/2,
                (1-r**2)*(1+s)/2,        (1-r)*(1-s**2)/2,
            ])

        t    = np.linspace(-1, 1, n_sub + 1)
        wts  = np.array([[N8(r, s) for s in t] for r in t])
        all_rest = []; all_dofs = []; all_wts = []; surface_quads = []

        def add_face(c4):
            c4 = [np.array(x) for x in c4]
            nodes = np.array([c4[0], c4[1], c4[2], c4[3],
                              (c4[0]+c4[1])/2, (c4[1]+c4[2])/2,
                              (c4[2]+c4[3])/2, (c4[3]+c4[0])/2])
            _, dof_idx = tree.query(nodes)
            start = len(all_rest)
            for ri in range(n_sub + 1):
                for si in range(n_sub + 1):
                    w = wts[ri, si]
                    all_rest.append(w @ nodes)
                    all_dofs.append(dof_idx)
                    all_wts.append(w)
            for ri in range(n_sub):
                for si in range(n_sub):
                    p00 = start + ri*(n_sub+1) + si; p10 = p00 + 1
                    p11 = start + (ri+1)*(n_sub+1) + si + 1; p01 = p11 - 1
                    surface_quads.append((p00, p10, p11, p01))

        # y=0 (bottom) and y=ny*dy (top)
        for i in range(nx):
            for k in range(nz):
                add_face([[x0+i*dx,     0,       k*dz],
                          [x0+(i+1)*dx, 0,       k*dz],
                          [x0+(i+1)*dx, 0,       (k+1)*dz],
                          [x0+i*dx,     0,       (k+1)*dz]])
                add_face([[x0+i*dx,     ny*dy,   (k+1)*dz],
                          [x0+(i+1)*dx, ny*dy,   (k+1)*dz],
                          [x0+(i+1)*dx, ny*dy,   k*dz],
                          [x0+i*dx,     ny*dy,   k*dz]])
        # z=0 (front) and z=nz*dz (back)
        for i in range(nx):
            for j in range(ny):
                add_face([[x0+i*dx,     j*dy,     0],
                          [x0+i*dx,     (j+1)*dy, 0],
                          [x0+(i+1)*dx, (j+1)*dy, 0],
                          [x0+(i+1)*dx, j*dy,     0]])
                add_face([[x0+(i+1)*dx, j*dy,     nz*dz],
                          [x0+(i+1)*dx, (j+1)*dy, nz*dz],
                          [x0+i*dx,     (j+1)*dy, nz*dz],
                          [x0+i*dx,     j*dy,     nz*dz]])
        # x=x0 (left) and x=x0+nx*dx (right)
        for j in range(ny):
            for k in range(nz):
                add_face([[x0,          j*dy,     (k+1)*dz],
                          [x0,          (j+1)*dy, (k+1)*dz],
                          [x0,          (j+1)*dy, k*dz],
                          [x0,          j*dy,     k*dz]])
                add_face([[x0+nx*dx,    j*dy,     k*dz],
                          [x0+nx*dx,    (j+1)*dy, k*dz],
                          [x0+nx*dx,    (j+1)*dy, (k+1)*dz],
                          [x0+nx*dx,    j*dy,     (k+1)*dz]])

        self.sample_rest  = np.array(all_rest, dtype=np.float64)
        self.sample_dofs  = np.array(all_dofs, dtype=np.int32)
        self.sample_wts   = np.array(all_wts,  dtype=np.float64)
        self.surface_quads = surface_quads

    # ── Slicer visualisation ──────────────────────────────────────────────
    def _deformed_samples_mm(self):
        u_vals = self.u_field.dof_values.numpy()
        u_at   = u_vals[self.sample_dofs].astype(np.float64)
        disp   = np.einsum('ni,nij->nj', self.sample_wts, u_at)
        return (self.sample_rest + disp) * MM_PER_M

    def _apply_tissue_colors(self):
        import vtk, slicer
        layer_defs = [
            (0.,     0.04,  0, "Liver",  (139/255, 69/255,  19/255)),
            (0.04,   0.055, 1, "Muscle", (210/255, 60/255,  60/255)),
            (0.055,  0.07,  2, "Fat",    (250/255, 220/255, 120/255)),
            (0.07,   0.08,  3, "Skin",   (240/255, 195/255, 160/255)),
        ]
        def layer_idx(y_m):
            y_c = max(0., min(float(y_m), BLOCK_SIZE))
            for y_lo, y_hi, idx, *_ in layer_defs[:-1]:
                if y_c < y_hi: return idx
            return layer_defs[-1][2]

        y_rest = self.sample_rest[:, 1]
        scalar_arr = vtk.vtkFloatArray(); scalar_arr.SetName('TissueLayer')
        scalar_arr.SetNumberOfTuples(len(y_rest))
        for i, y in enumerate(y_rest): scalar_arr.SetValue(i, layer_idx(float(y)))
        self._raw_poly.GetPointData().SetScalars(scalar_arr)
        self._raw_poly.Modified()
        self._normals_filter.Update()
        self.vtk_model.GetPolyData().Modified()

        ct_name = f'TissueLayerColors_{self.name}'
        ct = slicer.mrmlScene.GetFirstNodeByName(ct_name)
        if ct is None:
            ct = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLColorTableNode', ct_name)
        ct.SetTypeToUser(); ct.SetNumberOfColors(4)
        lut = ct.GetLookupTable(); lut.SetNumberOfTableValues(4)
        for y_lo, y_hi, idx, name, rgb in layer_defs:
            ct.SetColor(idx, name, rgb[0], rgb[1], rgb[2], 1.)
            lut.SetTableValue(idx, rgb[0], rgb[1], rgb[2], 1.)
        lut.Build()
        dn = self.vtk_model.GetDisplayNode()
        dn.SetAndObserveColorNodeID(ct.GetID()); dn.SetScalarVisibility(1)
        dn.SetActiveScalarName('TissueLayer')
        dn.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseManualScalarRange)
        dn.SetScalarRange(0, 3)

    def createModel(self):
        import vtk, slicer
        pts_np  = self._deformed_samples_mm()
        vtk_pts = vtk.vtkPoints()
        for p in pts_np: vtk_pts.InsertNextPoint(*p)
        cells = vtk.vtkCellArray()
        for q in self.surface_quads:
            cells.InsertNextCell(4)
            for vi in q: cells.InsertCellPoint(vi)
        self._raw_poly = vtk.vtkPolyData()
        self._raw_poly.SetPoints(vtk_pts); self._raw_poly.SetPolys(cells)
        nf = vtk.vtkPolyDataNormals()
        nf.SetInputData(self._raw_poly); nf.SetFeatureAngle(30.)
        nf.SplittingOn(); nf.ComputePointNormalsOn(); nf.Update()
        self._normals_filter = nf
        node_name = f'ResCompare_{self.name}'
        node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', node_name)
        node.SetAndObservePolyData(nf.GetOutput()); node.CreateDefaultDisplayNodes()
        dn = node.GetDisplayNode()
        dn.SetColor(*self.color); dn.SetOpacity(0.85)
        dn.SetEdgeVisibility(1); dn.SetEdgeColor(0.25, 0.25, 0.25)
        dn.SetBackfaceCulling(0)
        self.vtk_model = node

    def updateModel(self):
        if self.vtk_model is None: return
        pts_np  = self._deformed_samples_mm()
        vtk_pts = self._raw_poly.GetPoints()
        for i, p in enumerate(pts_np): vtk_pts.SetPoint(i, *p)
        self._raw_poly.GetPoints().Modified(); self._raw_poly.Modified()
        self._normals_filter.Update(); self.vtk_model.GetPolyData().Modified()


# ── Coordinator with shared markup ───────────────────────────────────────────
class ResolutionComparison:
    """Three resolution models sharing one palpation fiducial, solved sequentially."""

    def __init__(self, device=None):
        wp.init()
        if device is None:
            device = 'cuda:0' if wp.is_cuda_available() else 'cpu'
        self.device = device

        n = len(CONFIGS)
        total_span = (n - 1) * SEPARATION_M
        x_offsets  = [-total_span / 2 + i * SEPARATION_M for i in range(n)]

        self.models = []
        for i, (name, res, color) in enumerate(CONFIGS):
            print(f'  Building {name} model  res={res}  x_offset={x_offsets[i]*MM_PER_M:.1f} mm ...')
            m = ResolutionModel(name, res, x_offsets[i], color, device)
            self.models.append(m)

        # Shared markup rests at the HIGH model's palpation DOF rest position
        high = self.models[-1]
        self._markup_rest_m = high.dof_positions[high.palp_dof].copy()

        self.fiducial_list = None
        self._palp_fid_idx = None
        self._updating     = False
        self._solve_state  = {}

    # ── Shared fiducial ───────────────────────────────────────────────────
    def createSharedMarkup(self):
        import slicer
        ml = slicer.modules.markups.logic()
        slicer.mrmlScene.StartState(slicer.mrmlScene.BatchProcessState)
        self.fiducial_list = ml.AddNewMarkupsNode(
            'vtkMRMLMarkupsFiducialNode', 'ResCompare_BC')
        dn = self.fiducial_list.GetDisplayNode()
        dn.SetTextScale(3.0); dn.SetGlyphScale(6.0)
        dn.SetGlyphTypeFromString('Sphere3D')
        dn.SetColor(0.2, 0.2, 0.6)
        dn.SetSelectedColor(1.0, 0.9, 0.1)
        dn.SetActiveColor(1.0, 0.6, 0.0)
        dn.SetSnapMode(slicer.vtkMRMLMarkupsFiducialDisplayNode.SnapModeUnconstrained)
        dn.SetVisibility(True)
        ml.SetActiveListID(self.fiducial_list)

        rest_mm = self._markup_rest_m * MM_PER_M
        # Also account for the initial small displacement
        high = self.models[-1]
        u_vals = high.u_field.dof_values.numpy()
        init_u = u_vals[high.palp_dof].astype(np.float64)
        pos_mm = rest_mm + init_u * MM_PER_M

        self.fiducial_list.AddControlPoint(*pos_mm.tolist())
        self._palp_fid_idx = self.fiducial_list.GetNumberOfControlPoints() - 1
        self.fiducial_list.SetNthControlPointLabel(self._palp_fid_idx, 'drag me')
        self.fiducial_list.SetNthControlPointLocked(self._palp_fid_idx, False)
        self.fiducial_list.SetNthControlPointSelected(self._palp_fid_idx, True)

        self.fiducial_list.AddObserver(
            self.fiducial_list.PointEndInteractionEvent,
            lambda c, e: self.onMarkupMoved())
        slicer.mrmlScene.EndState(slicer.mrmlScene.BatchProcessState)

    def _get_target_disp_m(self):
        """Return displacement [m] from markup current position to rest position."""
        pt_mm = [0., 0., 0.]
        self.fiducial_list.GetNthControlPointPosition(self._palp_fid_idx, pt_mm)
        pt_m = np.array(pt_mm) / MM_PER_M
        return (pt_m - self._markup_rest_m).astype(np.float32)

    # ── Sequential solve control ──────────────────────────────────────────
    def onMarkupMoved(self):
        if self._updating:
            return
        target_disp = self._get_target_disp_m()
        if float(np.linalg.norm(target_disp)) < 1e-7:
            return
        self._updating = True
        self._solve_state = {'target_disp': target_disp}
        self._setup_model_solve(0)
        self._schedule_next_step()

    def _setup_model_solve(self, model_idx: int):
        """Populate _solve_state for model_idx and set step=0."""
        state = self._solve_state
        state['model_idx'] = model_idx
        if model_idx >= len(self.models):
            return
        model        = self.models[model_idx]
        target_disp  = state['target_disp']
        current_palp = model._last_good_u[model.palp_dof].copy()
        target_palp  = target_disp.astype(np.float32)
        delta        = target_palp - current_palp
        delta_mag    = float(np.linalg.norm(delta))
        n_steps      = max(4, int(delta_mag / 0.00025) + 1) if delta_mag > 1e-7 else 1
        state.update({
            'step':        0,
            'n_steps':     n_steps,
            'current_palp': current_palp,
            'delta':       delta,
            'last_good_u': model._last_good_u.copy(),
            'last_good_p': model._last_good_p.copy(),
        })

    def _schedule_next_step(self):
        try:
            import qt
            qt.QTimer.singleShot(0, self._do_one_step)
        except Exception:
            # Fallback: run synchronously when Qt is unavailable
            while self._updating:
                self._do_one_step()

    def _do_one_step(self):
        state     = self._solve_state
        model_idx = state.get('model_idx', len(self.models))

        if model_idx >= len(self.models):
            self._updating = False
            return

        model   = self.models[model_idx]
        step    = state['step']
        n_steps = state['n_steps']

        if step >= n_steps:
            self._finish_model(model_idx, success=True)
            return

        try:
            wp.set_device(self.device)
            frac   = (step + 1) / n_steps
            corner = state['current_palp'] + state['delta'] * frac

            u_vals = state['last_good_u'].copy()
            u_vals[model.palp_dof] = corner
            for di in model.bottom_dofs: u_vals[di] = [0, 0, 0]
            model.u_field.dof_values.assign(wp.array(u_vals, dtype=wp.vec3))
            model.p_field.dof_values.assign(wp.array(state['last_good_p'], dtype=float))

            prev_du = float('inf'); converged = False
            for _ in range(12):
                K  = fem.integrate(_up_Kuu,   fields={'u': model.trial, 'v': model.test,
                                   'u_cur': model.u_field, 'p_cur': model.p_field,
                                   **model._up_fields()})
                fi = fem.integrate(_up_force, fields={'v': model.test,
                                   'u_cur': model.u_field, 'p_cur': model.p_field,
                                   **model._up_fields()}, output_dtype=wp.vec3d)
                rhs_np = -fi.numpy()
                for di in model.bc_dofs: rhs_np[di] = [0., 0., 0.]
                rhs = wp.array(rhs_np, dtype=wp.vec3d)
                fem.project_linear_system(K, rhs, model.bc_matrix)
                _project_dof_bc(K, rhs, model.palp_dof)
                du = wp.zeros(model.n_dof, dtype=wp.vec3d)
                fem_utils.bsr_cg(K, b=rhs, x=du, quiet=True, tol=1e-10, max_iters=500)
                du_np = du.numpy(); dn = float(np.linalg.norm(du_np))
                if np.isnan(dn) or dn > 1e4:
                    model.u_field.dof_values.assign(wp.array(state['last_good_u'], dtype=wp.vec3))
                    model.p_field.dof_values.assign(wp.array(state['last_good_p'], dtype=float))
                    self._finish_model(model_idx, success=False)
                    return
                alpha = 1.0
                if dn > prev_du * 1.5: alpha = 0.5
                if dn > prev_du * 3.0: alpha = 0.25
                u_vals = model.u_field.dof_values.numpy()
                u_vals += (du_np * alpha).astype(np.float32)
                u_vals[model.palp_dof] = corner
                for di in model.bottom_dofs: u_vals[di] = [0, 0, 0]
                model.u_field.dof_values.assign(wp.array(u_vals, dtype=wp.vec3))
                prev_du = dn
                if dn < 5e-3: converged = True; break

            if converged:
                Jm1 = fem.integrate(_J_m1_integral,
                    fields={'q': model.p_test, 'u_cur': model.u_field},
                    output_dtype=float).numpy() / model.elem_vol
                pv = model.p_field.dof_values.numpy(); pv += model.kappa_vals * Jm1
                model.p_field.dof_values.assign(wp.array(pv, dtype=float))
                state['last_good_u'] = model.u_field.dof_values.numpy().copy()
                state['last_good_p'] = model.p_field.dof_values.numpy().copy()
                model.updateModel()
                self._render()
            else:
                # Revert this sub-step and stop for this model
                model.u_field.dof_values.assign(wp.array(state['last_good_u'], dtype=wp.vec3))
                model.p_field.dof_values.assign(wp.array(state['last_good_p'], dtype=float))
                self._finish_model(model_idx, success=False)
                return

            state['step'] += 1
            self._schedule_next_step()

        except Exception:
            import traceback; traceback.print_exc()
            self._finish_model(model_idx, success=False)

    def _finish_model(self, model_idx: int, success: bool):
        """Save converged state (or keep last good on failure), then move to next model."""
        state = self._solve_state
        model = self.models[model_idx]

        if success:
            model._last_good_u = state['last_good_u'].copy()
            model._last_good_p = state['last_good_p'].copy()
            print(f'  {model.name}: converged  (palp disp = '
                  f'{np.linalg.norm(model._last_good_u[model.palp_dof])*MM_PER_M:.2f} mm)')
        else:
            # Revert to last confirmed good state; markup stays where the user left it
            model.u_field.dof_values.assign(wp.array(model._last_good_u, dtype=wp.vec3))
            model.p_field.dof_values.assign(wp.array(model._last_good_p, dtype=float))
            achieved = np.linalg.norm(model._last_good_u[model.palp_dof]) * MM_PER_M
            target   = np.linalg.norm(state['target_disp']) * MM_PER_M
            print(f'  {model.name}: DID NOT CONVERGE  '
                  f'(achieved {achieved:.2f} mm / target {target:.2f} mm) — '
                  f'markup kept at user position')

        model.updateModel()
        self._render()

        # Advance to next model (still using the same user-specified target_disp)
        next_idx = model_idx + 1
        if next_idx >= len(self.models):
            self._updating = False
            return

        self._setup_model_solve(next_idx)
        self._schedule_next_step()

    @staticmethod
    def _render():
        try:
            import slicer as _s
            _s.app.layoutManager().threeDWidget(0).threeDView().renderWindow().Render()
            _s.app.processEvents()
        except Exception:
            pass

    # ── Top-level setup ───────────────────────────────────────────────────
    def run(self):
        import slicer
        for m in self.models:
            m.createModel()
            m.updateModel()
            m._apply_tissue_colors()
        self.createSharedMarkup()

        slicer.res_compare = self
        slicer.app.layoutManager().setLayout(
            slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
        v        = slicer.app.layoutManager().threeDWidget(0).threeDView()
        renderer = v.renderWindow().GetRenderers().GetFirstRenderer()
        camera   = renderer.GetActiveCamera()
        # Centre camera on the middle model
        cx = 0.0; cy = BLOCK_SIZE * MM_PER_M / 2; cz = BLOCK_SIZE * MM_PER_M / 2
        camera.SetFocalPoint(cx, cy, cz)
        camera.SetPosition(cx, cy - 350, cz + 200)
        camera.SetViewUp(0, 0, 1)
        renderer.ResetCamera()
        v.renderWindow().Render()

        print('\nResolution comparison ready.')
        for m in self.models:
            print(f'  {m.name:8s}  res={m.res}  DOFs={m.n_dof}  elements={m.res[0]*m.res[1]*m.res[2]}')
        print('\nDrag the yellow sphere to deform all three models.')
        print('Models solve sequentially: Low → Medium → High.')
        print('If a model cannot converge, the markup stays at your chosen position.')
        print('Access:  slicer.res_compare')


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__' or ('slicer' in dir() and slicer is not None):
    import slicer
    slicer.mrmlScene.Clear(0)
    print('=' * 60)
    print('Resolution comparison: Low / Medium / High mesh density')
    print('=' * 60)
    rc = ResolutionComparison()
    rc.run()
