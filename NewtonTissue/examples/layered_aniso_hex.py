"""Interactive anisotropic multi-layer tissue simulation (20-node serendipity hex).

Uses AnisotropicMaterial (HGO fiber term) and a mixed u-p formulation.
Incremental BC: dragging the fiducial advances in 0.25mm steps via QTimer.
No Uzawa outer loop -- single Newton per step with element-size step cap.
Access via: slicer.layered_aniso_hex
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
from newton_tissue import HexTissueModel, AnisotropicMaterial

BLOCK_SIZE = 0.08
RES        = (4, 8, 4)
MM_PER_M   = 1000.0

ANISO_LAYERS = [
    (0.000, 0.040, 10_000., 0.45, 0.,     1., [1.,0.,0.], 1060.),
    (0.040, 0.055, 60_000., 0.40, 5_000., 3., [1.,0.,0.], 1050.),
    (0.055, 0.070,  3_000., 0.49, 1_000., 2., [1.,0.,0.],  900.),
    (0.070, 0.080,100_000., 0.40, 6_000., 3., [0.707,0.,0.707], 1100.),
]

def _project_dof_bc(K, rhs, dof_idx):
    offsets = K.offsets.numpy(); columns = K.columns.numpy(); values = K.values.numpy()
    rs, re = offsets[dof_idx], offsets[dof_idx + 1]
    for k in range(rs, re):
        values[k] = (np.eye(3, dtype=values.dtype) if columns[k] == dof_idx
                     else np.zeros((3, 3), dtype=values.dtype))
    for row in range(len(offsets) - 1):
        if row == dof_idx: continue
        for k in range(offsets[row], offsets[row + 1]):
            if columns[k] == dof_idx:
                values[k] = np.zeros((3, 3), dtype=values.dtype); break
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
    p_r = p + r * (J - 1.0)
    J_m23 = wp.exp(-2.0 / 3.0 * wp.log(J_s))
    K_dev     = mu * J_m23 * wp.ddot(gdu, gv)
    FiTgdu    = wp.ddot(F_inv_T, gdu)
    FiTgv     = wp.ddot(F_inv_T, gv)
    # K_vol_sym only (Gauss-Newton for volumetric part).
    # K_vol_geo = -p_r*J*ddot(F^{-T}*gdu^T*F^{-T}, gv) is dropped:
    # when p_r > 0 (stretched elements) it makes K indefinite,
    # causing CG to find negative-curvature directions and diverge.
    K_vol_sym = (p_r * J + r * J * J) * FiTgdu * FiTgv
    a0 = fiber_f(s); k1 = k1_f(s); k2 = k2_f(s)
    C  = wp.transpose(F) @ F
    I4 = wp.dot(a0, C @ a0)
    E4 = wp.min(wp.max(I4 - 1.0, 0.0), wp.sqrt(20.0 / wp.max(k2, 0.001)))
    d2W = k1 * (1.0 + 2.0 * k2 * E4 * E4) * wp.exp(k2 * E4 * E4)
    A   = wp.outer(a0, a0)
    return K_dev + K_vol_sym + 4.0 * d2W * wp.ddot(A, gdu) * wp.ddot(A, gv)

@fem.integrand
def _J_m1_integral(s: fem.Sample, q: fem.Field, u_cur: fem.Field):
    F = wp.identity(n=3, dtype=float) + fem.grad(u_cur, s)
    return (wp.determinant(F) - 1.0) * q(s)


class LayeredAnisotropicTissueHex:

    def __init__(self, device=None):
        wp.init()
        if device is None:
            device = "cuda:0" if wp.is_cuda_available() else "cpu"
        self.device = device
        self.tissue_model = HexTissueModel.anisotropic_layered_block(
            res=RES, size=BLOCK_SIZE, aniso_layers=ANISO_LAYERS, fixed_bottom=True)
        print(repr(self.tissue_model))
        nx, ny, nz = RES
        with wp.ScopedDevice(self.device):
            self.geo = fem.Grid3D(res=wp.vec3i(nx, ny, nz),
                                  bounds_lo=wp.vec3(0., 0., 0.),
                                  bounds_hi=wp.vec3(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            self.u_space = fem.make_polynomial_space(self.geo, degree=2, dtype=wp.vec3,
                element_basis=fem.ElementBasis.SERENDIPITY)
            self.u_field = self.u_space.make_field()
            self.n_dof   = self.u_space.node_count()
            domain     = fem.Cells(geometry=self.geo)
            self.test  = fem.make_test(space=self.u_space, domain=domain)
            self.trial = fem.make_trial(space=self.u_space, domain=domain)
            mat = self.tissue_model.material; n = self.tissue_model.num_elements
            sc  = fem.make_polynomial_space(self.geo, degree=0, discontinuous=True, dtype=float)
            fs  = fem.make_polynomial_space(self.geo, degree=0, discontinuous=True, dtype=wp.vec3)
            self.mu_field    = sc.make_field(); self.lam_field   = sc.make_field()
            self.k1_field    = sc.make_field(); self.k2_field    = sc.make_field()
            self.fiber_field = fs.make_field()
            k_mu, k_lam = mat.to_lame_arrays(n)
            self.mu_field.dof_values.assign(wp.array(k_mu.astype(np.float64),         dtype=float))
            self.lam_field.dof_values.assign(wp.array(k_lam.astype(np.float64),        dtype=float))
            self.k1_field.dof_values.assign(wp.array(mat.get_k1().astype(np.float64),  dtype=float))
            self.k2_field.dof_values.assign(wp.array(mat.get_k2().astype(np.float64),  dtype=float))
            self.fiber_field.dof_values.assign(wp.array(mat.get_fiber_dirs().astype(np.float32), dtype=wp.vec3))
            p_space = fem.make_polynomial_space(self.geo, degree=0, discontinuous=True, dtype=float)
            self.p_field   = p_space.make_field()
            self.p_test    = fem.make_test(space=p_space, domain=domain)
            kappa_vals     = k_lam + (2.0 / 3.0) * k_mu
            self.kappa_field = sc.make_field()
            self.kappa_field.dof_values.assign(wp.array(kappa_vals.astype(np.float64), dtype=float))
            self.kappa_vals = kappa_vals.astype(np.float64)
            self.elem_vol   = float((BLOCK_SIZE/nx) * (BLOCK_SIZE/ny) * (BLOCK_SIZE/nz))
            M = fem.integrate(_pos_mass, fields={'u': self.trial, 'v': self.test})
            b = fem.integrate(_pos_rhs,  fields={'v': self.test}, output_dtype=wp.vec3d)
            pos = wp.zeros(self.n_dof, dtype=wp.vec3d)
            fem_utils.bsr_cg(M, b=b, x=pos, quiet=True, tol=1e-10)
            self.dof_positions = pos.numpy().astype(np.float64)
            tol_bc = BLOCK_SIZE / max(RES) * 0.1
            self.bottom_dofs = [i for i in range(self.n_dof) if self.dof_positions[i][1] < tol_bc]
            top = [i for i in range(self.n_dof) if self.dof_positions[i][1] > BLOCK_SIZE - tol_bc]
            self.palp_dof = top[int(np.argmin(
                [np.linalg.norm(self.dof_positions[i] - np.array([BLOCK_SIZE]*3)) for i in top]))]
            self.bc_dofs = set(self.bottom_dofs) | {self.palp_dof}
            boundary = fem.BoundarySides(self.geo)
            bd_test  = fem.make_test(space=self.u_space, domain=boundary)
            bd_trial = fem.make_trial(space=self.u_space, domain=boundary)
            self.bc_matrix = fem.integrate(_bottom_proj,
                fields={'u': bd_trial, 'v': bd_test}, assembly='nodal')
        self._build_subdivided_surface(n_sub=3)
        self._newton_solve_up(target_palp_disp=[0., -0.005, 0.], n_load_steps=4)
        self._last_converged_u = self.u_field.dof_values.numpy().copy()
        self._last_converged_p = self.p_field.dof_values.numpy().copy()
        self._inc              = {'steps_converged': 0}
        self.vtk_model = None; self.fiducial_list = None; self._updating = False

    def _up_fields(self):
        return {'mu_f': self.mu_field, 'r_f': self.kappa_field,
                'k1_f': self.k1_field, 'k2_f': self.k2_field, 'fiber_f': self.fiber_field}

    def _newton_solve_up(self, target_palp_disp, n_load_steps=None, max_newton=12, tol=5e-3):
        """Incremental Newton from current state. No Uzawa outer loop.
        Element-size step cap prevents oscillation.
        """
        wp.set_device(self.device)
        u_current    = self.u_field.dof_values.numpy().copy()
        current_palp = u_current[self.palp_dof].copy()
        target_palp  = np.asarray(target_palp_disp, dtype=np.float32)
        delta        = target_palp - current_palp
        delta_mag    = float(np.linalg.norm(delta))
        if delta_mag < 1e-7: return
        if n_load_steps is None:
            n_load_steps = max(4, int(delta_mag / 0.00025) + 1)
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
                K = fem.integrate(_up_Kuu,
                    fields={'u': self.trial, 'v': self.test,
                            'u_cur': self.u_field, 'p_cur': self.p_field, **self._up_fields()})
                fi = fem.integrate(_up_force,
                    fields={'v': self.test, 'u_cur': self.u_field, 'p_cur': self.p_field,
                            'mu_f': self.mu_field, 'r_f': self.kappa_field,
                            'k1_f': self.k1_field, 'k2_f': self.k2_field,
                            'fiber_f': self.fiber_field}, output_dtype=wp.vec3d)
                rhs_np = -fi.numpy()
                for di in self.bc_dofs: rhs_np[di] = [0., 0., 0.]
                rhs = wp.array(rhs_np, dtype=wp.vec3d)
                fem.project_linear_system(K, rhs, self.bc_matrix)
                _project_dof_bc(K, rhs, self.palp_dof)
                du = wp.zeros(self.n_dof, dtype=wp.vec3d)
                fem_utils.bsr_cg(K, b=rhs, x=du, quiet=True, tol=1e-10, max_iters=500)
                du_np = du.numpy(); du_norm = float(np.linalg.norm(du_np))
                if np.isnan(du_norm) or du_norm > 1e4:
                    self.u_field.dof_values.assign(wp.array(last_good_u, dtype=wp.vec3))
                    self.p_field.dof_values.assign(wp.array(last_good_p, dtype=float)); return
                alpha = 1.0
                if du_norm > prev_du * 1.5: alpha = 0.5
                if du_norm > prev_du * 3.0: alpha = 0.25
                u_vals = self.u_field.dof_values.numpy()
                u_vals += (du_np * alpha).astype(np.float32)
                u_vals[self.palp_dof] = corner
                for di in self.bottom_dofs: u_vals[di] = [0, 0, 0]
                self.u_field.dof_values.assign(wp.array(u_vals, dtype=wp.vec3))
                prev_du = du_norm
                if du_norm < tol: converged = True; break
            if converged:
                J_m1 = fem.integrate(_J_m1_integral,
                    fields={'q': self.p_test, 'u_cur': self.u_field},
                    output_dtype=float).numpy() / self.elem_vol
                p_vals = self.p_field.dof_values.numpy(); p_vals += self.kappa_vals * J_m1
                self.p_field.dof_values.assign(wp.array(p_vals, dtype=float))
                last_good_u = self.u_field.dof_values.numpy().copy()
                last_good_p = self.p_field.dof_values.numpy().copy()
                if getattr(self, 'vtk_model', None) is not None:
                    self.updateModel()
                    try:
                        import slicer as _s
                        _s.app.layoutManager().threeDWidget(0).threeDView().renderWindow().Render()
                        _s.app.processEvents()
                    except Exception: pass
            else:
                self.u_field.dof_values.assign(wp.array(last_good_u, dtype=wp.vec3))
                self.p_field.dof_values.assign(wp.array(last_good_p, dtype=float)); break

    def onControlPointMoved(self):
        if self._updating: return
        pt_mm = [0., 0., 0.]
        self.fiducial_list.GetNthControlPointPosition(self._palp_fid_idx, pt_mm)
        pt_m   = np.array(pt_mm) / MM_PER_M
        rest_m = self.dof_positions[self.palp_dof]
        target_disp = (pt_m - rest_m).astype(np.float32)
        # Always start from rest (zero displacement) for each drag.
        n_p    = len(self.p_field.dof_values.numpy())
        zero_u = np.zeros((self.n_dof, 3), dtype=np.float32)
        zero_p = np.zeros(n_p, dtype=np.float64)
        cur_palp  = np.zeros(3, dtype=np.float32)
        delta     = target_disp
        delta_mag = float(np.linalg.norm(delta))
        if delta_mag < 1e-7: return
        n_steps = max(4, int(delta_mag / 0.00025) + 1)
        self._inc = {
            'current_palp': cur_palp, 'delta': delta,
            'n_steps': n_steps, 'step': 0,
            'last_good_u': zero_u,
            'last_good_p': zero_p,
            'steps_converged': 0,
        }
        self._updating = True
        self._schedule_step()

    def _schedule_step(self):
        try:
            import qt; qt.QTimer.singleShot(0, self._do_one_step)
        except Exception:
            while self._inc['step'] < self._inc['n_steps']: self._do_one_step()

    def _do_one_step(self):
        try:
            wp.set_device(self.device)
            inc = self._inc; step = inc['step']; n_steps = inc['n_steps']
            if step >= n_steps: self._finish_solve(); return
            frac   = (step + 1) / n_steps
            corner = inc['current_palp'] + inc['delta'] * frac
            u_vals = inc['last_good_u'].copy()
            u_vals[self.palp_dof] = corner
            for di in self.bottom_dofs: u_vals[di] = [0, 0, 0]
            self.u_field.dof_values.assign(wp.array(u_vals, dtype=wp.vec3))
            self.p_field.dof_values.assign(wp.array(inc['last_good_p'], dtype=float))
            prev_du = float('inf'); converged = False
            for _ in range(12):
                K = fem.integrate(_up_Kuu,
                    fields={'u': self.trial, 'v': self.test,
                            'u_cur': self.u_field, 'p_cur': self.p_field, **self._up_fields()})
                fi = fem.integrate(_up_force,
                    fields={'v': self.test, 'u_cur': self.u_field, 'p_cur': self.p_field,
                            'mu_f': self.mu_field, 'r_f': self.kappa_field,
                            'k1_f': self.k1_field, 'k2_f': self.k2_field,
                            'fiber_f': self.fiber_field}, output_dtype=wp.vec3d)
                rhs_np = -fi.numpy()
                for di in self.bc_dofs: rhs_np[di] = [0., 0., 0.]
                rhs = wp.array(rhs_np, dtype=wp.vec3d)
                fem.project_linear_system(K, rhs, self.bc_matrix)
                _project_dof_bc(K, rhs, self.palp_dof)
                du = wp.zeros(self.n_dof, dtype=wp.vec3d)
                fem_utils.bsr_cg(K, b=rhs, x=du, quiet=True, tol=1e-10, max_iters=500)
                du_np = du.numpy(); dn = float(np.linalg.norm(du_np))
                if np.isnan(dn) or dn > 1e4:
                    self.u_field.dof_values.assign(wp.array(inc['last_good_u'], dtype=wp.vec3))
                    self.p_field.dof_values.assign(wp.array(inc['last_good_p'], dtype=float))
                    self._finish_solve(); return
                alpha = 1.0
                if dn > prev_du * 1.5: alpha = 0.5
                if dn > prev_du * 3.0: alpha = 0.25
                u_vals = self.u_field.dof_values.numpy()
                u_vals += (du_np * alpha).astype(np.float32)
                u_vals[self.palp_dof] = corner
                for di in self.bottom_dofs: u_vals[di] = [0, 0, 0]
                self.u_field.dof_values.assign(wp.array(u_vals, dtype=wp.vec3))
                prev_du = dn
                if dn < 5e-3: converged = True; break
            if converged:
                Jm1 = fem.integrate(_J_m1_integral,
                    fields={'q': self.p_test, 'u_cur': self.u_field},
                    output_dtype=float).numpy() / self.elem_vol
                pv = self.p_field.dof_values.numpy(); pv += self.kappa_vals * Jm1
                self.p_field.dof_values.assign(wp.array(pv, dtype=float))
                inc['last_good_u'] = self.u_field.dof_values.numpy().copy()
                inc['last_good_p'] = self.p_field.dof_values.numpy().copy()
                inc['steps_converged'] = inc.get('steps_converged', 0) + 1
                self.updateModel()
                try:
                    import slicer as _s
                    _s.app.layoutManager().threeDWidget(0).threeDView().renderWindow().Render()
                except Exception: pass
            else:
                self.u_field.dof_values.assign(wp.array(inc['last_good_u'], dtype=wp.vec3))
                self.p_field.dof_values.assign(wp.array(inc['last_good_p'], dtype=float))
                self._finish_solve(); return
            inc['step'] += 1
            self._schedule_step()
        except Exception:
            import traceback; traceback.print_exc(); self._finish_solve()

    def _finish_solve(self):
        wp.set_device(self.device)
        self.u_field.dof_values.assign(
            wp.array(self._inc['last_good_u'], dtype=wp.vec3))
        self.p_field.dof_values.assign(
            wp.array(self._inc['last_good_p'], dtype=float))
        self.updateModel(); self._sync_fiducials(); self._updating = False

    def _sync_fiducials(self):
        self._updating = True
        u_vals = self.u_field.dof_values.numpy()
        for fi, di in zip(self._bottom_fid_indices, sorted(self.bottom_dofs)[:4]):
            pos_mm = (self.dof_positions[di] + u_vals[di].astype(np.float64)) * MM_PER_M
            self.fiducial_list.SetNthControlPointPosition(fi, *pos_mm.tolist())
        palp_mm = (self.dof_positions[self.palp_dof] +
                   u_vals[self.palp_dof].astype(np.float64)) * MM_PER_M
        self.fiducial_list.SetNthControlPointPosition(self._palp_fid_idx, *palp_mm.tolist())
        self._updating = False

    def _build_subdivided_surface(self, n_sub=3):
        from scipy.spatial import cKDTree
        nx, ny, nz = RES; dx, dy, dz = BLOCK_SIZE/nx, BLOCK_SIZE/ny, BLOCK_SIZE/nz
        tree = cKDTree(self.dof_positions)
        def N8(r, s):
            return np.array([(1-r)*(1-s)*(-1-r-s)/4,(1+r)*(1-s)*(-1+r-s)/4,
                              (1+r)*(1+s)*(-1+r+s)/4,(1-r)*(1+s)*(-1-r+s)/4,
                              (1-r**2)*(1-s)/2,(1+r)*(1-s**2)/2,
                              (1-r**2)*(1+s)/2,(1-r)*(1-s**2)/2])
        t = np.linspace(-1,1,n_sub+1); wts = np.array([[N8(r,s) for s in t] for r in t])
        all_rest=[]; all_dofs=[]; all_wts=[]; surface_quads=[]
        def add_face(c4):
            c4=[np.array(x) for x in c4]
            nodes=np.array([c4[0],c4[1],c4[2],c4[3],
                             (c4[0]+c4[1])/2,(c4[1]+c4[2])/2,
                             (c4[2]+c4[3])/2,(c4[3]+c4[0])/2])
            _,dof_idx=tree.query(nodes); start=len(all_rest)
            for ri in range(n_sub+1):
                for si in range(n_sub+1):
                    w=wts[ri,si]; all_rest.append(w@nodes); all_dofs.append(dof_idx); all_wts.append(w)
            for ri in range(n_sub):
                for si in range(n_sub):
                    p00=start+ri*(n_sub+1)+si; p10=p00+1
                    p11=start+(ri+1)*(n_sub+1)+si+1; p01=p11-1
                    surface_quads.append((p00,p10,p11,p01))
        for i in range(nx):
            for k in range(nz):
                add_face([[i*dx,0,k*dz],[(i+1)*dx,0,k*dz],[(i+1)*dx,0,(k+1)*dz],[i*dx,0,(k+1)*dz]])
                add_face([[i*dx,ny*dy,(k+1)*dz],[(i+1)*dx,ny*dy,(k+1)*dz],[(i+1)*dx,ny*dy,k*dz],[i*dx,ny*dy,k*dz]])
        for i in range(nx):
            for j in range(ny):
                add_face([[i*dx,j*dy,0],[i*dx,(j+1)*dy,0],[(i+1)*dx,(j+1)*dy,0],[(i+1)*dx,j*dy,0]])
                add_face([[(i+1)*dx,j*dy,nz*dz],[(i+1)*dx,(j+1)*dy,nz*dz],[i*dx,(j+1)*dy,nz*dz],[i*dx,j*dy,nz*dz]])
        for j in range(ny):
            for k in range(nz):
                add_face([[0,j*dy,(k+1)*dz],[0,(j+1)*dy,(k+1)*dz],[0,(j+1)*dy,k*dz],[0,j*dy,k*dz]])
                add_face([[nx*dx,j*dy,k*dz],[nx*dx,(j+1)*dy,k*dz],[nx*dx,(j+1)*dy,(k+1)*dz],[nx*dx,j*dy,(k+1)*dz]])
        self.sample_rest=np.array(all_rest,dtype=np.float64); self.sample_dofs=np.array(all_dofs,dtype=np.int32)
        self.sample_wts=np.array(all_wts,dtype=np.float64); self.surface_quads=surface_quads

    def _deformed_samples_mm(self):
        u_vals=self.u_field.dof_values.numpy(); u_at=u_vals[self.sample_dofs].astype(np.float64)
        disp=np.einsum('ni,nij->nj',self.sample_wts,u_at); return (self.sample_rest+disp)*MM_PER_M

    def _apply_tissue_colors(self):
        import vtk, slicer
        layer_defs=[(0.,0.04,0,"Liver",(139/255,69/255,19/255)),(0.04,0.055,1,"Muscle",(210/255,60/255,60/255)),
                    (0.055,0.07,2,"Fat",(250/255,220/255,120/255)),(0.07,0.08,3,"Skin",(240/255,195/255,160/255))]
        def layer_idx(y_m):
            y_c=max(0.,min(float(y_m),BLOCK_SIZE))
            for y_lo,y_hi,idx,*_ in layer_defs[:-1]:
                if y_c<y_hi: return idx
            return layer_defs[-1][2]
        y_rest=self.sample_rest[:,1]; scalar_arr=vtk.vtkFloatArray(); scalar_arr.SetName("TissueLayer")
        scalar_arr.SetNumberOfTuples(len(y_rest))
        for i,y in enumerate(y_rest): scalar_arr.SetValue(i,layer_idx(float(y)))
        self._raw_poly.GetPointData().SetScalars(scalar_arr); self._raw_poly.Modified()
        self._normals_filter.Update(); self.vtk_model.GetPolyData().Modified()
        ct=slicer.mrmlScene.GetFirstNodeByName('TissueLayerColors')
        if ct is None: ct=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLColorTableNode','TissueLayerColors')
        ct.SetTypeToUser(); ct.SetNumberOfColors(4); lut=ct.GetLookupTable(); lut.SetNumberOfTableValues(4)
        for y_lo,y_hi,idx,name,rgb in layer_defs:
            ct.SetColor(idx,name,rgb[0],rgb[1],rgb[2],1.); lut.SetTableValue(idx,rgb[0],rgb[1],rgb[2],1.)
        lut.Build(); dn=self.vtk_model.GetDisplayNode()
        dn.SetAndObserveColorNodeID(ct.GetID()); dn.SetScalarVisibility(1)
        dn.SetActiveScalarName("TissueLayer")
        dn.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseManualScalarRange); dn.SetScalarRange(0,3)

    def createModel(self):
        import vtk, slicer
        pts_np=self._deformed_samples_mm(); vtk_pts=vtk.vtkPoints()
        for p in pts_np: vtk_pts.InsertNextPoint(*p)
        cells=vtk.vtkCellArray()
        for q in self.surface_quads:
            cells.InsertNextCell(4)
            for vi in q: cells.InsertCellPoint(vi)
        self._raw_poly=vtk.vtkPolyData(); self._raw_poly.SetPoints(vtk_pts); self._raw_poly.SetPolys(cells)
        nf=vtk.vtkPolyDataNormals(); nf.SetInputData(self._raw_poly); nf.SetFeatureAngle(30.)
        nf.SplittingOn(); nf.ComputePointNormalsOn(); nf.Update(); self._normals_filter=nf
        node=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode','LayeredAnisotropicTissueHex')
        node.SetAndObservePolyData(nf.GetOutput()); node.CreateDefaultDisplayNodes()
        dn=node.GetDisplayNode(); dn.SetColor(0.85,0.75,0.65); dn.SetOpacity(0.85)
        dn.SetEdgeVisibility(1); dn.SetEdgeColor(0.25,0.25,0.25); dn.SetBackfaceCulling(0)
        self.vtk_model=node

    def updateModel(self):
        pts_np=self._deformed_samples_mm(); vtk_pts=self._raw_poly.GetPoints()
        for i,p in enumerate(pts_np): vtk_pts.SetPoint(i,*p)
        self._raw_poly.GetPoints().Modified(); self._raw_poly.Modified()
        self._normals_filter.Update(); self.vtk_model.GetPolyData().Modified()

    def createControlPoints(self):
        import slicer; ml=slicer.modules.markups.logic()
        slicer.mrmlScene.StartState(slicer.mrmlScene.BatchProcessState)
        self.fiducial_list=ml.AddNewMarkupsNode('vtkMRMLMarkupsFiducialNode','AnisotropicTissueHex_controls')
        dn=self.fiducial_list.GetDisplayNode()
        dn.SetTextScale(0.); dn.SetGlyphScale(5.); dn.SetGlyphTypeFromString('Sphere3D')
        dn.SetColor(0.2,0.2,0.6); dn.SetSelectedColor(1.,0.9,0.1); dn.SetActiveColor(1.,0.6,0.)
        dn.SetSnapMode(slicer.vtkMRMLMarkupsFiducialDisplayNode.SnapModeUnconstrained); dn.SetVisibility(True)
        u_vals=self.u_field.dof_values.numpy(); self._bottom_fid_indices=[]
        for di in sorted(self.bottom_dofs)[:4]:
            pos_mm=(self.dof_positions[di]+u_vals[di].astype(np.float64))*MM_PER_M
            self.fiducial_list.AddControlPoint(*pos_mm.tolist())
            idx=self.fiducial_list.GetNumberOfControlPoints()-1
            self.fiducial_list.SetNthControlPointLabel(idx,'')
            self.fiducial_list.SetNthControlPointLocked(idx,True)
            self.fiducial_list.SetNthControlPointSelected(idx,False)
            self._bottom_fid_indices.append(idx)
        palp_mm=(self.dof_positions[self.palp_dof]+u_vals[self.palp_dof].astype(np.float64))*MM_PER_M
        self.fiducial_list.AddControlPoint(*palp_mm.tolist())
        self._palp_fid_idx=self.fiducial_list.GetNumberOfControlPoints()-1
        self.fiducial_list.SetNthControlPointLabel(self._palp_fid_idx,'drag me')
        self.fiducial_list.SetNthControlPointLocked(self._palp_fid_idx,False)
        self.fiducial_list.SetNthControlPointSelected(self._palp_fid_idx,True)
        self.fiducial_list.AddObserver(self.fiducial_list.PointEndInteractionEvent,
                                       lambda c,e: self.onControlPointMoved())
        slicer.mrmlScene.EndState(slicer.mrmlScene.BatchProcessState)

    def run(self):
        import slicer
        self.createModel(); self.updateModel(); self._apply_tissue_colors(); self.createControlPoints()
        slicer.layered_aniso_hex=self
        slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
        v=slicer.app.layoutManager().threeDWidget(0).threeDView()
        renderer=v.renderWindow().GetRenderers().GetFirstRenderer()
        camera=renderer.GetActiveCamera()
        camera.SetPosition(200,-200,200); camera.SetFocalPoint(40,40,40); camera.SetViewUp(0,0,1)
        renderer.ResetCamera(); v.renderWindow().Render()
        print("Layered ANISOTROPIC tissue hex ready.")
        print(f"  DOFs: {self.n_dof}  Elements: {self.tissue_model.num_elements}")
        print(f"  Palpation DOF: {self.palp_dof}  rest: {self.dof_positions[self.palp_dof]}")
        print("  Drag yellow fiducial -- 0.25mm steps, element-size step cap.")
        print("  Access: slicer.layered_aniso_hex")


if __name__ == '__main__' or ('slicer' in dir() and slicer is not None):
    import slicer
    slicer.mrmlScene.Clear(0)
    sim = LayeredAnisotropicTissueHex()  # auto-selects cuda:0 if available
    sim.run()
