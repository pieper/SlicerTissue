"""MPM laparoscopic partial-nephrectomy demo for 3D Slicer (NewtonTissue).

Scenario: the abdomen is insufflated and the perinephric fat has been
cleared, exposing the kidney.  A renal mass (the 'Mass' segment) at the
inferior pole is grasped and retracted toward the patient midline and
inferiorly while the most-superior third of the kidney is held fixed.  The
parenchyma deforms through its elastin/collagen fiber network.  A cautery
resection along the kidney-mass interface is added as a later phase.

PHASE 1 (this file): lesion grasp + retraction (the "pull").

Mechanics: explicit MLS-MPM (newton_tissue.mpm.MPMSimulator) on the GPU.
Kidney + mass are one fiber-connected continuum.  The mass particles form a
rigid kinematic "grasp" whose prescribed motion drags the kidney through the
fiber bonds.  Gravity is disabled (insufflated abdomen) and the segmented
shape is the stress-free reference, so no warm-up is needed.

Coordinate convention (inherited from mpm_ct_head): simulation units are
metres, the lattice axes are aligned with RAS (R->X, A->Y, S->Z), and
    pos_ras_mm = pos_sim_m * 1000 + ras_offset_mm

Assumes the scene already contains a vtkMRMLSegmentationNode with segments
named 'Kidney' and 'Mass'.  Runs inside Slicer via the MCP bridge; device
auto-selects cuda:0.
"""

from __future__ import annotations
import os, sys
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
_NEWTON_DIR = os.path.dirname(_SCRIPT_DIR) if os.path.basename(_SCRIPT_DIR) == 'examples' else _SCRIPT_DIR
_SRC_DIR    = os.path.join(_NEWTON_DIR, 'src')
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import warp as wp
from newton_tissue.mpm import MPMMaterial, MPMSimulator


# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
DX_MM   = 2.0      # Eulerian grid spacing [mm]; 3 mm for dev, 2 mm for production
PPC     = 2        # particles per cell per dimension (2 -> 8 per cell)
# Timestep is CFL-limited and scales with dx: dt = CFL_FACTOR * dx / c_s, where
# c_s = sqrt(E/rho).  CFL_FACTOR=0.18 reproduces the stable dx=3mm/dt~1.8e-4
# regime; at dx=2mm it yields ~1.2e-4 (dt=2e-4 there was unstable -> NaN).
CFL_FACTOR = 0.18
DT_MAX     = 2.0e-4   # timestep cap [s]
VELOCITY_DAMPING = 0.995
GRAVITY = np.array([0.0, 0.0, 0.0])   # insufflated abdomen: gravity disabled

# Soft-tissue Neo-Hookean + fiber network (matches mpm_ct_head soft tissue).
MATERIAL = MPMMaterial(E=10_000.0, nu=0.48, rho=1_060.0,
                       k_elastin=0.05, k_collagen=0.25, collagen_crimp=0.05,
                       k_curve=0.0)

KIDNEY_SEGMENT_NAME = 'Kidney'
MASS_SEGMENT_NAME   = 'Mass'

# Boundary condition: anchor the most-superior fraction of the kidney S-extent.
FIXED_SUPERIOR_FRACTION = 1.0 / 3.0

# Pull (lesion retraction) phase.
T_PULL_S         = 10.0    # ramp duration [s]
SETTLE_S         = 3.0     # hold/settle after the pull, before cutting [s]
PULL_MIDLINE_MM  = 12.0    # retraction toward the midline (x -> 0) [mm]
PULL_INFERIOR_MM = 12.0    # retraction inferiorly (-S) [mm]

# Recording: one sequence frame per 0.1 s of sim time -> 10 fps playback.
RECORD_DT_S = 0.1

# Cautery resection phase.
CUT_MARGIN_MM       = 3.0    # resection margin into the kidney from the mass boundary
CAUTERY_LEN_MM      = 5.0    # cautery tip length = stroke advance per cut [mm]
T_CUT_S             = 5.0    # cutting (relaxation) duration per stroke [s]
T_PAUSE_S           = 3.0    # pause between strokes [s]
FINAL_SETTLE_S      = 10.0   # settle after full severance, so the kidney bed
                             # springs back to rest while the mass is held [s]
# Continue retracting the grasped specimen during cutting, along the pull
# direction, so it visibly lifts/peels away as each stroke frees it.
RETRACT_SPEED_MM_S  = 0.3    # extra retraction rate during the cut phase [mm/s]

# Non-blocking run loop (Qt timer): keeps Slicer responsive and avoids the MCP
# call timeout on long (minutes-scale) GPU runs.
STEPS_PER_TICK   = 40
TICK_INTERVAL_MS = 5

# Display colors (RGB 0..1).
COLOR_KIDNEY_FREE  = (0.93, 0.60, 0.62)   # salmon (free parenchyma)
COLOR_KIDNEY_FIXED = (0.40, 0.55, 0.95)   # blue (anchored superior third)
COLOR_MASS         = (0.45, 0.85, 0.45)   # green (lesion / grasp)
COLOR_CUFF         = (0.95, 0.65, 0.20)   # orange (3 mm resection margin cuff)
COLOR_CAUTERY      = (1.0, 0.85, 0.1)     # yellow cautery tip


@wp.kernel
def _grasp_kernel(x: wp.array(dtype=wp.vec3),
                  v: wp.array(dtype=wp.vec3),
                  F: wp.array(dtype=wp.mat33),
                  driven: wp.array(dtype=int),
                  anchor: wp.array(dtype=wp.vec3),
                  disp: wp.vec3,
                  vel: wp.vec3):
    """Rigid kinematic grasp.

    Pins each driven particle to anchor+disp, sets its velocity to the
    prescribed grasp velocity (so P2G scatters the correct momentum and drags
    fiber-connected neighbours), and keeps it stress-free (F = I) so the rigid
    grasp introduces no internal stress.  Applied AFTER sim.step().
    """
    p = wp.tid()
    if driven[p] != 0:
        x[p] = anchor[p] + disp
        v[p] = vel
        F[p] = wp.mat33(1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 1.0)


class MPMKidneyResection:
    """Phase-1 kidney-resection demo: grasp the lesion and retract it."""

    def __init__(self, segmentation_node=None, reference_volume=None,
                 dx_mm=DX_MM, ppc=PPC, device=None):
        wp.init()
        if device is None:
            device = "cpu"
            try:
                _ = wp.zeros(1, dtype=float, device="cuda:0")
                device = "cuda:0"
            except Exception:
                pass
        self.device = device

        seg, ref = self._resolve_nodes(segmentation_node, reference_volume)
        self.segmentation_node = seg
        self.reference_volume  = ref

        kid_id  = self._segment_id(seg, KIDNEY_SEGMENT_NAME)
        mass_id = self._segment_id(seg, MASS_SEGMENT_NAME)

        import slicer, vtk
        kid_full = slicer.util.arrayFromSegmentBinaryLabelmap(seg, kid_id, ref) > 0   # KJI
        mass     = slicer.util.arrayFromSegmentBinaryLabelmap(seg, mass_id, ref) > 0

        ijk2ras = vtk.vtkMatrix4x4(); ref.GetIJKToRASMatrix(ijk2ras)
        self._ijk2ras = np.array([[ijk2ras.GetElement(r, c) for c in range(4)] for r in range(4)])
        ras2ijk = vtk.vtkMatrix4x4(); ref.GetRASToIJKMatrix(ras2ijk)
        self._ras2ijk = np.array([[ras2ijk.GetElement(r, c) for c in range(4)] for r in range(4)])

        # Host kidney = connected component nearest the mass (drops the
        # contralateral kidney and segmentation specks).
        host_kid = self._host_kidney(kid_full, mass)

        # ROI in RAS mm = (host kidney U mass) bbox, padded by two cells.
        roi_lo, roi_hi = self._roi_bounds(host_kid | mass, pad_mm=2.0 * dx_mm)

        dx_m   = dx_mm / 1000.0
        lo_m   = roi_lo / 1000.0
        hi_m   = roi_hi / 1000.0
        block_size = float((hi_m - lo_m).max())
        n_grid = int(np.ceil(block_size / dx_m))
        self._ras_offset_mm = lo_m * 1000.0

        c_s = (MATERIAL.E / MATERIAL.rho) ** 0.5
        dt  = min(DT_MAX, CFL_FACTOR * dx_m / c_s)
        # WIP (tissue-model revisit): total_lagrangian=True recomputes F from
        # positions every step (drift-free) so the kidney bed would spring back
        # to its true rest pose -- but it currently blows up because the
        # kinematic grasp's sharp displacement jump makes the finite-difference
        # F explode at the mass<->tissue interface.  The lattice-neighbour build
        # and cut/disconnect link-severing scaffolding is in place; the missing
        # piece is severing the mass<->tissue links up front.  Left False here so
        # the demo runs on stable updated-Lagrangian (bed relaxes ~6 mm short of
        # rest due to F-drift); flip to True when the tissue model is revisited.
        self.sim = MPMSimulator(
            block_lo=np.zeros(3), block_hi=np.full(3, block_size),
            n_grid=n_grid, dt=dt, material=MATERIAL, device=device,
            velocity_damping=VELOCITY_DAMPING, total_lagrangian=False)
        print("MPMKidneyResection: dx=%.2f mm, dt=%.2e s (CFL=%.2f), n_grid=%d"
              % (self.sim.dx * 1000, self.sim.dt,
                 self.sim.dt * c_s / self.sim.dx, n_grid))

        self._populate(host_kid, mass, lo_m, hi_m, dx_m, ppc)
        self._apply_bc()
        self._setup_grasp()
        self._apply_grasp(0.0)   # pin grasp at rest (disp = 0)

        # Simulation clock / recording cursor.
        self._t = 0.0
        self._next_record_t = 0.0

        # Viz / sequence state.
        self.vtk_model     = None
        self._vtk_points   = None
        self._vtk_poly     = None
        self.sequence_node = None
        self.browser_node  = None
        self.proxy_model   = None
        self.cautery_sequence_node = None
        self._cautery_model     = None
        self._cautery_transform = None

        # Continued-retraction state (enabled during the cut phase).
        self._retracting       = False
        self._retract_start_t  = 0.0
        self._retract_speed    = RETRACT_SPEED_MM_S / 1000.0   # m/s
        self._retract_extra    = 0.0   # accumulated retraction beyond the pull [m]

        # Non-blocking scenario state machine.
        self._run_active     = False
        self._scenario_done  = False
        self._cut_started    = False
        self._disconnected   = False
        self._n_strokes      = 0
        self._strokes_done   = 0
        self._phase          = 'idle'
        self._timer          = None

    # ------------------------------------------------------------------
    # Scene / segmentation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_nodes(seg, ref):
        import slicer
        if seg is None:
            segs = slicer.util.getNodesByClass('vtkMRMLSegmentationNode')
            if not segs:
                raise RuntimeError("No vtkMRMLSegmentationNode in the scene")
            seg = segs[0]
        if ref is None:
            try:
                ref = slicer.util.getNode('7: arterial')
            except Exception:
                vols = slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode')
                if not vols:
                    raise RuntimeError("No reference scalar volume in the scene")
                ref = vols[0]
        return seg, ref

    @staticmethod
    def _segment_id(seg, name):
        sid = seg.GetSegmentation().GetSegmentIdBySegmentName(name)
        if not sid:
            raise RuntimeError("Segment '%s' not found" % name)
        return sid

    def _ras_of_voxels(self, mask):
        kji = np.argwhere(mask)
        ijk = kji[:, ::-1]
        homo = np.column_stack([ijk, np.ones(len(ijk))])
        return (self._ijk2ras @ homo.T).T[:, :3]

    def _host_kidney(self, kid, mass, min_voxels=1000):
        import scipy.ndimage as ndi
        lab, n = ndi.label(kid)
        mass_c = self._ras_of_voxels(mass).mean(0)
        best, bestd = None, 1e18
        for L in range(1, n + 1):
            comp = (lab == L)
            if comp.sum() < min_voxels:
                continue
            d = float(np.linalg.norm(self._ras_of_voxels(comp).mean(0) - mass_c))
            if d < bestd:
                bestd, best = d, L
        return kid if best is None else (lab == best)

    def _roi_bounds(self, mask, pad_mm):
        ras = self._ras_of_voxels(mask)
        return ras.min(0) - pad_mm, ras.max(0) + pad_mm

    # ------------------------------------------------------------------
    # Particle setup
    # ------------------------------------------------------------------

    def _populate(self, host_kid, mass, lo_m, hi_m, dx_m, ppc):
        step = dx_m / ppc
        off  = 0.5 * step
        extent = hi_m - lo_m
        xs = np.arange(off, extent[0], step)
        ys = np.arange(off, extent[1], step)
        zs = np.arange(off, extent[2], step)
        nx, ny, nz = len(xs), len(ys), len(zs)
        ix, iy, iz = np.meshgrid(xs, ys, zs, indexing='ij')
        pos = np.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=1).astype(np.float32)

        # Sim coords -> RAS mm -> IJK of the reference volume.
        pos_ras = pos * 1000.0 + self._ras_offset_mm
        homo = np.column_stack([pos_ras, np.ones(len(pos_ras))])
        ijk = (self._ras2ijk @ homo.T).T[:, :3]
        ijk_int = np.round(ijk).astype(np.int32)

        dims = np.array([host_kid.shape[2], host_kid.shape[1], host_kid.shape[0]])  # I,J,K
        in_b = ((ijk_int >= 0) & (ijk_int < dims)).all(axis=1)
        kid_s  = np.zeros(len(pos), dtype=bool)
        mass_s = np.zeros(len(pos), dtype=bool)
        vi = ijk_int[in_b]
        kid_s[in_b]  = host_kid[vi[:, 2], vi[:, 1], vi[:, 0]]
        mass_s[in_b] = mass[vi[:, 2], vi[:, 1], vi[:, 0]]

        is_mass   = mass_s
        is_kidney = kid_s & ~mass_s
        keep = is_kidney | is_mass

        pos = pos[keep]
        is_mass_k   = is_mass[keep]
        is_kidney_k = is_kidney[keep]
        n = len(pos)
        self.sim.n_particles = n

        vol_p  = float(step ** 3)
        mass_p = float(MATERIAL.rho * vol_p)
        F_np = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
        C_np = np.zeros((n, 3, 3), dtype=np.float32)
        with wp.ScopedDevice(self.device):
            self.sim.x     = wp.array(pos,               dtype=wp.vec3)
            self.sim.x0    = wp.array(pos.copy(),        dtype=wp.vec3)
            self.sim.v     = wp.zeros(n,                 dtype=wp.vec3)
            self.sim.F     = wp.array(F_np,              dtype=wp.mat33)
            self.sim.C     = wp.array(C_np,              dtype=wp.mat33)
            self.sim.m_p   = wp.array(np.full(n, mass_p, dtype=np.float32), dtype=float)
            self.sim.vol_p = wp.array(np.full(n, vol_p,  dtype=np.float32), dtype=float)
            self.sim.fixed = wp.zeros(n, dtype=int)

        self._pos0      = pos.copy()
        self._is_mass   = is_mass_k
        self._is_kidney = is_kidney_k

        # Resection margin cuff: kidney particles within CUT_MARGIN of the mass
        # at rest -- this is the parenchyma that leaves with the specimen.
        from scipy.spatial import cKDTree
        margin_m = CUT_MARGIN_MM / 1000.0
        if is_mass_k.any():
            tree = cKDTree(pos[is_mass_k])
            dmin, _ = tree.query(pos)
            self._is_cuff = is_kidney_k & (dmin < margin_m)
        else:
            self._is_cuff = np.zeros(n, dtype=bool)

        # Fibers across ALL kept particles (kidney + mass = one structure).
        keep_flat = np.where(keep)[0]
        idx3d = np.unravel_index(keep_flat, (nx, ny, nz))
        lookup = np.full((nx, ny, nz), -1, dtype=np.int32)
        lookup[idx3d[0], idx3d[1], idx3d[2]] = np.arange(n, dtype=np.int32)
        self._build_bonds(lookup, pos, np.ones(n, dtype=bool), nx, ny, nz, step)

        # Lattice-neighbour arrays for total-Lagrangian (drift-free) F.
        ii, jj, kk = idx3d
        def _nbr(di, dj, dk):
            ni, nj, nk = ii + di, jj + dj, kk + dk
            valid = (ni >= 0) & (ni < nx) & (nj >= 0) & (nj < ny) & (nk >= 0) & (nk < nz)
            arr = np.full(n, -1, dtype=np.int32)
            arr[valid] = lookup[ni[valid], nj[valid], nk[valid]]
            return arr
        with wp.ScopedDevice(self.device):
            self.sim.nbr_px = wp.array(_nbr( 1, 0, 0), dtype=int)
            self.sim.nbr_mx = wp.array(_nbr(-1, 0, 0), dtype=int)
            self.sim.nbr_py = wp.array(_nbr( 0, 1, 0), dtype=int)
            self.sim.nbr_my = wp.array(_nbr( 0,-1, 0), dtype=int)
            self.sim.nbr_pz = wp.array(_nbr( 0, 0, 1), dtype=int)
            self.sim.nbr_mz = wp.array(_nbr( 0, 0,-1), dtype=int)
        self.sim._lattice_step = float(step)

        print("MPMKidneyResection: %d particles (%d kidney, %d mass), dx=%.2f mm, n_grid=%d"
              % (n, int(is_kidney_k.sum()), int(is_mass_k.sum()),
                 self.sim.dx * 1000, self.sim.n_grid))

    def _build_bonds(self, lookup, pos, is_tissue, nx, ny, nz, step):
        """Lattice-based elastin/collagen fiber bonds between kept particles."""
        bonds_i, bonds_j, bonds_l0, bonds_t = [], [], [], []

        if MATERIAL.k_elastin > 0.0:
            offsets = [(di, dj, dk)
                       for di in (-1, 0, 1)
                       for dj in (-1, 0, 1)
                       for dk in (-1, 0, 1)
                       if (di, dj, dk) > (0, 0, 0)]
            for di, dj, dk in offsets:
                i0, i1 = max(0, -di), nx - max(0, di)
                j0, j1 = max(0, -dj), ny - max(0, dj)
                k0, k1 = max(0, -dk), nz - max(0, dk)
                src = lookup[i0:i1,       j0:j1,       k0:k1].ravel()
                tgt = lookup[i0+di:i1+di, j0+dj:j1+dj, k0+dk:k1+dk].ravel()
                valid = (src >= 0) & (tgt >= 0) & is_tissue[src] & is_tissue[tgt]
                if valid.any():
                    s, t = src[valid], tgt[valid]
                    d = np.linalg.norm(pos[s] - pos[t], axis=1).astype(np.float32)
                    bonds_i.append(s); bonds_j.append(t)
                    bonds_l0.append(d); bonds_t.append(np.zeros(len(s), dtype=np.int32))

        if MATERIAL.k_collagen > 0.0:
            for di, dj, dk in [(2, 0, 0), (0, 2, 0), (0, 0, 2)]:
                i0, i1 = 0, nx - di
                j0, j1 = 0, ny - dj
                k0, k1 = 0, nz - dk
                src = lookup[i0:i1,       j0:j1,       k0:k1].ravel()
                tgt = lookup[i0+di:i1+di, j0+dj:j1+dj, k0+dk:k1+dk].ravel()
                valid = (src >= 0) & (tgt >= 0) & is_tissue[src] & is_tissue[tgt]
                if valid.any():
                    s, t = src[valid], tgt[valid]
                    d = np.linalg.norm(pos[s] - pos[t], axis=1).astype(np.float32)
                    bonds_i.append(s); bonds_j.append(t)
                    bonds_l0.append(d); bonds_t.append(np.ones(len(s), dtype=np.int32))

        if not bonds_i:
            self.sim.n_bonds = 0
            return
        all_i  = np.concatenate(bonds_i).astype(np.int32)
        all_j  = np.concatenate(bonds_j).astype(np.int32)
        all_l0 = np.concatenate(bonds_l0).astype(np.float32)
        all_t  = np.concatenate(bonds_t).astype(np.int32)
        n_e = int((all_t == 0).sum()); n_c = int((all_t == 1).sum())
        with wp.ScopedDevice(self.device):
            self.sim.fiber_i  = wp.array(all_i,  dtype=int)
            self.sim.fiber_j  = wp.array(all_j,  dtype=int)
            self.sim.fiber_l0 = wp.array(all_l0, dtype=float)
            self.sim.fiber_t  = wp.array(all_t,  dtype=int)
            self.sim.fiber_broken = wp.zeros(len(all_i), dtype=int)
        self.sim.n_bonds = len(all_i)
        print("MPMKidneyResection: %d fiber bonds (elastin=%d, collagen=%d)"
              % (len(all_i), n_e, n_c))

    def _apply_bc(self):
        pos = self.sim.x.numpy()
        kz = pos[self._is_kidney, 2]
        z_lo, z_hi = float(kz.min()), float(kz.max())
        z_fix = z_lo + (1.0 - FIXED_SUPERIOR_FRACTION) * (z_hi - z_lo)
        fixed = self._is_kidney & (pos[:, 2] >= z_fix)
        with wp.ScopedDevice(self.device):
            self.sim.fixed = wp.array(fixed.astype(np.int32), dtype=int)
        self._is_fixed = fixed
        s_ras = z_fix * 1000.0 + self._ras_offset_mm[2]
        print("MPMKidneyResection: fixed %d superior kidney particles (S >= %.1f mm)"
              % (int(fixed.sum()), s_ras))

    # ------------------------------------------------------------------
    # Grasp + stepping
    # ------------------------------------------------------------------

    def _setup_grasp(self):
        pos = self.sim.x.numpy()
        with wp.ScopedDevice(self.device):
            self._driven = wp.array(self._is_mass.astype(np.int32), dtype=int)
            self._anchor = wp.array(pos.copy(), dtype=wp.vec3)
        mass_x_ras = float((pos[self._is_mass, 0] * 1000.0 + self._ras_offset_mm[0]).mean())
        midline_sign = -1.0 if mass_x_ras > 0.0 else 1.0
        self._pull_vec_m = np.array([midline_sign * PULL_MIDLINE_MM / 1000.0,
                                     0.0,
                                     -PULL_INFERIOR_MM / 1000.0], dtype=np.float64)
        nrm = float(np.linalg.norm(self._pull_vec_m))
        self._pull_unit = self._pull_vec_m / nrm if nrm > 1e-9 else np.zeros(3)
        print("MPMKidneyResection: grasp %d mass particles; pull vec (mm) = [%.1f, %.1f, %.1f]"
              % (int(self._is_mass.sum()),
                 self._pull_vec_m[0] * 1000, self._pull_vec_m[1] * 1000,
                 self._pull_vec_m[2] * 1000))

    def _apply_grasp(self, t):
        if t <= T_PULL_S:
            # Phase 1: ramp to the retracted pose.
            frac = (t / T_PULL_S) if T_PULL_S > 0 else 1.0
            disp = self._pull_vec_m * frac
            vel  = self._pull_vec_m / T_PULL_S
        elif self._retracting:
            # Cut phase: keep retracting along the pull direction so the freed
            # specimen lifts away from the bed.  Remember the accumulated extra.
            self._retract_extra = self._retract_speed * (t - self._retract_start_t)
            disp = self._pull_vec_m + self._pull_unit * self._retract_extra
            vel  = self._pull_unit * self._retract_speed
        else:
            # Hold at the LAST retracted pose.  Must keep _retract_extra here --
            # dropping it makes the grasp yank the specimen back toward the
            # kidney (the t=61 s snap-back bug).
            disp = self._pull_vec_m + self._pull_unit * self._retract_extra
            vel  = np.zeros(3)
        wp.launch(_grasp_kernel, dim=self.sim.n_particles,
                  inputs=[self.sim.x, self.sim.v, self.sim.F,
                          self._driven, self._anchor,
                          wp.vec3(float(disp[0]), float(disp[1]), float(disp[2])),
                          wp.vec3(float(vel[0]),  float(vel[1]),  float(vel[2]))],
                  device=self.device)

    def _step(self):
        self.sim.step(GRAVITY)
        self._t += self.sim.dt
        self._apply_grasp(self._t)

    def advance(self, sim_seconds, update_every_s=0.05):
        """Run sim_seconds of physical time, recording frames + updating view."""
        import slicer
        dt = self.sim.dt
        n_steps = int(round(sim_seconds / dt))
        upd = max(1, int(round(update_every_s / dt)))
        for k in range(n_steps):
            self._step()
            if self._t + 1e-9 >= self._next_record_t:
                self._record_frame()
                self._next_record_t += RECORD_DT_S
            if (k % upd) == 0:
                self.update_model()
                slicer.app.processEvents()
        self.update_model()
        return self.telemetry()

    def telemetry(self):
        pos  = self.sim.get_positions()
        disp = (pos - self._pos0) * 1000.0
        dmag = np.linalg.norm(disp, axis=1)
        spd  = np.linalg.norm(self.sim.v.numpy(), axis=1)
        return {
            't_s': round(self._t, 3),
            'n_frames': (self.sequence_node.GetNumberOfDataNodes()
                         if self.sequence_node else 0),
            'mass_disp_mm':       round(float(dmag[self._is_mass].mean()), 2),
            'kidney_max_disp_mm': round(float(dmag[self._is_kidney].max()), 2),
            'max_speed_mm_s':     round(float(spd.max()) * 1000.0, 3),
            'nan': bool(np.isnan(pos).any()),
        }

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _colors_uint8(self):
        c = np.zeros((self.sim.n_particles, 3), dtype=np.uint8)
        c[self._is_kidney] = (np.array(COLOR_KIDNEY_FREE)  * 255).astype(np.uint8)
        c[self._is_cuff]   = (np.array(COLOR_CUFF)         * 255).astype(np.uint8)
        c[self._is_fixed]  = (np.array(COLOR_KIDNEY_FIXED) * 255).astype(np.uint8)
        c[self._is_mass]   = (np.array(COLOR_MASS)         * 255).astype(np.uint8)
        return c

    def _positions_mm(self):
        return (self.sim.get_positions() * 1000.0 + self._ras_offset_mm).astype(np.float32)

    def create_model(self):
        import slicer, vtk, vtk.util.numpy_support as ns
        pts = vtk.vtkPoints()
        pts.SetData(ns.numpy_to_vtk(self._positions_mm(), deep=True, array_type=vtk.VTK_FLOAT))
        colors = ns.numpy_to_vtk(self._colors_uint8(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        colors.SetName("Colors"); colors.SetNumberOfComponents(3)
        src = vtk.vtkPolyData(); src.SetPoints(pts); src.GetPointData().SetScalars(colors)
        glyph = vtk.vtkVertexGlyphFilter(); glyph.SetInputData(src); glyph.Update()
        poly = vtk.vtkPolyData(); poly.DeepCopy(glyph.GetOutput())
        self.vtk_model = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', 'MPMKidneyResection')
        self.vtk_model.SetAndObservePolyData(poly)
        self.vtk_model.CreateDefaultDisplayNodes()
        dn = self.vtk_model.GetDisplayNode()
        dn.SetPointSize(3); dn.SetScalarVisibility(True)
        dn.SetActiveScalarName("Colors")
        dn.SetScalarRangeFlagFromString("UseDirectMapping")
        self._vtk_points = poly.GetPoints()
        self._vtk_poly   = poly

    def update_model(self):
        if self._vtk_points is None:
            return
        import vtk, vtk.util.numpy_support as ns
        self._vtk_points.SetData(ns.numpy_to_vtk(self._positions_mm(), deep=True,
                                                 array_type=vtk.VTK_FLOAT))
        self._vtk_points.Modified(); self._vtk_poly.Modified()
        if self.vtk_model:
            self.vtk_model.GetPolyData().Modified()

    def _setup_view(self):
        import slicer
        lm = slicer.app.layoutManager()
        lm.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        slicer.util.setSliceViewerLayers(background=self.reference_volume)
        slicer.app.processEvents()
        slicer.util.resetThreeDViews()
        slicer.app.processEvents()

    # ------------------------------------------------------------------
    # Sequence recording
    # ------------------------------------------------------------------

    def setup_sequence(self):
        """Create the recording sequence only.

        The playback browser/proxy is created later in finalize_playback() so
        that during the run there is a single visible cloud (the live authoring
        model) with no overlapping proxy.
        """
        import slicer
        self.sequence_node = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLSequenceNode', 'KidneyResectionSeq')
        self.sequence_node.SetIndexName('time')
        self.sequence_node.SetIndexUnit('s')
        self.cautery_sequence_node = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLSequenceNode', 'KidneyResectionCauterySeq')
        self.cautery_sequence_node.SetIndexName('time')
        self.cautery_sequence_node.SetIndexUnit('s')
        self.browser_node = None

    def _record_frame(self):
        if self.sequence_node is None or self.vtk_model is None:
            return
        tval = "%.2f" % self._t
        self.sequence_node.SetDataNodeAtValue(self.vtk_model, tval)
        if self.cautery_sequence_node is not None and self._cautery_transform is not None:
            self.cautery_sequence_node.SetDataNodeAtValue(self._cautery_transform, tval)

    def finalize_playback(self):
        """Build the playback browser and show ONLY the animated proxy.

        The live model (self.vtk_model) is updated in place during the run; the
        sequence stores per-frame copies.  When the browser observes the
        sequence it creates its own proxy model node.  We hide the live model so
        scrubbing/playing shows a single, correctly animated cloud -- otherwise
        the static live model (frozen at the final pose) overlaps and masks the
        proxy's motion.  Call once, after all phases are recorded.
        """
        import slicer
        if self.sequence_node is None:
            return None
        if self.browser_node is None:
            self.browser_node = slicer.mrmlScene.AddNewNodeByClass(
                'vtkMRMLSequenceBrowserNode', 'KidneyResectionBrowser')
            self.browser_node.SetAndObserveMasterSequenceNodeID(self.sequence_node.GetID())
        # Animate the cautery during playback by syncing its transform sequence.
        if (self.cautery_sequence_node is not None
                and self._cautery_transform is not None
                and self.cautery_sequence_node.GetNumberOfDataNodes() > 0):
            try:
                slicer.modules.sequences.logic().AddSynchronizedNode(
                    self.cautery_sequence_node, self._cautery_transform, self.browser_node)
            except Exception:
                pass
        try:
            slicer.modules.sequences.logic().UpdateProxyNodesFromSequences(self.browser_node)
        except Exception:
            pass
        try:
            self.browser_node.SetPlaybackRateFps(1.0 / RECORD_DT_S)
        except Exception:
            pass
        proxy = self.browser_node.GetProxyNode(self.sequence_node)
        if proxy is not None and proxy.GetDisplayNode() is not None:
            pdn = proxy.GetDisplayNode()
            pdn.SetVisibility(True)
            pdn.SetPointSize(4)
            pdn.SetScalarVisibility(True)
            pdn.SetActiveScalarName("Colors")
            pdn.SetScalarRangeFlagFromString("UseDirectMapping")
        if self.vtk_model.GetDisplayNode() is not None:
            self.vtk_model.GetDisplayNode().SetVisibility(False)
        self.proxy_model = proxy
        return proxy

    def show_live_model(self):
        """Re-show the live authoring model (e.g. to run another phase)."""
        if self.vtk_model and self.vtk_model.GetDisplayNode():
            self.vtk_model.GetDisplayNode().SetVisibility(True)
        if self.proxy_model and self.proxy_model.GetDisplayNode():
            self.proxy_model.GetDisplayNode().SetVisibility(False)

    # ------------------------------------------------------------------
    # Cautery resection phase
    # ------------------------------------------------------------------

    def _mass_sdf_grid(self):
        """Signed distance (m) to the current mass-particle region, on the MPM
        grid.  Negative inside the mass, positive outside.  Shape (ng,ng,ng).

        Recomputed from the *current* deformed positions, so the resection
        surface (and thus the cautery trajectory) tracks the tissue as it moves
        and as earlier strokes release it.
        """
        from scipy.ndimage import distance_transform_edt
        ng  = self.sim.n_grid
        dx  = self.sim.dx
        inv = self.sim.inv_dx
        mp = self.sim.get_positions()[self._is_mass]
        occ = np.zeros((ng, ng, ng), dtype=bool)
        gi = np.clip(np.round(mp[:, 0] * inv).astype(int), 0, ng - 1)
        gj = np.clip(np.round(mp[:, 1] * inv).astype(int), 0, ng - 1)
        gk = np.clip(np.round(mp[:, 2] * inv).astype(int), 0, ng - 1)
        occ[gi, gj, gk] = True
        d_out = distance_transform_edt(~occ).astype(np.float32) * dx
        d_in  = distance_transform_edt( occ).astype(np.float32) * dx
        return np.where(occ, -d_in, d_out).astype(np.float32)

    def _grid_x_coords(self):
        ng = self.sim.n_grid
        dx = self.sim.dx
        X = np.zeros((ng, ng, ng), dtype=np.float32)
        X += (np.arange(ng) * dx).astype(np.float32)[:, None, None]
        return X

    def _cut_front_x(self, frac):
        """Absolute grid-x of the cautery front for a swept fraction of the
        CURRENT mass extent.  Re-evaluated each stroke, so the sweep tracks the
        moving (still-retracting) specimen instead of drifting off it.
        """
        margin_m = CUT_MARGIN_MM / 1000.0
        mx = self.sim.get_positions()[self._is_mass, 0]
        lo = float(mx.min()) - margin_m
        hi = float(mx.max()) + margin_m
        return lo + float(frac) * (hi - lo)

    def _build_cut_sdf(self, mass_sdf, frac):
        """Cumulative resection cut SDF, swept up to `frac` of the current mass
        x-extent (left -> right).

        Cut surface = isosurface {mass_sdf = margin}, i.e. margin_m into the
        kidney from the mass boundary (margin sits on the kidney side, so the
        specimen = mass + margin cuff).  Sign < 0 inside the specimen shell,
        > 0 in the kidney beyond.  Ahead of the cautery front (x >= x_front)
        the field is a large positive constant (uncut kidney side).  One
        coherent signed field = the single composite cut step() consumes via
        cut_sdfs[-1].
        """
        dx = self.sim.dx
        margin_m = CUT_MARGIN_MM / 1000.0
        x_front = self._cut_front_x(frac)
        cut = (mass_sdf - margin_m).astype(np.float32)
        X = self._grid_x_coords()
        active = X < x_front
        large = np.float32(10.0 * dx)
        return np.where(active, cut, large).astype(np.float32).ravel()

    def _create_cautery_model(self):
        if self._cautery_model is not None:
            return
        import slicer, vtk
        src = vtk.vtkCylinderSource()
        src.SetRadius(1.5)
        src.SetHeight(CAUTERY_LEN_MM)
        src.SetResolution(16)
        src.Update()
        self._cautery_model = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLModelNode', 'MPMCautery')
        self._cautery_model.SetAndObservePolyData(src.GetOutput())
        self._cautery_model.CreateDefaultDisplayNodes()
        dn = self._cautery_model.GetDisplayNode()
        dn.SetColor(*COLOR_CAUTERY)
        dn.SetOpacity(1.0)
        self._cautery_transform = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLLinearTransformNode', 'MPMCauteryXform')
        self._cautery_model.SetAndObserveTransformNodeID(
            self._cautery_transform.GetID())

    def _update_cautery(self, mass_sdf, frac):
        """Place the cautery on the resection surface at the current front."""
        import vtk
        if self._cautery_transform is None:
            return
        dx = self.sim.dx
        margin_m = CUT_MARGIN_MM / 1000.0
        x_front = self._cut_front_x(frac)
        band = np.abs(mass_sdf - margin_m) < dx
        X = self._grid_x_coords()
        slab = (X >= x_front - CAUTERY_LEN_MM / 1000.0) & (X < x_front)
        sel = band & slab
        if not sel.any():
            sel = band
        if not sel.any():
            return
        idx = np.argwhere(sel).astype(np.float64)
        centroid_ras = idx.mean(0) * dx * 1000.0 + self._ras_offset_mm
        m = vtk.vtkMatrix4x4(); m.Identity()
        m.SetElement(0, 3, float(centroid_ras[0]))
        m.SetElement(1, 3, float(centroid_ras[1]))
        m.SetElement(2, 3, float(centroid_ras[2]))
        self._cautery_transform.SetMatrixTransformToParent(m)

    def _begin_cut(self):
        """Initialise the cut phase: count strokes from the current mass extent
        and enable continued retraction."""
        margin_m = CUT_MARGIN_MM / 1000.0
        cl_m     = CAUTERY_LEN_MM / 1000.0
        mx = self.sim.get_positions()[self._is_mass, 0]
        extent = (float(mx.max()) - float(mx.min())) + 2.0 * margin_m
        self._n_strokes = max(1, int(np.ceil(extent / cl_m)))
        self._cut_started = True
        self._retracting = True
        self._retract_start_t = self._t
        print("MPMKidneyResection: cut phase begins at t=%.1f s, %d strokes"
              % (self._t, self._n_strokes))

    def _apply_cut_no_reset(self, sdf_np):
        """Register the composite cut and break crossing bonds, but WITHOUT
        resetting F/x0 near the cut.

        MPMSimulator.apply_cut re-references near-cut particles to their current
        (deformed) pose, which would freeze the freed kidney edge there.  We
        skip that so the kidney retains its elastic state and springs back to
        rest once the specimen is severed.
        """
        import newton_tissue.mpm as _ntm
        with wp.ScopedDevice(self.device):
            cut_sdf = wp.array(sdf_np.astype(np.float32), dtype=float)
            self.sim.cut_sdfs = [cut_sdf]       # single composite cut
            if self.sim.n_bonds > 0 and self.sim.fiber_broken is not None:
                wp.launch(_ntm._break_bonds_across_cut, dim=self.sim.n_bonds,
                          inputs=[self.sim.x, self.sim.fiber_i, self.sim.fiber_j,
                                  self.sim.fiber_broken, cut_sdf,
                                  self.sim.n_grid, float(self.sim.inv_dx)])
        self._sever_neighbors_across_cut(sdf_np)

    def _sever_neighbors_across_cut(self, sdf_np):
        """Cut the total-Lagrangian lattice-neighbour links that cross the cut,
        so F is never estimated across the gap (which would otherwise give a
        huge, unstable deformation gradient)."""
        ng = self.sim.n_grid
        inv = self.sim.inv_dx
        pos = self.sim.get_positions()
        gi = np.clip(np.round(pos[:, 0] * inv).astype(int), 0, ng - 1)
        gj = np.clip(np.round(pos[:, 1] * inv).astype(int), 0, ng - 1)
        gk = np.clip(np.round(pos[:, 2] * inv).astype(int), 0, ng - 1)
        p_sdf = sdf_np[gi * ng * ng + gj * ng + gk]
        for attr in ('nbr_px', 'nbr_mx', 'nbr_py', 'nbr_my', 'nbr_pz', 'nbr_mz'):
            arr = getattr(self.sim, attr, None)
            if arr is None:
                continue
            nbr = arr.numpy()
            has = nbr >= 0
            cross = np.zeros(len(nbr), dtype=bool)
            cross[has] = (p_sdf[has] * p_sdf[nbr[has]]) < 0.0
            if cross.any():
                nbr[cross] = -1
                with wp.ScopedDevice(self.device):
                    setattr(self.sim, attr, wp.array(nbr, dtype=int))

    def _disconnect_specimen(self):
        """Guarantee complete severance: break every remaining bond linking the
        specimen (mass + 3 mm cuff) to the rest of the kidney, so the two bodies
        respond independently."""
        spec = (self._is_mass | self._is_cuff)
        fi = self.sim.fiber_i.numpy(); fj = self.sim.fiber_j.numpy()
        broken = self.sim.fiber_broken.numpy()
        cross = spec[fi] != spec[fj]
        n_new = int((cross & (broken == 0)).sum())
        broken[cross] = 1
        with wp.ScopedDevice(self.device):
            self.sim.fiber_broken = wp.array(broken.astype(np.int32), dtype=int)
        # Also sever the total-Lagrangian neighbour links across the boundary.
        for attr in ('nbr_px', 'nbr_mx', 'nbr_py', 'nbr_my', 'nbr_pz', 'nbr_mz'):
            arr = getattr(self.sim, attr, None)
            if arr is None:
                continue
            nbr = arr.numpy()
            has = nbr >= 0
            xcut = np.zeros(len(nbr), dtype=bool)
            xcut[has] = spec[has] != spec[nbr[has]]
            if xcut.any():
                nbr[xcut] = -1
                with wp.ScopedDevice(self.device):
                    setattr(self.sim, attr, wp.array(nbr, dtype=int))
        self._disconnected = True
        print("MPMKidneyResection: specimen fully disconnected (+%d bonds, %d total)"
              % (n_new, int(broken.sum())))
        return n_new

    def _apply_stroke(self, idx):
        """Apply the cumulative composite cut up to stroke `idx` and move the
        cautery there."""
        frac = float(idx + 1) / float(self._n_strokes)
        mass_sdf = self._mass_sdf_grid()
        sdf = self._build_cut_sdf(mass_sdf, frac)
        self._apply_cut_no_reset(sdf)
        self._update_cautery(mass_sdf, frac)

    def _scenario_progress(self):
        """Drive the pull -> settle -> cut(strokes) -> settle state machine,
        evaluated once per substep (cheap unless a stroke fires)."""
        cut_start = T_PULL_S + SETTLE_S
        if self._t < T_PULL_S:
            self._phase = 'pull'
            return
        if self._t < cut_start:
            self._phase = 'settle'
            return
        if not self._cut_started:
            self._begin_cut()
        if self._strokes_done < self._n_strokes:
            next_t = cut_start + self._strokes_done * (T_CUT_S + T_PAUSE_S)
            if self._t >= next_t:
                self._apply_stroke(self._strokes_done)
                self._strokes_done += 1
            self._phase = 'cut %d/%d' % (self._strokes_done, self._n_strokes)
        else:
            if not self._disconnected:
                self._disconnect_specimen()      # guarantee full severance
            last_end = cut_start + self._n_strokes * (T_CUT_S + T_PAUSE_S)
            if self._t >= last_end and self._retracting:
                self._retracting = False         # hold the mass; let kidney relax
            self._phase = 'final settle'
            if self._t >= last_end + FINAL_SETTLE_S:
                self._scenario_done = True

    def start_run(self):
        """Run the full pull + resection scenario non-blocking via a Qt timer.

        Returns immediately; poll status().  Keeps Slicer responsive and avoids
        the MCP call timeout on the multi-minute GPU run.
        """
        self._create_cautery_model()
        self._update_cautery(self._mass_sdf_grid(), 0.0)   # park near the start
        self._record_frame()
        self._run_active    = True
        self._scenario_done = False
        self._cut_started   = False
        self._strokes_done  = 0
        self._phase         = 'pull'
        self._schedule_tick()
        return self.status()

    def _schedule_tick(self):
        import qt
        qt.QTimer.singleShot(TICK_INTERVAL_MS, self._tick)

    def _tick(self):
        if not self._run_active:
            return
        import slicer
        for _ in range(STEPS_PER_TICK):
            self._step()
            if self._t + 1e-9 >= self._next_record_t:
                self._record_frame()
                self._next_record_t += RECORD_DT_S
            self._scenario_progress()
            if self._scenario_done:
                break
        self.update_model()
        slicer.app.processEvents()
        if self._scenario_done:
            self._finish_run()
        else:
            self._schedule_tick()

    def _finish_run(self):
        self._run_active = False
        self.update_model()
        self.finalize_playback()
        print("MPMKidneyResection: scenario complete - t=%.1f s, %d frames, "
              "%d/%d bonds broken"
              % (self._t, self.sequence_node.GetNumberOfDataNodes(),
                 int(self.sim.fiber_broken.numpy().sum()), int(self.sim.n_bonds)))

    def stop_run(self):
        self._run_active = False

    def status(self):
        return {
            't_s': round(self._t, 2),
            'phase': self._phase,
            'strokes_done': self._strokes_done,
            'n_strokes': self._n_strokes,
            'run_active': self._run_active,
            'done': self._scenario_done,
            'retracting': self._retracting,
            'n_frames': self.sequence_node.GetNumberOfDataNodes() if self.sequence_node else 0,
            'broken_bonds': int(self.sim.fiber_broken.numpy().sum()),
            'nan': bool(np.isnan(self.sim.get_positions()).any()),
        }

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self):
        self.create_model()
        self.setup_sequence()
        self._record_frame()   # t = 0 frame
        self._setup_view()
        return self


def run(start=True):
    """Build the demo, attach it to the slicer object, and start the scenario.

    The full pull + cautery resection runs non-blocking via a Qt timer; poll
    inst.status().  Pass start=False to only build (then drive manually with
    inst.advance(seconds) or inst.start_run()).  State persists on
    slicer.kidneyResection.
    """
    import slicer
    inst = MPMKidneyResection()
    inst.run()
    slicer.kidneyResection = inst
    if start:
        inst.start_run()
    return inst


if __name__ == '__main__':
    print("MPMKidneyResection is intended to run inside Slicer (MCP).")
