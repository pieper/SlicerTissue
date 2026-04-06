"""MPM tissue simulation driven by CT head geometry and HU values.

Creates an MPM particle simulation where:
  - The Eulerian grid covers the CT volume bounds
  - Particles are placed on a regular sub-cell lattice
  - HU values control particle classification:
      bone  (HU > 300):  fixed boundary condition
      air   (HU < -200): skipped entirely
      tissue (else):     soft tissue material with fiber bonds
  - Gravity acts in the inferior (-S) direction in RAS coordinates
  - A Qt slider on the toolbar controls gravity magnitude
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
# GPU kernel for deformed CT resampling
# ---------------------------------------------------------------------------

@wp.func
def _trilinear_sample_f(data: wp.array(dtype=float),
                         nx: int, ny: int, nz: int,
                         fx: float, fy: float, fz: float) -> float:
    """Trilinear interpolation of a flat 3-D float array (IJK layout)."""
    ix = int(wp.floor(fx));  iy = int(wp.floor(fy));  iz = int(wp.floor(fz))
    dx = fx - float(ix);     dy = fy - float(iy);     dz = fz - float(iz)
    ix = wp.clamp(ix, 0, nx - 2);  iy = wp.clamp(iy, 0, ny - 2)
    iz = wp.clamp(iz, 0, nz - 2)
    i000 = ix * ny * nz + iy * nz + iz
    i001 = i000 + 1;           i010 = i000 + nz
    i011 = i010 + 1;           i100 = i000 + ny * nz
    i101 = i100 + 1;           i110 = i100 + nz;   i111 = i110 + 1
    c00 = data[i000] * (1.0 - dz) + data[i001] * dz
    c01 = data[i010] * (1.0 - dz) + data[i011] * dz
    c10 = data[i100] * (1.0 - dz) + data[i101] * dz
    c11 = data[i110] * (1.0 - dz) + data[i111] * dz
    c0 = c00 * (1.0 - dy) + c01 * dy
    c1 = c10 * (1.0 - dy) + c11 * dy
    return c0 * (1.0 - dx) + c1 * dx


@wp.kernel
def _resample_deformed_ct(
    # displacement grid (ng^3 flat, 3 separate arrays) [metres]
    disp_x: wp.array(dtype=float),
    disp_y: wp.array(dtype=float),
    disp_z: wp.array(dtype=float),
    mass:   wp.array(dtype=float),
    cut_sdf: wp.array(dtype=float),
    ng: int,
    # CT data (flat float array in IJK order)
    ct_data: wp.array(dtype=float),
    ct_ni: int, ct_nj: int, ct_nk: int,
    # output IJKToRAS (row-major, maps output IJK → RAS mm)
    a00: float, a01: float, a02: float, a03: float,
    a10: float, a11: float, a12: float, a13: float,
    a20: float, a21: float, a22: float, a23: float,
    # RAS-to-CT-IJK (row-major, maps RAS mm → source CT IJK)
    b00: float, b01: float, b02: float, b03: float,
    b10: float, b11: float, b12: float, b13: float,
    b20: float, b21: float, b22: float, b23: float,
    # output dims
    out_ni: int, out_nj: int, out_nk: int,
    # MPM grid origin (RAS mm) and spacing (mm)
    mpm_ox: float, mpm_oy: float, mpm_oz: float,
    mpm_dx_mm: float,
    # output
    output: wp.array(dtype=float),
):
    tid = wp.tid()
    oi = tid % out_ni
    oj = (tid // out_ni) % out_nj
    ok = tid // (out_ni * out_nj)

    fi = float(oi); fj = float(oj); fk = float(ok)

    # output IJK → RAS mm (full matrix, handles rotation + spacing)
    rx = a00 * fi + a01 * fj + a02 * fk + a03
    ry = a10 * fi + a11 * fj + a12 * fk + a13
    rz = a20 * fi + a21 * fj + a22 * fk + a23

    # RAS mm → MPM grid float coords
    mx = (rx - mpm_ox) / mpm_dx_mm
    my = (ry - mpm_oy) / mpm_dx_mm
    mz = (rz - mpm_oz) / mpm_dx_mm

    # Check if inside the MPM grid; if outside, pass through original CT
    if mx < 0.0 or mx > float(ng - 1) or my < 0.0 or my > float(ng - 1) or mz < 0.0 or mz > float(ng - 1):
        # Outside MPM domain — sample original CT undeformed
        ci = b00 * rx + b01 * ry + b02 * rz + b03
        cj = b10 * rx + b11 * ry + b12 * rz + b13
        ck = b20 * rx + b21 * ry + b22 * rz + b23
        output[tid] = _trilinear_sample_f(ct_data, ct_ni, ct_nj, ct_nk, ci, cj, ck)
        return

    # Trilinear sample displacement (metres)
    ddx = _trilinear_sample_f(disp_x, ng, ng, ng, mx, my, mz)
    ddy = _trilinear_sample_f(disp_y, ng, ng, ng, mx, my, mz)
    ddz = _trilinear_sample_f(disp_z, ng, ng, ng, mx, my, mz)

    # Source RAS = output RAS − displacement (m → mm)
    sx = rx - ddx * 1000.0
    sy = ry - ddy * 1000.0
    sz = rz - ddz * 1000.0

    # Source RAS → CT IJK
    ci = b00 * sx + b01 * sy + b02 * sz + b03
    cj = b10 * sx + b11 * sy + b12 * sz + b13
    ck = b20 * sx + b21 * sy + b22 * sz + b23

    hu = _trilinear_sample_f(ct_data, ct_ni, ct_nj, ct_nk, ci, cj, ck)

    # Wound mask: near cut, no mass → air
    m_val = _trilinear_sample_f(mass, ng, ng, ng, mx, my, mz)
    s_val = _trilinear_sample_f(cut_sdf, ng, ng, ng, mx, my, mz)
    if m_val < 1.0e-10 and wp.abs(s_val) > 0.0 and hu > -500.0:
        hu = -1000.0

    output[tid] = hu


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DT = 2e-4
VELOCITY_DAMPING = 0.995

MATERIAL = MPMMaterial(E=10_000.0, nu=0.48, rho=1_060.0,
                       k_elastin=0.05, k_collagen=0.25, collagen_crimp=0.05,
                       k_curve=0.0)

HU_AIR_MAX  = -200
HU_BONE_MIN = 300

EARTH_G = 9.81   # m/s²


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_ct_head(cache_dir=None):
    """Download CTHead.nrrd from Google Drive, return local path."""
    if cache_dir is None:
        cache_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'tests', 'artifacts')
    os.makedirs(cache_dir, exist_ok=True)

    cached = os.path.join(cache_dir, 'CTHead.nrrd')
    if os.path.exists(cached):
        return cached

    file_id = '1a0tt9_Uu7whrYs2VKbBezwKi7gJGG823'
    url = f'https://drive.google.com/uc?export=download&id={file_id}&confirm=t'
    import urllib.request
    print(f"Downloading CTHead to {cached} …")
    urllib.request.urlretrieve(url, cached)
    print(f"  {os.path.getsize(cached) / 1e6:.1f} MB")
    return cached


# ---------------------------------------------------------------------------
# MPMCTHead
# ---------------------------------------------------------------------------

class MPMCTHead:
    """MPM tissue simulation seeded from a CT volume."""

    def __init__(self, volume_node, dx_mm=4.0, ppc=2, device=None):
        """
        Args:
            volume_node:  vtkMRMLScalarVolumeNode with CT (HU) data.
            dx_mm:        Eulerian grid spacing [mm].  2 mm gives highest
                          fidelity but needs a large GPU; 4 mm is practical.
            ppc:          Particles per cell per dimension (2 → 8 per cell).
        """
        wp.init()
        if device is None:
            device = "cpu"
            try:
                _ = wp.zeros(1, dtype=float, device="cuda:0")
                device = "cuda:0"
            except Exception:
                pass
        self.device = device
        self.volume_node = volume_node

        # --- Volume geometry in RAS mm --------------------------------
        bounds = [0.0] * 6
        volume_node.GetRASBounds(bounds)
        ras_lo = np.array([bounds[0], bounds[2], bounds[4]])
        ras_hi = np.array([bounds[1], bounds[3], bounds[5]])

        # Convert to metres with one-cell margin
        margin_m = dx_mm / 1000.0
        lo_m = ras_lo / 1000.0 - margin_m
        hi_m = ras_hi / 1000.0 + margin_m
        block_size = float((hi_m - lo_m).max())
        n_grid = int(np.ceil(block_size / (dx_mm / 1000.0)))
        dx_m = block_size / n_grid

        self._ras_offset_mm = lo_m * 1000.0   # add to sim coords for RAS mm
        self._lo_m = lo_m

        # --- MPM simulator --------------------------------------------
        self.sim = MPMSimulator(
            block_lo=np.zeros(3),
            block_hi=np.full(3, block_size),
            n_grid=n_grid, dt=DT,
            material=MATERIAL,
            device=device,
            velocity_damping=VELOCITY_DAMPING,
            total_lagrangian=False,
        )

        # --- Particles ------------------------------------------------
        self._populate_particles(volume_node, lo_m, hi_m, dx_m, ppc)

        # --- Gravity --------------------------------------------------
        # Inferior = −S in RAS = −Z in our sim (R→X, A→Y, S→Z)
        self._gravity_scale = 1.0
        self.gravity = np.array([0.0, 0.0, -EARTH_G])

        # Warm-up: let tissue settle under gravity
        print("MPMCTHead: gravity warm-up …")
        for _ in range(400):
            self.sim.step(self.gravity)
        self.sim.sample_equilibrium()
        print("MPMCTHead: ready.")

        # UI / loop state
        self.vtk_model      = None
        self._vtk_points    = None
        self._vtk_poly      = None
        self._loop_running  = False
        self._steps_per_tick = 10
        self._tick_interval_ms = 50
        self._idle_ticks    = 0
        self._idle_ticks_to_stop = 20
        self._prev_tick_pos = None
        self._observer_tags = []
        self._contact_sphere = None
        self._slider        = None
        self._toolbar       = None
        self._deformed_ct        = None
        self._last_ct_update     = 0.0
        self._ct_update_interval = 0.2   # seconds between CT resamples

    # ------------------------------------------------------------------
    # Particle setup
    # ------------------------------------------------------------------

    def _populate_particles(self, vol_node, lo_m, hi_m, dx_m, ppc):
        """Place particles, sample HU, classify, upload to GPU."""
        import vtk

        step = dx_m / ppc          # particle spacing [m]
        off  = 0.5 * step
        vol_extent = hi_m - lo_m   # sim-space extents [m]

        xs = np.arange(off, vol_extent[0], step)
        ys = np.arange(off, vol_extent[1], step)
        zs = np.arange(off, vol_extent[2], step)
        nx, ny, nz = len(xs), len(ys), len(zs)
        ix, iy, iz = np.meshgrid(xs, ys, zs, indexing='ij')
        pos = np.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=1).astype(np.float32)

        # --- Sample HU -----------------------------------------------
        # Convert sim coords → RAS mm
        pos_ras = pos * 1000.0 + self._ras_offset_mm

        # Build RAS→IJK matrix as numpy 4×4
        ras2ijk = vtk.vtkMatrix4x4()
        vol_node.GetRASToIJKMatrix(ras2ijk)
        M = np.array([[ras2ijk.GetElement(r, c) for c in range(4)]
                       for r in range(4)])

        homo = np.column_stack([pos_ras, np.ones(len(pos_ras), dtype=np.float32)])
        ijk = (M @ homo.T).T[:, :3]
        ijk_int = np.round(ijk).astype(np.int32)

        # Volume array (Slicer returns KJI order)
        arr = self._volume_array(vol_node)       # shape (K, J, I)
        dims_ijk = np.array([arr.shape[2], arr.shape[1], arr.shape[0]])

        in_bounds = ((ijk_int >= 0) & (ijk_int < dims_ijk)).all(axis=1)
        hu = np.full(len(pos), -1000.0, dtype=np.float32)
        valid = ijk_int[in_bounds]
        hu[in_bounds] = arr[valid[:, 2], valid[:, 1], valid[:, 0]].astype(np.float32)

        # --- Classify -------------------------------------------------
        is_bone   = hu > HU_BONE_MIN
        is_air    = hu < HU_AIR_MAX
        is_tissue = ~is_bone & ~is_air
        keep      = is_tissue | is_bone

        pos  = pos[keep]
        hu   = hu[keep]
        is_bone_k   = is_bone[keep]
        is_tissue_k = is_tissue[keep]

        n = len(pos)
        self.sim.n_particles = n
        vol_p  = float(step ** 3)
        mass_p = float(MATERIAL.rho * vol_p)

        F_np = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
        C_np = np.zeros((n, 3, 3), dtype=np.float32)

        with wp.ScopedDevice(self.device):
            self.sim.x     = wp.array(pos,                        dtype=wp.vec3)
            self.sim.x0    = wp.array(pos.copy(),                 dtype=wp.vec3)
            self.sim.v     = wp.zeros(n,                          dtype=wp.vec3)
            self.sim.F     = wp.array(F_np,                       dtype=wp.mat33)
            self.sim.C     = wp.array(C_np,                       dtype=wp.mat33)
            self.sim.m_p   = wp.array(np.full(n, mass_p, dtype=np.float32), dtype=float)
            self.sim.vol_p = wp.array(np.full(n, vol_p,  dtype=np.float32), dtype=float)
            self.sim.fixed = wp.array(is_bone_k.astype(np.int32), dtype=int)

        self._hu       = hu
        self._is_bone  = is_bone_k
        self._is_tissue = is_tissue_k

        # --- Fiber bonds (tissue↔tissue only) -------------------------
        # Build a 3-D index map for fast lattice-based neighbour lookup.
        # Particles that were kept occupy a subset of the full (nx,ny,nz) lattice.
        keep_flat = np.where(keep)[0]                     # original flat indices
        idx_3d = np.unravel_index(keep_flat, (nx, ny, nz))  # (ix, iy, iz) arrays
        lookup = np.full((nx, ny, nz), -1, dtype=np.int32)
        new_idx = np.arange(n, dtype=np.int32)
        lookup[idx_3d[0], idx_3d[1], idx_3d[2]] = new_idx

        if MATERIAL.k_elastin > 0.0 or MATERIAL.k_collagen > 0.0:
            self._build_bonds(lookup, pos, is_tissue_k, nx, ny, nz, step)

        # --- Bone contact: grid-level BC + particle-level SDF projection ---
        self._build_bone_grid_mask(pos, is_bone_k)
        self._build_bone_sdf(pos, is_bone_k)

        # --- Tissue surface SDF (for cutting outward-normal estimation) ---
        self._build_tissue_sdf(pos, is_tissue_k | is_bone_k)

        n_bone = int(is_bone_k.sum())
        n_tissue = int(is_tissue_k.sum())
        print(f"MPMCTHead: {n} particles ({n_tissue} tissue, {n_bone} bone), "
              f"dx={self.sim.dx*1000:.2f} mm, n_grid={self.sim.n_grid}")

    @staticmethod
    def _volume_array(vol_node):
        """Return the volume as a numpy array (K, J, I)."""
        try:
            import slicer
            return slicer.util.arrayFromVolume(vol_node)
        except ImportError:
            import vtk.util.numpy_support as ns
            img = vol_node.GetImageData()
            sc = img.GetPointData().GetScalars()
            dims = img.GetDimensions()
            return ns.vtk_to_numpy(sc).reshape(dims[2], dims[1], dims[0])

    def _build_bone_grid_mask(self, pos, is_bone):
        """Mark grid nodes influenced by bone particles as fixed BCs.

        For each bone particle, the quadratic B-spline stencil touches a
        3×3×3 block of grid nodes.  All of these are marked fixed so the
        grid velocity is zeroed there every step, acting as a rigid wall.
        """
        ng     = self.sim.n_grid
        inv_dx = self.sim.inv_dx
        bone_pos = pos[is_bone]
        if len(bone_pos) == 0:
            return

        mask = np.zeros(ng ** 3, dtype=np.int32)

        # Vectorised: compute base cell for every bone particle
        base = np.floor(bone_pos * inv_dx - 0.5).astype(np.int32)  # (n_bone, 3)

        for di in range(3):
            for dj in range(3):
                for dk in range(3):
                    gi = base[:, 0] + di
                    gj = base[:, 1] + dj
                    gk = base[:, 2] + dk
                    valid = ((gi >= 0) & (gi < ng) &
                             (gj >= 0) & (gj < ng) &
                             (gk >= 0) & (gk < ng))
                    flat = gi[valid] * ng * ng + gj[valid] * ng + gk[valid]
                    mask[flat] = 1

        with wp.ScopedDevice(self.device):
            self.sim.grid_bc_fixed = wp.array(mask, dtype=int)

    def _build_bone_sdf(self, pos, is_bone):
        """Build a signed distance field of the bone on the MPM grid.

        SDF < 0 means inside bone.  The gradient points outward (away from
        bone interior).  Used by _apply_bone_sdf_contact to project tissue
        particles out of bone each step.
        """
        from scipy.ndimage import distance_transform_edt

        ng  = self.sim.n_grid
        dx  = self.sim.dx     # [m]
        inv_dx = self.sim.inv_dx

        # Sample bone occupancy on the MPM grid
        bone_grid = np.zeros((ng, ng, ng), dtype=bool)
        bone_pos = pos[is_bone]
        if len(bone_pos) == 0:
            return

        gi = np.clip(np.round(bone_pos[:, 0] * inv_dx).astype(int), 0, ng - 1)
        gj = np.clip(np.round(bone_pos[:, 1] * inv_dx).astype(int), 0, ng - 1)
        gk = np.clip(np.round(bone_pos[:, 2] * inv_dx).astype(int), 0, ng - 1)
        bone_grid[gi, gj, gk] = True

        # EDT: distance from each non-bone node to nearest bone surface
        dist_outside = distance_transform_edt(~bone_grid).astype(np.float32) * dx
        dist_inside  = distance_transform_edt( bone_grid).astype(np.float32) * dx

        # SDF: positive outside, negative inside
        sdf_3d = np.where(bone_grid, -dist_inside, dist_outside)

        # Gradient via central differences (outward normal)
        grad = np.zeros((ng, ng, ng, 3), dtype=np.float32)
        grad[1:-1, :, :, 0] = (sdf_3d[2:, :, :] - sdf_3d[:-2, :, :]) / (2.0 * dx)
        grad[:, 1:-1, :, 1] = (sdf_3d[:, 2:, :] - sdf_3d[:, :-2, :]) / (2.0 * dx)
        grad[:, :, 1:-1, 2] = (sdf_3d[:, :, 2:] - sdf_3d[:, :, :-2]) / (2.0 * dx)

        sdf_flat  = sdf_3d.ravel().astype(np.float32)
        grad_flat = grad.reshape(-1, 3).astype(np.float32)

        with wp.ScopedDevice(self.device):
            self.sim.bone_sdf      = wp.array(sdf_flat,  dtype=float)
            self.sim.bone_sdf_grad = wp.array(grad_flat, dtype=wp.vec3)

        self._bone_sdf_3d = sdf_3d   # (ng, ng, ng) for visualization
        print(f"MPMCTHead: bone SDF built — "
              f"{int(bone_grid.sum())} bone grid nodes, "
              f"SDF range [{sdf_flat.min()*1000:.1f}, {sdf_flat.max()*1000:.1f}] mm")

    def _build_tissue_sdf(self, pos, is_body):
        """Build a signed distance field of the tissue+bone body surface.

        SDF > 0 outside the body, < 0 inside.  The gradient at the surface
        points outward — used by build_scalpel_sdf() to orient the cutting
        ribbon perpendicular to the tissue surface.

        Args:
            pos:      (n, 3) particle positions [m].
            is_body:  (n,) bool mask — True for tissue OR bone particles
                      (everything that forms the solid body).
        """
        from scipy.ndimage import distance_transform_edt

        ng     = self.sim.n_grid
        dx     = self.sim.dx
        inv_dx = self.sim.inv_dx

        body_grid = np.zeros((ng, ng, ng), dtype=bool)
        body_pos  = pos[is_body]
        if len(body_pos) == 0:
            return

        gi = np.clip(np.round(body_pos[:, 0] * inv_dx).astype(int), 0, ng - 1)
        gj = np.clip(np.round(body_pos[:, 1] * inv_dx).astype(int), 0, ng - 1)
        gk = np.clip(np.round(body_pos[:, 2] * inv_dx).astype(int), 0, ng - 1)
        body_grid[gi, gj, gk] = True

        dist_outside = distance_transform_edt(~body_grid).astype(np.float32) * dx
        dist_inside  = distance_transform_edt( body_grid).astype(np.float32) * dx

        sdf_3d = np.where(body_grid, -dist_inside, dist_outside)

        # Gradient via central differences (outward normal)
        grad = np.zeros((ng, ng, ng, 3), dtype=np.float32)
        grad[1:-1, :, :, 0] = (sdf_3d[2:, :, :] - sdf_3d[:-2, :, :]) / (2.0 * dx)
        grad[:, 1:-1, :, 1] = (sdf_3d[:, 2:, :] - sdf_3d[:, :-2, :]) / (2.0 * dx)
        grad[:, :, 1:-1, 2] = (sdf_3d[:, :, 2:] - sdf_3d[:, :, :-2]) / (2.0 * dx)

        grad_flat = grad.reshape(-1, 3).astype(np.float32)

        with wp.ScopedDevice(self.device):
            self.sim.tissue_sdf_grad = wp.array(grad_flat, dtype=wp.vec3)

        self._tissue_sdf_3d = sdf_3d  # (ng, ng, ng) for visualization
        print(f"MPMCTHead: tissue SDF built — "
              f"{int(body_grid.sum())} body grid nodes, "
              f"SDF range [{sdf_3d.min()*1000:.1f}, {sdf_3d.max()*1000:.1f}] mm")

    def _build_bonds(self, lookup, pos, is_tissue, nx, ny, nz, step):
        """Lattice-based fiber bonds between tissue particles."""
        bonds_i, bonds_j, bonds_l0, bonds_t = [], [], [], []

        # Elastin: 26 nearest neighbours (di,dj,dk ∈ {-1,0,1})
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
                # Both must be present and both must be tissue
                valid = (src >= 0) & (tgt >= 0) & is_tissue[src] & is_tissue[tgt]
                if valid.any():
                    s, t = src[valid], tgt[valid]
                    d = np.linalg.norm(pos[s] - pos[t], axis=1).astype(np.float32)
                    bonds_i.append(s); bonds_j.append(t)
                    bonds_l0.append(d)
                    bonds_t.append(np.zeros(len(s), dtype=np.int32))

        # Collagen: 2-step axis-aligned
        if MATERIAL.k_collagen > 0.0:
            for di, dj, dk in [(2,0,0), (0,2,0), (0,0,2)]:
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
                    bonds_l0.append(d)
                    bonds_t.append(np.ones(len(s), dtype=np.int32))

        if not bonds_i:
            return
        all_i  = np.concatenate(bonds_i).astype(np.int32)
        all_j  = np.concatenate(bonds_j).astype(np.int32)
        all_l0 = np.concatenate(bonds_l0).astype(np.float32)
        all_t  = np.concatenate(bonds_t).astype(np.int32)
        n_e = int((all_t == 0).sum()); n_c = int((all_t == 1).sum())
        print(f"MPMCTHead: {len(all_i)} fiber bonds (elastin={n_e}, collagen={n_c})")
        with wp.ScopedDevice(self.device):
            self.sim.fiber_i  = wp.array(all_i,  dtype=int)
            self.sim.fiber_j  = wp.array(all_j,  dtype=int)
            self.sim.fiber_l0 = wp.array(all_l0, dtype=float)
            self.sim.fiber_t  = wp.array(all_t,  dtype=int)
            self.sim.fiber_broken = wp.zeros(len(all_i), dtype=int)
        self.sim.n_bonds = len(all_i)

    # ------------------------------------------------------------------
    # Run / loop
    # ------------------------------------------------------------------

    def run(self):
        """Create Slicer visualisation and start simulation loop."""
        try:
            import slicer
        except ImportError:
            return
        self._create_vtk_model()
        self._create_gravity_slider()
        # Show the deformed CT volume rendering from the start so the
        # user can place curve points on the skin surface.  The point
        # model is hidden; it can be re-enabled from the Data module.
        self.setup_deformed_ct()
        self._setup_view()
        self.start_simulation_loop()

    def start_simulation_loop(self):
        if self._loop_running:
            return
        self._loop_running = True
        self._idle_ticks = 0
        self._prev_tick_pos = None
        try:
            import slicer
            tag = slicer.mrmlScene.AddObserver(
                slicer.mrmlScene.StartCloseEvent,
                self._on_scene_about_to_close)
            self._observer_tags.append((slicer.mrmlScene, tag))
        except Exception:
            pass
        self._schedule_tick()

    def stop_simulation_loop(self):
        self._loop_running = False
        if hasattr(self, '_start_stop_btn') and self._start_stop_btn:
            self._start_stop_btn.text = "Start"
        for node, tag in self._observer_tags:
            try:
                node.RemoveObserver(tag)
            except Exception:
                pass
        self._observer_tags.clear()

    def _schedule_tick(self):
        try:
            import qt
            qt.QTimer.singleShot(self._tick_interval_ms, self._timer_tick)
        except Exception:
            pass

    def _timer_tick(self):
        if not self._loop_running:
            return
        try:
            import slicer
            if self.vtk_model and not slicer.mrmlScene.IsNodePresent(self.vtk_model):
                self.stop_simulation_loop()
                return
        except Exception:
            self.stop_simulation_loop()
            return

        g = self.gravity * self._gravity_scale
        for _ in range(self._steps_per_tick):
            if self._contact_sphere is not None:
                cs = self._contact_sphere
                self.sim.step_with_contact(g, cs['center'], cs['radius'])
            else:
                self.sim.step(g)

        pos_now = self.sim.get_positions()
        if self._deformed_ct is not None:
            import time
            now = time.time()
            if now - self._last_ct_update > self._ct_update_interval:
                self.update_deformed_ct()
                self._last_ct_update = now
        else:
            self.update_model()

        if self._prev_tick_pos is not None:
            if float(np.abs(pos_now - self._prev_tick_pos).max()) < 2e-5:
                self._idle_ticks += 1
            else:
                self._idle_ticks = 0
        self._prev_tick_pos = pos_now

        if self._idle_ticks >= self._idle_ticks_to_stop:
            self._loop_running = False
            return
        self._schedule_tick()

    def _on_scene_about_to_close(self, _s, _e):
        self.stop_simulation_loop()

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def update_model(self):
        if self._vtk_points is None:
            return
        import vtk, vtk.util.numpy_support as ns
        pos_mm = (self.sim.get_positions() * 1000.0
                  + self._ras_offset_mm).astype(np.float32)
        self._vtk_points.SetData(
            ns.numpy_to_vtk(pos_mm, deep=True, array_type=vtk.VTK_FLOAT))
        self._vtk_points.Modified()
        self._vtk_poly.Modified()
        if self.vtk_model:
            self.vtk_model.GetPolyData().Modified()

    def setup_deformed_ct(self):
        """Create a resampled deformed CT with wound cavity visualization.

        Called once after a cut is applied.  Creates:
          - An output volume at MPM grid resolution (~4 mm)
          - Volume rendering with a soft-tissue transfer function
          - Pre-computed coordinate mapping for fast per-tick updates

        Subsequent calls to update_deformed_ct() resample the CT
        through the current displacement field ~5-10× per second.
        """
        import slicer, vtk
        from scipy.ndimage import map_coordinates

        ng     = self.sim.n_grid
        dx_m   = self.sim.dx
        dx_mm  = dx_m * 1000.0
        inv_dx = self.sim.inv_dx
        offset = self._ras_offset_mm

        # --- Pre-compute output grid → CT source mapping -----------------
        # Output volume matches the original CT geometry (full resolution)
        self._ct_arr = self._volume_array(self.volume_node)  # (K,J,I)
        nK, nJ, nI = self._ct_arr.shape

        ijk2ras = vtk.vtkMatrix4x4()
        self.volume_node.GetIJKToRASMatrix(ijk2ras)
        self._out_ijk2ras = vtk.vtkMatrix4x4()
        self._out_ijk2ras.DeepCopy(ijk2ras)

        ras2ijk = vtk.vtkMatrix4x4()
        self.volume_node.GetRASToIJKMatrix(ras2ijk)
        self._ct_ras2ijk = np.array(
            [[ras2ijk.GetElement(r, c) for c in range(4)] for r in range(4)])

        # --- Upload CT to GPU (once) --------------------------------------
        ct_ijk = self._ct_arr.transpose(2, 1, 0).astype(np.float32).ravel()
        nK, nJ, nI = self._ct_arr.shape
        n_out = nI * nJ * nK
        with wp.ScopedDevice(self.device):
            self._ct_gpu = wp.array(ct_ijk, dtype=float)
            self._out_gpu = wp.zeros(n_out, dtype=float)

        # Store IJKToRAS matrix elements for the kernel
        self._ijk2ras_elems = np.array(
            [[ijk2ras.GetElement(r, c) for c in range(4)] for r in range(4)])
        self._out_ni = nI
        self._out_nj = nJ
        self._out_nk = nK

        # --- Create the DeformedCT volume (same geometry as CT) ----------
        old = slicer.mrmlScene.GetFirstNodeByName('DeformedCT')
        if old:
            slicer.mrmlScene.RemoveNode(old)

        self._deformed_ct = slicer.util.addVolumeFromArray(
            self._ct_arr.copy(), ijkToRAS=ijk2ras, name='DeformedCT')

        # Initial resample
        self.update_deformed_ct()

        print(f"MPMCTHead: CT uploaded to GPU ({n_out/1e6:.1f}M voxels), "
              f"output {nI}×{nJ}×{nK} at full CT resolution")

        # --- Volume rendering with soft-tissue TF ------------------------
        vrLogic = slicer.modules.volumerendering.logic()
        vrdn = vrLogic.CreateDefaultVolumeRenderingNodes(self._deformed_ct)
        vrdn = vrLogic.GetFirstVolumeRenderingDisplayNode(self._deformed_ct)
        vrdn.SetVisibility(True)

        vp = vrdn.GetVolumePropertyNode().GetVolumeProperty()
        otf = vp.GetScalarOpacity()
        otf.RemoveAllPoints()
        otf.AddPoint(-1000, 0.0); otf.AddPoint(-200, 0.0)
        otf.AddPoint(-50, 0.0); otf.AddPoint(0, 0.5)
        otf.AddPoint(100, 0.7); otf.AddPoint(300, 0.85)
        otf.AddPoint(500, 0.9); otf.AddPoint(2000, 1.0)
        ctf = vp.GetRGBTransferFunction()
        ctf.RemoveAllPoints()
        ctf.AddRGBPoint(-1000, 0, 0, 0); ctf.AddRGBPoint(-50, 0, 0, 0)
        ctf.AddRGBPoint(0, 0.87, 0.72, 0.60)
        ctf.AddRGBPoint(60, 0.83, 0.58, 0.50)
        ctf.AddRGBPoint(200, 0.90, 0.82, 0.75)
        ctf.AddRGBPoint(400, 0.95, 0.92, 0.88)
        ctf.AddRGBPoint(1500, 1.0, 1.0, 0.95)
        gof = vp.GetGradientOpacity()
        gof.RemoveAllPoints()
        gof.AddPoint(0, 0.0); gof.AddPoint(15, 0.7); gof.AddPoint(80, 1.0)
        vp.SetShade(True)
        vp.SetAmbient(0.15); vp.SetDiffuse(0.75); vp.SetSpecular(0.3)
        vrdn.GetVolumePropertyNode().Modified()

        # Hide point model, show deformed CT instead
        if self.vtk_model:
            self.vtk_model.GetDisplayNode().SetVisibility(False)

        slicer.util.setSliceViewerLayers(background=self._deformed_ct)
        print("MPMCTHead: deformed CT visualization ready")

    def update_deformed_ct(self):
        """Resample CT on GPU through current displacement field + wound mask.

        Called periodically (controlled by _ct_update_interval).
        The warp kernel _resample_deformed_ct handles trilinear sampling
        of both the displacement grid and the CT volume on the GPU.
        """
        if self._deformed_ct is None or self._ct_gpu is None:
            return
        import slicer

        pos    = self.sim.get_positions()
        x0     = self.sim.x0.numpy()
        disp   = pos - x0
        mass_p = self.sim.m_p.numpy()
        ng     = self.sim.n_grid
        inv_dx = self.sim.inv_dx

        # Scatter displacement and mass to MPM grid (CPU — small grid)
        disp_grid = np.zeros((ng, ng, ng, 3), dtype=np.float32)
        mass_grid = np.zeros((ng, ng, ng), dtype=np.float32)
        gi = np.clip(np.round(pos[:, 0] * inv_dx).astype(int), 0, ng - 1)
        gj = np.clip(np.round(pos[:, 1] * inv_dx).astype(int), 0, ng - 1)
        gk = np.clip(np.round(pos[:, 2] * inv_dx).astype(int), 0, ng - 1)
        np.add.at(mass_grid, (gi, gj, gk), mass_p)
        for d in range(3):
            np.add.at(disp_grid[:, :, :, d], (gi, gj, gk),
                       mass_p * disp[:, d])
        valid = mass_grid > 0
        for d in range(3):
            disp_grid[:, :, :, d][valid] /= mass_grid[valid]

        # Upload displacement grid to GPU
        dx_flat = disp_grid[:, :, :, 0].ravel().astype(np.float32)
        dy_flat = disp_grid[:, :, :, 1].ravel().astype(np.float32)
        dz_flat = disp_grid[:, :, :, 2].ravel().astype(np.float32)
        m_flat  = mass_grid.ravel().astype(np.float32)

        with wp.ScopedDevice(self.device):
            disp_x_gpu = wp.array(dx_flat, dtype=float)
            disp_y_gpu = wp.array(dy_flat, dtype=float)
            disp_z_gpu = wp.array(dz_flat, dtype=float)
            mass_gpu   = wp.array(m_flat, dtype=float)

            cut_sdf_gpu = (self.sim.cut_sdfs[-1]
                           if self.sim.cut_sdfs
                           else wp.zeros(ng ** 3, dtype=float))

            A = self._ijk2ras_elems   # output IJK → RAS
            B = self._ct_ras2ijk      # RAS → source CT IJK
            offset = self._ras_offset_mm
            dx_mm = self.sim.dx * 1000.0
            nI, nJ, nK = self._out_ni, self._out_nj, self._out_nk

            wp.launch(_resample_deformed_ct, dim=nI * nJ * nK, inputs=[
                disp_x_gpu, disp_y_gpu, disp_z_gpu, mass_gpu, cut_sdf_gpu,
                ng,
                self._ct_gpu, nI, nJ, nK,
                # output IJK → RAS (A matrix)
                float(A[0, 0]), float(A[0, 1]), float(A[0, 2]), float(A[0, 3]),
                float(A[1, 0]), float(A[1, 1]), float(A[1, 2]), float(A[1, 3]),
                float(A[2, 0]), float(A[2, 1]), float(A[2, 2]), float(A[2, 3]),
                # RAS → source CT IJK (B matrix)
                float(B[0, 0]), float(B[0, 1]), float(B[0, 2]), float(B[0, 3]),
                float(B[1, 0]), float(B[1, 1]), float(B[1, 2]), float(B[1, 3]),
                float(B[2, 0]), float(B[2, 1]), float(B[2, 2]), float(B[2, 3]),
                nI, nJ, nK,
                float(offset[0]), float(offset[1]), float(offset[2]),
                float(dx_mm),
                self._out_gpu,
            ])

        # Download and update Slicer volume
        nI, nJ, nK = self._out_ni, self._out_nj, self._out_nk
        # Kernel output has I fastest (tid % nI), K slowest — reshape as
        # (K,J,I) in C order so the last axis (I) is fastest, matching
        # both the kernel layout and Slicer's KJI array convention.
        out_kji = self._out_gpu.numpy().reshape(nK, nJ, nI).astype(np.int16)

        arr = slicer.util.arrayFromVolume(self._deformed_ct)
        arr[:] = out_kji
        slicer.util.arrayFromVolumeModified(self._deformed_ct)

    def rebuild_colors(self):
        """Rebuild particle colors (e.g. after a cut to show sides)."""
        if self._vtk_poly is None:
            return
        import vtk, vtk.util.numpy_support as ns
        colors_np = self._build_colors()
        ca = ns.numpy_to_vtk(colors_np, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        ca.SetName("Colors"); ca.SetNumberOfComponents(3)
        self._vtk_poly.GetPointData().SetScalars(ca)
        self._vtk_poly.Modified()
        if self.vtk_model:
            self.vtk_model.GetPolyData().Modified()

    def _build_colors(self):
        """Bone=ivory, tissue=HU gradient, cut sides=red/blue."""
        n = self.sim.n_particles
        colors = np.zeros((n, 3), dtype=np.uint8)
        colors[self._is_bone] = [240, 230, 210]         # ivory

        # Tissue: warm colour gradient by HU
        tissue = self._is_tissue
        if tissue.any():
            hu_t = self._hu[tissue]
            t = np.clip((hu_t - HU_AIR_MAX) / (HU_BONE_MIN - HU_AIR_MAX), 0, 1)
            colors[tissue, 0] = (200 + 55 * t).astype(np.uint8)
            colors[tissue, 1] = (100 + 80 * t).astype(np.uint8)
            colors[tissue, 2] = (80  + 40 * t).astype(np.uint8)

        # Override near-cut particles with red/blue by side.
        # Only color particles within ~25mm of the cut surface (small
        # |SDF|) — the full SDF extends much further for physics but
        # the visual indicator should be a narrow band near the cut.
        if self.sim.cut_sdfs:
            cut_sdf_np = self.sim.cut_sdfs[-1].numpy()
            pos = self.sim.get_positions()
            ng = self.sim.n_grid
            inv_dx = self.sim.inv_dx
            gi = np.clip(np.round(pos[:, 0] * inv_dx).astype(int), 0, ng - 1)
            gj = np.clip(np.round(pos[:, 1] * inv_dx).astype(int), 0, ng - 1)
            gk = np.clip(np.round(pos[:, 2] * inv_dx).astype(int), 0, ng - 1)
            p_sdf = cut_sdf_np[gi * ng * ng + gj * ng + gk]
            color_band = 0.025   # 25mm in sim coords [m]
            near = np.abs(p_sdf) < color_band
            pos_side = (p_sdf > 0) & near & self._is_tissue
            neg_side = (p_sdf < 0) & near & self._is_tissue

            # Checkerboard based on initial positions so deformation
            # is visible as distortion of the checker pattern
            x0 = self.sim.x0.numpy()
            checker_size = 2.0 * self.sim.dx  # ~8mm squares
            ci = (x0[:, 0] / checker_size).astype(int) % 2
            cj = (x0[:, 1] / checker_size).astype(int) % 2
            ck = (x0[:, 2] / checker_size).astype(int) % 2
            checker = (ci ^ cj ^ ck).astype(bool)

            colors[pos_side &  checker] = [220, 80, 80]   # red dark
            colors[pos_side & ~checker] = [255, 160, 140]  # red light
            colors[neg_side &  checker] = [80, 120, 220]   # blue dark
            colors[neg_side & ~checker] = [140, 180, 255]  # blue light

        return colors

    def _create_vtk_model(self):
        import slicer, vtk, vtk.util.numpy_support as ns

        pos_mm = (self.sim.get_positions() * 1000.0
                  + self._ras_offset_mm).astype(np.float32)
        pts = vtk.vtkPoints()
        pts.SetData(ns.numpy_to_vtk(pos_mm, deep=True, array_type=vtk.VTK_FLOAT))

        colors_np = self._build_colors()
        ca = ns.numpy_to_vtk(colors_np, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        ca.SetName("Colors"); ca.SetNumberOfComponents(3)

        src = vtk.vtkPolyData()
        src.SetPoints(pts)
        src.GetPointData().SetScalars(ca)
        gf = vtk.vtkVertexGlyphFilter(); gf.SetInputData(src); gf.Update()
        poly = vtk.vtkPolyData(); poly.DeepCopy(gf.GetOutput())

        self.vtk_model = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLModelNode', 'MPMCTHead')
        self.vtk_model.SetAndObservePolyData(poly)
        self.vtk_model.CreateDefaultDisplayNodes()
        dn = self.vtk_model.GetDisplayNode()
        dn.SetPointSize(2)
        dn.SetScalarVisibility(True)
        dn.SetActiveScalarName("Colors")
        dn.SetScalarRangeFlagFromString("UseDirectMapping")
        self._vtk_points = poly.GetPoints()
        self._vtk_poly   = poly

    def _setup_view(self):
        import slicer
        lm = slicer.app.layoutManager()
        lm.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        slicer.app.processEvents()
        slicer.util.resetThreeDViews()
        slicer.app.processEvents()

    # ------------------------------------------------------------------
    # Gravity slider
    # ------------------------------------------------------------------

    def _create_gravity_slider(self):
        import qt, slicer

        # Remove any leftover gravity toolbars from previous runs
        main_window = slicer.util.mainWindow()
        for tb in main_window.findChildren(qt.QToolBar):
            if tb.windowTitle == "MPM Gravity":
                main_window.removeToolBar(tb)
                tb.deleteLater()

        self._toolbar = qt.QToolBar("MPM Gravity")
        slicer.util.mainWindow().addToolBar(self._toolbar)

        label = qt.QLabel("  Gravity: ")
        self._toolbar.addWidget(label)

        slider = qt.QSlider(qt.Qt.Horizontal)
        slider.setMinimum(-400)    # -4 × g
        slider.setMaximum(400)     #  4 × g
        slider.setValue(100)       #  1 × g (normal earth gravity inferior)
        slider.setFixedWidth(200)
        slider.setToolTip("Gravity scale (−2g … +2g).  "
                          "Positive = inferior, negative = superior.")
        slider.valueChanged.connect(self._on_gravity_changed)
        self._toolbar.addWidget(slider)
        self._slider = slider

        self._grav_label = qt.QLabel(" 1.00 g")
        self._toolbar.addWidget(self._grav_label)

        self._toolbar.addSeparator()

        self._start_stop_btn = qt.QPushButton("Stop")
        self._start_stop_btn.setFixedWidth(60)
        self._start_stop_btn.setToolTip("Start / stop the simulation loop")
        self._start_stop_btn.clicked.connect(self._on_start_stop)
        self._toolbar.addWidget(self._start_stop_btn)

    def _on_start_stop(self):
        if self._loop_running:
            self.stop_simulation_loop()
            self._start_stop_btn.text = "Start"
        else:
            self.start_simulation_loop()
            self._start_stop_btn.text = "Stop"

    def _on_gravity_changed(self, value):
        self._gravity_scale = value / 100.0
        self._grav_label.text = f" {self._gravity_scale:+.2f} g"
        self._idle_ticks = 0
        if not self._loop_running:
            self.start_simulation_loop()
            if hasattr(self, '_start_stop_btn'):
                self._start_stop_btn.text = "Stop"

    def cleanup_toolbar(self):
        """Remove the gravity toolbar."""
        if self._toolbar is not None:
            import slicer
            slicer.util.mainWindow().removeToolBar(self._toolbar)
            self._toolbar.deleteLater()
            self._toolbar = None


if __name__ == '__main__':
    print("MPMCTHead requires Slicer — run via the TissueSimulation module.")
