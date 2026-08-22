"""Explicit MLS-MPM soft tissue solver using NVIDIA Warp.

References:
  Hu et al., "A Moving Least Squares Material Point Method with
  Displacement Discontinuity and Two-Way Rigid Body Coupling" (SIGGRAPH 2018).

  Ou & Tavakoli, "CRESSim-MPM: A Material Point Method Library for Surgical
  Soft Body Simulation with Cutting and Suturing", arXiv:2502.18437v3, 2025.
  (Side-aware P2G/G2P transfer blocking for tissue cutting.)

Each simulation step:
  1. zero_grid   — clear grid mass and momentum
  2. [opt] recompute_F — total-Lagrangian: estimate F from particle positions
  3. p2g         — scatter particle state → grid (B-spline weights, APIC + stress)
  4. grid_update — apply gravity + inferior-face no-penetration BC
  5. g2p         — gather grid velocities → particles, update C and positions
                   (F update from G2P is skipped in total-Lagrangian mode)
  6. [opt] fiber — Cosserat-style cable bond forces

Deformation-gradient drift ("numerical plasticity"):
  In updated-Lagrangian MPM, F accumulates small errors each step via the
  multiplicative update F = (I + dt*C) @ F.  After a large-deformation cycle
  (e.g. deep palpation followed by release), the drifted F causes the tissue
  to settle at a wrong equilibrium a few mm from the true rest position.

  Fix — on-demand F reset via reset_F_from_positions():
    After the tissue has elastically rebounded and all free particles are within
    a fraction of the lattice spacing of their reference positions x0, call
    reset_F_from_positions().  This recomputes F from finite differences on the
    initial lattice — valid because deformations are small at that moment — and
    corrects the drift without disrupting the ongoing dynamics.

  MPMTissueBlock.recover() does this automatically using check_near_rest().

  total_lagrangian=True (opt-in): recomputes F EVERY step before P2G.  This is
  correct for small deformations but fails catastrophically for palpation > 1
  lattice spacing because the fixed initial topology produces nonsensical
  finite differences in the transition zone.  Not recommended for palpation.
"""

import numpy as np
import warp as wp
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------

@dataclass
class MPMMaterial:
    """Neo-Hookean isotropic material with optional Cosserat fiber network.

    Fiber network parameters model elastin and collagen-like bonds between
    adjacent particles.  Set k_elastin or k_collagen > 0 to enable.

    Elastin bonds:
      Short-range (+-1 particle step, all 26 face/edge/corner neighbors).
      Bidirectional spring: resists both stretch and compression.

    Collagen bonds:
      Medium-range (+-2 particle steps, axis-aligned only).
      Tension-only: only activates above the crimp strain threshold.
    """
    E: float = 3_000.0    # Young's modulus [Pa]
    nu: float = 0.45      # Poisson's ratio
    rho: float = 1_060.0  # density [kg/m^3]

    # --- fiber network (Cosserat-style cable bonds) ---
    k_elastin: float = 0.0    # elastin bond stiffness [N/m]; 0 = disabled
    k_collagen: float = 0.0   # collagen bond stiffness [N/m]; 0 = disabled
    collagen_crimp: float = 0.03  # strain threshold for collagen activation

    # --- curvature / positional-homeostasis springs ---
    # Discrete Laplacian spring: for each axis-aligned triplet (A-, A, A+),
    # penalises deviation of A from the midpoint of its two axis neighbors.
    # Resists bending of lattice lines, prevents topology crossing, and models
    # the positional homeostasis of cells and ECM at the continuum scale.
    # With k_curve > 0, total_lagrangian=True becomes stable even during large
    # palpation because particles can never pass their axis neighbors.
    # Stability limit: k_curve < mass_particle / (2 * dt^2); at dt=2e-4 s and
    # rho=1060 kg/m^3, ppc=2, this is ~200 N/m.  Start around 10-50 N/m.
    k_curve: float = 0.0      # curvature spring stiffness [N/m]; 0 = disabled

    @property
    def mu(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def lam(self) -> float:
        return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))


# ---------------------------------------------------------------------------
# Warp helper function
# ---------------------------------------------------------------------------

@wp.func
def _bspline_w(fx: float, i: int) -> float:
    """Quadratic B-spline weight for MLS-MPM (Hu 2018)."""
    if i == 0:
        d = 1.5 - fx
        return 0.5 * d * d
    elif i == 1:
        d = fx - 1.0
        return 0.75 - d * d
    else:
        d = fx - 0.5
        return 0.5 * d * d


# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------

@wp.kernel
def _zero_grid(
    grid_v: wp.array(dtype=wp.vec3),
    grid_m: wp.array(dtype=float),
):
    i = wp.tid()
    grid_v[i] = wp.vec3(0.0, 0.0, 0.0)
    grid_m[i] = 0.0


@wp.kernel
def _recompute_F_total_lagrangian(
    x:      wp.array(dtype=wp.vec3),
    F:      wp.array(dtype=wp.mat33),
    nbr_px: wp.array(dtype=int),   # +X lattice neighbour index (-1 = none)
    nbr_mx: wp.array(dtype=int),   # -X
    nbr_py: wp.array(dtype=int),   # +Y
    nbr_my: wp.array(dtype=int),   # -Y
    nbr_pz: wp.array(dtype=int),   # +Z
    nbr_mz: wp.array(dtype=int),   # -Z
    step:   float,                  # initial lattice spacing [m]
):
    """Recompute F as a finite-difference estimate from current positions.

    F_ij = dx_i / dX_j  (deformation gradient, total Lagrangian)

    Each column of F is estimated by central (or one-sided) differences along
    the corresponding initial lattice axis.  Because F is derived fresh from
    positions every step, it never accumulates drift — the Neo-Hookean stress
    always points correctly toward the equilibrium configuration.

    Warp stores mat33 in row-major order:
        wp.mat33(F[0,0], F[0,1], F[0,2],
                 F[1,0], F[1,1], F[1,2],
                 F[2,0], F[2,1], F[2,2])
    Columns of F correspond to material directions:
        col_j = [F[0,j], F[1,j], F[2,j]]
    """
    p = wp.tid()

    # When BOTH axis neighbours are missing there is nothing to difference.
    # Falling back to the identity column would assert "unstretched along this
    # axis", which is wrong exactly where it matters most: on a fresh cut face,
    # next to two genuinely stretched columns, it manufactures spurious shear.
    # Keep whatever F already holds for that direction instead.
    Fp = F[p]

    # --- Column 0: X material direction (i-axis) ---
    pp = nbr_px[p]
    pm = nbr_mx[p]
    if pp >= 0 and pm >= 0:
        col0 = (x[pp] - x[pm]) * (0.5 / step)
    elif pp >= 0:
        col0 = (x[pp] - x[p]) * (1.0 / step)
    elif pm >= 0:
        col0 = (x[p] - x[pm]) * (1.0 / step)
    else:
        col0 = wp.vec3(Fp[0, 0], Fp[1, 0], Fp[2, 0])  # keep current column

    # --- Column 1: Y material direction (j-axis) ---
    pp = nbr_py[p]
    pm = nbr_my[p]
    if pp >= 0 and pm >= 0:
        col1 = (x[pp] - x[pm]) * (0.5 / step)
    elif pp >= 0:
        col1 = (x[pp] - x[p]) * (1.0 / step)
    elif pm >= 0:
        col1 = (x[p] - x[pm]) * (1.0 / step)
    else:
        col1 = wp.vec3(Fp[0, 1], Fp[1, 1], Fp[2, 1])  # keep current column

    # --- Column 2: Z material direction (k-axis) ---
    pp = nbr_pz[p]
    pm = nbr_mz[p]
    if pp >= 0 and pm >= 0:
        col2 = (x[pp] - x[pm]) * (0.5 / step)
    elif pp >= 0:
        col2 = (x[pp] - x[p]) * (1.0 / step)
    elif pm >= 0:
        col2 = (x[p] - x[pm]) * (1.0 / step)
    else:
        col2 = wp.vec3(Fp[0, 2], Fp[1, 2], Fp[2, 2])  # keep current column

    # Assemble F[i,j] = col_j[i]  (row-major Warp storage)
    F[p] = wp.mat33(col0[0], col1[0], col2[0],
                    col0[1], col1[1], col2[1],
                    col0[2], col1[2], col2[2])


@wp.kernel
def _p2g(
    x:      wp.array(dtype=wp.vec3),
    v:      wp.array(dtype=wp.vec3),
    F:      wp.array(dtype=wp.mat33),
    C:      wp.array(dtype=wp.mat33),
    m_p:    wp.array(dtype=float),
    vol_p:  wp.array(dtype=float),
    grid_v: wp.array(dtype=wp.vec3),
    grid_m: wp.array(dtype=float),
    mu_p:   wp.array(dtype=float),
    lam_p:  wp.array(dtype=float),
    n_grid: int,
    inv_dx: float,
    dt:     float,
):
    """Particle → Grid transfer (one thread per particle)."""
    p = wp.tid()
    mu  = mu_p[p]
    lam = lam_p[p]
    xp = x[p]
    mp = m_p[p]
    Vp = vol_p[p]
    dx = 1.0 / inv_dx

    Fp     = F[p]
    J      = wp.determinant(Fp)
    J_safe = wp.max(J, 0.1)
    FFt    = Fp @ wp.transpose(Fp)
    I3     = wp.mat33(1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0)
    tau    = mu * (FFt - I3) + lam * wp.log(J_safe) * I3

    stress = (-4.0 * dt * inv_dx * inv_dx * Vp) * tau
    affine = stress + mp * C[p]

    bx = int(wp.floor(xp[0] * inv_dx - 0.5))
    by = int(wp.floor(xp[1] * inv_dx - 0.5))
    bz = int(wp.floor(xp[2] * inv_dx - 0.5))

    fx_x = xp[0] * inv_dx - float(bx)
    fx_y = xp[1] * inv_dx - float(by)
    fx_z = xp[2] * inv_dx - float(bz)

    for di in range(3):
        wx = _bspline_w(fx_x, di)
        gi = bx + di
        if gi < 0 or gi >= n_grid:
            continue
        for dj in range(3):
            wy = _bspline_w(fx_y, dj)
            gj = by + dj
            if gj < 0 or gj >= n_grid:
                continue
            for dk in range(3):
                wz = _bspline_w(fx_z, dk)
                gk = bz + dk
                if gk < 0 or gk >= n_grid:
                    continue
                w    = wx * wy * wz
                flat = gi * n_grid * n_grid + gj * n_grid + gk
                xi   = wp.vec3(float(gi) * dx, float(gj) * dx, float(gk) * dx)
                dpos = xi - xp
                wp.atomic_add(grid_v, flat, w * (mp * v[p] + affine @ dpos))
                wp.atomic_add(grid_m, flat, w * mp)


@wp.kernel
def _p2g_cut(
    x:       wp.array(dtype=wp.vec3),
    v:       wp.array(dtype=wp.vec3),
    F:       wp.array(dtype=wp.mat33),
    C:       wp.array(dtype=wp.mat33),
    m_p:     wp.array(dtype=float),
    vol_p:   wp.array(dtype=float),
    grid_v:  wp.array(dtype=wp.vec3),
    grid_m:  wp.array(dtype=float),
    cut_sdf: wp.array(dtype=float),   # signed distance per grid node
    mu_p:    wp.array(dtype=float),
    lam_p:   wp.array(dtype=float),
    n_grid:  int,
    inv_dx:  float,
    dt:      float,
):
    """P2G with cut-aware transfer: skip scatter across a cut surface.

    If a particle and a grid node are on opposite sides of the cut SDF
    (different signs), the transfer is blocked.  This creates a velocity
    discontinuity at the cut, allowing tissue on opposite sides to separate.
    """
    p = wp.tid()
    mu  = mu_p[p]
    lam = lam_p[p]
    xp = x[p]
    mp = m_p[p]
    Vp = vol_p[p]
    dx = 1.0 / inv_dx

    # Particle's SDF side: sample at nearest grid node
    pi = int(wp.round(xp[0] * inv_dx))
    pj = int(wp.round(xp[1] * inv_dx))
    pk = int(wp.round(xp[2] * inv_dx))
    pi = wp.clamp(pi, 0, n_grid - 1)
    pj = wp.clamp(pj, 0, n_grid - 1)
    pk = wp.clamp(pk, 0, n_grid - 1)
    p_sdf = cut_sdf[pi * n_grid * n_grid + pj * n_grid + pk]

    Fp     = F[p]
    J      = wp.determinant(Fp)
    J_safe = wp.max(J, 0.1)
    FFt    = Fp @ wp.transpose(Fp)
    I3     = wp.mat33(1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0)
    tau    = mu * (FFt - I3) + lam * wp.log(J_safe) * I3

    stress = (-4.0 * dt * inv_dx * inv_dx * Vp) * tau
    affine = stress + mp * C[p]

    bx = int(wp.floor(xp[0] * inv_dx - 0.5))
    by = int(wp.floor(xp[1] * inv_dx - 0.5))
    bz = int(wp.floor(xp[2] * inv_dx - 0.5))

    fx_x = xp[0] * inv_dx - float(bx)
    fx_y = xp[1] * inv_dx - float(by)
    fx_z = xp[2] * inv_dx - float(bz)

    for di in range(3):
        wx = _bspline_w(fx_x, di)
        gi = bx + di
        if gi < 0 or gi >= n_grid:
            continue
        for dj in range(3):
            wy = _bspline_w(fx_y, dj)
            gj = by + dj
            if gj < 0 or gj >= n_grid:
                continue
            for dk in range(3):
                wz = _bspline_w(fx_z, dk)
                gk = bz + dk
                if gk < 0 or gk >= n_grid:
                    continue
                flat = gi * n_grid * n_grid + gj * n_grid + gk
                # Block transfer if particle and node are on opposite sides
                g_sdf = cut_sdf[flat]
                if p_sdf * g_sdf < 0.0:
                    continue
                w    = wx * wy * wz
                xi   = wp.vec3(float(gi) * dx, float(gj) * dx, float(gk) * dx)
                dpos = xi - xp
                wp.atomic_add(grid_v, flat, w * (mp * v[p] + affine @ dpos))
                wp.atomic_add(grid_m, flat, w * mp)


@wp.kernel
def _grid_update(
    grid_v:  wp.array(dtype=wp.vec3),
    grid_m:  wp.array(dtype=float),
    n_grid:  int,
    dt:      float,
    gravity: wp.vec3,
    damping: float,
    floor_j0: int,
):
    """Grid momentum update: divide by mass, add gravity, damp, apply BCs.

    floor_j0 enables the frictionless no-penetration floor at grid row j == 0.
    It is on by default for backward compatibility, but scenarios that supply
    their own support surface (SARRTS) should turn it off -- otherwise the
    most-posterior grid plane silently acts as a second, invisible wall.
    """
    flat = wp.tid()
    m = grid_m[flat]
    if m > 0.0:
        v = (grid_v[flat] / m + dt * gravity) * damping

        j = (flat // n_grid) % n_grid

        if floor_j0 != 0 and j == 0 and v[1] < 0.0:
            v = wp.vec3(v[0], 0.0, v[2])

        grid_v[flat] = v
    else:
        grid_v[flat] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def _apply_grid_fixed_bc(
    grid_v:       wp.array(dtype=wp.vec3),
    grid_bc_fixed: wp.array(dtype=int),
):
    """Zero velocity at grid nodes marked as fixed (e.g. bone).

    This makes fixed regions act as rigid walls in the grid velocity field,
    physically preventing free particles from flowing through them.
    """
    flat = wp.tid()
    if grid_bc_fixed[flat] != 0:
        grid_v[flat] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def _apply_grid_sdf_bc(
    grid_v:      wp.array(dtype=wp.vec3),
    sdf:         wp.array(dtype=float),
    sdf_grad:    wp.array(dtype=wp.vec3),
    dx:          float,
    friction_mu: float,
):
    """SDF-based grid boundary condition for a static obstacle.

    - SDF < 0 (inside the obstacle): zero velocity — hard wall.
    - 0 ≤ SDF < dx (near surface): remove the inward velocity component so
      tissue can slide along the surface but not penetrate.
    - SDF ≥ dx: free.

    friction_mu applies Coulomb friction to the sliding component: the
    tangential velocity is reduced by at most mu * |v_n|, never reversed.
    friction_mu = 0 reproduces frictionless sliding.

    Applied after grid_update and before G2P so the grid velocity field can
    never push tissue into the obstacle.
    """
    flat = wp.tid()
    d = sdf[flat]
    if d < 0.0:
        grid_v[flat] = wp.vec3(0.0, 0.0, 0.0)
    elif d < dx:
        n = sdf_grad[flat]
        n_len = wp.length(n)
        if n_len > 1.0e-8:
            normal = n / n_len
            v = grid_v[flat]
            v_n = wp.dot(v, normal)
            if v_n < 0.0:
                v_t = v - v_n * normal
                if friction_mu > 0.0:
                    t_len = wp.length(v_t)
                    if t_len > 1.0e-8:
                        # Coulomb: drop at most mu*|v_n| of tangential speed
                        scale = wp.max(0.0, 1.0 - friction_mu * wp.abs(v_n) / t_len)
                        v_t = v_t * scale
                grid_v[flat] = v_t


@wp.kernel
def _apply_bone_sdf_contact(
    x:        wp.array(dtype=wp.vec3),
    v:        wp.array(dtype=wp.vec3),
    fixed:    wp.array(dtype=int),
    sdf:      wp.array(dtype=float),     # SDF value per grid node (neg = inside bone)
    sdf_grad: wp.array(dtype=wp.vec3),   # precomputed SDF gradient per grid node
    n_grid:   int,
    inv_dx:   float,
    friction_mu: float,
):
    """Project tissue particles out of an obstacle using a signed distance field.

    For each free particle inside bone (SDF < 0), the particle is pushed to
    the bone surface along the precomputed SDF gradient.  The inward velocity
    component is also removed so the particle slides along the surface rather
    than re-entering.  Runs after G2P each step.
    """
    p = wp.tid()
    if fixed[p] != 0:
        return

    xp = x[p]
    # Nearest grid node
    gi = int(wp.round(xp[0] * inv_dx))
    gj = int(wp.round(xp[1] * inv_dx))
    gk = int(wp.round(xp[2] * inv_dx))

    if gi < 0 or gi >= n_grid or gj < 0 or gj >= n_grid or gk < 0 or gk >= n_grid:
        return

    flat = gi * n_grid * n_grid + gj * n_grid + gk
    d = sdf[flat]

    if d < 0.0:
        # Particle is inside bone — push to surface
        n = sdf_grad[flat]
        n_len = wp.length(n)
        if n_len > 1.0e-8:
            normal = n / n_len
            # Project position to surface (move by -d along normal)
            x[p] = xp + (-d) * normal
            # Cancel inward velocity component
            v_n = wp.dot(v[p], normal)
            if v_n < 0.0:
                v[p] = v[p] - v_n * normal


@wp.kernel
def _g2p(
    x:      wp.array(dtype=wp.vec3),
    v:      wp.array(dtype=wp.vec3),
    F:      wp.array(dtype=wp.mat33),
    C:      wp.array(dtype=wp.mat33),
    fixed:  wp.array(dtype=int),
    grid_v: wp.array(dtype=wp.vec3),
    n_grid: int,
    inv_dx: float,
    dt:     float,
):
    """Grid → Particle: update v, C, x and F (standard updated-Lagrangian mode)."""
    p = wp.tid()
    xp = x[p]
    dx = 1.0 / inv_dx

    bx = int(wp.floor(xp[0] * inv_dx - 0.5))
    by = int(wp.floor(xp[1] * inv_dx - 0.5))
    bz = int(wp.floor(xp[2] * inv_dx - 0.5))

    fx_x = xp[0] * inv_dx - float(bx)
    fx_y = xp[1] * inv_dx - float(by)
    fx_z = xp[2] * inv_dx - float(bz)

    new_v = wp.vec3(0.0, 0.0, 0.0)
    new_C = wp.mat33(0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0)

    for di in range(3):
        wx = _bspline_w(fx_x, di)
        gi = bx + di
        if gi < 0 or gi >= n_grid:
            continue
        for dj in range(3):
            wy = _bspline_w(fx_y, dj)
            gj = by + dj
            if gj < 0 or gj >= n_grid:
                continue
            for dk in range(3):
                wz = _bspline_w(fx_z, dk)
                gk = bz + dk
                if gk < 0 or gk >= n_grid:
                    continue
                w    = wx * wy * wz
                flat = gi * n_grid * n_grid + gj * n_grid + gk
                vi   = grid_v[flat]
                xi   = wp.vec3(float(gi) * dx, float(gj) * dx, float(gk) * dx)
                dpos = xi - xp
                new_v = new_v + w * vi
                new_C = new_C + (4.0 * inv_dx * inv_dx * w) * wp.outer(vi, dpos)

    I3 = wp.mat33(1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0)
    F[p] = (I3 + dt * new_C) @ F[p]
    C[p] = new_C

    if fixed[p] == 0:
        v[p] = new_v
        x[p] = xp + dt * new_v


@wp.kernel
def _g2p_cut(
    x:       wp.array(dtype=wp.vec3),
    v:       wp.array(dtype=wp.vec3),
    F:       wp.array(dtype=wp.mat33),
    C:       wp.array(dtype=wp.mat33),
    fixed:   wp.array(dtype=int),
    grid_v:  wp.array(dtype=wp.vec3),
    cut_sdf: wp.array(dtype=float),
    n_grid:  int,
    inv_dx:  float,
    dt:      float,
):
    """G2P with cut-aware transfer: skip gather across a cut surface."""
    p = wp.tid()
    xp = x[p]
    dx = 1.0 / inv_dx

    # Particle's SDF side
    pi = wp.clamp(int(wp.round(xp[0] * inv_dx)), 0, n_grid - 1)
    pj = wp.clamp(int(wp.round(xp[1] * inv_dx)), 0, n_grid - 1)
    pk = wp.clamp(int(wp.round(xp[2] * inv_dx)), 0, n_grid - 1)
    p_sdf = cut_sdf[pi * n_grid * n_grid + pj * n_grid + pk]

    bx = int(wp.floor(xp[0] * inv_dx - 0.5))
    by = int(wp.floor(xp[1] * inv_dx - 0.5))
    bz = int(wp.floor(xp[2] * inv_dx - 0.5))

    fx_x = xp[0] * inv_dx - float(bx)
    fx_y = xp[1] * inv_dx - float(by)
    fx_z = xp[2] * inv_dx - float(bz)

    new_v = wp.vec3(0.0, 0.0, 0.0)
    new_C = wp.mat33(0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0)

    for di in range(3):
        wx = _bspline_w(fx_x, di)
        gi = bx + di
        if gi < 0 or gi >= n_grid:
            continue
        for dj in range(3):
            wy = _bspline_w(fx_y, dj)
            gj = by + dj
            if gj < 0 or gj >= n_grid:
                continue
            for dk in range(3):
                wz = _bspline_w(fx_z, dk)
                gk = bz + dk
                if gk < 0 or gk >= n_grid:
                    continue
                flat = gi * n_grid * n_grid + gj * n_grid + gk
                g_sdf = cut_sdf[flat]
                if p_sdf * g_sdf < 0.0:
                    continue
                w    = wx * wy * wz
                vi   = grid_v[flat]
                xi   = wp.vec3(float(gi) * dx, float(gj) * dx, float(gk) * dx)
                dpos = xi - xp
                new_v = new_v + w * vi
                new_C = new_C + (4.0 * inv_dx * inv_dx * w) * wp.outer(vi, dpos)

    # No weight renormalization — following CRESSim-MPM (Ou & Tavakoli 2025).
    # Particles near the cut receive less total weight because some nodes
    # are blocked.  This correctly reduces their velocity rather than
    # amplifying from the surviving nodes.

    I3 = wp.mat33(1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0)
    F[p] = (I3 + dt * new_C) @ F[p]
    C[p] = new_C

    if fixed[p] == 0:
        v[p] = new_v
        x[p] = xp + dt * new_v


@wp.kernel
def _g2p_cut_no_F_update(
    x:       wp.array(dtype=wp.vec3),
    v:       wp.array(dtype=wp.vec3),
    C:       wp.array(dtype=wp.mat33),
    fixed:   wp.array(dtype=int),
    grid_v:  wp.array(dtype=wp.vec3),
    cut_sdf: wp.array(dtype=float),
    n_grid:  int,
    inv_dx:  float,
    dt:      float,
):
    """Cut-aware G2P that does NOT update F — for total-Lagrangian mode.

    Without this kernel, total_lagrangian=True silently loses cut awareness on
    the gather side: step() would pick _g2p_no_F_update, so momentum was
    blocked across the cut during P2G but not during G2P.  That asymmetry made
    "fix TL to get true spring-back" impossible.  F is recomputed from
    positions by _recompute_F_total_lagrangian each step, so it must not also
    be integrated here.
    """
    p = wp.tid()
    xp = x[p]
    dx = 1.0 / inv_dx

    # Particle's SDF side
    pi = wp.clamp(int(wp.round(xp[0] * inv_dx)), 0, n_grid - 1)
    pj = wp.clamp(int(wp.round(xp[1] * inv_dx)), 0, n_grid - 1)
    pk = wp.clamp(int(wp.round(xp[2] * inv_dx)), 0, n_grid - 1)
    p_sdf = cut_sdf[pi * n_grid * n_grid + pj * n_grid + pk]

    bx = int(wp.floor(xp[0] * inv_dx - 0.5))
    by = int(wp.floor(xp[1] * inv_dx - 0.5))
    bz = int(wp.floor(xp[2] * inv_dx - 0.5))

    fx_x = xp[0] * inv_dx - float(bx)
    fx_y = xp[1] * inv_dx - float(by)
    fx_z = xp[2] * inv_dx - float(bz)

    new_v = wp.vec3(0.0, 0.0, 0.0)
    new_C = wp.mat33(0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0)

    for di in range(3):
        wx = _bspline_w(fx_x, di)
        gi = bx + di
        if gi < 0 or gi >= n_grid:
            continue
        for dj in range(3):
            wy = _bspline_w(fx_y, dj)
            gj = by + dj
            if gj < 0 or gj >= n_grid:
                continue
            for dk in range(3):
                wz = _bspline_w(fx_z, dk)
                gk = bz + dk
                if gk < 0 or gk >= n_grid:
                    continue
                flat = gi * n_grid * n_grid + gj * n_grid + gk
                g_sdf = cut_sdf[flat]
                if p_sdf * g_sdf < 0.0:
                    continue
                w    = wx * wy * wz
                vi   = grid_v[flat]
                xi   = wp.vec3(float(gi) * dx, float(gj) * dx, float(gk) * dx)
                dpos = xi - xp
                new_v = new_v + w * vi
                new_C = new_C + (4.0 * inv_dx * inv_dx * w) * wp.outer(vi, dpos)

    # No weight renormalization — following CRESSim-MPM (Ou & Tavakoli 2025).
    # Particles near the cut receive less total weight because some nodes
    # are blocked.  This correctly reduces their velocity rather than
    # amplifying from the surviving nodes.

    C[p] = new_C

    if fixed[p] == 0:
        v[p] = new_v
        x[p] = xp + dt * new_v


@wp.kernel
def _break_bonds_across_cut(
    x:            wp.array(dtype=wp.vec3),
    fib_i:        wp.array(dtype=int),
    fib_j:        wp.array(dtype=int),
    fiber_broken:  wp.array(dtype=int),
    cut_sdf:      wp.array(dtype=float),
    n_grid:       int,
    inv_dx:       float,
):
    """Break fiber bonds where the two particles are on opposite sides of a cut."""
    b = wp.tid()
    if fiber_broken[b] != 0:
        return

    xi = x[fib_i[b]]
    xj = x[fib_j[b]]

    # SDF side for particle i
    gi = wp.clamp(int(wp.round(xi[0] * inv_dx)), 0, n_grid - 1)
    gj = wp.clamp(int(wp.round(xi[1] * inv_dx)), 0, n_grid - 1)
    gk = wp.clamp(int(wp.round(xi[2] * inv_dx)), 0, n_grid - 1)
    sdf_i = cut_sdf[gi * n_grid * n_grid + gj * n_grid + gk]

    # SDF side for particle j
    gi = wp.clamp(int(wp.round(xj[0] * inv_dx)), 0, n_grid - 1)
    gj = wp.clamp(int(wp.round(xj[1] * inv_dx)), 0, n_grid - 1)
    gk = wp.clamp(int(wp.round(xj[2] * inv_dx)), 0, n_grid - 1)
    sdf_j = cut_sdf[gi * n_grid * n_grid + gj * n_grid + gk]

    if sdf_i * sdf_j < 0.0:
        fiber_broken[b] = 1


@wp.kernel
def _g2p_no_F_update(
    x:      wp.array(dtype=wp.vec3),
    v:      wp.array(dtype=wp.vec3),
    C:      wp.array(dtype=wp.mat33),
    fixed:  wp.array(dtype=int),
    grid_v: wp.array(dtype=wp.vec3),
    n_grid: int,
    inv_dx: float,
    dt:     float,
):
    """Grid → Particle: update v, C, x only — F update is handled externally.

    Used in total-Lagrangian mode where F is recomputed from positions each step
    rather than accumulated multiplicatively.
    """
    p = wp.tid()
    xp = x[p]
    dx = 1.0 / inv_dx

    bx = int(wp.floor(xp[0] * inv_dx - 0.5))
    by = int(wp.floor(xp[1] * inv_dx - 0.5))
    bz = int(wp.floor(xp[2] * inv_dx - 0.5))

    fx_x = xp[0] * inv_dx - float(bx)
    fx_y = xp[1] * inv_dx - float(by)
    fx_z = xp[2] * inv_dx - float(bz)

    new_v = wp.vec3(0.0, 0.0, 0.0)
    new_C = wp.mat33(0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0)

    for di in range(3):
        wx = _bspline_w(fx_x, di)
        gi = bx + di
        if gi < 0 or gi >= n_grid:
            continue
        for dj in range(3):
            wy = _bspline_w(fx_y, dj)
            gj = by + dj
            if gj < 0 or gj >= n_grid:
                continue
            for dk in range(3):
                wz = _bspline_w(fx_z, dk)
                gk = bz + dk
                if gk < 0 or gk >= n_grid:
                    continue
                w    = wx * wy * wz
                flat = gi * n_grid * n_grid + gj * n_grid + gk
                vi   = grid_v[flat]
                xi   = wp.vec3(float(gi) * dx, float(gj) * dx, float(gk) * dx)
                dpos = xi - xp
                new_v = new_v + w * vi
                new_C = new_C + (4.0 * inv_dx * inv_dx * w) * wp.outer(vi, dpos)

    C[p] = new_C

    if fixed[p] == 0:
        v[p] = new_v
        x[p] = xp + dt * new_v


@wp.kernel
def _apply_probe_force(
    x:      wp.array(dtype=wp.vec3),
    v:      wp.array(dtype=wp.vec3),
    fixed:  wp.array(dtype=int),
    center: wp.vec3,
    accel:  wp.vec3,   # probe_pressure_pa / rho  [m/s²] — body acceleration at centre
    radius: float,
    dt:     float,
):
    """Apply a spatially-varying body force (pressure field) near the probe centre.

    The acceleration field has a cosine profile: a(r) = accel * w(r/R) where
    w = 0.5*(1 + cos(π*r/R)).  For pressure P [Pa] and tissue density ρ,
    accel = P/ρ * n̂  (n̂ = unit direction of force).

    This is resolution-independent: each particle receives the same dv regardless
    of particle size, because the body force per unit mass is P/ρ * w(r).
    Total force ≈ P * integral(w(r)*dV) over contact patch ≈ P * A_contact/2.
    """
    p = wp.tid()
    if fixed[p] != 0:
        return
    dist = wp.length(x[p] - center)
    if dist < radius:
        t = dist / radius
        w = 0.5 * (1.0 + wp.cos(3.14159265 * t))
        wp.atomic_add(v, p, (w * dt) * accel)


@wp.kernel
def _grid_rigid_sphere_bc(
    grid_v: wp.array(dtype=wp.vec3),
    grid_m: wp.array(dtype=float),
    n_grid: int,
    dx:     float,
    sphere_center: wp.vec3,
    sphere_radius: float,
    push_rate: float,   # outward velocity per metre of penetration [1/s]
    max_dv:   float,    # CFL cap on outward velocity correction [m/s]
    hemi_dir: wp.vec3,  # gate direction; (0,-1,0) = lower half, 0 = full sphere
):
    """Grid-level rigid sphere boundary condition (lower hemisphere only).

    Applied between grid_update and G2P, so that the constrained velocity field
    is properly captured by the APIC C matrix and the deformation gradient F.

    Only the lower hemisphere (nodes below sphere centre) is enforced, matching
    a finger pressing from above.  The outward velocity push is proportional
    to penetration depth but capped at max_dv for CFL stability.
    """
    flat = wp.tid()
    m = grid_m[flat]
    if m <= 0.0:
        return

    i = flat // (n_grid * n_grid)
    j = (flat // n_grid) % n_grid
    k = flat % n_grid
    xi = wp.vec3(float(i) * dx, float(j) * dx, float(k) * dx)

    delta = xi - sphere_center
    dist = wp.length(delta)
    gated = wp.length(hemi_dir) < 1.0e-9 or wp.dot(delta, hemi_dir) > 0.0
    if dist < sphere_radius and dist > 1.0e-12 and gated:
        penetration = sphere_radius - dist
        normal = delta / dist
        v = grid_v[flat]
        v_n = wp.dot(v, normal)
        dv = push_rate * penetration
        if dv > max_dv:
            dv = max_dv
        if v_n < 0.0:
            grid_v[flat] = v + (dv - v_n) * normal
        else:
            grid_v[flat] = v + dv * normal


@wp.kernel
def _apply_hemisphere_contact(
    x:      wp.array(dtype=wp.vec3),
    v:      wp.array(dtype=wp.vec3),
    fixed:  wp.array(dtype=int),
    m_p:    wp.array(dtype=float),
    sphere_center: wp.vec3,
    sphere_radius: float,
    response_rate: float,
    dt:     float,
    sphere_vel: wp.vec3,     # rigid-body translational velocity [m/s]
    stiction:   float,       # tangential velocity-match fraction per kick (0..1)
    contact_impulse: wp.array(dtype=wp.vec3),
    hemi_dir: wp.vec3,       # gate direction; (0,-1,0) = lower half, 0 = full sphere
):
    """Lower-hemisphere rigid contact — applied BEFORE P2G.

    Any free particle in the lower hemisphere of the sphere receives an outward
    velocity kick proportional to penetration depth.  Applied before P2G so the
    modified velocities are scattered to the grid and captured by G2P in C and F.

    Optional stiction: with stiction > 0 each in-contact particle's tangential
    velocity is nudged toward the sphere's tangential velocity by a fraction
    `stiction` per kick.  This is a velocity-match no-slip model — it engages
    even at zero shear velocity, so the finger sticks to the deformed surface
    instead of skidding off at large indentation.

    The negated per-particle impulse (m * dv) — including BOTH the normal kick
    and the tangential stiction kick — is atomically accumulated into
    contact_impulse[0]; this is the Newton-3rd-law reaction on the rigid sphere
    from the tissue, summed across all penetrating particles.
    """
    p = wp.tid()
    if fixed[p] != 0:
        return
    delta = x[p] - sphere_center
    dist = wp.length(delta)
    gated = wp.length(hemi_dir) < 1.0e-9 or wp.dot(delta, hemi_dir) > 0.0
    if dist < sphere_radius and dist > 1.0e-12 and gated:
        penetration = sphere_radius - dist
        normal = delta / dist

        # --- Normal kick (penetration push-back) ---
        dv_mag = response_rate * penetration
        v_n = wp.dot(v[p], normal)
        if v_n < 0.0:
            dv = (dv_mag - v_n) * normal
        else:
            dv = dv_mag * normal

        # --- Tangential stiction (velocity match against sphere) ---
        if stiction > 0.0:
            v_after    = v[p] + dv
            v_t_now    = v_after    - wp.dot(v_after,    normal) * normal
            v_t_target = sphere_vel - wp.dot(sphere_vel, normal) * normal
            dv = dv + stiction * (v_t_target - v_t_now)

        v[p] = v[p] + dv
        wp.atomic_add(contact_impulse, 0, -m_p[p] * dv)


@wp.kernel
def _project_particles_from_sphere(
    x:      wp.array(dtype=wp.vec3),
    v:      wp.array(dtype=wp.vec3),
    fixed:  wp.array(dtype=int),
    sphere_center: wp.vec3,
    sphere_radius: float,
    hemi_dir: wp.vec3,  # gate direction; (0,-1,0) = lower half, 0 = full sphere
):
    """Project any free particle inside the rigid sphere to its surface.

    Runs after G2P to enforce the no-penetration constraint at the particle
    level.  The grid-level BC (_grid_rigid_sphere_bc) already ensures the
    velocity gradient C and deformation gradient F are correct; this kernel
    handles the geometry — particles that the grid BC couldn't fully expel
    due to grid-resolution smearing are placed exactly on the sphere surface.

    The inward velocity component is also removed so the particle doesn't
    re-enter the sphere on the next step.
    """
    p = wp.tid()
    if fixed[p] != 0:
        return
    delta = x[p] - sphere_center
    dist = wp.length(delta)
    # Lower hemisphere only (consistent with grid BC)
    gated = wp.length(hemi_dir) < 1.0e-9 or wp.dot(delta, hemi_dir) > 0.0
    if dist < sphere_radius and dist > 1.0e-12 and gated:
        normal = delta / dist
        x[p] = sphere_center + sphere_radius * normal
        v_n = wp.dot(v[p], normal)
        if v_n < 0.0:
            v[p] = v[p] - v_n * normal


@wp.kernel
def _apply_fiber_forces(
    x:      wp.array(dtype=wp.vec3),
    v:      wp.array(dtype=wp.vec3),
    fixed:  wp.array(dtype=int),
    m_p:    wp.array(dtype=float),
    fib_i:  wp.array(dtype=int),
    fib_j:  wp.array(dtype=int),
    fib_l0: wp.array(dtype=float),
    fib_t:  wp.array(dtype=int),    # 0 = elastin, 1 = collagen
    fiber_broken: wp.array(dtype=int),   # 0 = intact, nonzero = broken
    fiber_k: wp.array(dtype=float),      # per-bond stiffness multiplier
    k_e:    float,
    k_c:    float,
    crimp:  float,
    dt:     float,
):
    """Cosserat-style cable bond forces (one thread per bond).

    Elastin: bidirectional linear spring.
    Collagen: tension-only, activates above crimp strain threshold.
    Broken bonds (fiber_broken != 0) are skipped.

    fiber_k scales both bond types per bond (1.0 = nominal).  This is how thin
    fibrous shells -- a renal capsule, a tumour pseudocapsule -- are modelled:
    at 1-2 mm particle spacing a shell is a few bonds thick, which bonds
    represent well and a bulk modulus does not.
    """
    b = wp.tid()
    if fiber_broken[b] != 0:
        return
    pi = fib_i[b]
    pj = fib_j[b]
    fi = fixed[pi]
    fj = fixed[pj]
    if fi != 0 and fj != 0:
        return

    xi = x[pi]
    xj = x[pj]
    delta = xj - xi
    l = wp.length(delta)
    l0 = fib_l0[b]

    if l < 1.0e-12:
        return

    strain = (l - l0) / l0
    unit = delta / l

    ftype = fib_t[b]
    force_mag = float(0.0)
    fk = fiber_k[b]

    if ftype == 0:
        force_mag = fk * k_e * (l - l0)
    else:
        eff = strain - crimp
        if eff > 0.0:
            force_mag = fk * k_c * eff * l0

    if force_mag == float(0.0):
        return

    force = force_mag * unit
    if fi == 0:
        wp.atomic_add(v, pi,  force * (dt / m_p[pi]))
    if fj == 0:
        wp.atomic_add(v, pj, -force * (dt / m_p[pj]))


@wp.kernel
def _apply_curvature_forces(
    x:       wp.array(dtype=wp.vec3),
    v:       wp.array(dtype=wp.vec3),
    fixed:   wp.array(dtype=int),
    m_p:     wp.array(dtype=float),
    nbr_px:  wp.array(dtype=int),
    nbr_mx:  wp.array(dtype=int),
    nbr_py:  wp.array(dtype=int),
    nbr_my:  wp.array(dtype=int),
    nbr_pz:  wp.array(dtype=int),
    nbr_mz:  wp.array(dtype=int),
    k_curve: float,
    dt:      float,
):
    """Discrete Laplacian (curvature) spring — one thread per center particle.

    For each axis direction where both neighbors exist, penalises deviation of
    this particle from the midpoint of its two axis neighbors:

        L = x[pp] - 2*x[p] + x[pm]   (= 0 at rest on regular lattice)

    Conservative 3-body potential  V = (k_curve/2) |L|^2  gives forces:
        F_center  = +2 * k_curve * L   (restores particle toward midpoint)
        F_pp      = -1 * k_curve * L   (reaction on + neighbor)
        F_pm      = -1 * k_curve * L   (reaction on - neighbor)

    This resists bending of lattice lines, preventing axis-neighbor topology
    crossing, and models the positional homeostasis of cells and ECM.
    """
    p = wp.tid()

    # X direction
    pp = nbr_px[p];  pm = nbr_mx[p]
    if pp >= 0 and pm >= 0:
        lap = x[pp] - 2.0 * x[p] + x[pm]
        fc  = k_curve * lap
        if fixed[p]  == 0: wp.atomic_add(v, p,  (2.0 * fc) * (dt / m_p[p]))
        if fixed[pp] == 0: wp.atomic_add(v, pp, (-fc)       * (dt / m_p[pp]))
        if fixed[pm] == 0: wp.atomic_add(v, pm, (-fc)       * (dt / m_p[pm]))

    # Y direction
    pp = nbr_py[p];  pm = nbr_my[p]
    if pp >= 0 and pm >= 0:
        lap = x[pp] - 2.0 * x[p] + x[pm]
        fc  = k_curve * lap
        if fixed[p]  == 0: wp.atomic_add(v, p,  (2.0 * fc) * (dt / m_p[p]))
        if fixed[pp] == 0: wp.atomic_add(v, pp, (-fc)       * (dt / m_p[pp]))
        if fixed[pm] == 0: wp.atomic_add(v, pm, (-fc)       * (dt / m_p[pm]))

    # Z direction
    pp = nbr_pz[p];  pm = nbr_mz[p]
    if pp >= 0 and pm >= 0:
        lap = x[pp] - 2.0 * x[p] + x[pm]
        fc  = k_curve * lap
        if fixed[p]  == 0: wp.atomic_add(v, p,  (2.0 * fc) * (dt / m_p[p]))
        if fixed[pp] == 0: wp.atomic_add(v, pp, (-fc)       * (dt / m_p[pp]))
        if fixed[pm] == 0: wp.atomic_add(v, pm, (-fc)       * (dt / m_p[pm]))


# ---------------------------------------------------------------------------
# Simulator class
# ---------------------------------------------------------------------------

@wp.kernel
def _apply_attachment_forces(
    x:         wp.array(dtype=wp.vec3),
    v:         wp.array(dtype=wp.vec3),
    fixed:     wp.array(dtype=int),
    m_p:       wp.array(dtype=float),
    att_p:     wp.array(dtype=int),      # particle index per attachment
    att_ref:   wp.array(dtype=wp.vec3),  # attach point in FRAME coordinates [m]
    att_k:     wp.array(dtype=float),    # stiffness [N/m]
    att_c:     wp.array(dtype=float),    # damping   [N s/m]
    att_state: wp.array(dtype=int),      # 1 = attached, 0 = released
    frame_o:   wp.vec3,                  # frame origin, world [m]
    frame_v:   wp.vec3,                  # frame velocity [m/s]
    f_break:   float,                    # per-attachment release force [N]; <=0 = never
    dt:        float,
    reaction:  wp.array(dtype=wp.vec3),  # [0] accumulates the impulse on the frame
):
    """Damped springs tying particles to a moving rigid frame.

    One kernel serves two things that look different but are not: the renal
    pedicle tether is a set of particles sprung to a *static* frame (the aorta
    anchor), and a suction cup is the same set sprung to a *moving* frame (the
    cup body).  Both are compliant, both report a reaction force, and both can
    let go under load.

    Each attachment pulls its particle toward ``frame_o + att_ref[a]``, so the
    attachment geometry is recorded once in frame coordinates and then rides
    along with the frame -- no snap when the frame starts moving.

    ``f_break`` is a per-attachment release threshold.  Checking it on-device
    rather than in Python avoids a per-step readback and makes the failure
    cascade emergent and correct: edge attachments stretch most, release
    first, the survivors then carry more, and the grip peels off rather than
    vanishing all at once.

    The Newton-3rd-law reaction impulse is accumulated so a tool can be driven
    by the force the tissue exerts back on it.
    """
    a = wp.tid()
    if att_state[a] == 0:
        return
    p = att_p[a]
    if fixed[p] != 0:
        return

    target = frame_o + att_ref[a]
    f = att_k[a] * (target - x[p]) + att_c[a] * (frame_v - v[p])

    if f_break > 0.0 and wp.length(f) > f_break:
        att_state[a] = 0
        return

    wp.atomic_add(v, p, f * (dt / m_p[p]))
    wp.atomic_add(reaction, 0, -f * dt)


class AttachmentSet:
    """A set of damped springs tying particles to a rigid frame.

    Two instances of the same thing:

    * **pedicle tether** -- hilar kidney particles sprung to a *static* frame
      at the aorta.  Replaces a hard Dirichlet BC with something compliant, so
      the organ can move and recoil instead of being nailed in place, and so
      the tether force is observable (and can avulse).
    * **suction cup** -- tumour-surface particles sprung to a *moving* frame
      (the cup body).  Force-controlled, so it can slip or pop off.

    Stiffness is specified as a total for the set and divided across
    attachments, which keeps behaviour resolution-independent: refining the
    particle lattice does not silently stiffen the tether.

    Damping defaults to a fraction ``zeta`` of critical for each attachment's
    own particle mass.

    Stability: an explicit spring needs ``k_per < 2 m_p / dt^2``, and a clean
    (non-ringing) response wants ``omega*dt < 0.2``, i.e. roughly
    ``k_per < 0.04 m_p / dt^2``.  ``check_stability()`` reports both margins.
    """

    def __init__(self, sim, particle_idx, frame_origin_m, k_total,
                 zeta=0.7, f_break_total=None, ref_positions=None):
        """
        Args:
            sim:            the MPMSimulator to attach to.
            particle_idx:   indices of the particles to attach.
            frame_origin_m: the frame's world origin [m].  Attachment offsets
                            are recorded relative to this, so the set rides
                            with the frame without snapping.
            k_total:        total stiffness of the set [N/m], divided evenly.
            zeta:           damping ratio per attachment (0.7 = well damped).
            f_break_total:  total force at which the set fully lets go [N].
                            None = never releases.  Divided per attachment, so
                            release is a peel rather than an all-at-once pop.
            ref_positions:  attachment points in WORLD coordinates [m].
                            Defaults to the particles' current positions,
                            which is what you want when attaching to whatever
                            the tissue is doing right now.  Pass rest
                            positions to tether tissue to its rest shape.
        """
        self.sim = sim
        idx = np.ascontiguousarray(np.asarray(particle_idx, dtype=np.int32).ravel())
        if idx.size == 0:
            raise ValueError("AttachmentSet needs at least one particle")
        n = idx.size

        pos = sim.x.numpy() if ref_positions is None else np.asarray(ref_positions,
                                                                     dtype=np.float32)
        if ref_positions is None:
            ref_world = pos[idx]
        else:
            ref_world = pos if pos.shape[0] == n else pos[idx]

        self._origin = np.asarray(frame_origin_m, dtype=np.float64).reshape(3)
        ref_frame = (ref_world - self._origin).astype(np.float32)

        m = sim.m_p.numpy()[idx].astype(np.float64)
        k_per = float(k_total) / n
        c_per = 2.0 * float(zeta) * np.sqrt(np.maximum(k_per * m, 0.0))

        self.k_total = float(k_total)
        self.zeta = float(zeta)
        self.f_break = (float(f_break_total) / n) if f_break_total else 0.0
        self._frame_v = np.zeros(3)
        self._n0 = n

        with wp.ScopedDevice(sim.device):
            self.att_p = wp.array(idx, dtype=int)
            self.att_ref = wp.array(np.ascontiguousarray(ref_frame), dtype=wp.vec3)
            self.att_k = wp.array(np.full(n, k_per, dtype=np.float32), dtype=float)
            self.att_c = wp.array(np.ascontiguousarray(c_per, dtype=np.float32),
                                  dtype=float)
            self.att_state = wp.array(np.ones(n, dtype=np.int32), dtype=int)
            self._reaction = wp.zeros(1, dtype=wp.vec3)
        self._last_impulse = np.zeros(3, dtype=np.float32)

    # -- frame ---------------------------------------------------------------

    def set_frame(self, origin_m, vel_m_s=None):
        """Move the frame.  Attachments follow, dragging their particles."""
        self._origin = np.asarray(origin_m, dtype=np.float64).reshape(3)
        self._frame_v = (np.zeros(3) if vel_m_s is None
                         else np.asarray(vel_m_s, dtype=np.float64).reshape(3))

    @property
    def frame_origin(self):
        return self._origin.copy()

    # -- state ---------------------------------------------------------------

    @property
    def n_attached(self):
        """How many attachments are still holding."""
        return int(self.att_state.numpy().sum())

    @property
    def n_total(self):
        return self._n0

    def release(self):
        """Let go entirely."""
        with wp.ScopedDevice(self.sim.device):
            self.att_state = wp.zeros(self._n0, dtype=int)

    @property
    def reaction_impulse(self):
        """Impulse applied to the frame over the last step [kg m/s]."""
        return self._last_impulse.copy()

    @property
    def reaction_force(self):
        """Force the tissue exerts on the frame, last step [N]."""
        return self._last_impulse / float(self.sim.dt)

    # -- diagnostics ---------------------------------------------------------

    def check_stability(self):
        """Return the explicit-integration margins for this set.

        ``hard`` < 1 is required for stability at all; ``clean`` < 1 keeps the
        spring from ringing at the timestep scale.
        """
        k_per = self.k_total / self._n0
        m_min = float(self.sim.m_p.numpy()[self.att_p.numpy()].min())
        dt = float(self.sim.dt)
        return {
            "k_per": k_per,
            "k_hard_limit": 2.0 * m_min / dt ** 2,
            "k_clean_limit": 0.04 * m_min / dt ** 2,
            "hard": k_per / (2.0 * m_min / dt ** 2),
            "clean": k_per / (0.04 * m_min / dt ** 2),
        }

    # -- called by MPMSimulator._step_core -----------------------------------

    def _launch(self, dt):
        self._reaction.zero_()
        wp.launch(_apply_attachment_forces, dim=self._n0,
                  inputs=[self.sim.x, self.sim.v, self.sim.fixed, self.sim.m_p,
                          self.att_p, self.att_ref, self.att_k, self.att_c,
                          self.att_state,
                          wp.vec3(*[float(c) for c in self._origin]),
                          wp.vec3(*[float(c) for c in self._frame_v]),
                          float(self.f_break), float(dt), self._reaction])

    def _read_reaction(self):
        self._last_impulse = self._reaction.numpy()[0].copy()


class MPMSimulator:
    """Explicit MLS-MPM simulator for a rectangular soft-tissue block.

    Coordinate system: Y is vertical (up), X and Z are horizontal.
    Units: SI (metres, kg, Pa, seconds).

    Key feature — curvature springs (k_curve > 0) + total_lagrangian=True:
      Set material.k_curve to a non-zero value (10–50 N/m) to activate
      discrete Laplacian springs that resist bending of lattice lines.
      These prevent axis-neighbor topology crossing during palpation, which
      in turn makes total_lagrangian=True safe at all deformation levels.
      With both enabled, F is always recomputed from actual particle positions
      → Neo-Hookean stress correctly drives elastic recovery after palpation,
      with no F-drift and no artificial state resets needed.
    """

    def __init__(
        self,
        block_lo,
        block_hi,
        n_grid: int = 32,
        dt: float = 5e-4,
        material: MPMMaterial = None,
        device: str = "cpu",
        velocity_damping: float = 0.995,
        total_lagrangian: bool = False,
    ):
        if material is None:
            material = MPMMaterial()
        self.block_lo = np.asarray(block_lo, dtype=np.float64)
        self.block_hi = np.asarray(block_hi, dtype=np.float64)
        self.n_grid   = n_grid
        self.dt       = dt
        self.material = material
        self.device           = device
        self.velocity_damping = velocity_damping
        self.total_lagrangian = total_lagrangian

        block_size  = float((self.block_hi - self.block_lo).max())
        self.dx     = block_size / n_grid
        self.inv_dx = 1.0 / self.dx

        with wp.ScopedDevice(device):
            n_nodes = n_grid ** 3
            self.grid_v = wp.zeros(n_nodes, dtype=wp.vec3)
            self.grid_m = wp.zeros(n_nodes, dtype=float)

        self.n_particles = 0

        # Lattice-neighbour arrays for total-Lagrangian F (set in initialize_block_particles)
        self.nbr_px = None;  self.nbr_mx = None
        self.nbr_py = None;  self.nbr_my = None
        self.nbr_pz = None;  self.nbr_mz = None
        self._lattice_step = 0.0

        # Gravity-equilibrium reference (set by sample_equilibrium())
        self.x_eq = None   # (n, 3) float32 positions at gravity equilibrium
        self.F_eq = None   # (n, 3, 3) float32 deformation gradients at gravity equilibrium

        # Optional grid-level fixed BC (e.g. bone).  Set to a warp array of
        # int (0=free, nonzero=fixed) with n_grid^3 elements to enforce zero
        # velocity at those grid nodes every step.
        self.grid_bc_fixed = None

        # Optional static-obstacle SDF (grid BC + particle projection).
        # Set obstacle_sdf (float per grid node, neg = inside the obstacle) and
        # obstacle_sdf_grad (vec3 per grid node, outward normal) to enable.
        # Historically named bone_sdf; it is a generic obstacle -- skull,
        # psoas, body wall, a synthetic support half-space -- so the neutral
        # name is primary and bone_sdf remains as an alias property.
        self.obstacle_sdf      = None
        self.obstacle_sdf_grad = None

        #: Coulomb friction coefficient for the obstacle surface.  0 = the
        #: historical frictionless sliding.  Retroperitoneal fat is slippery
        #: but not frictionless: ~0.15.
        self.obstacle_friction = 0.0

        #: Attachment spring sets (AttachmentSet).  Launched by _step_core in
        #: the pre-P2G position, so their force is scattered to the grid and
        #: seen by the same step's stress update.  Used for the renal-pedicle
        #: tether and the suction cup.
        self.attachments = []

        #: Frictionless no-penetration floor at grid row j == 0.  On by
        #: default for backward compatibility.  Turn it off in scenarios that
        #: supply their own support surface, or the most-posterior grid plane
        #: acts as a second, invisible wall.
        self.floor_j0 = True

        # Per-particle Lamé parameters.  Allocated lazily from self.material by
        # _ensure_material_arrays(), which every step entry point calls, so
        # callers that populate self.x / self.m_p by hand (mpm_ct_head,
        # mpm_kidney_resection) keep working unchanged.  Use
        # set_particle_material() for heterogeneous tissue.
        self.mu_p  = None   # wp.array(dtype=float), shear modulus [Pa]
        self.lam_p = None   # wp.array(dtype=float), first Lamé parameter [Pa]
        self._E_max = float(material.E)   # stiffest particle; drives CFL and
                                          # the contact response rate

        # Fiber bond arrays
        self.fiber_i  = None;  self.fiber_j  = None
        self.fiber_l0 = None;  self.fiber_t  = None
        self.fiber_broken = None  # wp.array(dtype=int), 0=intact, 1=broken
        self.fiber_k = None       # wp.array(dtype=float), per-bond stiffness scale
        self.n_bonds  = 0

        # Cut SDFs: list of wp.array(dtype=float) of length n_grid^3.
        # Sign encodes which side of the cut a point is on.  When active,
        # P2G/G2P block transfers across each cut surface, and bonds that
        # cross the cut are broken.
        self.cut_sdfs = []

        # Single-element vec3 buffer that step_with_contact zeros on entry and
        # the contact kernel atomically accumulates the Newton-3rd-law impulse
        # into.  Read after step_with_contact via last_contact_impulse / force.
        with wp.ScopedDevice(device):
            self._contact_impulse_buf = wp.zeros(1, dtype=wp.vec3)
        self._last_contact_impulse_np = np.zeros(3, dtype=np.float32)

    def initialize_block_particles(
        self,
        lo=None,
        hi=None,
        ppc: int = 2,
        fixed_y_max: float = None,
    ):
        """Fill a rectangular region with particles on a regular sub-cell grid."""
        lo = np.asarray(lo if lo is not None else self.block_lo, dtype=np.float64)
        hi = np.asarray(hi if hi is not None else self.block_hi, dtype=np.float64)
        if fixed_y_max is None:
            fixed_y_max = float(lo[1]) + 2.0 * self.dx

        step = self.dx / ppc
        off  = 0.5 * step

        xs = np.arange(lo[0] + off, hi[0], step)
        ys = np.arange(lo[1] + off, hi[1], step)
        zs = np.arange(lo[2] + off, hi[2], step)
        ix, iy, iz = np.meshgrid(xs, ys, zs, indexing='ij')
        positions = np.stack(
            [ix.ravel(), iy.ravel(), iz.ravel()], axis=1
        ).astype(np.float32)

        n = len(positions)
        self.n_particles = n

        self._warn_grid_margin(positions)

        vol_p  = float(step ** 3)
        mass_p = float(self.material.rho * vol_p)
        fixed_mask = (positions[:, 1] <= fixed_y_max).astype(np.int32)

        F_np = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
        C_np = np.zeros((n, 3, 3), dtype=np.float32)

        with wp.ScopedDevice(self.device):
            self.x     = wp.array(positions,          dtype=wp.vec3)
            self.x0    = wp.array(positions.copy(),   dtype=wp.vec3)
            self.v     = wp.zeros(n,                  dtype=wp.vec3)
            self.F     = wp.array(F_np,               dtype=wp.mat33)
            self.C     = wp.array(C_np,               dtype=wp.mat33)
            self.m_p   = wp.array(np.full(n, mass_p,  dtype=np.float32), dtype=float)
            self.vol_p = wp.array(np.full(n, vol_p,   dtype=np.float32), dtype=float)
            self.fixed = wp.array(fixed_mask,          dtype=int)

        print(f"MPMSimulator: {n} particles, {int(fixed_mask.sum())} fixed, "
              f"dx={self.dx*1000:.2f}mm, vol_p={vol_p*1e9:.3f}mm3")

        nx, ny, nz = len(xs), len(ys), len(zs)

        # Always build lattice neighbours — needed for reset_F_from_positions()
        # even when total_lagrangian=False (on-demand F correction after recovery).
        self._build_lattice_neighbors(nx, ny, nz)
        self._lattice_step = float(step)

        if self.material.k_elastin > 0.0 or self.material.k_collagen > 0.0:
            self._build_fiber_bonds(nx, ny, nz, step)

    def _build_lattice_neighbors(self, nx: int, ny: int, nz: int):
        """Build the 6 axis-aligned neighbour arrays needed for total-Lagrangian F.

        nbr_px[p] = flat index of the particle one step in the +X direction, or -1.
        nbr_mx[p] = flat index of the particle one step in the -X direction, or -1.
        Analogously for Y (py/my) and Z (pz/mz).
        """
        n = nx * ny * nz
        pi_arr = np.arange(n, dtype=np.int32).reshape(nx, ny, nz)
        NONE = np.int32(-1)

        def make_nbr(di, dj, dk):
            arr = np.full(n, NONE, dtype=np.int32)
            i0 = max(0, -di);  i1 = nx - max(0, di)
            j0 = max(0, -dj);  j1 = ny - max(0, dj)
            k0 = max(0, -dk);  k1 = nz - max(0, dk)
            src = pi_arr[i0:i1, j0:j1, k0:k1].ravel()
            tgt = pi_arr[i0+di:i1+di, j0+dj:j1+dj, k0+dk:k1+dk].ravel()
            arr[src] = tgt
            return arr

        with wp.ScopedDevice(self.device):
            self.nbr_px = wp.array(make_nbr( 1,  0,  0), dtype=int)
            self.nbr_mx = wp.array(make_nbr(-1,  0,  0), dtype=int)
            self.nbr_py = wp.array(make_nbr( 0,  1,  0), dtype=int)
            self.nbr_my = wp.array(make_nbr( 0, -1,  0), dtype=int)
            self.nbr_pz = wp.array(make_nbr( 0,  0,  1), dtype=int)
            self.nbr_mz = wp.array(make_nbr( 0,  0, -1), dtype=int)

        print(f"MPMSimulator: lattice neighbour arrays built ({nx}x{ny}x{nz})")

    def _build_fiber_bonds(self, nx: int, ny: int, nz: int, step: float):
        """Build fiber bond arrays for Cosserat-style cable forces."""
        pi_arr = np.arange(nx * ny * nz, dtype=np.int32).reshape(nx, ny, nz)

        bonds_i_list  = []
        bonds_j_list  = []
        bonds_l0_list = []
        bonds_t_list  = []

        if self.material.k_elastin > 0.0:
            elastin_offsets = [
                (di, dj, dk)
                for di in [-1, 0, 1]
                for dj in [-1, 0, 1]
                for dk in [-1, 0, 1]
                if (di, dj, dk) > (0, 0, 0)
            ]
            for di, dj, dk in elastin_offsets:
                i0 = max(0, -di);  i1 = nx - max(0, di)
                j0 = max(0, -dj);  j1 = ny - max(0, dj)
                k0 = max(0, -dk);  k1 = nz - max(0, dk)
                src = pi_arr[i0:i1, j0:j1, k0:k1].ravel()
                tgt = pi_arr[i0+di:i1+di, j0+dj:j1+dj, k0+dk:k1+dk].ravel()
                l0  = float(np.sqrt(di**2 + dj**2 + dk**2)) * step
                nb  = len(src)
                bonds_i_list.append(src);  bonds_j_list.append(tgt)
                bonds_l0_list.append(np.full(nb, l0, dtype=np.float32))
                bonds_t_list.append(np.zeros(nb, dtype=np.int32))

        if self.material.k_collagen > 0.0:
            collagen_offsets = [(2, 0, 0), (0, 2, 0), (0, 0, 2)]
            for di, dj, dk in collagen_offsets:
                i0, i1 = 0, nx - di
                j0, j1 = 0, ny - dj
                k0, k1 = 0, nz - dk
                src = pi_arr[i0:i1, j0:j1, k0:k1].ravel()
                tgt = pi_arr[i0+di:i1+di, j0+dj:j1+dj, k0+dk:k1+dk].ravel()
                l0  = float(di + dj + dk) * step
                nb  = len(src)
                bonds_i_list.append(src);  bonds_j_list.append(tgt)
                bonds_l0_list.append(np.full(nb, l0, dtype=np.float32))
                bonds_t_list.append(np.ones(nb, dtype=np.int32))

        if not bonds_i_list:
            return

        all_i  = np.concatenate(bonds_i_list).astype(np.int32)
        all_j  = np.concatenate(bonds_j_list).astype(np.int32)
        all_l0 = np.concatenate(bonds_l0_list).astype(np.float32)
        all_t  = np.concatenate(bonds_t_list).astype(np.int32)

        n_e = int((all_t == 0).sum());  n_c = int((all_t == 1).sum())
        print(f"MPMSimulator: {len(all_i)} fiber bonds (elastin={n_e}, collagen={n_c})")

        with wp.ScopedDevice(self.device):
            self.fiber_i  = wp.array(all_i,  dtype=int)
            self.fiber_j  = wp.array(all_j,  dtype=int)
            self.fiber_l0 = wp.array(all_l0, dtype=float)
            self.fiber_t  = wp.array(all_t,  dtype=int)
            self.fiber_broken = wp.zeros(len(all_i), dtype=int)
        self.n_bonds = len(all_i)

        # Particle/bond counts just changed: drop any per-particle material so
        # _ensure_material_arrays() rebuilds it uniformly from self.material.
        # Call set_particle_material() AFTER this to make tissue heterogeneous.
        self.mu_p = None
        self.lam_p = None
        self.fiber_k = None
        self._E_max = float(self.material.E)

    # ------------------------------------------------------------------
    # Per-particle material
    # ------------------------------------------------------------------

    def _warn_grid_margin(self, positions):
        """Warn when particles reach the outermost grid cells.

        P2G and G2P *skip* out-of-range nodes rather than clamping, so
        momentum scattered beyond the grid is silently lost and G2P gathers a
        reduced velocity.  The result reads as a mysterious drag that pins the
        body in place -- easy to mistake for a physics result.  Keep at least
        two cells of padding between the particles and block_lo/block_hi.
        """
        margin = 2.0 * self.dx
        lo_edge = positions.min(axis=0) - self.block_lo
        hi_edge = self.block_hi - positions.max(axis=0)
        if (lo_edge < margin).any() or (hi_edge < margin).any():
            print("MPMSimulator: WARNING particles come within 2*dx (%.1f mm) of "
                  "the grid boundary; margins lo=%s mm hi=%s mm.  P2G/G2P drop "
                  "out-of-range nodes, so the boundary acts as an artificial "
                  "drag.  Enlarge block_lo/block_hi relative to the particles."
                  % (margin * 1000,
                     np.round(lo_edge * 1000, 1), np.round(hi_edge * 1000, 1)))

    def _ensure_material_arrays(self):
        """Allocate mu_p / lam_p / fiber_k if absent or stale.

        Called at the top of every step entry point.  Examples that build the
        particle arrays by hand (rather than via initialize_block_particles)
        therefore need no changes: they get uniform arrays broadcast from
        self.material on the first step.
        """
        n = self.n_particles
        if n and (self.mu_p is None or len(self.mu_p) != n):
            with wp.ScopedDevice(self.device):
                self.mu_p = wp.array(
                    np.full(n, self.material.mu, dtype=np.float32), dtype=float)
                self.lam_p = wp.array(
                    np.full(n, self.material.lam, dtype=np.float32), dtype=float)
            self._E_max = float(self.material.E)
        if self.n_bonds and (self.fiber_k is None or len(self.fiber_k) != self.n_bonds):
            with wp.ScopedDevice(self.device):
                self.fiber_k = wp.array(
                    np.ones(self.n_bonds, dtype=np.float32), dtype=float)

    def set_particle_material(self, mu=None, lam=None, E=None, nu=None):
        """Set per-particle Lamé parameters.

        Accepts scalars (broadcast) or (n,) arrays.  Either give (mu, lam)
        directly or (E, nu) to be converted.  Updates _E_max, which drives
        cfl_dt() and the contact response rate -- so call this BEFORE choosing
        the timestep.

        Example, a tumour softer than the surrounding parenchyma::

            E = np.where(is_tumor, 13_000.0, 19_000.0)
            sim.set_particle_material(E=E, nu=0.48)
            sim.dt = sim.cfl_dt()
        """
        n = self.n_particles
        if not n:
            raise RuntimeError("no particles yet; populate the simulator first")

        if E is not None:
            E = np.broadcast_to(np.asarray(E, dtype=np.float64), (n,))
            nu = self.material.nu if nu is None else nu
            nu = np.broadcast_to(np.asarray(nu, dtype=np.float64), (n,))
            mu = E / (2.0 * (1.0 + nu))
            lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        else:
            if mu is None or lam is None:
                raise ValueError("give either (E, nu) or both mu and lam")
            mu = np.broadcast_to(np.asarray(mu, dtype=np.float64), (n,))
            lam = np.broadcast_to(np.asarray(lam, dtype=np.float64), (n,))
            # E = mu(3 lam + 2 mu) / (lam + mu), for the CFL bound
            E = mu * (3.0 * lam + 2.0 * mu) / np.maximum(lam + mu, 1e-12)

        with wp.ScopedDevice(self.device):
            self.mu_p = wp.array(np.ascontiguousarray(mu, dtype=np.float32), dtype=float)
            self.lam_p = wp.array(np.ascontiguousarray(lam, dtype=np.float32), dtype=float)
        self._E_max = float(np.max(E))

    def set_bond_stiffness(self, scale):
        """Set the per-bond stiffness multiplier (scalar or (n_bonds,) array).

        Used for fibrous shells such as the renal capsule and the tumour
        pseudocapsule, which are thin enough that bonds -- not bulk moduli --
        are the right representation at 1-2 mm particle spacing.
        """
        if not self.n_bonds:
            raise RuntimeError("no bonds; build the fiber network first")
        scale = np.broadcast_to(np.asarray(scale, dtype=np.float32), (self.n_bonds,))
        with wp.ScopedDevice(self.device):
            self.fiber_k = wp.array(np.ascontiguousarray(scale, dtype=np.float32),
                                    dtype=float)

    @property
    def wave_speed(self):
        """Rod wave speed of the stiffest particle, sqrt(E_max/rho) [m/s].

        NOT the CFL bound -- see p_wave_speed.  This is kept as the reference
        speed for the contact response rate, where it is a tuned parameter
        rather than a stability limit.
        """
        return float(np.sqrt(self._E_max / self.material.rho))

    @property
    def p_wave_speed(self):
        """Dilatational (P-wave) speed of the stiffest particle [m/s].

        sqrt((lam + 2 mu) / rho).  This is the fastest signal an explicit
        scheme has to resolve, so it -- not sqrt(E/rho) -- sets the CFL bound.

        The difference is not academic for soft tissue.  At nu = 0.48 the
        P-wave speed is ~3x the rod speed, because lam blows up as nu -> 0.5:

            nu = 0.30   ratio 1.16
            nu = 0.45   ratio 2.11
            nu = 0.48   ratio 2.96
            nu = 0.49   ratio 4.15

        Sizing dt from sqrt(E/rho) at nu = 0.48 therefore runs at roughly 3x
        the intended Courant number, which shows up as slow energy growth
        rather than an immediate blow-up -- tissue that keeps creeping and
        never settles.
        """
        if self.mu_p is None or self.lam_p is None:
            mu, lam = self.material.mu, self.material.lam
        else:
            mu = float(self.mu_p.numpy().max())
            lam = float(self.lam_p.numpy().max())
        return float(np.sqrt((lam + 2.0 * mu) / self.material.rho))

    def courant(self, dt=None):
        """Courant number c_p * dt / dx for the current (or a proposed) dt."""
        dt = float(self.dt if dt is None else dt)
        return float(self.p_wave_speed * dt / self.dx)

    def cfl_dt(self, cfl: float = 0.18, dt_max: float = 2.0e-4) -> float:
        """CFL-limited timestep, min(dt_max, cfl * dx / c_p).

        Uses the dilatational speed of the stiffest particle, so a
        heterogeneous, nearly-incompressible model is stable everywhere.
        """
        return float(min(dt_max, cfl * self.dx / self.p_wave_speed))

    def _step_core(self, gravity, pre_p2g=(), post_g2p=(), post_forces=()):
        """The single MPM pipeline. Every step entry point goes through here.

        step(), step_with_contact() and step_with_probe() used to be near-
        duplicate copies of this sequence, and that duplication was a bug, not
        just repetition: step_with_contact's copy always launched the plain
        _p2g/_g2p, so an active cut silently stopped blocking momentum during
        palpation.  Defining the pipeline once fixes that by construction.

        The hooks are sequences of zero-arg callables that launch their own
        kernels at documented points, all inside the ScopedDevice:

        pre_p2g     before the grid is zeroed, so the effect is scattered to
                    the grid and captured by the APIC C matrix (and hence F).
                    Sustained loads -- contact, tethers, suction -- belong
                    here; applied only after G2P they are invisible to that
                    step's stress update, which is what drives F drift.
        post_g2p    immediately after G2P, for the geometric half of a
                    dual-kick contact.
        post_forces after the fiber and curvature forces.
        """
        if gravity is None:
            gravity = np.array([0.0, -9.8, 0.0])
        self._ensure_material_arrays()
        ng  = self.n_grid
        n   = self.n_particles
        dt  = float(self.dt)
        g   = wp.vec3(float(gravity[0]), float(gravity[1]), float(gravity[2]))
        cut = self.cut_sdfs[-1] if self.cut_sdfs else None

        with wp.ScopedDevice(self.device):
            for hook in pre_p2g:
                hook()

            # Attachment springs are a SUSTAINED load, so they belong before
            # P2G: applied after G2P they would be invisible to this step's
            # stress update, which is exactly the mechanism that drives F
            # drift (see the step_with_contact docstring).
            for att in self.attachments:
                att._launch(dt)

            # Total-Lagrangian: recompute F from current positions BEFORE P2G
            # so the stress uses a drift-free deformation gradient.
            if self.total_lagrangian and self.nbr_px is not None:
                wp.launch(_recompute_F_total_lagrangian, dim=n,
                          inputs=[self.x, self.F,
                                  self.nbr_px, self.nbr_mx,
                                  self.nbr_py, self.nbr_my,
                                  self.nbr_pz, self.nbr_mz,
                                  float(self._lattice_step)])

            wp.launch(_zero_grid, dim=ng**3, inputs=[self.grid_v, self.grid_m])

            # P2G — cut-aware when a cut is active.  Only the most recent cut
            # SDF is used; multiple cuts must be merged into one composite
            # field by the caller (see apply_cut(replace=True)).
            if cut is not None:
                wp.launch(_p2g_cut, dim=n,
                          inputs=[self.x, self.v, self.F, self.C,
                                  self.m_p, self.vol_p,
                                  self.grid_v, self.grid_m, cut,
                                  self.mu_p, self.lam_p,
                                  ng, float(self.inv_dx), dt])
            else:
                wp.launch(_p2g, dim=n,
                          inputs=[self.x, self.v, self.F, self.C,
                                  self.m_p, self.vol_p,
                                  self.grid_v, self.grid_m,
                                  self.mu_p, self.lam_p,
                                  ng, float(self.inv_dx), dt])

            wp.launch(_grid_update, dim=ng**3,
                      inputs=[self.grid_v, self.grid_m, ng, dt, g,
                              float(self.velocity_damping),
                              int(bool(self.floor_j0))])

            if self.grid_bc_fixed is not None:
                wp.launch(_apply_grid_fixed_bc, dim=ng**3,
                          inputs=[self.grid_v, self.grid_bc_fixed])

            if self.obstacle_sdf is not None:
                wp.launch(_apply_grid_sdf_bc, dim=ng**3,
                          inputs=[self.grid_v, self.obstacle_sdf,
                                  self.obstacle_sdf_grad, float(self.dx),
                                  float(self.obstacle_friction)])

            # G2P — a clean 2x2 over (cut active?, total-Lagrangian?)
            if cut is not None and self.total_lagrangian:
                wp.launch(_g2p_cut_no_F_update, dim=n,
                          inputs=[self.x, self.v, self.C, self.fixed,
                                  self.grid_v, cut,
                                  ng, float(self.inv_dx), dt])
            elif cut is not None:
                wp.launch(_g2p_cut, dim=n,
                          inputs=[self.x, self.v, self.F, self.C, self.fixed,
                                  self.grid_v, cut,
                                  ng, float(self.inv_dx), dt])
            elif self.total_lagrangian:
                wp.launch(_g2p_no_F_update, dim=n,
                          inputs=[self.x, self.v, self.C, self.fixed,
                                  self.grid_v, ng, float(self.inv_dx), dt])
            else:
                wp.launch(_g2p, dim=n,
                          inputs=[self.x, self.v, self.F, self.C, self.fixed,
                                  self.grid_v, ng, float(self.inv_dx), dt])

            for hook in post_g2p:
                hook()

            if self.obstacle_sdf is not None:
                wp.launch(_apply_bone_sdf_contact, dim=n,
                          inputs=[self.x, self.v, self.fixed,
                                  self.obstacle_sdf, self.obstacle_sdf_grad,
                                  ng, float(self.inv_dx),
                                  float(self.obstacle_friction)])

            if self.n_bonds > 0:
                wp.launch(_apply_fiber_forces, dim=self.n_bonds,
                          inputs=[
                              self.x, self.v, self.fixed, self.m_p,
                              self.fiber_i, self.fiber_j, self.fiber_l0, self.fiber_t,
                              self.fiber_broken, self.fiber_k,
                              float(self.material.k_elastin),
                              float(self.material.k_collagen),
                              float(self.material.collagen_crimp),
                              dt,
                          ])

            if self.material.k_curve > 0.0 and self.nbr_px is not None:
                wp.launch(_apply_curvature_forces, dim=n,
                          inputs=[
                              self.x, self.v, self.fixed, self.m_p,
                              self.nbr_px, self.nbr_mx,
                              self.nbr_py, self.nbr_my,
                              self.nbr_pz, self.nbr_mz,
                              float(self.material.k_curve),
                              dt,
                          ])

            for hook in post_forces:
                hook()

        for att in self.attachments:
            att._read_reaction()

    def step(self, gravity=None):
        """Advance one explicit timestep."""
        self._step_core(gravity)

    def sample_equilibrium(self):
        """Snapshot current positions as the gravity-equilibrium reference.

        Call this after a sufficient gravity settlement (before any palpation).
        The saved positions are used by check_near_equilibrium() to detect when
        the tissue has returned to its resting configuration after palpation.
        """
        self.x_eq = self.x.numpy().copy()
        self.F_eq = self.F.numpy().copy()

    def check_near_equilibrium(self) -> float:
        """Return max displacement of any free particle from the gravity-equilibrium positions.

        Returns float('inf') if sample_equilibrium() has not been called.
        Use this to decide when reset_F_from_positions() is safe to call:
        safe when result < ~0.4 * lattice_step (guarantees no lattice topology crossing).

        Returns:
            max displacement [m] from x_eq, or inf if no equilibrium sampled.
        """
        if self.x_eq is None:
            return float('inf')
        pos  = self.x.numpy()
        free = (self.fixed.numpy() == 0)
        return float(np.abs(pos[free] - self.x_eq[free]).max())

    # Keep old name as alias for backward compatibility
    def check_near_rest(self, max_dist_fraction: float = 0.4) -> float:
        """Deprecated: use check_near_equilibrium() instead.

        Returns max displacement from x_eq if available, else from x0.
        """
        if self.x_eq is not None:
            return self.check_near_equilibrium()
        pos  = self.x.numpy()
        x0   = self.x0.numpy()
        free = (self.fixed.numpy() == 0)
        return float(np.abs(pos[free] - x0[free]).max())

    def reset_F_from_positions(self):
        """Recompute F from current particle positions using the initial lattice topology.

        Corrects accumulated deformation-gradient drift ("numerical plasticity") that
        builds up during large palpation cycles.  Safe to call when all free particles
        are within ~0.4 * lattice_step of x_eq (the gravity equilibrium positions).

        Call sample_equilibrium() first, then check_near_equilibrium() before calling.
        """
        if self.nbr_px is None or self._lattice_step == 0.0:
            return
        n = self.n_particles
        with wp.ScopedDevice(self.device):
            wp.launch(_recompute_F_total_lagrangian, dim=n,
                      inputs=[self.x, self.F,
                               self.nbr_px, self.nbr_mx,
                               self.nbr_py, self.nbr_my,
                               self.nbr_pz, self.nbr_mz,
                               float(self._lattice_step)])

    # ------------------------------------------------------------------
    # Cutting
    # ------------------------------------------------------------------

    def apply_cut(self, cut_sdf_np: np.ndarray,
                  retract_mm: float = 5.0, retract_steps: int = 100,
                  reset_reference: bool = True, replace: bool = False):
        """Register a cut surface and retract the edges apart.

        The SDF is sampled on the MPM grid (n_grid^3 flat array, float32).
        The sign encodes which side of the cut a point is on.  During P2G
        and G2P, transfers between particles and grid nodes on opposite
        sides are blocked, creating a velocity discontinuity that lets
        the tissue separate.  Fiber bonds crossing the cut are broken.

        After applying the cut, particles near the cut surface are
        displaced apart (like surgical retractors) over retract_steps
        simulation steps.  This opens the wound visibly.

        Based on the CDF approach from:
          Ou & Tavakoli, "CRESSim-MPM: A Material Point Method Library for
          Surgical Soft Body Simulation with Cutting and Suturing",
          arXiv:2502.18437v3, 2025.

        Args:
            cut_sdf_np:   (n_grid^3,) float32 SDF on the MPM grid.
            retract_mm:   How far to retract each side [mm]. 0 = no retraction.
            retract_steps: Number of sim steps over which to apply retraction.
            reset_reference: re-reference near-cut particles to their current
                          (deformed) pose by setting F->I and x0->x.  Correct
                          when cutting tissue that is at rest, but it FREEZES
                          the freed edge in its deformed shape -- so pass
                          False when the tissue is under load and you want it
                          to spring back after severance.
            replace:      replace the cut list with this SDF instead of
                          appending.  Only cut_sdfs[-1] is used by the
                          pipeline, so a caller compositing progressive cuts
                          into one field should pass True to avoid growing an
                          unused list.
        """
        with wp.ScopedDevice(self.device):
            cut_sdf = wp.array(cut_sdf_np.astype(np.float32), dtype=float)
        if replace:
            self.cut_sdfs = [cut_sdf]
        else:
            self.cut_sdfs.append(cut_sdf)

        # Break fiber bonds that cross the cut
        if self.n_bonds > 0 and self.fiber_broken is not None:
            with wp.ScopedDevice(self.device):
                wp.launch(_break_bonds_across_cut, dim=self.n_bonds,
                          inputs=[self.x, self.fiber_i, self.fiber_j,
                                  self.fiber_broken, cut_sdf,
                                  self.n_grid, float(self.inv_dx)])
            n_broken = int(self.fiber_broken.numpy().sum())
            print(f"MPMSimulator: cut applied — {n_broken} bonds broken "
                  f"({len(self.cut_sdfs)} active cuts)")

        # Reset F to identity and update x0 for particles near the cut.
        # Without this, the Neo-Hookean stress from F tries to restore
        # the pre-cut shape, pulling cut edges back together.  Resetting
        # F and x0 to the current state makes the post-cut configuration
        # the new stress-free reference.
        sdf_np = cut_sdf_np.astype(np.float32)
        pos    = self.x.numpy()
        F_np   = self.F.numpy()
        x0_np  = self.x0.numpy()
        fixed  = self.fixed.numpy()
        inv_dx = float(self.inv_dx)
        ng     = self.n_grid

        gi = np.clip(np.round(pos[:, 0] * inv_dx).astype(int), 0, ng - 1)
        gj = np.clip(np.round(pos[:, 1] * inv_dx).astype(int), 0, ng - 1)
        gk = np.clip(np.round(pos[:, 2] * inv_dx).astype(int), 0, ng - 1)
        p_sdf = sdf_np[gi * ng * ng + gj * ng + gk]

        near_cut = (np.abs(p_sdf) > 0) & (np.abs(p_sdf) < 3.0 * float(self.dx))
        near_cut = near_cut & (fixed == 0)
        n_reset = int(near_cut.sum())
        if n_reset > 0 and reset_reference:
            F_np[near_cut] = np.eye(3, dtype=np.float32)
            x0_np[near_cut] = pos[near_cut]
            with wp.ScopedDevice(self.device):
                self.F  = wp.array(F_np, dtype=wp.mat33)
                self.x0 = wp.array(x0_np, dtype=wp.vec3)
            print(f"MPMSimulator: reset F→I and x0→x for {n_reset} "
                  f"particles near cut (within {3*self.dx*1000:.0f} mm)")

        # --- Retraction: displace near-cut particles apart ----------------
        # Retraction is concentrated in the center of the incision
        # (cosine-bell weight) so the tissue mechanics naturally produce
        # a curved opening — wide in the middle, closed at the ends.
        if retract_mm > 0 and retract_steps > 0:
            retract_band = 2.0 * float(self.dx)
            retract_pos = (p_sdf > 0) & (p_sdf < retract_band) & (fixed == 0)
            retract_neg = (p_sdf < 0) & (p_sdf > -retract_band) & (fixed == 0)
            retract_any = retract_pos | retract_neg

            if retract_any.sum() < 2:
                return

            # Retraction direction: pos centroid → neg centroid
            pos_centroid = pos[retract_pos].mean(axis=0)
            neg_centroid = pos[retract_neg].mean(axis=0)
            cut_dir = pos_centroid - neg_centroid
            cut_dir_norm = float(np.linalg.norm(cut_dir))
            if cut_dir_norm > 1e-8:
                cut_dir = cut_dir / cut_dir_norm
            else:
                cut_dir = np.array([0.0, 0.0, 1.0])

            # Along-curve direction: 1st principal component of
            # retractor particle positions (longest spread direction)
            all_retract_pos = pos[retract_any]
            centered = all_retract_pos - all_retract_pos.mean(axis=0)
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            curve_axis = Vt[0]  # first principal component

            # Per-particle weight: cosine bell along the curve axis
            # t=0 at one end, t=1 at the other, max retraction at t=0.5
            proj = pos @ curve_axis  # all particles
            proj_min = float(proj[retract_any].min())
            proj_max = float(proj[retract_any].max())
            proj_range = max(proj_max - proj_min, 1e-8)
            t = (proj - proj_min) / proj_range  # 0..1

            # cos²(π(t-0.5)) = 1 at center, 0 at endpoints
            weight = np.cos(np.pi * (t - 0.5)) ** 2
            weight = weight.astype(np.float32)

            step_disp_max = (retract_mm / 1000.0) / retract_steps
            gravity_zero = np.array([0.0, 0.0, 0.0])

            n_pos_r = int(retract_pos.sum())
            n_neg_r = int(retract_neg.sum())
            for _ in range(retract_steps):
                x_np = self.x.numpy()
                x_np[retract_pos] += (step_disp_max * weight[retract_pos])[:, None] * cut_dir
                x_np[retract_neg] -= (step_disp_max * weight[retract_neg])[:, None] * cut_dir
                with wp.ScopedDevice(self.device):
                    self.x = wp.array(x_np, dtype=wp.vec3)
                self.step(gravity_zero)

            print(f"MPMSimulator: retracted {n_pos_r}+{n_neg_r} particles "
                  f"by ±{retract_mm:.1f} mm (center-weighted) "
                  f"over {retract_steps} steps")

    def set_prestress(self, stretch: float = 1.02):
        """Initialize F with isotropic stretch to create tissue pre-tension.

        WARNING: modifying F directly creates an F/position mismatch that
        causes updated-Lagrangian mode to diverge.  Prefer
        set_prestress_fibers() which achieves tension through the fiber
        network without corrupting F.

        Args:
            stretch:  isotropic stretch ratio (1.0 = no pre-stress).
        """
        F_np = self.F.numpy()
        I_stretched = np.eye(3, dtype=np.float32) * stretch
        F_np[:] = I_stretched
        with wp.ScopedDevice(self.device):
            self.F = wp.array(F_np, dtype=wp.mat33)

    def set_prestress_fibers(self, stretch: float = 1.05):
        """Create tissue pre-tension by shortening fiber rest lengths.

        Scales all fiber rest lengths by 1/stretch, making the current
        particle spacing longer than the rest length.  This creates
        isotropic tension in the bond network without modifying F or
        creating any F/position inconsistency.

        When a cut is applied, cross-cut bonds are broken and the
        tension in surviving bonds pulls the cut edges apart, producing
        realistic wound gaping.

        Typical values: 1.02–1.10 (2–10% pre-stretch).
        At 5% with k_elastin=0.05, k_collagen=0.25, this produces
        moderate tension comparable to skin turgor.

        Args:
            stretch:  isotropic stretch ratio (1.0 = no pre-stress).
        """
        if self.n_bonds == 0 or self.fiber_l0 is None:
            return
        l0_np = self.fiber_l0.numpy()
        l0_new = (l0_np / stretch).astype(np.float32)
        with wp.ScopedDevice(self.device):
            self.fiber_l0 = wp.array(l0_new, dtype=float)
        print(f"MPMSimulator: fiber pre-stress — rest lengths scaled by "
              f"1/{stretch:.3f}, tension strain {stretch - 1:.1%}")

    def get_positions(self) -> np.ndarray:
        """Return current particle positions as (n_particles, 3) float32 [m]."""
        return self.x.numpy()

    def get_deformation_norms(self) -> np.ndarray:
        """Return ||F - I|| per particle as a scalar deformation measure."""
        F_np = self.F.numpy()
        I3   = np.eye(3, dtype=np.float32)
        return np.linalg.norm(F_np - I3, axis=(1, 2))

    def step_with_probe(
        self,
        gravity,
        probe_center: np.ndarray,
        probe_pressure_pa: float,
        probe_normal: np.ndarray,
        probe_radius: float,
    ):
        """Advance one timestep and apply a spatially-smoothed probe pressure.

        The probe exerts a contact-pressure body force on all free particles
        within probe_radius of probe_center.  The pressure profile is a cosine
        bell: P(r) = probe_pressure_pa * 0.5*(1 + cos(π*r/R)).

        Force model: finger contact pressure P [Pa] over circular area πR² is
        distributed as body acceleration to particles in sphere volume (4/3)πR³:
            a = P × πR² / (ρ × 4/3 × πR³) = P × 3 / (4ρR)  [m/s²]
        This is dimensionally correct (Pa/m / (kg/m³) = m/s²).

        Stability limit: P_max = ρ × c_s × 4R / (3 × dt)
        At E=10 kPa, R=25 mm, dt=2e-4 s: P_max ≈ 540 kPa.

        Args:
            gravity:           gravity vector [m/s²]
            probe_center:      3-vector, probe contact centre [m]
            probe_pressure_pa: peak contact pressure [Pa] — positive pushes in
                               probe_normal direction.  ~10 kPa gives ~20 mm
                               deflection for E=10 kPa, 80 mm block.
                               Clinical palpation: 5–25 kPa (1–5 N over 1–3 cm²).
            probe_normal:      unit vector of force direction (e.g. [0,-1,0] for
                               downward press on top face)
            probe_radius:      spatial falloff radius [m]
        """
        rho   = float(self.material.rho)
        # Dimensionally correct: contact pressure P over area πR² distributed
        # into sphere volume (4/3)πR³ → body accel = P × 3/(4ρR) [m/s²]
        area_vol_ratio = 3.0 / (4.0 * float(probe_radius))  # [1/m]
        accel = np.asarray(probe_normal, dtype=np.float64) * (probe_pressure_pa * area_vol_ratio / rho)
        c  = wp.vec3(float(probe_center[0]), float(probe_center[1]), float(probe_center[2]))
        a  = wp.vec3(float(accel[0]),        float(accel[1]),        float(accel[2]))
        # _step_core runs the hooks inside its own ScopedDevice.
        def probe_kick():
            wp.launch(_apply_probe_force, dim=self.n_particles,
                      inputs=[self.x, self.v, self.fixed,
                              c, a, float(probe_radius), float(self.dt)])

        # post_forces preserves the historical ordering (the probe force used
        # to be applied after a complete step).  A sustained load would be
        # better placed in pre_p2g so the stress update sees it.
        self._step_core(gravity, post_forces=[probe_kick])

    def step_with_contact(
        self,
        gravity,
        sphere_center: np.ndarray,
        sphere_radius: float,
        sphere_vel: np.ndarray = None,
        stiction: float = 0.0,
        hemi_dir: np.ndarray = None,
    ):
        """Advance one timestep with rigid sphere contact (dual kick).

        The contact kick is applied twice: BEFORE P2G, so the modified
        velocities scatter to the grid and G2P computes C and F from a
        velocity field that includes the contact (this is what prevents
        F-drift); and again after G2P, which supplies the actual geometric
        push.

        Because this now runs through _step_core, an active cut is respected
        here too.  It previously was not: this method carried its own copy of
        the pipeline that always launched the plain _p2g/_g2p.

        Args:
            gravity:        gravity vector [m/s^2]
            sphere_center:  3-vector, centre of the rigid probe sphere [m]
            sphere_radius:  probe sphere radius [m] (finger tip ~0.008-0.012 m)
            sphere_vel:     optional 3-vector, sphere translational velocity
                            [m/s].  Used by the stiction term; ignored if
                            stiction == 0.  Defaults to zero.
            stiction:       tangential velocity-match fraction per kick (0..1).
                            0 = frictionless (default), 1 = full no-slip lock
                            in one substep.  ~0.3 gives a noticeable grip.
            hemi_dir:       which half of the sphere makes contact.  Defaults
                            to (0, -1, 0), the historical lower-hemisphere
                            behaviour of a finger pressing from above.  Pass
                            a zero vector for a full sphere -- correct for a
                            laparoscopic tip, which contacts from any side
                            once embedded.  NOTE the default is an axis-1 gate
                            regardless of the scenario's gravity direction, so
                            a tool pressing along another axis must pass this.
        """
        if sphere_vel is None:
            sphere_vel = np.zeros(3)
        if hemi_dir is None:
            hemi_dir = np.array([0.0, -1.0, 0.0])

        self._ensure_material_arrays()
        dt = float(self.dt)
        n = self.n_particles

        # Stiffest particle: with heterogeneous tissue the probe must respond
        # at the fastest local wave speed or it under-pushes stiff regions.
        response_rate = self.wave_speed / float(sphere_radius)
        sc = wp.vec3(*[float(c) for c in sphere_center])
        sv = wp.vec3(*[float(c) for c in sphere_vel])
        hd = wp.vec3(*[float(c) for c in hemi_dir])
        st = float(stiction)

        def kick():
            wp.launch(_apply_hemisphere_contact, dim=n,
                      inputs=[self.x, self.v, self.fixed, self.m_p,
                              sc, float(sphere_radius),
                              float(response_rate), dt, sv, st,
                              self._contact_impulse_buf, hd])

        with wp.ScopedDevice(self.device):
            self._contact_impulse_buf.zero_()

        self._step_core(gravity, pre_p2g=[kick], post_g2p=[kick])

        self._last_contact_impulse_np = self._contact_impulse_buf.numpy()[0].copy()

    @property
    def bone_sdf(self):
        """Deprecated alias for obstacle_sdf (kept for mpm_ct_head)."""
        return self.obstacle_sdf

    @bone_sdf.setter
    def bone_sdf(self, value):
        self.obstacle_sdf = value

    @property
    def bone_sdf_grad(self):
        """Deprecated alias for obstacle_sdf_grad."""
        return self.obstacle_sdf_grad

    @bone_sdf_grad.setter
    def bone_sdf_grad(self, value):
        self.obstacle_sdf_grad = value

    @property
    def last_contact_impulse(self):
        """Newton-3rd-law impulse on the rigid contact sphere from tissue [kg·m/s].

        Total impulse summed across both pre-P2G and post-G2P kicks during the
        most recent step_with_contact() call.  Zero before any contact step.
        """
        return self._last_contact_impulse_np

    @property
    def last_contact_force(self):
        """Effective tissue→sphere reaction force over the last contact step [N].

        impulse / dt — average force; appropriate for driving a force-coupled
        rigid body whose state is updated once per step.
        """
        return self._last_contact_impulse_np / float(self.dt)
