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
        col0 = wp.vec3(1.0, 0.0, 0.0)   # fallback: identity

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
        col1 = wp.vec3(0.0, 1.0, 0.0)

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
        col2 = wp.vec3(0.0, 0.0, 1.0)

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
    n_grid: int,
    inv_dx: float,
    dt:     float,
    mu:     float,
    lam:    float,
):
    """Particle → Grid transfer (one thread per particle)."""
    p = wp.tid()
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
    n_grid:  int,
    inv_dx:  float,
    dt:      float,
    mu:      float,
    lam:     float,
):
    """P2G with cut-aware transfer: skip scatter across a cut surface.

    If a particle and a grid node are on opposite sides of the cut SDF
    (different signs), the transfer is blocked.  This creates a velocity
    discontinuity at the cut, allowing tissue on opposite sides to separate.
    """
    p = wp.tid()
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
):
    """Grid momentum update: divide by mass, add gravity, damp, apply BCs."""
    flat = wp.tid()
    m = grid_m[flat]
    if m > 0.0:
        v = (grid_v[flat] / m + dt * gravity) * damping

        i = flat // (n_grid * n_grid)
        j = (flat // n_grid) % n_grid
        k = flat % n_grid

        if j == 0 and v[1] < 0.0:
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
    grid_v:   wp.array(dtype=wp.vec3),
    sdf:      wp.array(dtype=float),
    sdf_grad: wp.array(dtype=wp.vec3),
    dx:       float,
):
    """SDF-based grid boundary condition for bone.

    - SDF < 0 (inside bone): zero velocity — hard wall.
    - 0 ≤ SDF < dx (near surface): remove the inward velocity component
      so tissue can slide along bone but not penetrate.
    - SDF ≥ dx: free.

    Applied after grid_update and before G2P to prevent the grid velocity
    field from ever pushing tissue into bone.
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
                grid_v[flat] = v - v_n * normal


@wp.kernel
def _apply_bone_sdf_contact(
    x:        wp.array(dtype=wp.vec3),
    v:        wp.array(dtype=wp.vec3),
    fixed:    wp.array(dtype=int),
    sdf:      wp.array(dtype=float),     # SDF value per grid node (neg = inside bone)
    sdf_grad: wp.array(dtype=wp.vec3),   # precomputed SDF gradient per grid node
    n_grid:   int,
    inv_dx:   float,
):
    """Project tissue particles out of bone using a signed distance field.

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
    if dist < sphere_radius and dist > 1.0e-12 and delta[1] < 0.0:
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
    sphere_center: wp.vec3,
    sphere_radius: float,
    response_rate: float,
    dt:     float,
):
    """Lower-hemisphere rigid contact — applied BEFORE P2G.

    Any free particle in the lower hemisphere of the sphere receives an outward
    velocity kick proportional to penetration depth.  Applied before P2G so the
    modified velocities are scattered to the grid and captured by G2P in C and F.
    """
    p = wp.tid()
    if fixed[p] != 0:
        return
    delta = x[p] - sphere_center
    dist = wp.length(delta)
    if dist < sphere_radius and dist > 1.0e-12 and delta[1] < 0.0:
        penetration = sphere_radius - dist
        normal = delta / dist
        dv_mag = response_rate * penetration
        v_n = wp.dot(v[p], normal)
        if v_n < 0.0:
            v[p] = v[p] + (dv_mag - v_n) * normal
        else:
            v[p] = v[p] + dv_mag * normal


@wp.kernel
def _project_particles_from_sphere(
    x:      wp.array(dtype=wp.vec3),
    v:      wp.array(dtype=wp.vec3),
    fixed:  wp.array(dtype=int),
    sphere_center: wp.vec3,
    sphere_radius: float,
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
    if dist < sphere_radius and dist > 1.0e-12 and delta[1] < 0.0:
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
    k_e:    float,
    k_c:    float,
    crimp:  float,
    dt:     float,
):
    """Cosserat-style cable bond forces (one thread per bond).

    Elastin: bidirectional linear spring.
    Collagen: tension-only, activates above crimp strain threshold.
    Broken bonds (fiber_broken != 0) are skipped.
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

    if ftype == 0:
        force_mag = k_e * (l - l0)
    else:
        eff = strain - crimp
        if eff > 0.0:
            force_mag = k_c * eff * l0

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

        # Optional bone SDF contact (particle-level, after G2P).
        # Set bone_sdf (float per grid node, neg=inside bone) and
        # bone_sdf_grad (vec3 per grid node, outward normal) to enable.
        self.bone_sdf      = None
        self.bone_sdf_grad = None

        # Fiber bond arrays
        self.fiber_i  = None;  self.fiber_j  = None
        self.fiber_l0 = None;  self.fiber_t  = None
        self.fiber_broken = None  # wp.array(dtype=int), 0=intact, 1=broken
        self.n_bonds  = 0

        # Cut SDFs: list of wp.array(dtype=float) of length n_grid^3.
        # Sign encodes which side of the cut a point is on.  When active,
        # P2G/G2P block transfers across each cut surface, and bonds that
        # cross the cut are broken.
        self.cut_sdfs = []

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

    def step(self, gravity=None):
        """Advance one explicit timestep."""
        if gravity is None:
            gravity = np.array([0.0, -9.8, 0.0])
        ng  = self.n_grid
        n   = self.n_particles
        dt  = float(self.dt)
        mu  = float(self.material.mu)
        lam = float(self.material.lam)
        g   = wp.vec3(float(gravity[0]), float(gravity[1]), float(gravity[2]))

        with wp.ScopedDevice(self.device):
            # In total-Lagrangian mode, recompute F from current positions
            # BEFORE P2G so the stress uses a drift-free deformation gradient.
            if self.total_lagrangian and self.nbr_px is not None:
                wp.launch(_recompute_F_total_lagrangian, dim=n,
                          inputs=[self.x, self.F,
                                  self.nbr_px, self.nbr_mx,
                                  self.nbr_py, self.nbr_my,
                                  self.nbr_pz, self.nbr_mz,
                                  float(self._lattice_step)])

            wp.launch(_zero_grid, dim=ng**3,
                      inputs=[self.grid_v, self.grid_m])

            # P2G: use cut-aware version if any cut is active.
            # Uses the most recent cut SDF (multiple cuts are accumulated
            # by merging into a single composite SDF via apply_cut).
            if self.cut_sdfs:
                wp.launch(_p2g_cut, dim=n,
                          inputs=[self.x, self.v, self.F, self.C,
                                   self.m_p, self.vol_p,
                                   self.grid_v, self.grid_m,
                                   self.cut_sdfs[-1],
                                   ng, float(self.inv_dx), dt, mu, lam])
            else:
                wp.launch(_p2g, dim=n,
                          inputs=[self.x, self.v, self.F, self.C,
                                   self.m_p, self.vol_p,
                                   self.grid_v, self.grid_m,
                                   ng, float(self.inv_dx), dt, mu, lam])

            wp.launch(_grid_update, dim=ng**3,
                      inputs=[self.grid_v, self.grid_m, ng, dt, g,
                               float(self.velocity_damping)])

            if self.grid_bc_fixed is not None:
                wp.launch(_apply_grid_fixed_bc, dim=ng**3,
                          inputs=[self.grid_v, self.grid_bc_fixed])

            if self.bone_sdf is not None:
                wp.launch(_apply_grid_sdf_bc, dim=ng**3,
                          inputs=[self.grid_v, self.bone_sdf,
                                  self.bone_sdf_grad, float(self.dx)])

            # G2P: cut-aware version blocks gather across cut surfaces
            if self.cut_sdfs and not self.total_lagrangian:
                wp.launch(_g2p_cut, dim=n,
                          inputs=[self.x, self.v, self.F, self.C, self.fixed,
                                  self.grid_v, self.cut_sdfs[-1],
                                  ng, float(self.inv_dx), dt])
            elif self.total_lagrangian:
                wp.launch(_g2p_no_F_update, dim=n,
                          inputs=[self.x, self.v, self.C, self.fixed,
                                  self.grid_v, ng, float(self.inv_dx), dt])
            else:
                wp.launch(_g2p, dim=n,
                          inputs=[self.x, self.v, self.F, self.C, self.fixed,
                                  self.grid_v, ng, float(self.inv_dx), dt])

            if self.bone_sdf is not None:
                wp.launch(_apply_bone_sdf_contact, dim=n,
                          inputs=[self.x, self.v, self.fixed,
                                  self.bone_sdf, self.bone_sdf_grad,
                                  ng, float(self.inv_dx)])

            if self.n_bonds > 0:
                wp.launch(_apply_fiber_forces, dim=self.n_bonds,
                          inputs=[
                              self.x, self.v, self.fixed, self.m_p,
                              self.fiber_i, self.fiber_j, self.fiber_l0, self.fiber_t,
                              self.fiber_broken,
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

    def apply_cut(self, cut_sdf_np: np.ndarray):
        """Register a cut surface defined by a signed distance field.

        The SDF is sampled on the MPM grid (n_grid^3 flat array, float32).
        The sign encodes which side of the cut a point is on.  During P2G
        and G2P, transfers between particles and grid nodes on opposite
        sides are blocked, creating a velocity discontinuity that lets
        the tissue separate.  Fiber bonds crossing the cut are broken.

        Based on the CDF approach from:
          Ou & Tavakoli, "CRESSim-MPM: A Material Point Method Library for
          Surgical Soft Body Simulation with Cutting and Suturing",
          arXiv:2502.18437v3, 2025.

        Args:
            cut_sdf_np:  (n_grid^3,) float32 array — signed distance on the
                         MPM grid.  Positive on one side, negative on the other.
        """
        with wp.ScopedDevice(self.device):
            cut_sdf = wp.array(cut_sdf_np.astype(np.float32), dtype=float)
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

    def set_prestress(self, stretch: float = 1.02):
        """Initialize F with isotropic stretch to create tissue pre-tension.

        When a cut is made, the pre-stress drives the tissue edges apart,
        producing a realistic gaping wound.  Typical values: 1.01–1.05
        (1–5% isotropic pre-stretch).

        Args:
            stretch:  isotropic stretch ratio (1.0 = no pre-stress).
        """
        F_np = self.F.numpy()
        I_stretched = np.eye(3, dtype=np.float32) * stretch
        F_np[:] = I_stretched
        with wp.ScopedDevice(self.device):
            self.F = wp.array(F_np, dtype=wp.mat33)

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
        self.step(gravity)
        rho   = float(self.material.rho)
        # Dimensionally correct: contact pressure P over area πR² distributed
        # into sphere volume (4/3)πR³ → body accel = P × 3/(4ρR) [m/s²]
        area_vol_ratio = 3.0 / (4.0 * float(probe_radius))  # [1/m]
        accel = np.asarray(probe_normal, dtype=np.float64) * (probe_pressure_pa * area_vol_ratio / rho)
        c  = wp.vec3(float(probe_center[0]), float(probe_center[1]), float(probe_center[2]))
        a  = wp.vec3(float(accel[0]),        float(accel[1]),        float(accel[2]))
        with wp.ScopedDevice(self.device):
            wp.launch(_apply_probe_force, dim=self.n_particles,
                      inputs=[self.x, self.v, self.fixed,
                               c, a, float(probe_radius), float(self.dt)])

    def step_with_contact(
        self,
        gravity,
        sphere_center: np.ndarray,
        sphere_radius: float,
    ):
        """Advance one timestep with rigid hemisphere contact.

        The contact velocity kick is applied to particles BEFORE P2G so that
        the modified velocities are scattered to the grid.  G2P then computes
        C and F from grid velocities that include the contact effect — this
        prevents the F-drift that occurs when contact bypasses the MPM pipeline.

        Only the lower hemisphere is enforced (finger pressing from above).

        Args:
            gravity:        gravity vector [m/s²]
            sphere_center:  3-vector, centre of rigid probe sphere [m]
            sphere_radius:  probe sphere radius [m] (finger tip ≈ 0.008–0.012 m)
        """
        if gravity is None:
            gravity = np.array([0.0, -9.8, 0.0])
        ng  = self.n_grid
        n   = self.n_particles
        dt  = float(self.dt)
        mu  = float(self.material.mu)
        lam = float(self.material.lam)
        g   = wp.vec3(float(gravity[0]), float(gravity[1]), float(gravity[2]))

        c_s = (float(self.material.E) / float(self.material.rho)) ** 0.5
        response_rate = c_s / float(sphere_radius)
        sc = wp.vec3(float(sphere_center[0]), float(sphere_center[1]),
                      float(sphere_center[2]))

        with wp.ScopedDevice(self.device):
            if self.total_lagrangian and self.nbr_px is not None:
                wp.launch(_recompute_F_total_lagrangian, dim=n,
                          inputs=[self.x, self.F,
                                  self.nbr_px, self.nbr_mx,
                                  self.nbr_py, self.nbr_my,
                                  self.nbr_pz, self.nbr_mz,
                                  float(self._lattice_step)])

            # Contact velocity kick BEFORE P2G — gets scattered to grid
            # so that G2P correctly captures the contact in C and F.
            wp.launch(_apply_hemisphere_contact, dim=n,
                      inputs=[self.x, self.v, self.fixed,
                              sc, float(sphere_radius),
                              float(response_rate), dt])

            wp.launch(_zero_grid, dim=ng**3,
                      inputs=[self.grid_v, self.grid_m])

            wp.launch(_p2g, dim=n,
                      inputs=[self.x, self.v, self.F, self.C,
                               self.m_p, self.vol_p,
                               self.grid_v, self.grid_m,
                               ng, float(self.inv_dx), dt, mu, lam])

            wp.launch(_grid_update, dim=ng**3,
                      inputs=[self.grid_v, self.grid_m, ng, dt, g,
                               float(self.velocity_damping)])

            if self.grid_bc_fixed is not None:
                wp.launch(_apply_grid_fixed_bc, dim=ng**3,
                          inputs=[self.grid_v, self.grid_bc_fixed])

            if self.bone_sdf is not None:
                wp.launch(_apply_grid_sdf_bc, dim=ng**3,
                          inputs=[self.grid_v, self.bone_sdf,
                                  self.bone_sdf_grad, float(self.dx)])

            if self.total_lagrangian:
                wp.launch(_g2p_no_F_update, dim=n,
                          inputs=[self.x, self.v, self.C, self.fixed,
                                  self.grid_v, ng, float(self.inv_dx), dt])
            else:
                wp.launch(_g2p, dim=n,
                          inputs=[self.x, self.v, self.F, self.C, self.fixed,
                                  self.grid_v, ng, float(self.inv_dx), dt])

            # Second contact kick after G2P — provides actual deformation.
            wp.launch(_apply_hemisphere_contact, dim=n,
                      inputs=[self.x, self.v, self.fixed,
                              sc, float(sphere_radius),
                              float(response_rate), dt])

            if self.bone_sdf is not None:
                wp.launch(_apply_bone_sdf_contact, dim=n,
                          inputs=[self.x, self.v, self.fixed,
                                  self.bone_sdf, self.bone_sdf_grad,
                                  ng, float(self.inv_dx)])

            if self.n_bonds > 0:
                wp.launch(_apply_fiber_forces, dim=self.n_bonds,
                          inputs=[
                              self.x, self.v, self.fixed, self.m_p,
                              self.fiber_i, self.fiber_j, self.fiber_l0, self.fiber_t,
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
