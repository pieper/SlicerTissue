"""Unified step-pipeline tests for the MPM solver.

Pins the behaviour of ``_step_core`` and the two real bugs it fixed:

* ``step_with_contact`` carried its own copy of the pipeline that always
  launched the plain ``_p2g``/``_g2p``, so an **active cut silently stopped
  blocking momentum** during palpation.
* ``total_lagrangian=True`` selected ``_g2p_no_F_update``, which is not
  cut-aware, so TL + cutting ran **scatter-blocked but gather-unblocked**.

Imports mpm.py directly by path — see test_mpm_heterogeneous for why.
"""

import importlib.util
import os

import numpy as np
import pytest

wp = pytest.importorskip("warp", reason="warp not installed")

_MPM_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "src", "newton_tissue", "mpm.py")
_spec = importlib.util.spec_from_file_location("_mpm_pipeline_under_test", _MPM_PATH)
mpm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mpm)

MPMMaterial = mpm.MPMMaterial
MPMSimulator = mpm.MPMSimulator

L = 0.04
DEVICE = "cpu"


def _sim(total_lagrangian=False, **matkw):
    mat = MPMMaterial(E=3_000.0, nu=0.3, rho=1060.0, **matkw)
    sim = MPMSimulator([0, 0, 0], [L, L, L], n_grid=12, dt=1e-4, material=mat,
                       device=DEVICE, total_lagrangian=total_lagrangian)
    sim.initialize_block_particles(ppc=2)
    return sim


def _plane_cut_sdf(sim, axis=0, offset_cells=0.5):
    """Signed distance to a plane, offset off the grid nodes.

    The offset matters: a plane landing exactly on a node makes that node's SDF
    exactly 0, and the ``p_sdf * g_sdf < 0`` side test never fires there.
    """
    ng = sim.n_grid
    coord = np.arange(ng) * sim.dx
    shape = [1, 1, 1]
    shape[axis] = ng
    grid = coord.reshape(shape) * np.ones((ng, ng, ng))
    return (grid - (L / 2 + offset_cells * sim.dx)).astype(np.float32).ravel()


# ---------------------------------------------------------------------------
# Bug 1: cuts must be respected by every entry point
# ---------------------------------------------------------------------------

def test_cut_is_respected_under_contact():
    """An active cut must still block momentum during step_with_contact.

    Drive the block with a probe and compare a cut run against an uncut one.
    Before the _step_core refactor these were identical, because
    step_with_contact never launched the cut-aware kernels.
    """
    def run(with_cut):
        sim = _sim()
        if with_cut:
            sim.apply_cut(_plane_cut_sdf(sim), retract_mm=0.0, retract_steps=0)
        g = np.array([0.0, -9.8, 0.0])
        c = np.array([L / 2, L + 0.004, L / 2])
        for _ in range(40):
            c[1] -= 2.0e-4
            sim.step_with_contact(g, c, 0.008,
                                  sphere_vel=np.array([0.0, -2.0, 0.0]),
                                  stiction=0.3)
        return sim.x.numpy().copy()

    uncut = run(False)
    cut = run(True)
    assert np.isfinite(cut).all(), "cut + contact produced non-finite positions"
    assert not np.allclose(uncut, cut), (
        "an active cut made no difference under step_with_contact -- the "
        "cut-aware P2G/G2P are not being used on this path")


def test_cut_separates_under_contact():
    """The two sides of the cut must move differently once contact drives them."""
    sim = _sim()
    sdf = _plane_cut_sdf(sim)
    sim.apply_cut(sdf, retract_mm=0.0, retract_steps=0)

    pos0 = sim.x.numpy().copy()
    ng, inv_dx = sim.n_grid, float(sim.inv_dx)
    gi = np.clip(np.round(pos0[:, 0] * inv_dx).astype(int), 0, ng - 1)
    gj = np.clip(np.round(pos0[:, 1] * inv_dx).astype(int), 0, ng - 1)
    gk = np.clip(np.round(pos0[:, 2] * inv_dx).astype(int), 0, ng - 1)
    side = sdf[gi * ng * ng + gj * ng + gk]
    free = sim.fixed.numpy() == 0
    pos_side = free & (side > 0)
    neg_side = free & (side < 0)
    assert pos_side.sum() > 50 and neg_side.sum() > 50

    g = np.array([0.0, -9.8, 0.0])
    # press on the negative side only
    c = np.array([L / 4, L + 0.004, L / 2])
    for _ in range(40):
        c[1] -= 2.0e-4
        sim.step_with_contact(g, c, 0.008,
                              sphere_vel=np.array([0.0, -2.0, 0.0]))
    pos1 = sim.x.numpy()
    dy_neg = float((pos1[neg_side, 1] - pos0[neg_side, 1]).mean())
    dy_pos = float((pos1[pos_side, 1] - pos0[pos_side, 1]).mean())
    assert abs(dy_neg) > abs(dy_pos), (
        "pressed side moved %.4g mm, far side %.4g mm -- the cut is not "
        "decoupling them" % (1e3 * dy_neg, 1e3 * dy_pos))


def test_total_lagrangian_cut_uses_cut_aware_g2p():
    """TL + cut must stay finite and must differ from TL without a cut.

    This exercises _g2p_cut_no_F_update, which did not exist before: TL mode
    fell through to _g2p_no_F_update, blocking scatter but not gather.
    """
    def run(with_cut):
        sim = _sim(total_lagrangian=True, k_curve=25.0)
        if with_cut:
            sim.apply_cut(_plane_cut_sdf(sim), retract_mm=0.0, retract_steps=0)
        g = np.array([0.0, -9.8, 0.0])
        for _ in range(60):
            sim.step(g)
        return sim

    plain = run(False)
    cut = run(True)
    assert np.isfinite(cut.x.numpy()).all(), "TL + cut produced non-finite positions"
    assert np.isfinite(cut.F.numpy()).all(), "TL + cut produced non-finite F"
    assert not np.allclose(plain.x.numpy(), cut.x.numpy()), (
        "TL run was unaffected by the cut -- cut-aware G2P is not being used")


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------

def _halfspace(sim, height):
    """SDF and gradient for the half-space y > height."""
    ng = sim.n_grid
    gy = (np.arange(ng) * sim.dx)[None, :, None] * np.ones((ng, ng, ng))
    with wp.ScopedDevice(DEVICE):
        sdf = wp.array((gy - height).astype(np.float32).ravel(), dtype=float)
        grad = wp.array(np.tile(np.array([0, 1, 0], np.float32), (ng ** 3, 1)),
                        dtype=wp.vec3)
    return sdf, grad


def test_floor_j0_can_be_disabled():
    """floor_j0=False must let particles fall past grid row j == 0."""
    def run(floor):
        sim = _sim()
        sim.floor_j0 = floor
        with wp.ScopedDevice(DEVICE):
            sim.fixed = wp.zeros(sim.n_particles, dtype=int)   # nothing clamped
        g = np.array([0.0, -9.8, 0.0])
        for _ in range(200):
            sim.step(g)
        return sim.x.numpy()[:, 1].min()

    assert run(False) < run(True), "disabling floor_j0 did not remove the floor"


def test_obstacle_friction_kernel():
    """Coulomb friction on the sliding layer: bounded, monotonic, never reversing.

    Exercised at the kernel, not through a full sim.  In a sim only the single
    grid-node row that falls in the 0 <= sdf < dx band is affected, so a mean
    over all particles is swamped by the free bulk and measures nothing.

    Setup: one node in the sliding band with velocity (vt, -vn, 0) and an
    outward normal +y.  The normal component is removed in all cases; friction
    then scales the tangential part by max(0, 1 - mu*|v_n|/|v_t|).
    """
    sim = _sim()
    ng = sim.n_grid
    dx = float(sim.dx)

    v_t, v_n = 0.10, 0.02   # |v_t| = 5 * |v_n|, so mu = 0.2 gives exactly zero

    def run(mu):
        with wp.ScopedDevice(DEVICE):
            grid_v = wp.array(
                np.tile(np.array([v_t, -v_n, 0.0], np.float32), (ng ** 3, 1)),
                dtype=wp.vec3)
            # every node sits in the sliding band 0 <= d < dx
            sdf = wp.array(np.full(ng ** 3, 0.5 * dx, np.float32), dtype=float)
            grad = wp.array(np.tile(np.array([0, 1, 0], np.float32), (ng ** 3, 1)),
                            dtype=wp.vec3)
            wp.launch(mpm._apply_grid_sdf_bc, dim=ng ** 3,
                      inputs=[grid_v, sdf, grad, dx, float(mu)])
        return grid_v.numpy()[0]

    free = run(0.0)
    assert free[1] == pytest.approx(0.0, abs=1e-7), "normal component not removed"
    assert free[0] == pytest.approx(v_t, rel=1e-5), "frictionless sliding was altered"

    mild = run(0.1)
    expected = v_t * (1.0 - 0.1 * v_n / v_t)
    assert mild[0] == pytest.approx(expected, rel=1e-5)
    assert mild[0] < free[0]

    # mu * |v_n| == |v_t| exactly: sliding stops
    stopped = run(v_t / v_n)
    assert stopped[0] == pytest.approx(0.0, abs=1e-7)

    # beyond that, friction must clamp to zero rather than reverse the motion
    over = run(10.0 * v_t / v_n)
    assert over[0] == pytest.approx(0.0, abs=1e-7), \
        "friction reversed the tangential velocity; it must only oppose it"


def test_bone_sdf_alias():
    """bone_sdf remains a working alias for obstacle_sdf (mpm_ct_head uses it)."""
    sim = _sim()
    sdf, grad = _halfspace(sim, 0.004)
    sim.bone_sdf = sdf
    sim.bone_sdf_grad = grad
    assert sim.obstacle_sdf is sdf
    assert sim.obstacle_sdf_grad is grad
    assert sim.bone_sdf is sim.obstacle_sdf


# ---------------------------------------------------------------------------
# Contact gate
# ---------------------------------------------------------------------------

def test_hemi_dir_selects_which_half_contacts():
    """The default lower-half gate and a full sphere must differ.

    A sphere centred inside the block touches particles all around it; the
    default (0,-1,0) gate only kicks those below the centre.
    """
    def run(hemi_dir):
        sim = _sim()
        g = np.array([0.0, 0.0, 0.0])
        sim.step_with_contact(g, np.array([L / 2, L / 2, L / 2]), 0.010,
                              hemi_dir=hemi_dir)
        return float(np.linalg.norm(sim.last_contact_force))

    lower = run(None)                 # default (0, -1, 0)
    full = run(np.zeros(3))           # whole sphere
    assert lower > 0 and full > 0
    assert not np.isclose(lower, full), (
        "full-sphere and lower-hemisphere contact gave the same reaction "
        "(%.4g vs %.4g N)" % (lower, full))


# ---------------------------------------------------------------------------
# apply_cut options
# ---------------------------------------------------------------------------

def test_apply_cut_replace_keeps_one_sdf():
    sim = _sim()
    sim.apply_cut(_plane_cut_sdf(sim, offset_cells=0.5), retract_mm=0.0, retract_steps=0)
    sim.apply_cut(_plane_cut_sdf(sim, offset_cells=1.5), retract_mm=0.0, retract_steps=0)
    assert len(sim.cut_sdfs) == 2
    sim.apply_cut(_plane_cut_sdf(sim, offset_cells=2.5), retract_mm=0.0,
                  retract_steps=0, replace=True)
    assert len(sim.cut_sdfs) == 1


def test_apply_cut_reset_reference_false_preserves_F():
    """reset_reference=False must leave F alone so loaded tissue can recoil."""
    def run(reset):
        sim = _sim()
        g = np.array([0.0, -9.8, 0.0])
        for _ in range(40):
            sim.step(g)          # build up some deformation
        F_before = sim.F.numpy().copy()
        sim.apply_cut(_plane_cut_sdf(sim), retract_mm=0.0, retract_steps=0,
                      reset_reference=reset)
        return F_before, sim.F.numpy().copy()

    before_keep, after_keep = run(False)
    assert np.array_equal(before_keep, after_keep), \
        "reset_reference=False still modified F"

    before_reset, after_reset = run(True)
    assert not np.array_equal(before_reset, after_reset), \
        "reset_reference=True did not re-reference F"
