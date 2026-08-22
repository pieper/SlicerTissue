"""Attachment-spring tests for the MPM solver.

``AttachmentSet`` is one mechanism serving two roles in the SARRTS scenario:
the renal-pedicle tether (static frame) and the suction cup (moving frame).
These tests pin the physics that both rely on -- static equilibrium against a
known load, dragging by a moving frame, and force-limited release.

Imports mpm.py directly by path; see test_mpm_heterogeneous for why.
"""

import importlib.util
import os

import numpy as np
import pytest

wp = pytest.importorskip("warp", reason="warp not installed")

_MPM_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "src", "newton_tissue", "mpm.py")
_spec = importlib.util.spec_from_file_location("_mpm_attach_under_test", _MPM_PATH)
mpm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mpm)

MPMMaterial = mpm.MPMMaterial
MPMSimulator = mpm.MPMSimulator
AttachmentSet = mpm.AttachmentSet

L = 0.04          # block side [m]
PAD = 0.02        # grid padding around the block [m]
S = L + 2 * PAD   # grid box side [m]
DEVICE = "cpu"
G = 9.8


def _free_block(E=30_000.0, dt=5e-5, damping=0.98, ppc=2, n_grid=16, **matkw):
    """A block with nothing clamped and no floor -- held only by attachments.

    The grid is deliberately much larger than the block.  P2G/G2P *skip*
    out-of-range nodes rather than clamping, so particles within ~2 cells of
    the grid boundary silently lose momentum, which shows up as a drag that
    pins the body in place.  With the block filling the grid, a hanging block
    "supported" 98% of its own weight on that artefact.
    """
    mat = MPMMaterial(E=E, nu=0.3, rho=1060.0, **matkw)
    sim = MPMSimulator([0, 0, 0], [S, S, S], n_grid=n_grid, dt=dt, material=mat,
                       device=DEVICE, velocity_damping=damping)
    sim.initialize_block_particles(lo=[PAD] * 3, hi=[PAD + L] * 3, ppc=ppc,
                                   fixed_y_max=-1.0)
    sim.floor_j0 = False
    return sim


def _top_layer(sim):
    pos = sim.x.numpy()
    spacing = sim.dx / 2.0
    return np.where(pos[:, 1] > pos[:, 1].max() - 1.5 * spacing)[0]


def _total_weight(sim):
    return float(sim.m_p.numpy().sum()) * G


# ---------------------------------------------------------------------------
# Static equilibrium -- the headline acceptance test
# ---------------------------------------------------------------------------

def test_hanging_block_reaction_equals_weight():
    """A block hung from its top layer must load the frame with its own weight.

    The sharpest available check that the spring force, the particle masses and
    the Newton-3rd-law bookkeeping are all consistent: at equilibrium the frame
    must carry exactly m_total * g, whatever the stiffness.

    The reaction is NEGATIVE in y: reaction_force is the force the tissue
    exerts *on the frame*, and a hanging block pulls its support down.
    """
    sim = _free_block()
    idx = _top_layer(sim)
    weight = _total_weight(sim)

    att = AttachmentSet(sim, idx, frame_origin_m=[0.0, 0.0, 0.0],
                        k_total=400.0, zeta=1.0)
    sim.attachments.append(att)
    assert att.check_stability()["hard"] < 1.0, "test setup is already unstable"

    g = np.array([0.0, -G, 0.0])
    for _ in range(12000):
        sim.step(g)

    assert np.isfinite(sim.x.numpy()).all(), "hanging block went non-finite"

    reaction = att.reaction_force
    assert reaction[1] == pytest.approx(-weight, rel=0.02), (
        "reaction %.6f N vs -weight %.6f N (%.1f%%)"
        % (reaction[1], -weight, 100 * abs(reaction[1] + weight) / weight))
    # nothing sideways at equilibrium
    assert abs(reaction[0]) < 0.02 * weight
    assert abs(reaction[2]) < 0.02 * weight


def test_stiffness_is_resolution_independent():
    """k_total is a set total, so refining the lattice must not stiffen it.

    Sag at equilibrium is set by k_total and the weight, not by how many
    attachments share the load.
    """
    def sag(ppc):
        sim = _free_block(ppc=ppc)
        idx = _top_layer(sim)
        att = AttachmentSet(sim, idx, [0.0, 0.0, 0.0], k_total=400.0, zeta=1.0)
        sim.attachments.append(att)
        y0 = float(sim.x.numpy()[idx, 1].mean())
        for _ in range(8000):
            sim.step(np.array([0.0, -G, 0.0]))
        return float(sim.x.numpy()[idx, 1].mean() - y0)

    coarse, fine = sag(1), sag(2)
    assert coarse < 0 and fine < 0, "block did not sag at all"
    assert fine == pytest.approx(coarse, rel=0.25), \
        "sag changed with resolution: ppc=1 %.4g mm, ppc=2 %.4g mm" % (
            1e3 * coarse, 1e3 * fine)


# ---------------------------------------------------------------------------
# Moving frame -- the suction-cup role
# ---------------------------------------------------------------------------

def test_moving_frame_drags_particles():
    """Moving the frame must drag the attached particles with it, without snap."""
    sim = _free_block()
    idx = _top_layer(sim)
    att = AttachmentSet(sim, idx, frame_origin_m=[0.0, 0.0, 0.0],
                        k_total=4000.0, zeta=0.7)
    sim.attachments.append(att)

    zero_g = np.zeros(3)
    pos0 = sim.x.numpy()[idx].mean(axis=0)

    # Attachment refs were recorded at the current pose, so with the frame
    # unmoved there must be no initial jump.
    for _ in range(200):
        sim.step(zero_g)
    assert sim.x.numpy()[idx].mean(axis=0) == pytest.approx(pos0, abs=2e-5), \
        "attaching produced an initial snap"

    lift, n = 0.005, 3000
    for i in range(n):
        frac = (i + 1) / n
        att.set_frame([0.0, lift * frac, 0.0],
                      vel_m_s=[0.0, lift / (n * sim.dt), 0.0])
        sim.step(zero_g)
    for _ in range(6000):
        att.set_frame([0.0, lift, 0.0])
        sim.step(zero_g)

    moved = float(sim.x.numpy()[idx, 1].mean() - pos0[1])
    assert moved == pytest.approx(lift, rel=0.15), (
        "frame moved %.2f mm but particles moved %.2f mm" % (1e3 * lift, 1e3 * moved))


# ---------------------------------------------------------------------------
# Force-limited release
# ---------------------------------------------------------------------------

def test_f_break_releases_under_overload():
    """Pulling past the hold force must release attachments; below it must not."""
    def run(f_break_total, lift):
        sim = _free_block()
        idx = _top_layer(sim)
        att = AttachmentSet(sim, idx, [0.0, 0.0, 0.0], k_total=2000.0,
                            zeta=0.7, f_break_total=f_break_total)
        sim.attachments.append(att)
        n = 600
        for i in range(n):
            att.set_frame([0.0, lift * (i + 1) / n, 0.0],
                          vel_m_s=[0.0, lift / (n * sim.dt), 0.0])
            sim.step(np.array([0.0, -G, 0.0]))
        return att

    gentle = run(50.0, 0.002)
    assert gentle.n_attached == gentle.n_total, \
        "attachments released under a load far below f_break"

    violent = run(0.05, 0.05)
    assert violent.n_attached < violent.n_total, \
        "attachments held through a load far above f_break"


def test_release_peels_under_a_load_gradient():
    """Under a load gradient, attachments must release progressively.

    Three things the setup has to get right, all of which bit earlier versions
    of this test:

    * **The block must be anchored.** Lifting a *free* block only accelerates
      it, so the springs never load up and nothing releases.
    * **There must be a load gradient.** A uniform patch on a flat face pulled
      by a rigid frame loads every attachment almost identically, so they all
      cross f_break within a few steps of each other.  That is correct
      physics, not a defect -- a peel needs a reason for one side to carry
      more.  Here only half the bottom is anchored, so the free half deforms
      more.  (The real scenario has this by construction: a suction cup pulls
      a tumour off a kidney tethered at the hilum.)
    * **The frame velocity must be eased in.** A step change puts the whole
      damping term c*(frame_v - v_p) on every attachment at once, tripping
      f_break simultaneously.  That is a real consideration for driving a cup,
      not just a test artefact.
    """
    sim = _free_block(ppc=1)
    pos = sim.x.numpy()
    spacing = sim.dx

    # Anchor only the -x half of the bottom face.
    bottom = ((pos[:, 1] < pos[:, 1].min() + 1.5 * spacing)
              & (pos[:, 0] < PAD + L / 2))
    with wp.ScopedDevice(DEVICE):
        sim.fixed = wp.array(bottom.astype(np.int32), dtype=int)

    idx = np.where(pos[:, 1] > pos[:, 1].max() - 1.5 * spacing)[0]
    att = AttachmentSet(sim, idx, [0.0, 0.0, 0.0], k_total=2000.0, zeta=0.7,
                        f_break_total=6.0)
    sim.attachments.append(att)

    counts = []
    lift, n = 0.015, 4000
    for i in range(n):
        frac = (i + 1) / n
        att.set_frame([0.0, lift * frac ** 2, 0.0],
                      vel_m_s=[0.0, 2.0 * lift * frac / (n * sim.dt), 0.0])
        sim.step(np.array([0.0, -G, 0.0]))
        if i % 25 == 0:
            counts.append(att.n_attached)
    counts.append(att.n_attached)

    assert counts[0] == att.n_total, \
        "attachments released on the very first sample; the drive is too abrupt"
    assert counts[-1] < att.n_total, \
        "nothing released; the pull never reached f_break"
    partial = [c for c in counts if 0 < c < att.n_total]
    assert len(partial) >= 3, (
        "release was effectively all-at-once (%d partial samples); counts %s"
        % (len(partial), counts))
    assert counts == sorted(counts, reverse=True), \
        "attachment count went up; release must be monotonic"


def test_release_method_drops_all_force():
    sim = _free_block()
    idx = _top_layer(sim)
    att = AttachmentSet(sim, idx, [0.0, 0.0, 0.0], k_total=2000.0, zeta=0.7)
    sim.attachments.append(att)
    g = np.array([0.0, -G, 0.0])
    for _ in range(200):
        sim.step(g)
    assert np.linalg.norm(att.reaction_force) > 1e-4

    att.release()
    sim.step(g)
    assert att.n_attached == 0
    assert np.linalg.norm(att.reaction_force) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# The tether role: a soft replacement for a Dirichlet BC
# ---------------------------------------------------------------------------

def test_tether_to_rest_positions_holds_shape():
    """Springing particles to their own rest positions is a compliant anchor.

    This is how the renal pedicle is modelled: the tissue is held near its
    rest shape, but it can move and recoil rather than being nailed in place.
    """
    sim = _free_block()
    idx = _top_layer(sim)
    rest = sim.x.numpy().copy()

    # ref_positions in world coords = the rest pose; frame at the origin.
    att = AttachmentSet(sim, idx, frame_origin_m=[0.0, 0.0, 0.0],
                        k_total=800.0, zeta=1.0, ref_positions=rest[idx])
    sim.attachments.append(att)

    g = np.array([0.0, -G, 0.0])
    for _ in range(3000):
        sim.step(g)

    pos = sim.x.numpy()
    drift = float(np.abs(pos[idx, 1] - rest[idx, 1]).max())
    assert drift < 2e-3, "tethered layer drifted %.2f mm from rest" % (1e3 * drift)
    # and it is genuinely compliant, not pinned
    assert drift > 1e-6, "tether behaved like a rigid constraint"
