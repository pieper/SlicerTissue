"""Per-particle material tests for the MPM solver.

These import ``newton_tissue/mpm.py`` **directly by path** rather than through
``newton_tissue/__init__.py``, because that package imports ``newton`` (the
NVIDIA physics package) which is not available on every dev machine.  mpm.py
itself needs only warp + numpy.

Two things are being pinned down here:

1. **No regression.** Moving mu/lam from scalar kernel arguments to
   per-particle arrays must not change homogeneous behaviour at all.
   ``test_uniform_arrays_match_scalar_material`` asserts bit-identical
   positions, which is a far sharper instrument than the 20% analytical
   tolerance in the FEBio tier-2 suite.

2. **Heterogeneity works.** A stiff region must actually deflect less, the
   per-bond multiplier must scale bond forces, and the CFL bound must follow
   the stiffest particle rather than the nominal material.
"""

import importlib.util
import os

import numpy as np
import pytest

wp = pytest.importorskip("warp", reason="warp not installed")

_MPM_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "src", "newton_tissue", "mpm.py")
_spec = importlib.util.spec_from_file_location("_mpm_under_test", _MPM_PATH)
mpm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mpm)

MPMMaterial = mpm.MPMMaterial
MPMSimulator = mpm.MPMSimulator

DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Analytical reference (duplicated from test_febio_tier1_validation so this
# module stays importable without the newton-dependent package)
# ---------------------------------------------------------------------------

def nh_cauchy(F, mu, lam):
    """Neo-Hookean Cauchy stress, matching the tau used in _p2g."""
    J = float(np.linalg.det(F))
    B = F @ F.T
    tau = mu * (B - np.eye(3)) + lam * np.log(max(J, 1e-10)) * np.eye(3)
    return tau / max(J, 1e-10)


def uniaxial_lateral_stretch(lam_axial, mu, lam_param):
    """Lateral stretch giving zero transverse stress, by Newton iteration."""
    nu_approx = lam_param / (2.0 * (lam_param + mu))
    lam_T = max(1.0 - nu_approx * (lam_axial - 1.0), 1e-4)
    for _ in range(60):
        J = lam_axial * lam_T ** 2
        f = mu * (lam_T ** 2 - 1.0) + lam_param * np.log(J)
        df = 2.0 * mu * lam_T + 2.0 * lam_param / lam_T
        delta = -f / df
        lam_T = max(lam_T + delta, 1e-8)
        if abs(delta) < 1e-12:
            break
    return lam_T


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

L = 0.04          # block side [m]
E_NOM = 3_000.0   # Pa
NU = 0.30
RHO = 1060.0
LAM_AXIAL = 1.20
N_GRID = 16
PPC = 2


def _build(material=None, total_lagrangian=True, damping=0.92):
    mat = material or MPMMaterial(E=E_NOM, nu=NU, rho=RHO)
    sim = MPMSimulator(
        block_lo=[0.0, 0.0, 0.0],
        block_hi=[L, L * 1.5, L],
        n_grid=N_GRID, dt=1e-4, material=mat, device=DEVICE,
        velocity_damping=damping, total_lagrangian=total_lagrangian,
    )
    sim.initialize_block_particles(lo=[0.0, 0.0, 0.0], hi=[L, L, L],
                                   ppc=PPC, fixed_y_max=-1.0)
    return sim


def _clamp_and_stretch(sim):
    """Fix the bottom layer, fix + stretch the top layer. Returns the masks."""
    pos = sim.x.numpy().copy()
    spacing = sim.dx / PPC
    fixed = np.zeros(sim.n_particles, dtype=np.int32)
    bottom = pos[:, 1] < 1.5 * spacing
    top = pos[:, 1] > L - 1.5 * spacing
    fixed[bottom] = 1
    fixed[top] = 1
    pos[top, 1] *= LAM_AXIAL
    with wp.ScopedDevice(DEVICE):
        sim.x = wp.array(pos, dtype=wp.vec3)
        sim.fixed = wp.array(fixed, dtype=int)
    return bottom, top


def _run(sim, n_steps):
    zero_g = np.array([0.0, 0.0, 0.0])
    for _ in range(n_steps):
        sim.step(gravity=zero_g)
    return sim.x.numpy().copy()


# ---------------------------------------------------------------------------
# 1. No regression
# ---------------------------------------------------------------------------

def test_uniform_arrays_match_scalar_material():
    """Explicit uniform per-particle material == lazily broadcast material.

    This is the regression guard for moving mu/lam out of the kernel
    signature: the two runs must agree exactly, not approximately.
    """
    sim_a = _build()
    _clamp_and_stretch(sim_a)
    pos_a = _run(sim_a, 200)

    sim_b = _build()
    _clamp_and_stretch(sim_b)
    sim_b.set_particle_material(E=E_NOM, nu=NU)   # explicit, still uniform
    pos_b = _run(sim_b, 200)

    assert np.array_equal(pos_a, pos_b), (
        "explicit uniform material diverged from the lazily broadcast one; "
        "max |dx| = %g" % np.abs(pos_a - pos_b).max())


def test_homogeneous_stress_matches_analytical():
    """Homogeneous block still reproduces the Neo-Hookean uniaxial solution."""
    sim = _build()
    _clamp_and_stretch(sim)
    _run(sim, 500)

    mu = E_NOM / (2.0 * (1.0 + NU))
    lam = E_NOM * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
    lat = uniaxial_lateral_stretch(LAM_AXIAL, mu, lam)
    sigma_ref = nh_cauchy(np.diag([lat, LAM_AXIAL, lat]), mu, lam)[1, 1]

    pos = sim.x.numpy()
    F = sim.F.numpy()
    spacing = sim.dx / PPC
    interior = (pos[:, 1] > 3.0 * spacing) & (pos[:, 1] < L * LAM_AXIAL - 3.0 * spacing)
    assert interior.sum() > 100

    sigma_yy = np.array([nh_cauchy(F[i], mu, lam)[1, 1] for i in np.where(interior)[0]])
    mean_yy = float(sigma_yy.mean())
    assert np.isfinite(mean_yy)
    rel = abs(mean_yy - sigma_ref) / abs(sigma_ref)
    assert rel < 0.20, "sigma_yy %.1f Pa vs analytical %.1f Pa (%.1f%%)" % (
        mean_yy, sigma_ref, 100 * rel)

    J = np.linalg.det(F[interior])
    assert (J > 0).all(), "non-positive Jacobian in %d particles" % (J <= 0).sum()


# ---------------------------------------------------------------------------
# 2. Heterogeneity
# ---------------------------------------------------------------------------

def test_stiff_half_stretches_less():
    """In series along the pull axis, the stiff half must take less strain.

    The split has to be along the stretch axis.  Splitting across it puts the
    two halves in *parallel*, where the prescribed top-face displacement fixes
    both halves' elongation kinematically and only the stress differs -- so a
    displacement assertion there would be measuring nothing.
    """
    sim = _build()
    _clamp_and_stretch(sim)

    pos0 = sim.x.numpy().copy()
    spacing = sim.dx / PPC
    # Series split along y (the pull axis), excluding the clamped layers.
    stiff0 = pos0[:, 1] > 0.5 * L
    E = np.where(stiff0, 10.0 * E_NOM, E_NOM)
    sim.set_particle_material(E=E, nu=NU)
    sim.dt = sim.cfl_dt()

    free = sim.fixed.numpy() == 0
    soft_band = free & ~stiff0 & (pos0[:, 1] > 2.0 * spacing)
    stiff_band = free & stiff0 & (pos0[:, 1] < L - 2.0 * spacing)
    assert soft_band.sum() > 100 and stiff_band.sum() > 100

    def extent(pos, mask):
        return float(pos[mask, 1].max() - pos[mask, 1].min())

    soft0, hard0 = extent(pos0, soft_band), extent(pos0, stiff_band)
    pos1 = _run(sim, 500)
    soft1, hard1 = extent(pos1, soft_band), extent(pos1, stiff_band)

    soft_stretch = soft1 / soft0
    stiff_stretch = hard1 / hard0
    assert np.isfinite(soft_stretch) and np.isfinite(stiff_stretch)
    assert stiff_stretch < soft_stretch, (
        "stiff half stretched %.4f, soft half %.4f -- the 10x stiffer region "
        "should take less of the imposed elongation"
        % (stiff_stretch, soft_stretch))


def test_E_max_and_cfl_follow_stiffest_particle():
    sim = _build()
    _clamp_and_stretch(sim)
    dt_uniform = sim.cfl_dt(dt_max=1.0)

    n = sim.n_particles
    E = np.full(n, E_NOM)
    E[0] = 100.0 * E_NOM
    sim.set_particle_material(E=E, nu=NU)

    assert sim._E_max == pytest.approx(100.0 * E_NOM)
    # c_s scales as sqrt(E), so dt must shrink by 10x
    assert sim.cfl_dt(dt_max=1.0) == pytest.approx(dt_uniform / 10.0, rel=1e-6)


def test_mu_lam_conversion_roundtrip():
    """set_particle_material(E, nu) must produce the standard Lame values."""
    sim = _build()
    E = 19_000.0
    nu = 0.48
    sim.set_particle_material(E=E, nu=nu)
    mu = sim.mu_p.numpy()
    lam = sim.lam_p.numpy()
    assert mu == pytest.approx(E / (2 * (1 + nu)), rel=1e-5)
    assert lam == pytest.approx(E * nu / ((1 + nu) * (1 - 2 * nu)), rel=1e-5)


# ---------------------------------------------------------------------------
# 3. Per-bond stiffness
# ---------------------------------------------------------------------------

def test_bond_stiffness_scale_changes_bond_forces():
    """fiber_k=0 must disable bond forces; the default must equal fiber_k=1."""
    mat = MPMMaterial(E=E_NOM, nu=NU, rho=RHO, k_elastin=0.5, k_collagen=2.0,
                      collagen_crimp=0.05)

    def run(scale):
        sim = _build(material=mat)
        assert sim.n_bonds > 0, "fiber network was not built"
        _clamp_and_stretch(sim)
        if scale is not None:
            sim._ensure_material_arrays()
            sim.set_bond_stiffness(scale)
        return _run(sim, 200)

    pos_default = run(None)
    pos_one = run(1.0)
    pos_zero = run(0.0)

    assert np.array_equal(pos_default, pos_one), \
        "default bond stiffness should be exactly fiber_k = 1"
    assert not np.allclose(pos_zero, pos_one), \
        "fiber_k = 0 should disable bond forces and change the result"
