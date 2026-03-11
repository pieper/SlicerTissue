"""Large-deformation cantilever: nonlinear Newton vs linear comparison.

Runs the Newton VBD solver on a soft cantilever beam under gravity,
producing bending well beyond 45 degrees. Generates a comparison figure:

- Left column: Nonlinear (Newton VBD) result with von Mises stress.
  The beam curves naturally and the tip cross-section rotates to stay
  perpendicular to the beam centerline -- shape is preserved.
- Right column: Linear (Euler-Bernoulli) prediction applied to the
  same mesh. The tip plunges straight down without rotation, producing
  unrealistic element distortion at large displacement.

Output: tests/artifacts/large_deformation.png
"""

import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib.cm as cm
import numpy as np
import pytest

from newton_tissue import TissueModel, TissueSolver, IsotropicMaterial, FixedByBox, Gravity
from newton_tissue.beam_analytical import BeamProperties, deflection_self_weight
from tests.conftest import make_cantilever_mesh


# ── Beam parameters ───────────────────────────────────────────────────────
# Soft silicone-like material: large deformation under self-weight

Lx = 0.30        # 30 cm beam length
Ly = 0.05        # 5 cm height
Lz = 0.05        # 5 cm width
E_SOFT = 15_000.0    # 15 kPa
NU = 0.3
RHO = 1100.0

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


# ── Stress from deformation gradients ─────────────────────────────────────

def compute_deformation_gradients(ref_nodes, def_nodes, elements):
    """Per-element deformation gradient F = Ds @ inv(Dm)."""
    X0 = ref_nodes[elements[:, 0]]
    x0 = def_nodes[elements[:, 0]]
    Dm = np.stack([
        ref_nodes[elements[:, k]] - X0 for k in (1, 2, 3)
    ], axis=-1)
    Ds = np.stack([
        def_nodes[elements[:, k]] - x0 for k in (1, 2, 3)
    ], axis=-1)
    return Ds @ np.linalg.inv(Dm)


def compute_von_mises(F, mu, lam):
    """Per-element von Mises from Neo-Hookean Cauchy stress."""
    J = np.maximum(np.linalg.det(F), 1e-10)
    B = F @ np.transpose(F, (0, 2, 1))
    I3 = np.eye(3)[np.newaxis]
    sigma = ((mu / J)[:, None, None] * (B - I3)
             + (lam * np.log(J) / J)[:, None, None] * I3)
    tr_s = np.trace(sigma, axis1=1, axis2=2)
    dev = sigma - (tr_s / 3.0)[:, None, None] * I3
    return np.sqrt(1.5 * np.sum(dev * dev, axis=(1, 2)))


def element_to_node(values, elements, num_nodes):
    """Average per-element scalar to nodes."""
    total = np.zeros(num_nodes)
    count = np.zeros(num_nodes)
    for i, tet in enumerate(elements):
        for j in tet:
            total[j] += values[i]
            count[j] += 1
    count[count == 0] = 1
    return total / count


# ── Surface extraction ────────────────────────────────────────────────────

def extract_surface_triangles(elements):
    """Extract boundary triangles from tet mesh."""
    face_patterns = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    face_count = Counter()
    face_to_nodes = {}
    for tet in elements:
        for p in face_patterns:
            key = tuple(sorted(tet[list(p)]))
            face_count[key] += 1
            face_to_nodes[key] = tet[list(p)]
    return np.array(
        [face_to_nodes[k] for k, c in face_count.items() if c == 1],
        dtype=np.int32,
    )


def get_tip_surface_tris(surface_tris, ref_nodes, tol):
    """Surface triangles at the beam tip (all 3 vertices at x_max)."""
    x_max = ref_nodes[:, 0].max()
    mask = np.all(np.abs(ref_nodes[surface_tris, 0] - x_max) < tol, axis=1)
    return surface_tris[mask]


def linear_deform_nodes(nodes, beam):
    """Euler-Bernoulli self-weight deflection (vertical only, no rotation)."""
    deformed = nodes.copy()
    deformed[:, 1] -= deflection_self_weight(nodes[:, 0], beam)
    return deformed


# ── Plotting ──────────────────────────────────────────────────────────────

def plot_beam_side(ax, nodes_def, surface_tris, node_stress, title,
                   beam, vmax, nodes_ref=None):
    """Side view (x-y) of deformed beam with stress coloring and mesh edges."""
    x = nodes_def[:, 0]
    y = nodes_def[:, 1]
    tri_stress = node_stress[surface_tris].mean(axis=1)
    verts = np.stack([x[surface_tris], y[surface_tris]], axis=-1)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)

    poly = PolyCollection(
        verts, array=tri_stress, cmap=cm.plasma, norm=norm,
        edgecolors="k", linewidths=0.15,
    )
    ax.add_collection(poly)

    # Undeformed outline
    if nodes_ref is not None:
        xr, yr = nodes_ref[:, 0], nodes_ref[:, 1]
        ax.plot(
            [xr.min(), xr.max(), xr.max(), xr.min(), xr.min()],
            [yr.min(), yr.min(), yr.max(), yr.max(), yr.min()],
            color="0.5", ls="--", linewidth=0.8, label="undeformed",
        )

    # Wall hatch
    y_lo = min(y.min(), -0.01) - beam.h * 0.3
    y_hi = max(y.max(), beam.h + 0.01) + beam.h * 0.3
    wall_y = np.linspace(y_lo, y_hi, 30)
    for yw in wall_y:
        ax.plot([-beam.L * 0.02, 0], [yw - beam.h * 0.06, yw],
                "k-", linewidth=0.4)
    ax.axvline(0, color="k", linewidth=1.5)

    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("x [m]", fontsize=8)
    ax.set_ylabel("y [m]", fontsize=8)
    ax.tick_params(labelsize=7)
    return poly, norm


def plot_tip_face(ax, nodes_def, tip_tris, node_stress, title, vmax):
    """Tip face cross-section in y-z plane."""
    y = nodes_def[:, 1]
    z = nodes_def[:, 2]
    tri_stress = node_stress[tip_tris].mean(axis=1)
    verts = np.stack([z[tip_tris], y[tip_tris]], axis=-1)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)

    poly = PolyCollection(
        verts, array=tri_stress, cmap=cm.plasma, norm=norm,
        edgecolors="k", linewidths=0.5,
    )
    ax.add_collection(poly)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("z [m]", fontsize=8)
    ax.set_ylabel("y [m]", fontsize=8)
    ax.tick_params(labelsize=7)
    return poly


# ── Solver helper ─────────────────────────────────────────────────────────

def run_simulation(model, max_frames=4000):
    """Run VBD quasi-static solve with tuned parameters for large deformation."""
    solver = TissueSolver(
        model, dt=5e-4, num_substeps=1, iterations=30,
        solver_type="vbd", k_damp=0.1,
    )
    return solver.solve_static(max_frames=max_frames, tol=1e-3)


def build_model(nx=12, ny=3, nz=3):
    """Build the soft cantilever model."""
    nodes, elements = make_cantilever_mesh(nx, ny, nz, Lx, Ly, Lz)
    material = IsotropicMaterial(E=E_SOFT, nu=NU, density=RHO)
    dx = Lx / nx
    bc = FixedByBox(
        [-0.001, -0.001, -0.001],
        [dx + 0.001, Ly + 0.001, Lz + 0.001],
    )
    return TissueModel(
        nodes=nodes, elements=elements, material=material,
        boundary_conditions=[bc], loading_conditions=[Gravity()],
    )


# ── Tests ─────────────────────────────────────────────────────────────────

class TestLargeDeformation:

    def test_large_deformation_rendering(self):
        """Run Newton solver, render nonlinear vs linear side-by-side."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Build and solve
        nx, ny, nz = 12, 3, 3
        model = build_model(nx, ny, nz)
        nodes = model.nodes
        elements = model.elements
        beam = BeamProperties(L=Lx, b=Lz, h=Ly, E=E_SOFT, nu=NU, density=RHO)
        dx = Lx / nx

        result = run_simulation(model)
        print(f"\n  Newton VBD: converged={result.converged}, "
              f"max_disp={result.max_displacement():.4f} m")

        # Stress: nonlinear
        mu, lam = model.material.mu, model.material.lam
        F_nl = compute_deformation_gradients(nodes, result.positions, elements)
        vm_nl = element_to_node(
            compute_von_mises(F_nl, mu, lam), elements, model.num_nodes)

        # Stress: linear
        linear_def = linear_deform_nodes(nodes, beam)
        F_lin = compute_deformation_gradients(nodes, linear_def, elements)
        vm_lin = element_to_node(
            compute_von_mises(F_lin, mu, lam), elements, model.num_nodes)

        # Surfaces
        surface_tris = extract_surface_triangles(elements)
        tip_tris = get_tip_surface_tris(surface_tris, nodes, dx * 0.6)

        vmax = max(vm_nl.max(), vm_lin.max(), 1.0)
        delta_lin = deflection_self_weight(np.array([Lx]), beam)[0]

        # Tip displacement angle
        tip_mask = nodes[:, 0] > Lx - dx * 1.5
        tip_d = result.positions[tip_mask].mean(axis=0) - nodes[tip_mask].mean(axis=0)
        angle = np.degrees(np.arctan2(abs(tip_d[1]), Lx + tip_d[0]))

        # ── Figure ────────────────────────────────────────────
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[2.5, 1.2],
                              hspace=0.35, wspace=0.3)

        # --- Top-left: Newton nonlinear ---
        ax_nl = fig.add_subplot(gs[0, 0])
        poly_nl, norm_nl = plot_beam_side(
            ax_nl, result.positions, surface_tris, vm_nl,
            f"Nonlinear (Newton VBD)\nNeo-Hookean, tip angle $\\approx$ {angle:.0f}$^\\circ$",
            beam, vmax, nodes_ref=nodes,
        )
        pad = 0.03
        ax_nl.set_xlim(-Lx * 0.08, Lx * 1.08)
        ax_nl.set_ylim(
            min(result.positions[:, 1].min(), -pad) - pad,
            max(result.positions[:, 1].max(), Ly) + pad,
        )

        # --- Top-right: Linear EB ---
        ax_lin = fig.add_subplot(gs[0, 1])
        poly_lin, _ = plot_beam_side(
            ax_lin, linear_def, surface_tris, vm_lin,
            f"Linear (Euler-Bernoulli)\ntip $\\delta$ = {delta_lin:.2f} m "
            f"({delta_lin/Lx:.1f}$\\times$ beam length!)",
            beam, vmax, nodes_ref=nodes,
        )
        ax_lin.set_xlim(-Lx * 0.08, Lx * 1.08)
        ax_lin.set_ylim(
            min(linear_def[:, 1].min(), -pad) - pad,
            max(linear_def[:, 1].max(), Ly) + pad,
        )

        # --- Bottom-left: Tip face nonlinear ---
        ax_tip_nl = fig.add_subplot(gs[1, 0])
        if len(tip_tris) > 0:
            plot_tip_face(ax_tip_nl, result.positions, tip_tris, vm_nl,
                          "Tip face (nonlinear): rotated, shape preserved", vmax)
            tn = np.unique(tip_tris.flatten())
            tp = result.positions[tn]
            pad_t = Ly * 0.4
            ax_tip_nl.set_xlim(tp[:, 2].min() - pad_t, tp[:, 2].max() + pad_t)
            ax_tip_nl.set_ylim(tp[:, 1].min() - pad_t, tp[:, 1].max() + pad_t)

            # Draw reference square for comparison
            ref_tip = nodes[tn]
            ax_tip_nl.plot(
                [ref_tip[:, 2].min(), ref_tip[:, 2].max(),
                 ref_tip[:, 2].max(), ref_tip[:, 2].min(),
                 ref_tip[:, 2].min()],
                [ref_tip[:, 1].min(), ref_tip[:, 1].min(),
                 ref_tip[:, 1].max(), ref_tip[:, 1].max(),
                 ref_tip[:, 1].min()],
                "k--", linewidth=0.8, alpha=0.4, label="original",
            )
            ax_tip_nl.legend(fontsize=7, loc="upper right")
        else:
            ax_tip_nl.text(0.5, 0.5, "(no tip triangles)", ha="center",
                           va="center", transform=ax_tip_nl.transAxes)

        # --- Bottom-right: Tip face linear ---
        ax_tip_lin = fig.add_subplot(gs[1, 1])
        if len(tip_tris) > 0:
            plot_tip_face(ax_tip_lin, linear_def, tip_tris, vm_lin,
                          "Tip face (linear): no rotation, displaced straight down",
                          vmax)
            tp_lin = linear_def[tn]
            ax_tip_lin.set_xlim(tp_lin[:, 2].min() - pad_t,
                                tp_lin[:, 2].max() + pad_t)
            ax_tip_lin.set_ylim(tp_lin[:, 1].min() - pad_t,
                                tp_lin[:, 1].max() + pad_t)

            ax_tip_lin.plot(
                [ref_tip[:, 2].min(), ref_tip[:, 2].max(),
                 ref_tip[:, 2].max(), ref_tip[:, 2].min(),
                 ref_tip[:, 2].min()],
                [ref_tip[:, 1].min(), ref_tip[:, 1].min(),
                 ref_tip[:, 1].max(), ref_tip[:, 1].max(),
                 ref_tip[:, 1].min()],
                "k--", linewidth=0.8, alpha=0.4, label="original",
            )
            ax_tip_lin.legend(fontsize=7, loc="upper right")

        # Colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        cb = fig.colorbar(
            cm.ScalarMappable(norm=norm_nl, cmap=cm.plasma), cax=cbar_ax)
        cb.set_label("von Mises stress [Pa]", fontsize=9)

        fig.suptitle(
            f"Large-Deformation Cantilever: E = {E_SOFT/1e3:.0f} kPa, "
            f"$\\nu$ = {NU}, $\\rho$ = {RHO:.0f} kg/m$^3$\n"
            f"Beam: {Lx*100:.0f} cm $\\times$ {Ly*100:.0f} cm $\\times$ "
            f"{Lz*100:.0f} cm  |  "
            f"Newton tip: {angle:.0f}$^\\circ$  |  "
            f"Linear predicts {delta_lin/Lx:.1f}$\\times$ beam-length deflection",
            fontsize=11, y=0.98,
        )

        out_path = os.path.join(OUTPUT_DIR, "large_deformation.png")
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        file_size = os.path.getsize(out_path)
        print(f"  Artifact: {out_path} ({file_size / 1024:.0f} KB)")

        assert os.path.exists(out_path)
        assert file_size > 20_000

    def test_bending_exceeds_45_degrees(self):
        """Verify the nonlinear solver produces >45 degree bending."""
        model = build_model(nx=8, ny=2, nz=2)
        nodes = model.nodes
        dx = Lx / 8

        result = run_simulation(model)

        tip_mask = nodes[:, 0] > Lx - dx * 1.5
        tip_ref = nodes[tip_mask].mean(axis=0)
        tip_def = result.positions[tip_mask].mean(axis=0)
        delta = tip_def - tip_ref
        angle = np.degrees(np.arctan2(abs(delta[1]), Lx + delta[0]))

        print(f"\n  Tip: dx={delta[0]:.4f}, dy={delta[1]:.4f}, angle~{angle:.0f} deg")

        assert angle > 45, (
            f"Tip bending angle {angle:.1f} deg < 45 deg -- not enough deformation"
        )

    def test_elements_not_inverted(self):
        """Verify no elements are inverted in the nonlinear result."""
        model = build_model(nx=8, ny=2, nz=2)
        nodes = model.nodes
        elements = model.elements

        result = run_simulation(model)

        F = compute_deformation_gradients(nodes, result.positions, elements)
        J = np.linalg.det(F)

        n_inverted = np.sum(J <= 0)
        assert n_inverted == 0, (
            f"{n_inverted}/{len(elements)} elements inverted in nonlinear result"
        )
        print(f"\n  All {len(elements)} elements positive J "
              f"(min={J.min():.4f}, max={J.max():.4f})")

        # Compare: linear deformation has distorted/near-inverted elements
        beam = BeamProperties(L=Lx, b=Lz, h=Ly, E=E_SOFT, nu=NU, density=RHO)
        linear_def = linear_deform_nodes(nodes, beam)
        F_lin = compute_deformation_gradients(nodes, linear_def, elements)
        J_lin = np.linalg.det(F_lin)
        print(f"  Linear J range: [{J_lin.min():.4f}, {J_lin.max():.4f}]")
