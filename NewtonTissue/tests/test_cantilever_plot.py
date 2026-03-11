"""Visualization test: cantilever beam deformation and stress under various loads.

Produces a PNG artifact showing:
- Row 1: Side views (x-y plane) of the deformed beam colored by bending stress
- Row 2: Cross-section views (y-z plane) at mid-span colored by bending stress

Four loading conditions are compared:
1. Self-weight (gravity)
2. Tip point load
3. Heavy distributed load
4. Combined: self-weight + tip load

The analytical Euler-Bernoulli beam solution is used to compute realistic
deformations and stresses. The mesh comes from the NewtonTissue API.
"""

import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import PolyCollection
import matplotlib.cm as cm
import numpy as np
import pytest

from newton_tissue import TissueModel, IsotropicMaterial, FixedByBox, Gravity, PointForce
from newton_tissue.beam_analytical import (
    BeamProperties,
    bending_stress,
    deform_nodes,
    deflection_self_weight,
    deflection_tip_load,
    deflection_uniform_load,
    moment_self_weight,
    moment_tip_load,
    moment_uniform_load,
)
from tests.conftest import make_cantilever_mesh


# ── Beam parameters ─────────────────────────────────────────────────────────

Lx, Ly, Lz = 0.5, 0.05, 0.05  # 50cm x 5cm x 5cm beam
E_AL = 70e9      # Aluminum, Pa
NU_AL = 0.33
RHO_AL = 2700.0  # kg/m^3


def _beam_props():
    return BeamProperties(L=Lx, b=Lz, h=Ly, E=E_AL, nu=NU_AL, density=RHO_AL)


def _make_model(nx=20, ny=4, nz=4):
    """Build a cantilever TissueModel using the NewtonTissue API."""
    nodes, elements = make_cantilever_mesh(nx, ny, nz, Lx, Ly, Lz)
    material = IsotropicMaterial(E=E_AL, nu=NU_AL, density=RHO_AL)
    bc = FixedByBox([-0.001, -0.001, -0.001], [0.001, Ly + 0.001, Lz + 0.001])
    model = TissueModel(
        nodes=nodes,
        elements=elements,
        material=material,
        boundary_conditions=[bc],
        loading_conditions=[Gravity()],
    )
    return model


# ── Surface extraction ──────────────────────────────────────────────────────

def extract_surface_triangles(elements: np.ndarray) -> np.ndarray:
    """Extract surface triangles from a tetrahedral mesh.

    Each tet has 4 faces. Surface faces appear exactly once (interior faces
    are shared by two tets).

    Returns:
        (K, 3) int32 array of surface triangle connectivity.
    """
    # The 4 faces of a tet with vertices (a,b,c,d)
    face_patterns = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]

    face_count = Counter()
    face_to_nodes = {}
    for tet in elements:
        for pattern in face_patterns:
            face_nodes = tuple(sorted(tet[list(pattern)]))
            face_count[face_nodes] += 1
            face_to_nodes[face_nodes] = tet[list(pattern)]

    surface = []
    for face_key, count in face_count.items():
        if count == 1:
            surface.append(face_to_nodes[face_key])

    return np.array(surface, dtype=np.int32)


def extract_cross_section_nodes(nodes: np.ndarray, x_target: float, tol: float):
    """Find nodes near a given x-coordinate for cross-section plotting.

    Returns:
        indices: Node indices near x_target.
        yz_coords: (K, 2) array of (y, z) coordinates.
    """
    mask = np.abs(nodes[:, 0] - x_target) < tol
    indices = np.nonzero(mask)[0]
    yz = nodes[indices][:, 1:3]
    return indices, yz


# ── Loading cases ───────────────────────────────────────────────────────────

def _loading_cases(beam: BeamProperties):
    """Define loading cases with their analytical solution functions.

    Returns list of (name, deflection_fn, moment_fn, scale_note) tuples.
    The deflection/moment functions have signature fn(x, beam).
    """
    # Tip load: choose P so max deflection is ~5x self-weight for visibility
    q_self = beam.weight_per_length
    delta_sw = q_self * beam.L**4 / (8.0 * beam.E * beam.I)
    P_tip = 5.0 * delta_sw * 3.0 * beam.E * beam.I / beam.L**3

    # Heavy distributed load: 10x self-weight
    q_heavy = 10.0 * q_self

    cases = [
        (
            f"Self-weight\n(q={q_self:.1f} N/m)",
            lambda x, b: deflection_self_weight(x, b),
            lambda x, b: moment_self_weight(x, b),
        ),
        (
            f"Tip load\n(P={P_tip:.1f} N)",
            lambda x, b, P=P_tip: deflection_tip_load(x, P, b),
            lambda x, b, P=P_tip: moment_tip_load(x, P, b),
        ),
        (
            f"Distributed load\n(q={q_heavy:.1f} N/m)",
            lambda x, b, q=q_heavy: deflection_uniform_load(x, q, b),
            lambda x, b, q=q_heavy: moment_uniform_load(x, q, b),
        ),
        (
            f"Combined\n(self-weight + P={P_tip:.1f} N)",
            lambda x, b, P=P_tip: (
                deflection_self_weight(x, b) + deflection_tip_load(x, P, b)
            ),
            lambda x, b, P=P_tip: (
                moment_self_weight(x, b) + moment_tip_load(x, P, b)
            ),
        ),
    ]
    return cases


# ── Plotting ────────────────────────────────────────────────────────────────

def _plot_side_view(ax, nodes_deformed, surface_tris, stress, title, beam):
    """Plot the side view (x-y plane) of the deformed beam with stress colors."""
    # Project surface triangles onto x-y plane
    x = nodes_deformed[:, 0]
    y = nodes_deformed[:, 1]

    # Compute per-triangle average stress for coloring
    tri_stress = stress[surface_tris].mean(axis=1)

    # Build polygon collection for filled triangles
    verts = np.stack([x[surface_tris], y[surface_tris]], axis=-1)

    vmax = max(np.abs(stress).max(), 1e-6)
    norm = matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax)
    cmap = cm.RdBu_r

    poly = PolyCollection(
        verts,
        array=tri_stress,
        cmap=cmap,
        norm=norm,
        edgecolors="face",
        linewidths=0.1,
    )
    ax.add_collection(poly)

    # Draw the undeformed outline for reference
    ax.plot(
        [0, beam.L, beam.L, 0, 0],
        [0, 0, beam.h, beam.h, 0],
        "k--", linewidth=0.5, alpha=0.3, label="undeformed",
    )

    # Wall hatch at x=0
    wall_y = np.linspace(
        nodes_deformed[:, 1].min() - beam.h * 0.2,
        nodes_deformed[:, 1].max() + beam.h * 0.2,
        20,
    )
    for yw in wall_y:
        ax.plot([-beam.L * 0.02, 0], [yw - beam.h * 0.05, yw], "k-", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=1.5)

    ax.set_xlim(-beam.L * 0.05, beam.L * 1.05)
    y_range = max(nodes_deformed[:, 1].max() - nodes_deformed[:, 1].min(), beam.h)
    y_center = (nodes_deformed[:, 1].max() + nodes_deformed[:, 1].min()) / 2.0
    ax.set_ylim(y_center - y_range * 0.8, y_center + y_range * 0.8)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x [m]", fontsize=8)
    ax.set_ylabel("y [m]", fontsize=8)
    ax.tick_params(labelsize=7)

    return poly, norm, cmap


def _plot_cross_section(ax, yz, stress_vals, title, beam):
    """Plot cross-section (y-z plane) at a given x, colored by stress."""
    y = yz[:, 0]
    z = yz[:, 1]

    # Triangulate the cross-section points
    tri = mtri.Triangulation(z, y)

    vmax = max(np.abs(stress_vals).max(), 1e-6)
    norm = matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax)

    tpc = ax.tripcolor(
        tri, stress_vals,
        cmap=cm.RdBu_r, norm=norm,
        shading="gouraud",
    )

    # Outline the undeformed cross-section
    ax.plot(
        [0, beam.b, beam.b, 0, 0],
        [0, 0, beam.h, beam.h, 0],
        "k--", linewidth=0.5, alpha=0.3,
    )

    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("z [m]", fontsize=8)
    ax.set_ylabel("y [m]", fontsize=8)
    ax.tick_params(labelsize=7)

    return tpc


# ── Test ────────────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


class TestCantileverPlot:
    """Generate cantilever beam visualization under various loading conditions."""

    def test_cantilever_stress_plot(self):
        """Produce a multi-panel PNG of deformed beams with stress heatmaps."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        beam = _beam_props()
        model = _make_model(nx=24, ny=4, nz=4)
        nodes = model.nodes
        elements = model.elements

        surface_tris = extract_surface_triangles(elements)
        cases = _loading_cases(beam)

        # Cross-section at mid-span
        x_mid = beam.L / 2.0
        dx = beam.L / 24.0  # match mesh spacing
        cs_indices, cs_yz = extract_cross_section_nodes(nodes, x_mid, tol=dx * 0.6)

        fig, axes = plt.subplots(
            2, len(cases), figsize=(4.5 * len(cases), 7),
            gridspec_kw={"height_ratios": [2, 1.2]},
        )

        for col, (title, defl_fn, moment_fn) in enumerate(cases):
            # Deform nodes analytically
            deformed = deform_nodes(nodes, defl_fn, beam)

            # Bending stress at every node
            stress = bending_stress(
                nodes[:, 0], nodes[:, 1], moment_fn, beam
            )

            # --- Side view (top row) ---
            ax_side = axes[0, col]
            poly, norm, cmap_used = _plot_side_view(
                ax_side, deformed, surface_tris, stress, title, beam
            )

            # Colorbar for side view
            cb = fig.colorbar(
                cm.ScalarMappable(norm=norm, cmap=cmap_used),
                ax=ax_side, fraction=0.06, pad=0.04,
            )
            cb.set_label("σ_xx [Pa]", fontsize=7)
            cb.ax.tick_params(labelsize=6)

            # --- Cross-section at mid-span (bottom row) ---
            ax_cs = axes[1, col]
            cs_stress = stress[cs_indices]
            cs_yz_deformed = deformed[cs_indices][:, 1:3]

            tpc = _plot_cross_section(
                ax_cs, cs_yz_deformed, cs_stress,
                f"Cross-section at x={x_mid:.2f}m", beam
            )

            cb2 = fig.colorbar(tpc, ax=ax_cs, fraction=0.06, pad=0.04)
            cb2.set_label("σ_xx [Pa]", fontsize=7)
            cb2.ax.tick_params(labelsize=6)

        fig.suptitle(
            f"Cantilever Beam: Aluminum {Lx*100:.0f}cm × {Ly*100:.0f}cm × {Lz*100:.0f}cm\n"
            f"E = {E_AL/1e9:.0f} GPa, ν = {NU_AL}, ρ = {RHO_AL} kg/m³\n"
            "(Euler-Bernoulli analytical solution mapped onto tet mesh)",
            fontsize=11, y=1.02,
        )
        fig.tight_layout()

        out_path = os.path.join(OUTPUT_DIR, "cantilever_stress.png")
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        assert os.path.exists(out_path), f"Plot not saved to {out_path}"
        file_size = os.path.getsize(out_path)
        assert file_size > 10_000, f"Plot file suspiciously small: {file_size} bytes"

        print(f"\nArtifact saved: {out_path} ({file_size / 1024:.0f} KB)")

    def test_deflection_magnitudes(self):
        """Verify the analytical deflections are physically reasonable."""
        beam = _beam_props()
        x = np.array([beam.L])

        # Self-weight tip deflection
        delta_sw = deflection_self_weight(x, beam)[0]
        # For aluminum cantilever 50cm x 5cm x 5cm, should be very small
        assert 0 < delta_sw < 1e-3, f"Self-weight deflection {delta_sw} seems wrong"

        cases = _loading_cases(beam)
        for name, defl_fn, _ in cases:
            delta = defl_fn(x, beam)[0]
            assert delta >= 0, f"Deflection should be non-negative for {name}"
            assert delta < beam.L, f"Deflection {delta} exceeds beam length for {name}"

    def test_stress_distribution(self):
        """Verify bending stress is zero at neutral axis, antisymmetric."""
        beam = _beam_props()
        x_test = np.array([beam.L / 2.0])

        # At neutral axis, stress should be zero
        y_neutral = np.array([beam.y_neutral])
        sigma = bending_stress(x_test, y_neutral, moment_self_weight, beam)
        np.testing.assert_allclose(sigma, 0.0, atol=1e-6)

        # Top and bottom fibers should have opposite sign
        y_top = np.array([beam.h])
        y_bot = np.array([0.0])
        sigma_top = bending_stress(x_test, y_top, moment_self_weight, beam)
        sigma_bot = bending_stress(x_test, y_bot, moment_self_weight, beam)
        np.testing.assert_allclose(sigma_top, -sigma_bot, rtol=1e-10)

    def test_model_api_exercised(self):
        """Verify the test exercises TissueModel API correctly."""
        model = _make_model()
        assert model.num_elements > 0
        assert model.num_nodes > 0
        assert len(model.fixed_node_indices) > 0
        assert len(model.free_node_indices) > 0

        vols = model.compute_element_volumes()
        np.testing.assert_allclose(vols.sum(), Lx * Ly * Lz, rtol=1e-10)

        forces = model.assemble_forces()
        assert forces.shape == (model.num_nodes, 3)
        # Gravity should produce downward force
        assert forces[:, 1].sum() < 0
