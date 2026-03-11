"""Palpation simulation of layered abdominal tissue near the rib cage.

Builds a 10x10x10 cm block of tissue with anatomically-inspired layers:
  - Skin (superficial, stiff)
  - Subcutaneous fat (soft, compressible)
  - Abdominal muscle (moderate stiffness)
  - Liver (bulk organ, soft)
  - Rib (rigid bone at one edge)

Boundary conditions:
  - Bottom face (deep tissue attachment) and rib region are fixed.

Loading:
  - Downward palpation force on a patch of the skin surface,
    sized to produce ~2 cm indentation.

Output:
  Cross-section rendering (y-z plane at mid-x) showing tissue layers
  before and after palpation, plus displacement magnitude.

Usage:
  python -m examples.palpation
  # or: python examples/palpation.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
import matplotlib.cm as cm

# Ensure newton_tissue is importable when run as a script
_pkg_src = os.path.join(os.path.dirname(__file__), "..", "src")
if _pkg_src not in sys.path:
    sys.path.insert(0, os.path.abspath(_pkg_src))

from newton_tissue import TissueModel, TissueSolver, FixedBC, PointForce
from newton_tissue.materials import HeterogeneousMaterial


# ── Geometry ──────────────────────────────────────────────────────────────

BLOCK_SIZE = 0.10  # 10 cm cube

# Layer boundaries (y coordinate, bottom to top)
# y=0 is deep (posterior), y=0.10 is skin surface (anterior)
LAYER_BOUNDS = {
    "liver":  (0.000, 0.050),   # 5 cm
    "muscle": (0.050, 0.065),   # 1.5 cm
    "fat":    (0.065, 0.085),   # 2 cm
    "skin":   (0.085, 0.100),   # 1.5 cm
}

# Rib region (rigid, at one edge of the block)
# Runs along x, sits at high-z edge, spanning muscle/fat/lower-skin layers
RIB_BOX = {
    "z": (0.070, 0.100),
    "y": (0.040, 0.085),
}

# Material properties (literature-informed)
#                    E [Pa]    nu     rho [kg/m^3]
TISSUE_PROPS = {
    "skin":   (100_000.0, 0.40, 1100.0),
    "fat":    (  3_000.0, 0.49,  900.0),
    "muscle": ( 60_000.0, 0.40, 1050.0),
    "liver":  ( 10_000.0, 0.45, 1060.0),
    "rib":    ( 60_000.0,  0.40, 1900.0),  # fixed BC; use muscle-like E to avoid stiffness contrast
}

# Tissue colors for rendering
TISSUE_COLORS = {
    "skin":   "#F5CBA7",   # light tan
    "fat":    "#F9E79F",   # yellow
    "muscle": "#E74C3C",   # red
    "liver":  "#8B4513",   # saddle brown
    "rib":    "#ECF0F1",   # off-white
}


# ── Mesh generation ───────────────────────────────────────────────────────

def make_block_mesh(nx: int, ny: int, nz: int, L: float = BLOCK_SIZE):
    """Generate a structured tet mesh for a cube.

    Uses the same hex-to-5-tet decomposition as make_cantilever_mesh.

    Returns:
        nodes: (N, 3) float64 positions.
        elements: (M, 4) int32 tet connectivity.
    """
    xs = np.linspace(0, L, nx + 1)
    ys = np.linspace(0, L, ny + 1)
    zs = np.linspace(0, L, nz + 1)

    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)
    nodes = grid.reshape(-1, 3).astype(np.float64)

    def idx(i, j, k):
        return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k

    elements = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                n = [
                    idx(i, j, k),       idx(i+1, j, k),
                    idx(i+1, j+1, k),   idx(i, j+1, k),
                    idx(i, j, k+1),     idx(i+1, j, k+1),
                    idx(i+1, j+1, k+1), idx(i, j+1, k+1),
                ]
                parity = (i + j + k) % 2
                if parity == 0:
                    elements.append([n[0], n[1], n[3], n[4]])
                    elements.append([n[1], n[2], n[3], n[6]])
                    elements.append([n[4], n[6], n[5], n[1]])
                    elements.append([n[3], n[4], n[6], n[7]])
                    elements.append([n[1], n[3], n[4], n[6]])
                else:
                    elements.append([n[0], n[1], n[2], n[5]])
                    elements.append([n[0], n[2], n[3], n[7]])
                    elements.append([n[0], n[4], n[5], n[7]])
                    elements.append([n[2], n[5], n[6], n[7]])
                    elements.append([n[0], n[2], n[5], n[7]])

    return nodes, np.array(elements, dtype=np.int32)


# ── Tissue classification ─────────────────────────────────────────────────

def classify_elements(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """Assign a tissue label to each element based on centroid position.

    Returns:
        labels: (M,) array of strings.
    """
    centroids = nodes[elements].mean(axis=1)  # (M, 3)
    labels = np.empty(len(elements), dtype=object)

    for i, c in enumerate(centroids):
        y, z = c[1], c[2]

        # Check rib first (overrides layer assignment)
        if (RIB_BOX["z"][0] <= z <= RIB_BOX["z"][1]
                and RIB_BOX["y"][0] <= y <= RIB_BOX["y"][1]):
            labels[i] = "rib"
            continue

        # Layer assignment by y-coordinate
        assigned = False
        for tissue, (y_lo, y_hi) in LAYER_BOUNDS.items():
            if y_lo <= y < y_hi:
                labels[i] = tissue
                assigned = True
                break
        if not assigned:
            labels[i] = "liver"  # default for anything out of range

    return labels


def build_material(labels: np.ndarray) -> HeterogeneousMaterial:
    """Create a HeterogeneousMaterial from per-element tissue labels."""
    n = len(labels)
    k_mu = np.zeros(n, dtype=np.float64)
    k_lambda = np.zeros(n, dtype=np.float64)
    density = np.zeros(n, dtype=np.float64)

    for i, tissue in enumerate(labels):
        E, nu, rho = TISSUE_PROPS[tissue]
        mu = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        k_mu[i] = mu
        k_lambda[i] = lam
        density[i] = rho

    return HeterogeneousMaterial(k_mu=k_mu, k_lambda=k_lambda, density=density)


# ── Boundary conditions and loading ───────────────────────────────────────

def get_fixed_nodes(nodes: np.ndarray, labels: np.ndarray,
                    elements: np.ndarray) -> list[int]:
    """Return indices of fixed nodes: bottom face + rib elements."""
    dy = BLOCK_SIZE / 10  # approximate grid spacing
    fixed = set()

    # Bottom face
    for i, pos in enumerate(nodes):
        if pos[1] < dy * 0.5:
            fixed.add(i)

    # Rib element nodes
    for i, tissue in enumerate(labels):
        if tissue == "rib":
            for j in elements[i]:
                fixed.add(int(j))

    return sorted(fixed)


def get_palpation_nodes(nodes: np.ndarray, center_x: float, center_z: float,
                        radius: float) -> np.ndarray:
    """Find skin-surface nodes within a circular palpation patch."""
    y_top = nodes[:, 1].max()
    dy = BLOCK_SIZE / 10
    on_top = np.abs(nodes[:, 1] - y_top) < dy * 0.5
    dx = nodes[:, 0] - center_x
    dz = nodes[:, 2] - center_z
    in_circle = (dx**2 + dz**2) < radius**2
    return np.nonzero(on_top & in_circle)[0]


# ── Cross-section extraction and rendering ────────────────────────────────

def get_slice_elements(nodes, elements, x_target, tol):
    """Elements whose centroid is within tol of x_target."""
    centroids = nodes[elements].mean(axis=1)
    mask = np.abs(centroids[:, 0] - x_target) < tol
    return np.nonzero(mask)[0]


def tet_faces(tet):
    """Return the 4 triangular faces of a tet."""
    return [
        [tet[0], tet[1], tet[2]],
        [tet[0], tet[1], tet[3]],
        [tet[0], tet[2], tet[3]],
        [tet[1], tet[2], tet[3]],
    ]


def plot_cross_section(ax, nodes, elements, elem_indices, labels,
                       title, show_disp=False, displacements=None):
    """Plot a y-z cross-section colored by tissue type (or displacement)."""
    for tissue in TISSUE_COLORS:
        verts_list = []
        for ei in elem_indices:
            if labels[ei] != tissue:
                continue
            for face in tet_faces(elements[ei]):
                tri_yz = nodes[face][:, [2, 1]]  # (z, y) for x-axis=z, y-axis=y
                verts_list.append(tri_yz)
        if verts_list:
            poly = PolyCollection(
                verts_list,
                facecolors=TISSUE_COLORS[tissue],
                edgecolors="k",
                linewidths=0.1,
                alpha=0.85,
            )
            ax.add_collection(poly)

    # Overlay displacement magnitude if requested
    if show_disp and displacements is not None:
        disp_mag = np.linalg.norm(displacements, axis=1)
        vmax = max(disp_mag.max(), 1e-6)
        norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
        cmap = cm.viridis

        verts_list = []
        face_colors = []
        for ei in elem_indices:
            for face in tet_faces(elements[ei]):
                tri_yz = nodes[face][:, [2, 1]]
                verts_list.append(tri_yz)
                avg_disp = disp_mag[face].mean()
                face_colors.append(cmap(norm(avg_disp)))

        poly = PolyCollection(
            verts_list,
            facecolors=face_colors,
            edgecolors="k",
            linewidths=0.1,
            alpha=0.9,
        )
        ax.add_collection(poly)
        return norm, cmap

    ax.set_aspect("equal")
    ax.set_xlabel("z [m]", fontsize=9)
    ax.set_ylabel("y [m]", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=7)
    return None, None


# ── Main ──────────────────────────────────────────────────────────────────

def run_palpation(output_dir: str | None = None):
    """Run the palpation simulation and generate the rendering."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "..", "tests", "artifacts")
    os.makedirs(output_dir, exist_ok=True)

    # ── Build mesh ────────────────────────────────────────────
    nx, ny, nz = 6, 12, 6
    nodes, elements = make_block_mesh(nx, ny, nz)
    labels = classify_elements(nodes, elements)
    material = build_material(labels)

    tissue_counts = {}
    for t in TISSUE_COLORS:
        tissue_counts[t] = np.sum(labels == t)
    print(f"Mesh: {len(nodes)} nodes, {len(elements)} elements")
    for t, c in tissue_counts.items():
        print(f"  {t:8s}: {c} elements")

    # ── Boundary conditions ───────────────────────────────────
    fixed_indices = get_fixed_nodes(nodes, labels, elements)
    bc = FixedBC(fixed_indices)
    print(f"Fixed nodes: {len(fixed_indices)}")

    # ── Palpation loading ─────────────────────────────────────
    # Press on a patch of skin surface, centered away from the rib
    palp_center_x = BLOCK_SIZE / 2.0
    palp_center_z = 0.035  # away from rib (rib is at z > 0.07)
    palp_radius = 0.020    # 2 cm radius patch
    palp_nodes = get_palpation_nodes(nodes, palp_center_x, palp_center_z,
                                     palp_radius)
    print(f"Palpation nodes: {len(palp_nodes)}")

    # Total force distributed over the patch, pushing down (-y)
    # ~8 N produces approximately 2 cm skin indentation with these materials
    total_force = 8.0  # Newtons
    if len(palp_nodes) > 0:
        force_per_node = np.array([0.0, -total_force / len(palp_nodes), 0.0])
    else:
        force_per_node = np.array([0.0, -total_force, 0.0])

    loading = PointForce(palp_nodes, force_per_node)

    # ── Build model ───────────────────────────────────────────
    model = TissueModel(
        nodes=nodes,
        elements=elements,
        material=material,
        boundary_conditions=[bc],
        loading_conditions=[loading],
    )

    # ── Solve ─────────────────────────────────────────────────
    print("Solving (VBD quasi-static)...")
    solver = TissueSolver(
        model, dt=5e-4, num_substeps=1, iterations=30,
        solver_type="vbd", k_damp=0.1,
    )
    result = solver.solve_static(max_frames=4000, tol=1e-3)
    max_d = result.max_displacement()
    print(f"  converged={result.converged}, max_disp={max_d * 100:.2f} cm")

    # ── Cross-section rendering ───────────────────────────────
    x_mid = BLOCK_SIZE / 2.0
    dx = BLOCK_SIZE / nx
    slice_elems = get_slice_elements(nodes, elements, x_mid, dx * 0.6)
    print(f"Cross-section elements: {len(slice_elems)}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Undeformed, tissue type
    ax = axes[0]
    plot_cross_section(ax, nodes, elements, slice_elems, labels,
                       "Undeformed\n(tissue layers)")
    ax.set_xlim(-0.005, BLOCK_SIZE + 0.005)
    ax.set_ylim(-0.005, BLOCK_SIZE + 0.005)
    ax.set_aspect("equal")

    # Mark palpation zone on skin surface
    palp_z_lo = palp_center_z - palp_radius
    palp_z_hi = palp_center_z + palp_radius
    ax.annotate("", xy=(palp_center_z, BLOCK_SIZE),
                xytext=(palp_center_z, BLOCK_SIZE + 0.012),
                arrowprops=dict(arrowstyle="->", color="blue", lw=2))
    ax.text(palp_center_z, BLOCK_SIZE + 0.014, "palpation\nforce",
            ha="center", va="bottom", fontsize=8, color="blue")

    # Panel 2: Deformed, tissue type
    ax = axes[1]
    plot_cross_section(ax, result.positions, elements, slice_elems, labels,
                       f"After palpation (F={total_force:.0f} N)\n"
                       f"max indentation: {max_d * 100:.1f} cm")
    ax.set_xlim(-0.005, BLOCK_SIZE + 0.005)
    ax.set_ylim(-0.005, BLOCK_SIZE + 0.005)
    ax.set_aspect("equal")

    # Panel 3: Deformed, displacement magnitude
    ax = axes[2]
    norm_d, cmap_d = plot_cross_section(
        ax, result.positions, elements, slice_elems, labels,
        "Displacement magnitude [m]",
        show_disp=True, displacements=result.displacements,
    )
    ax.set_xlim(-0.005, BLOCK_SIZE + 0.005)
    ax.set_ylim(-0.005, BLOCK_SIZE + 0.005)
    ax.set_aspect("equal")
    ax.set_xlabel("z [m]", fontsize=9)
    ax.set_ylabel("y [m]", fontsize=9)
    ax.set_title("Displacement magnitude [m]", fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=7)
    if norm_d is not None:
        cb = fig.colorbar(cm.ScalarMappable(norm=norm_d, cmap=cmap_d),
                          ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("|u| [m]", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    # Legend
    legend_patches = [Patch(facecolor=TISSUE_COLORS[t], edgecolor="k",
                            linewidth=0.5, label=t.capitalize())
                      for t in TISSUE_COLORS]
    axes[0].legend(handles=legend_patches, loc="lower left", fontsize=8,
                   framealpha=0.9)

    fig.suptitle(
        "Abdominal Palpation Simulation (cross-section at x = 5 cm)\n"
        f"E: skin={TISSUE_PROPS['skin'][0]/1e3:.0f} kPa, "
        f"fat={TISSUE_PROPS['fat'][0]/1e3:.0f} kPa, "
        f"muscle={TISSUE_PROPS['muscle'][0]/1e3:.0f} kPa, "
        f"liver={TISSUE_PROPS['liver'][0]/1e3:.0f} kPa  |  "
        f"rib = rigid",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()

    out_path = os.path.join(output_dir, "palpation.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    file_size = os.path.getsize(out_path)
    print(f"Saved: {out_path} ({file_size / 1024:.0f} KB)")
    return out_path


if __name__ == "__main__":
    run_palpation()
