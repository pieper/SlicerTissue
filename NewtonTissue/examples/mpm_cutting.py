"""Cutting tools for MPM tissue simulation in 3D Slicer.

Provides:
  - ScalpelCut: generates a blade surface from a markup open curve,
    extending perpendicular to the tissue surface down to bone depth.
  - CurveObserver: watches the Slicer scene for completed open curves
    and automatically applies scalpel cuts to the active MPM simulation.

Based on the side-aware transfer approach from:
  Ou & Tavakoli, "CRESSim-MPM: A Material Point Method Library for
  Surgical Soft Body Simulation with Cutting and Suturing",
  arXiv:2502.18437v3, 2025.
"""

from __future__ import annotations
import numpy as np


def build_scalpel_sdf(sim, curve_points_ras_mm, depth_mm=20.0,
                      cut_normal=None):
    """Build a signed distance field for a scalpel cut on the MPM grid.

    The cut surface is a thin ribbon defined by the curve, extruded
    perpendicular to the tissue surface.  Points on one side of the
    ribbon get positive SDF, the other side gets negative.

    Args:
        sim:                MPMSimulator instance (for grid params).
        curve_points_ras_mm: (N, 3) array of curve points in RAS mm.
        depth_mm:           How deep below the surface the cut extends [mm].
        cut_normal:         Optional (3,) normal to the cutting plane.
                            If None, estimated from the curve and gravity.

    Returns:
        sdf_np:  (n_grid^3,) float32 array — signed distance on the MPM grid.
    """
    ng     = sim.n_grid
    dx     = sim.dx
    inv_dx = sim.inv_dx

    # Convert curve points to simulation coordinates [m]
    if hasattr(sim, '_ras_offset_mm'):
        # MPMCTHead stores the offset
        offset_mm = sim._ras_offset_mm
    else:
        offset_mm = np.zeros(3)

    # For external callers passing an MPMCTHead wrapper, check there too
    pts_m = (curve_points_ras_mm - offset_mm) / 1000.0  # shape (N, 3)

    if len(pts_m) < 2:
        return np.zeros(ng**3, dtype=np.float32)

    # Compute the cut plane.  The cut surface is defined by:
    #   - The curve path (a polyline in 3D)
    #   - An extrusion direction (depth direction, typically inferior = -Z in RAS)
    # The "side" is determined by the cross product of the curve tangent
    # and the depth direction, giving a normal to the cutting ribbon.

    # Curve tangent: average of segment directions
    segments = np.diff(pts_m, axis=0)          # (N-1, 3)
    tangent = segments.mean(axis=0)
    tangent /= max(np.linalg.norm(tangent), 1e-12)

    # Depth direction: default to -Z (inferior in RAS sim coords)
    depth_dir = np.array([0.0, 0.0, -1.0])

    # Cut plane normal: perpendicular to both tangent and depth
    if cut_normal is None:
        cut_normal = np.cross(tangent, depth_dir)
        norm = np.linalg.norm(cut_normal)
        if norm < 1e-6:
            # Tangent parallel to depth — use a fallback
            cut_normal = np.cross(tangent, np.array([1.0, 0.0, 0.0]))
            norm = np.linalg.norm(cut_normal)
        cut_normal /= norm

    # Build the SDF on the grid: signed distance from each grid node
    # to the cutting ribbon (polyline + depth extrusion).
    # For each grid node, project onto the nearest point on the polyline,
    # check depth, and compute signed distance using cut_normal.

    # Grid node positions
    gi = np.arange(ng, dtype=np.float32)
    gx, gy, gz = np.meshgrid(gi * dx, gi * dx, gi * dx, indexing='ij')
    grid_pos = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)  # (ng^3, 3)

    # For each grid node, find distance to the cutting ribbon.
    # Default 0.0 = not near any cut.  The side check in _p2g_cut/_g2p_cut
    # uses sign(p_sdf) * sign(g_sdf) < 0 to block transfers.  Zero means
    # "neutral" — transfers are never blocked for zero-SDF nodes.
    sdf = np.zeros(ng**3, dtype=np.float32)

    # Process each segment of the polyline
    depth_m = depth_mm / 1000.0
    for seg_idx in range(len(pts_m) - 1):
        p0 = pts_m[seg_idx]
        p1 = pts_m[seg_idx + 1]
        seg = p1 - p0
        seg_len = np.linalg.norm(seg)
        if seg_len < 1e-12:
            continue
        seg_dir = seg / seg_len

        # Project grid nodes onto this segment
        v = grid_pos - p0                       # (ng^3, 3)
        t = np.clip(v @ seg_dir / seg_len, 0, 1)  # parametric position
        proj = p0 + np.outer(t, seg)            # nearest point on segment

        # Vector from projection to grid node
        delta = grid_pos - proj                 # (ng^3, 3)

        # Decompose into: along cut_normal, along depth_dir, remaining
        d_normal = delta @ cut_normal           # signed distance across cut plane
        d_depth  = delta @ depth_dir            # distance along depth direction

        # The cut ribbon extends from the surface down by depth_m.
        # We want the SDF to be signed (by cut_normal) only for nodes
        # that are within the ribbon's depth range and close to the polyline.
        # Nodes far from the ribbon get large positive SDF (unaffected).

        # Within the ribbon's depth range?
        # The ribbon starts at the curve (d_depth=0) and goes down by depth_m
        in_depth = (d_depth > -depth_m - dx) & (d_depth < dx)

        # Distance along the segment axis: only affect nodes near the
        # polyline (within half a segment length + margin at each end)
        d_along = v @ seg_dir
        near_segment = (d_along > -2 * dx) & (d_along < seg_len + 2 * dx)

        # Active = within depth range AND near the segment.
        # NO lateral distance limit — we assign a side to ALL nodes
        # that are within the ribbon's extent, regardless of how far
        # they are from the cutting plane.  This ensures tissue on
        # both sides of the cut gets proper side assignment.
        active = in_depth & near_segment

        # The SDF value is the signed normal distance from the cut plane.
        # Positive = one side, negative = the other.
        candidate_sdf = d_normal

        # Update SDF: for active nodes, take the value with smallest
        # absolute distance (closest to the cut surface).  Nodes that
        # haven't been assigned yet (sdf=0) always get overwritten.
        unassigned = active & (sdf == 0.0)
        closer = active & (sdf != 0.0) & (np.abs(candidate_sdf) < np.abs(sdf))
        sdf[unassigned | closer] = candidate_sdf[unassigned | closer]

    return sdf


def build_scalpel_sdf_from_curve_node(mpm_wrapper, curve_node,
                                       depth_mm=20.0):
    """Build a scalpel cut SDF from a Slicer markup curve node.

    Args:
        mpm_wrapper:  MPMCTHead (or similar) with .sim and ._ras_offset_mm.
        curve_node:   vtkMRMLMarkupsCurveNode with the cutting path.
        depth_mm:     Cut depth below the surface [mm].

    Returns:
        sdf_np:  (n_grid^3,) float32 signed distance field.
    """
    n_pts = curve_node.GetNumberOfControlPoints()
    if n_pts < 2:
        return None

    pts = np.zeros((n_pts, 3))
    for i in range(n_pts):
        p = [0.0, 0.0, 0.0]
        curve_node.GetNthControlPointPositionWorld(i, p)
        pts[i] = p

    sim = mpm_wrapper.sim
    # Pass the RAS offset through the sim if stored on the wrapper
    if hasattr(mpm_wrapper, '_ras_offset_mm'):
        sim._ras_offset_mm = mpm_wrapper._ras_offset_mm

    return build_scalpel_sdf(sim, pts, depth_mm)


class CurveObserver:
    """Watches the Slicer scene for completed open curves and applies cuts.

    When a new vtkMRMLMarkupsOpenCurveNode is added to the scene and the
    user finishes placing points (PlaceModeFinished), the observer builds
    a scalpel SDF from the curve and applies it to the active MPM simulation.
    """

    def __init__(self, mpm_wrapper, depth_mm=20.0):
        """
        Args:
            mpm_wrapper:  MPMCTHead (or similar) with .sim attribute.
            depth_mm:     Default cut depth [mm].
        """
        import slicer
        self.mpm_wrapper = mpm_wrapper
        self.depth_mm = depth_mm
        self._observer_tags = []
        self._processed_curves = set()

        # Watch the interaction node for EndPlacementEvent — fires when the
        # user finishes placing points on any markup.
        interaction = slicer.app.applicationLogic().GetInteractionNode()
        tag = interaction.AddObserver(
            interaction.EndPlacementEvent,
            self._on_end_placement)
        self._observer_tags.append(('interaction', tag))
        print("CurveObserver: watching for completed open curve markups")

    def _on_end_placement(self, interaction_node, event):
        """Called when the user exits placement mode for any markup."""
        import slicer

        # Find any unprocessed curve nodes with enough points
        nodes = slicer.mrmlScene.GetNodesByClass('vtkMRMLMarkupsCurveNode')
        for i in range(nodes.GetNumberOfItems()):
            node = nodes.GetItemAsObject(i)
            node_id = node.GetID()
            if node_id in self._processed_curves:
                continue
            if node.GetNumberOfControlPoints() >= 2:
                self._apply_cut(node)

    def _apply_cut(self, curve_node):
        """Build SDF from the curve and apply the cut."""
        node_id = curve_node.GetID()
        if node_id in self._processed_curves:
            return
        self._processed_curves.add(node_id)

        print(f"CurveObserver: applying scalpel cut from '{curve_node.GetName()}' "
              f"({curve_node.GetNumberOfControlPoints()} points, "
              f"depth={self.depth_mm} mm)")

        sdf = build_scalpel_sdf_from_curve_node(
            self.mpm_wrapper, curve_node, self.depth_mm)

        if sdf is not None:
            self.mpm_wrapper.sim.apply_cut(sdf)

            # Update particle colors to show cut sides
            if hasattr(self.mpm_wrapper, 'rebuild_colors'):
                self.mpm_wrapper.rebuild_colors()

            # Restart the simulation loop if it was idle
            if hasattr(self.mpm_wrapper, '_loop_running'):
                self.mpm_wrapper._idle_ticks = 0
                if not self.mpm_wrapper._loop_running:
                    self.mpm_wrapper.start_simulation_loop()

    def cleanup(self):
        """Remove all observers."""
        import slicer
        for kind, tag in self._observer_tags:
            if kind == 'interaction':
                interaction = slicer.app.applicationLogic().GetInteractionNode()
                interaction.RemoveObserver(tag)
            elif kind == 'scene':
                slicer.mrmlScene.RemoveObserver(tag)
        self._observer_tags.clear()
