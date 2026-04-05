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
        return np.zeros(ng**3, dtype=np.float32), []

    # The cut surface is a ribbon defined by:
    #   - The curve path (a polyline in 3D on the tissue surface)
    #   - An extrusion into the tissue (depth direction)
    # The "side" is determined by cross(tangent, depth_dir), giving a
    # normal that separates left and right of the cutting ribbon.
    #
    # depth_dir must point OUTWARD from the tissue surface (away from
    # the body interior).  The in_depth check then admits grid nodes
    # with negative d_depth (= into tissue) up to depth_m.
    #
    # The outward direction is derived from the tissue body SDF gradient
    # (computed from the EDT of the tissue+bone occupancy grid).  This
    # gives the true surface outward normal at each curve point.

    # Tissue body SDF gradient for surface normal estimation
    tissue_grad_np = None
    if hasattr(sim, 'tissue_sdf_grad') and sim.tissue_sdf_grad is not None:
        tissue_grad_np = sim.tissue_sdf_grad.numpy()  # (ng^3, 3)

    # Grid node positions
    gi = np.arange(ng, dtype=np.float32)
    gx, gy, gz = np.meshgrid(gi * dx, gi * dx, gi * dx, indexing='ij')
    grid_pos = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)  # (ng^3, 3)

    sdf = np.zeros(ng**3, dtype=np.float32)
    depth_m = depth_mm / 1000.0

    ref_cut_normal = None   # first segment sets the reference sign
    ribbon_quads = []       # (p0, p1, p1_deep, p0_deep) per segment

    for seg_idx in range(len(pts_m) - 1):
        p0 = pts_m[seg_idx]
        p1 = pts_m[seg_idx + 1]
        seg = p1 - p0
        seg_len = np.linalg.norm(seg)
        if seg_len < 1e-12:
            continue
        seg_dir = seg / seg_len

        # --- Per-segment depth direction (outward from tissue) --------
        if cut_normal is not None:
            # User-supplied side normal: derive depth_dir from it
            local_depth = np.cross(cut_normal, seg_dir)
            nrm = np.linalg.norm(local_depth)
            if nrm < 1e-6:
                local_depth = np.array([0.0, 0.0, 1.0])
            else:
                local_depth = local_depth / nrm
            local_normal = cut_normal.copy()
        elif tissue_grad_np is not None:
            # Sample tissue body SDF gradient at segment midpoint.
            # The gradient points outward from the tissue surface —
            # the true surface normal at the curve location.
            mid = (p0 + p1) / 2.0
            mi = np.clip(int(round(mid[0] * inv_dx)), 0, ng - 1)
            mj = np.clip(int(round(mid[1] * inv_dx)), 0, ng - 1)
            mk = np.clip(int(round(mid[2] * inv_dx)), 0, ng - 1)
            flat_idx = mi * ng * ng + mj * ng + mk
            grad = tissue_grad_np[flat_idx]
            nrm = np.linalg.norm(grad)
            if nrm > 1e-6:
                local_depth = grad / nrm       # outward from tissue surface
            else:
                local_depth = np.array([0.0, 0.0, 1.0])  # fallback

            local_normal = np.cross(seg_dir, local_depth)
            nrm = np.linalg.norm(local_normal)
            if nrm < 1e-6:
                local_normal = np.cross(seg_dir, np.array([1.0, 0.0, 0.0]))
                nrm = np.linalg.norm(local_normal)
            local_normal = local_normal / nrm
        else:
            # No tissue SDF available — use +Z (superior in RAS = outward
            # on the top of the head).  Correct only for vertex cuts.
            local_depth = np.array([0.0, 0.0, 1.0])
            local_normal = np.cross(seg_dir, local_depth)
            nrm = np.linalg.norm(local_normal)
            if nrm < 1e-6:
                local_normal = np.cross(seg_dir, np.array([1.0, 0.0, 0.0]))
                nrm = np.linalg.norm(local_normal)
            local_normal = local_normal / nrm

        # Ensure consistent cut_normal sign across all segments so
        # the positive/negative side assignment doesn't flip mid-cut.
        if ref_cut_normal is None:
            ref_cut_normal = local_normal.copy()
        elif np.dot(local_normal, ref_cut_normal) < 0:
            local_normal = -local_normal

        if seg_idx == 0:
            print(f"  seg[0] depth_dir={local_depth}, "
                  f"cut_normal={local_normal}")

        # --- Collect ribbon quad geometry (for visualization) ---------
        # Each segment generates a quad: two points on the surface,
        # two points at depth.  Store in sim coords [m].
        ribbon_quads.append((p0.copy(), p1.copy(),
                             p1 - local_depth * depth_m,
                             p0 - local_depth * depth_m))

        # --- Compute SDF for this segment ----------------------------
        v = grid_pos - p0                          # (ng^3, 3)
        t = np.clip(v @ seg_dir / seg_len, 0, 1)  # parametric position
        proj = p0 + np.outer(t, seg)               # nearest point on segment
        delta = grid_pos - proj                    # (ng^3, 3)

        d_normal = delta @ local_normal            # signed side distance

        # Every grid node gets a side assignment — the cut plane divides
        # the entire grid into two half-spaces.  Any zero-SDF node acts
        # as a coupling bridge that lets momentum transfer across the
        # cut, defeating the separation.  The CRESSim-MPM approach
        # requires ALL nodes to have a sign.
        candidate_sdf = d_normal

        # Keep the value closest to the cut surface per node
        unassigned = sdf == 0.0
        closer = (sdf != 0.0) & (np.abs(candidate_sdf) < np.abs(sdf))
        sdf[unassigned | closer] = candidate_sdf[unassigned | closer]

    n_pos = int((sdf > 0).sum())
    n_neg = int((sdf < 0).sum())
    print(f"build_scalpel_sdf: {n_pos} pos, {n_neg} neg, "
          f"{ng**3 - n_pos - n_neg} neutral grid nodes")

    # Convert ribbon quads from sim coords [m] to RAS mm for visualization
    ribbon_quads_ras = []
    for q in ribbon_quads:
        ribbon_quads_ras.append(tuple(v * 1000.0 + offset_mm for v in q))

    return sdf, ribbon_quads_ras


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

    return build_scalpel_sdf(sim, pts, depth_mm)  # (sdf, ribbon_quads_ras)


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

        result = build_scalpel_sdf_from_curve_node(
            self.mpm_wrapper, curve_node, self.depth_mm)

        if result is not None:
            sdf, ribbon_quads_ras = result
            self.mpm_wrapper.sim.apply_cut(sdf)

            # Create a model of the cut ribbon geometry
            self._create_cut_model(ribbon_quads_ras, curve_node.GetName())

            # Update particle colors to show cut sides
            if hasattr(self.mpm_wrapper, 'rebuild_colors'):
                self.mpm_wrapper.rebuild_colors()

            # Restart the simulation loop if it was idle
            if hasattr(self.mpm_wrapper, '_loop_running'):
                self.mpm_wrapper._idle_ticks = 0
                if not self.mpm_wrapper._loop_running:
                    self.mpm_wrapper.start_simulation_loop()

    def _create_cut_model(self, ribbon_quads_ras, curve_name):
        """Create a VTK model node showing the cut ribbon surface."""
        import slicer
        import vtk

        points = vtk.vtkPoints()
        polys  = vtk.vtkCellArray()

        for q in ribbon_quads_ras:
            base = points.GetNumberOfPoints()
            for v in q:  # 4 corners of the quad
                points.InsertNextPoint(float(v[0]), float(v[1]), float(v[2]))
            quad = vtk.vtkQuad()
            for i in range(4):
                quad.GetPointIds().SetId(i, base + i)
            polys.InsertNextCell(quad)

        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetPolys(polys)

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(poly)
        normals.Update()

        name = f"Cut_{curve_name}"
        old = slicer.mrmlScene.GetFirstNodeByName(name)
        if old:
            slicer.mrmlScene.RemoveNode(old)

        model = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', name)
        model.SetAndObservePolyData(normals.GetOutput())
        model.CreateDefaultDisplayNodes()
        dn = model.GetDisplayNode()
        dn.SetColor(1.0, 1.0, 0.0)  # yellow
        dn.SetOpacity(0.5)
        dn.SetBackfaceCulling(False)
        print(f"CurveObserver: created cut model '{name}' "
              f"({len(ribbon_quads_ras)} quads)")

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
