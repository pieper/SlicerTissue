"""Explicit MLS-MPM tissue block experiment for 3D Slicer."""

from __future__ import annotations
import os, sys
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
_NEWTON_DIR = os.path.dirname(_SCRIPT_DIR) if os.path.basename(_SCRIPT_DIR) == 'examples' else _SCRIPT_DIR
_SRC_DIR    = os.path.join(_NEWTON_DIR, 'src')
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import warp as wp
from newton_tissue.mpm import MPMMaterial, MPMSimulator

BLOCK_SIZE = 0.08
N_GRID     = 32
PPC        = 2
DT         = 2e-4
GRAVITY    = np.array([0.0, -9.8, 0.0])

# Material calibrated against soft-tissue mechanical testing literature:
#   E = 10 kPa  — soft connective tissue / glandular parenchyma; sits within the
#                 measured range for breast glandular (Samani 2007: 3–28 kPa) and
#                 low-to-mid prostate peripheral zone (Zhang 2010: 17–24 kPa);
#                 collagen fiber network stiffens tissue to ~48 kPa at moderate strain.
#   nu = 0.48   — near-incompressible hydrated tissue (Palmeri 2020, Vossoughi 1994).
#   k_elastin   — 0.05 N/m: low-level bidirectional bonds maintain lattice topology
#                 without adding bulk stiffness (elastin is dominant at <2% strain).
#   k_collagen  — 0.25 N/m tension-only; effective modulus ~48 kPa at 5% post-crimp
#                 (Krouskop 1998 high-strain glandular: 100–220 kPa with dense stroma).
#   crimp=0.05  — 5% toe region (Frontiers Materials 2021: 2–10% for soft connective).
#   k_curve=0   — disabled: Updated Lagrangian mode handles large deformation correctly;
#                 the fiber network (elastin+collagen bonds) provides lattice connectivity.
#   damping     — 0.995: near-critically damped (Q≈1.1) vs overdamped (Q≈0.54 at 0.99).
MATERIAL         = MPMMaterial(E=10_000.0, nu=0.48, rho=1_060.0,
                               k_elastin=0.05, k_collagen=0.25, collagen_crimp=0.05,
                               k_curve=0.0)
VELOCITY_DAMPING = 0.995

# Probe geometry: rigid sphere contact for displacement-controlled palpation.
# Sphere radius ≈ fingertip: 4×dx ≈ 10 mm.
PROBE_RADIUS_DX  = 4.0     # probe radius in multiples of grid spacing


class MPMTissueBlock:

    def __init__(self, device=None):
        wp.init()
        if device is None:
            device = "cpu"
            try:
                _ = wp.zeros(1, dtype=float, device="cuda:0")
                device = "cuda:0"
            except Exception:
                pass
        self.device = device

        lo = np.zeros(3)
        hi = np.full(3, BLOCK_SIZE)

        self.sim = MPMSimulator(
            block_lo=lo, block_hi=hi,
            n_grid=N_GRID, dt=DT,
            material=MATERIAL,
            device=device,
            velocity_damping=VELOCITY_DAMPING,
            total_lagrangian=(MATERIAL.k_curve > 0.0),
        )
        self.sim.initialize_block_particles(lo=lo, hi=hi, ppc=PPC,
                                            fixed_y_max=float(lo[1]) + 2.0 * self.sim.dx)

        self._probe_radius = PROBE_RADIUS_DX * self.sim.dx   # [m]

        pos = self.sim.get_positions()
        center   = BLOCK_SIZE / 2.0
        top_y    = BLOCK_SIZE - 3.0 * self.sim.dx
        xz_range = 4.0 * self.sim.dx
        self._palp_mask = (
            (pos[:, 1] > top_y) &
            (np.abs(pos[:, 0] - center) < xz_range) &
            (np.abs(pos[:, 2] - center) < xz_range)
        )
        palp_pos = pos[self._palp_mask].mean(axis=0) * 1000.0
        self._palp_pos_mm = palp_pos.copy()

        # The fiducial sits at the tissue surface.  The probe sphere centre
        # is offset upward by sphere_radius so it just touches the surface
        # at rest and pushes inward when the fiducial is dragged down.
        self._surface_y = float(BLOCK_SIZE)   # top of block [m]

        self.sim.step(GRAVITY)
        for _ in range(749):  # 750 total: enough for damping=0.995 to reach <10um/tick
            self.sim.step(GRAVITY)
        self.sim.sample_equilibrium()

        self.vtk_model    = None
        self._vtk_points  = None
        self._vtk_poly    = None
        self.fiducial_list = None
        self._updating    = False
        self.grid_volume  = None

        # simulation loop state
        self._loop_running          = False
        self._steps_per_tick        = 10
        self._tick_interval_ms      = 50
        self._idle_ticks            = 0
        self._idle_ticks_to_stop    = 5
        self._contact_sphere        = None    # {'center': [m], 'radius': [m]} or None
        self._prev_tick_pos         = None
        self._observer_tags         = []

    def run(self):
        try:
            import slicer
            _has_slicer = True
        except ImportError:
            _has_slicer = False
        if _has_slicer:
            self._create_vtk_model()
            self._create_fiducial()
            self._create_grid_volume()
            self.update_model()
            self._setup_view()
            self.start_simulation_loop()

    def step_and_update(self, n_steps=10):
        for _ in range(n_steps):
            self.sim.step(GRAVITY)
        self.update_model()

    def update_model(self):
        if self._vtk_points is not None:
            import vtk, vtk.util.numpy_support as ns
            pos_mm = (self.sim.get_positions() * 1000.0).astype(np.float32)
            self._vtk_points.SetData(ns.numpy_to_vtk(pos_mm, deep=True, array_type=vtk.VTK_FLOAT))
            self._vtk_points.Modified()
            self._vtk_poly.Modified()
            if self.vtk_model:
                self.vtk_model.GetPolyData().Modified()
        self._update_grid_volume()

    # ------------------------------------------------------------------
    # Continuous simulation loop
    # ------------------------------------------------------------------

    def start_simulation_loop(self):
        if self._loop_running:
            return
        self._loop_running = True
        self._idle_ticks = 0
        self._prev_tick_pos = None
        try:
            import slicer
            tag = slicer.mrmlScene.AddObserver(
                slicer.mrmlScene.StartCloseEvent,
                self._on_scene_about_to_close)
            self._observer_tags.append((slicer.mrmlScene, tag))
        except Exception:
            pass
        self._schedule_tick()

    def stop_simulation_loop(self):
        self._loop_running = False
        for node, tag in self._observer_tags:
            try:
                node.RemoveObserver(tag)
            except Exception:
                pass
        self._observer_tags.clear()

    def _schedule_tick(self):
        try:
            import qt
            qt.QTimer.singleShot(self._tick_interval_ms, self._timer_tick)
        except Exception:
            pass

    def _timer_tick(self):
        if not self._loop_running:
            return
        try:
            import slicer
            if self.vtk_model and not slicer.mrmlScene.IsNodePresent(self.vtk_model):
                self.stop_simulation_loop()
                return
        except Exception:
            self.stop_simulation_loop()
            return

        for _ in range(self._steps_per_tick):
            if self._contact_sphere is not None:
                cs = self._contact_sphere
                self.sim.step_with_contact(GRAVITY, cs['center'], cs['radius'])
            else:
                self.sim.step(GRAVITY)

        pos_now = self.sim.get_positions()
        self.update_model()

        if self._prev_tick_pos is not None:
            # Threshold-based idle check: stop when tissue reaches equilibrium.
            # Works with or without active probe (probe holds tissue in place).
            if float(np.abs(pos_now - self._prev_tick_pos).max()) < 2e-5:
                self._idle_ticks += 1
            else:
                self._idle_ticks = 0
        self._prev_tick_pos = pos_now

        if self._idle_ticks >= self._idle_ticks_to_stop:
            self._loop_running = False
            return
        self._schedule_tick()

    def _on_scene_about_to_close(self, _scene, _event):
        self.stop_simulation_loop()

    # ------------------------------------------------------------------
    # Palpation (displacement-controlled rigid sphere contact)
    # ------------------------------------------------------------------

    def _sphere_center_for_fiducial(self, fiducial_pos_m):
        """Compute rigid sphere centre from fiducial position.

        The sphere centre sits one radius above the fiducial so that at rest
        (fiducial at tissue surface) the sphere just touches the surface.
        """
        return fiducial_pos_m + np.array([0.0, self._probe_radius, 0.0])

    def apply_palpation(self, push_depth_m=0.015, n_steps=500, show_every=0):
        """Push a rigid sphere into tissue by push_depth_m, then hold."""
        self._loop_running = False
        rest_m = self._palp_pos_mm / 1000.0
        for i in range(n_steps):
            frac = min(1.0, (i + 1) / n_steps)
            fid_pos = rest_m + np.array([0.0, -push_depth_m * frac, 0.0])
            sphere_c = self._sphere_center_for_fiducial(fid_pos)
            self.sim.step_with_contact(GRAVITY, sphere_c, self._probe_radius)
            if show_every > 0 and i % show_every == show_every - 1:
                self.update_model()
                try:
                    import slicer
                    slicer.app.processEvents()
                except ImportError:
                    pass

    def recover(self, n_steps=1500, show_every=5):
        """Release probe and let tissue recover elastically."""
        self._loop_running = False
        self._contact_sphere = None
        for i in range(n_steps):
            self.sim.step(GRAVITY)
            if show_every > 0 and i % show_every == show_every - 1:
                self.update_model()
                try:
                    import slicer
                    slicer.app.processEvents()
                except ImportError:
                    pass

    # ------------------------------------------------------------------
    # Slicer helpers
    # ------------------------------------------------------------------

    def _create_grid_volume(self):
        import slicer
        dx_mm = float(self.sim.dx * 1000.0)
        arr = self._grid_mass_uint8()
        self.grid_volume = slicer.util.addVolumeFromArray(arr, name='MPMGridDensity')
        self.grid_volume.SetSpacing(dx_mm, dx_mm, dx_mm)
        self.grid_volume.SetOrigin(0.0, 0.0, 0.0)
        dn = self.grid_volume.GetDisplayNode()
        dn.SetAutoWindowLevel(True)
        dn.SetAndObserveColorNodeID('vtkMRMLColorTableNodeGrey')
        slicer.util.setSliceViewerLayers(background=self.grid_volume)

    def _grid_mass_uint8(self):
        ng = self.sim.n_grid
        gm = self.sim.grid_m.numpy().reshape(ng, ng, ng)
        lo, hi = float(gm.min()), float(gm.max())
        if hi > lo:
            norm = ((gm - lo) / (hi - lo) * 255).astype(np.uint8)
        else:
            norm = np.zeros_like(gm, dtype=np.uint8)
        return norm.transpose(2, 1, 0)

    def _update_grid_volume(self):
        if self.grid_volume is None:
            return
        import slicer
        slicer.util.updateVolumeFromArray(self.grid_volume, self._grid_mass_uint8())
        self.grid_volume.GetDisplayNode().SetAutoWindowLevel(True)

    def _build_grid_colors(self):
        x0 = self.sim.x0.numpy()
        d  = 2.0 * self.sim.dx
        ci = (x0[:, 0] / d).astype(int) % 2
        cj = (x0[:, 1] / d).astype(int) % 2
        ck = (x0[:, 2] / d).astype(int) % 2
        idx = ci * 4 + cj * 2 + ck
        palette = np.array([
            [220, 150,  80], [180, 100,  50], [100, 160, 200], [ 60, 110, 170],
            [190, 200, 100], [130, 170,  70], [200, 120, 180], [150,  80, 140],
        ], dtype=np.uint8)
        return palette[idx]

    def _create_vtk_model(self):
        import slicer, vtk, vtk.util.numpy_support as ns
        pos_mm = (self.sim.get_positions() * 1000.0).astype(np.float32)
        pts = vtk.vtkPoints()
        pts.SetData(ns.numpy_to_vtk(pos_mm, deep=True, array_type=vtk.VTK_FLOAT))
        colors_np = self._build_grid_colors()
        color_arr = ns.numpy_to_vtk(colors_np, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        color_arr.SetName("Colors")
        color_arr.SetNumberOfComponents(3)
        src_poly = vtk.vtkPolyData()
        src_poly.SetPoints(pts)
        src_poly.GetPointData().SetScalars(color_arr)
        glyph = vtk.vtkVertexGlyphFilter()
        glyph.SetInputData(src_poly)
        glyph.Update()
        poly = vtk.vtkPolyData()
        poly.DeepCopy(glyph.GetOutput())
        self.vtk_model = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', 'MPMTissueBlock')
        self.vtk_model.SetAndObservePolyData(poly)
        self.vtk_model.CreateDefaultDisplayNodes()
        dn = self.vtk_model.GetDisplayNode()
        dn.SetPointSize(3)
        dn.SetScalarVisibility(True)
        dn.SetActiveScalarName("Colors")
        dn.SetScalarRangeFlagFromString("UseDirectMapping")
        self._vtk_points = poly.GetPoints()
        self._vtk_poly   = poly

    def _setup_view(self):
        import slicer
        lm = slicer.app.layoutManager()
        lm.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        slicer.app.processEvents()
        slicer.util.resetThreeDViews()
        slicer.app.processEvents()
        threeDWidget = lm.threeDWidget(0)
        rw = threeDWidget.threeDView().renderWindow()
        renderer = rw.GetRenderers().GetFirstRenderer()
        cam = renderer.GetActiveCamera()
        cam.Elevation(20)
        cam.Azimuth(-20)
        cam.OrthogonalizeViewUp()
        renderer.ResetCameraClippingRange()
        rw.Render()
        for color in ['Red', 'Green', 'Yellow']:
            sliceNode = lm.sliceWidget(color).sliceLogic().GetSliceNode()
            sliceNode.JumpSliceByCentering(40.0, 40.0, 40.0)
        slicer.app.processEvents()

    def _create_fiducial(self):
        import slicer
        self.fiducial_list = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLMarkupsFiducialNode', 'MPMPalpationPoint')
        dn = self.fiducial_list.GetDisplayNode()
        dn.SetGlyphTypeFromString('Sphere3D')
        dn.SetGlyphScale(4.0)
        dn.SetColor(1.0, 0.3, 0.3)
        x, y, z = self._palp_pos_mm
        self.fiducial_list.AddControlPoint(x, y, z)
        self.fiducial_list.SetNthControlPointLabel(0, 'palp')
        self._palp_ref_pos   = np.array([x, y, z])
        self._palp_initial_mm = np.array([x, y, z])   # never updated — rest position
        self.fiducial_list.AddObserver(
            self.fiducial_list.PointModifiedEvent,
            lambda c, e: self._on_fiducial_moved(c))

    def _on_fiducial_moved(self, fiducial_list):
        p = [0.0, 0.0, 0.0]
        fiducial_list.GetNthControlPointPosition(0, p)
        p_mm     = np.array(p)
        delta_mm = p_mm - self._palp_initial_mm   # displacement from rest
        dist_mm  = float(np.linalg.norm(delta_mm))

        if dist_mm < 0.5:
            # Probe returned to rest — release contact
            self._contact_sphere = None
            self._idle_ticks     = 0
            if not self._loop_running:
                self.start_simulation_loop()
            return

        # Displacement-controlled rigid sphere contact.
        # The sphere centre is offset upward by probe_radius from the fiducial
        # so it just touches the tissue surface at rest.  Dragging the fiducial
        # into the tissue pushes the sphere in, displacing particles.
        fid_pos_m = p_mm / 1000.0
        sphere_c  = self._sphere_center_for_fiducial(fid_pos_m)
        self._contact_sphere = {
            'center': sphere_c,
            'radius': self._probe_radius,
        }
        self._palp_pos_mm  = p_mm
        self._palp_ref_pos = p_mm
        self._idle_ticks   = 0
        if not self._loop_running:
            self.start_simulation_loop()


if __name__ == '__main__':
    print("Running MPMTissueBlock standalone...")
    sim = MPMTissueBlock(device="cpu")
    for _ in range(200):
        sim.sim.step(GRAVITY)
    pos  = sim.sim.get_positions()
    free = ~sim.sim.fixed.numpy().astype(bool)
    print(f"  Free particles: {free.sum()}")
    print(f"  Mean Y displacement: {(pos[free,1] - sim.sim.x0.numpy()[free,1]).mean()*1000:.2f} mm")
    print("Done.")
