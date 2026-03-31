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

MATERIAL         = MPMMaterial(E=3_000.0, nu=0.45, rho=1_060.0,
                               k_elastin=1e-3, k_collagen=3e-3, collagen_crimp=0.03,
                               k_curve=10.0)
VELOCITY_DAMPING = 0.999


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

        self.sim.step(GRAVITY)
        for _ in range(499):
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
        self._probe_params          = None
        self._probe_ticks_remaining = 0
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
            if self._probe_params is not None and self._probe_ticks_remaining > 0:
                pp = self._probe_params
                self.sim.step_with_probe(GRAVITY, pp['center'], pp['pressure_pa'],
                                         pp['normal'], pp['radius'])
            else:
                self.sim.step(GRAVITY)
        if self._probe_ticks_remaining > 0:
            self._probe_ticks_remaining -= 1
            if self._probe_ticks_remaining == 0:
                self._probe_params = None

        pos_now = self.sim.get_positions()
        self.update_model()

        if self._prev_tick_pos is not None and self._probe_ticks_remaining == 0:
            if np.array_equal(pos_now, self._prev_tick_pos):
                self._idle_ticks += 1
            else:
                self._idle_ticks = 0
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
    # Palpation
    # ------------------------------------------------------------------

    def apply_palpation(self, pressure_pa=200.0, n_steps=500, show_every=0):
        self._loop_running = False   # pause loop so timer ticks don't interfere
        probe_center = self._palp_pos_mm / 1000.0
        probe_normal = np.array([0.0, -1.0, 0.0])
        probe_radius = 10.0 * self.sim.dx
        for i in range(n_steps):
            self.sim.step_with_probe(GRAVITY, probe_center, pressure_pa,
                                     probe_normal, probe_radius)
            if show_every > 0 and i % show_every == show_every - 1:
                self.update_model()
                try:
                    import slicer
                    slicer.app.processEvents()
                except ImportError:
                    pass

    def recover(self, n_steps=1500, show_every=5):
        self._loop_running = False   # pause loop during scripted recovery
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
        self._palp_ref_pos = np.array([x, y, z])
        self.fiducial_list.AddObserver(
            self.fiducial_list.PointModifiedEvent,
            lambda c, e: self._on_fiducial_moved(c))

    def _on_fiducial_moved(self, fiducial_list):
        p = [0.0, 0.0, 0.0]
        fiducial_list.GetNthControlPointPosition(0, p)
        delta_mm = np.array(p) - self._palp_ref_pos
        dist_mm  = float(np.linalg.norm(delta_mm))
        if dist_mm < 0.5:
            return
        k_probe_pa_per_mm = 500_000.0
        self._probe_params = {
            'center':      self._palp_pos_mm / 1000.0,
            'pressure_pa': k_probe_pa_per_mm * dist_mm,
            'normal':      delta_mm / dist_mm,
            'radius':      10.0 * self.sim.dx,
        }
        ticks_per_second = 1000 // self._tick_interval_ms
        self._probe_ticks_remaining = 2 * ticks_per_second
        self._palp_pos_mm  = np.array(p)
        self._palp_ref_pos = np.array(p)
        self._idle_ticks = 0
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
