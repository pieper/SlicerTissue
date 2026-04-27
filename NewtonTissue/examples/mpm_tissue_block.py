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
# CFL: dt < dx / c_s.  At E=30 kPa, ν=0.48, ρ=1060 → c_s ≈ 15.8 m/s,
# dx = 2.5 mm gives dt_max ≈ 1.58e-4.  Run at 1e-4 for ~60% margin.
DT         = 1e-4
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
MATERIAL         = MPMMaterial(E=30_000.0, nu=0.48, rho=1_060.0,
                               k_elastin=0.5, k_collagen=2.0, collagen_crimp=0.05,
                               k_curve=0.0)
VELOCITY_DAMPING = 0.995

# Probe geometry: rigid sphere contact for force-driven palpation.
# Sphere radius ≈ palm/fingertip pad: 8×dx ≈ 20 mm.  Larger probe distributes
# contact force over a wider tissue patch so deformation spreads naturally.
PROBE_RADIUS_DX  = 8.0     # probe radius in multiples of grid spacing

# Two-body palpation: a fiducial-controlled "target" expresses the user's
# intended push direction/depth, and a force-driven "finger" sphere trails the
# target via a spring.  The finger never penetrates beyond what tissue
# resistance allows — so dragging the target deeper just makes the finger push
# harder, not deeper, until the spring force matches the tissue reaction.
#
# Spring law is a sublinear power form:
#     F = SPRING_K · ((1 + d/SPRING_D0)^SPRING_P − 1)
#
# Near rest (d ≪ d0): F ≈ (SPRING_K · SPRING_P / SPRING_D0) · d  — linear,
# with effective stiffness k_eff = K·p/d0.  Far from rest: F ∝ d^p.
# SPRING_P = 1 recovers a linear spring; SPRING_P → 0 recovers the
# logarithmic limit.  The default p=0.5 (square-root) sits between linear
# and log: gentler than linear (so fine penetration is controllable) but
# rises faster than log (so the operator still feels increasing resistance
# at deep target offsets).
FINGER_REST_GAP_DX = 0.5    # clearance between finger and tissue at rest [×dx]
FINGER_MASS_KG     = 0.050  # ~50 g, typical fingertip mass
SPRING_K           = 4.0    # N: power-spring scale
SPRING_D0          = 0.005  # m: characteristic distance — 5 mm
SPRING_P           = 0.5    # power-law exponent (0 < p < 1: sublinear)
FINGER_DAMPING     = 4.0    # N·s/m: ≈ 2·sqrt(k_eff·m) at small d
# Stiction: fraction per kick at which in-contact tissue particles are
# velocity-matched to the sphere's tangential motion.  Engages even at zero
# shear velocity, so the finger sticks to the deformed surface instead of
# skidding off when the user drags the fiducial laterally.  0 = frictionless.
FINGER_STICTION    = 0.3


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

        # Rest geometry of the two-body palpation system:
        #   surface  :  top of the tissue block.
        #   finger   :  sphere whose lower surface sits FINGER_REST_GAP above the
        #               tissue at rest, so no contact occurs with the probe idle.
        #   target   :  same xz as finger, same y at rest — the fiducial.  Drag
        #               the fiducial below the rest position to load the spring
        #               connecting target → finger.
        self._surface_y = float(BLOCK_SIZE)
        self._rest_gap_m = FINGER_REST_GAP_DX * self.sim.dx
        finger_y_rest = self._surface_y + self._probe_radius + self._rest_gap_m
        palp_xz = pos[self._palp_mask].mean(axis=0)[[0, 2]]
        self._palp_pos_mm = np.array([
            palp_xz[0] * 1000.0,
            finger_y_rest * 1000.0,
            palp_xz[1] * 1000.0,
        ], dtype=float)

        # 1500 steps × 1e-4 s = 0.15 s of warm-up — long enough for the
        # damping=0.995 envelope to reach <10 µm/tick after the half-step
        # CFL change.
        self.sim.step(GRAVITY)
        for _ in range(1499):
            self.sim.step(GRAVITY)
        self.sim.sample_equilibrium()

        # Two-body palpation state (m, m/s, kg).
        self._target_pos_m = (self._palp_pos_mm / 1000.0).astype(np.float64)
        self._finger_pos_m = self._target_pos_m.copy()
        self._finger_vel_m = np.zeros(3, dtype=np.float64)
        self._finger_mass  = float(FINGER_MASS_KG)
        self._spring_k     = float(SPRING_K)
        self._spring_d0    = float(SPRING_D0)
        self._spring_p     = float(SPRING_P)
        self._finger_damp  = float(FINGER_DAMPING)
        self._stiction     = float(FINGER_STICTION)

        self.vtk_model    = None
        self._vtk_points  = None
        self._vtk_poly    = None
        self.fiducial_list = None
        self._finger_model = None
        self._finger_transform = None
        self._updating    = False
        self.grid_volume  = None

        # simulation loop state
        self._loop_running          = False
        self._steps_per_tick        = 20    # 2× to compensate for halved dt
        self._tick_interval_ms      = 50
        self._idle_ticks            = 0
        self._idle_ticks_to_stop    = 5
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
            self._create_finger_model()
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
        self._update_finger_model()
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
            self._advance_finger_and_step()

        pos_now = self.sim.get_positions()
        self.update_model()

        if self._prev_tick_pos is not None:
            # Idle when both tissue and finger are at rest.  Always run a
            # contact step here — the finger may still be moving even when
            # the tissue is settled, and the kernel skips work cheaply when
            # the sphere is fully clear of every particle.
            tissue_idle = float(np.abs(pos_now - self._prev_tick_pos).max()) < 2e-5
            finger_idle = float(np.abs(self._finger_vel_m).max()) < 1e-4
            if tissue_idle and finger_idle:
                self._idle_ticks += 1
            else:
                self._idle_ticks = 0
        self._prev_tick_pos = pos_now

        if self._idle_ticks >= self._idle_ticks_to_stop:
            self._loop_running = False
            return
        self._schedule_tick()

    def _advance_finger_and_step(self):
        """Advance finger one substep (force-driven), then step the MPM sim.

        The finger is a 1-DOF rigid body coupled to the user's target by a
        sublinear power-spring (toward target) and a damper (on its own
        velocity), with the Newton-3rd-law tissue reaction read back from
        the previous step.
        """
        dt = float(self.sim.dt)
        delta = self._target_pos_m - self._finger_pos_m
        d = float(np.linalg.norm(delta))
        if d > 1e-9:
            f_mag = self._spring_k * ((1.0 + d / self._spring_d0) ** self._spring_p - 1.0)
            F_spring = (f_mag / d) * delta
        else:
            F_spring = np.zeros(3)
        F_damp   = -self._finger_damp * self._finger_vel_m
        F_react  = self.sim.last_contact_force          # 0 before any contact step
        F_total  = F_spring + F_damp + F_react

        self._finger_vel_m += (F_total / self._finger_mass) * dt
        self._finger_pos_m += self._finger_vel_m * dt

        self.sim.step_with_contact(GRAVITY, self._finger_pos_m, self._probe_radius,
                                   sphere_vel=self._finger_vel_m,
                                   stiction=self._stiction)

    # ------------------------------------------------------------------
    # Public API for driving the target
    # ------------------------------------------------------------------

    def set_target_position_m(self, pos_m):
        """Set the target (palpation goal) position in metres.

        The finger is pulled toward this point by the spring; tissue reaction
        decides how close the finger actually gets.
        """
        self._target_pos_m = np.asarray(pos_m, dtype=np.float64).copy()
        self._idle_ticks   = 0
        if not self._loop_running:
            self.start_simulation_loop()

    def set_target_depth_m(self, depth_m):
        """Drag the target straight down by depth_m relative to its rest pose.

        depth_m > 0 means the user is pushing into the tissue.  Convenience
        wrapper around set_target_position_m for the common 1-D test case.
        """
        rest = self._palp_pos_mm / 1000.0
        self.set_target_position_m(rest + np.array([0.0, -float(depth_m), 0.0]))

    def _on_scene_about_to_close(self, _scene, _event):
        self.stop_simulation_loop()

    # ------------------------------------------------------------------
    # Palpation (displacement-controlled rigid sphere contact)
    # ------------------------------------------------------------------

    def apply_palpation(self, push_depth_m=0.015, n_steps=500, show_every=0):
        """Ramp the target down by push_depth_m over n_steps and let the
        force-driven finger settle against the tissue.

        The finger does NOT move rigidly with the target — it trails behind
        and stops at force balance.  push_depth_m therefore represents user
        intent (how deep the user is pulling the fiducial), not measured
        deformation.  Use sim.last_contact_force to read the reaction force.
        """
        self._loop_running = False
        rest_m = self._palp_pos_mm / 1000.0
        for i in range(n_steps):
            frac = min(1.0, (i + 1) / n_steps)
            self._target_pos_m = rest_m + np.array([0.0, -push_depth_m * frac, 0.0])
            self._advance_finger_and_step()
            if show_every > 0 and i % show_every == show_every - 1:
                self.update_model()
                try:
                    import slicer
                    slicer.app.processEvents()
                except ImportError:
                    pass

    def recover(self, n_steps=1500, show_every=5):
        """Release probe (target back to rest above surface) and let tissue
        and finger relax."""
        self._loop_running = False
        self._target_pos_m = (self._palp_pos_mm / 1000.0).astype(np.float64).copy()
        for i in range(n_steps):
            self._advance_finger_and_step()
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

    def _create_finger_model(self):
        """Sphere model node showing the dynamic finger position.

        Linked to a transform node whose translation is updated each tick.
        """
        import slicer, vtk
        src = vtk.vtkSphereSource()
        src.SetRadius(self._probe_radius * 1000.0)   # mm
        src.SetThetaResolution(24)
        src.SetPhiResolution(24)
        src.Update()
        self._finger_model = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLModelNode', 'MPMFinger')
        self._finger_model.SetAndObservePolyData(src.GetOutput())
        self._finger_model.CreateDefaultDisplayNodes()
        dn = self._finger_model.GetDisplayNode()
        dn.SetColor(0.4, 0.7, 1.0)
        dn.SetOpacity(0.7)
        self._finger_transform = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLLinearTransformNode', 'MPMFingerTransform')
        self._finger_model.SetAndObserveTransformNodeID(
            self._finger_transform.GetID())
        self._update_finger_model()

    def _update_finger_model(self):
        if self._finger_transform is None:
            return
        import vtk
        m = vtk.vtkMatrix4x4()
        m.Identity()
        m.SetElement(0, 3, self._finger_pos_m[0] * 1000.0)
        m.SetElement(1, 3, self._finger_pos_m[1] * 1000.0)
        m.SetElement(2, 3, self._finger_pos_m[2] * 1000.0)
        self._finger_transform.SetMatrixTransformToParent(m)

    def _on_fiducial_moved(self, fiducial_list):
        """Fiducial drives the target only — finger trails it via the spring.

        The fiducial's RAS position (mm) is taken verbatim as the new target,
        no offset.  Force-driven contact handles the rest.
        """
        p = [0.0, 0.0, 0.0]
        fiducial_list.GetNthControlPointPosition(0, p)
        self.set_target_position_m(np.array(p) / 1000.0)
        self._palp_ref_pos = np.array(p)


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
