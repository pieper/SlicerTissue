import slicer

try:
    import warp
    import newton
except ModuleNotFoundError:
    slicer.util.pip_install("warp-lang")
    #slicer.util.pip_install("git+https://github.com/nvidia/warp")
    slicer.util.pip_install("git+https://github.com/newton-physics/newton")

import numpy as np
import warp as wp
import qt

import math

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM

toMilli = 1000.

def apply_attraction_force(
    particle_q: np.ndarray,
    target_point: np.ndarray,
    attraction_strength: float,
    particle_f: np.ndarray,
):
    """
    Applies an attraction force to each particle, pulling it towards a target point.
    This is a NumPy-based implementation.

    Args:
        particle_q: Input array of particle positions.
        target_point: The point in space to attract particles to.
        attraction_strength: The magnitude of the attraction force.
        particle_f: Output array of particle forces to which the attraction
                    force will be added.
    """
    direction_vectors = target_point - particle_q
    distances = np.linalg.norm(direction_vectors, axis=1)

    # Avoid division by zero
    mask = distances > 1.0e-6
    force_directions = np.zeros_like(direction_vectors)
    force_directions[mask] = direction_vectors[mask] / distances[mask, np.newaxis]
    forces = force_directions * attraction_strength
    particle_f += forces



class SlicerViewer():
    def __init__(self):
        self.points = {}
        self.radii = {}
        self.colors = {}
        self.running = True
        self.paused = False

    def is_running(self):
        return self.running

    def is_paused(self):
        return self.paused


    def set_model(self, model):
        self.model = model

        self.vtk_points = vtk.vtkPoints()
        initial_positions = model.particle_q.numpy()
        self.vtk_points.SetData(vtk.util.numpy_support.numpy_to_vtk(toMilli * initial_positions))

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(self.vtk_points)

        vertices = vtk.vtkCellArray()
        for i in range(polydata.GetNumberOfPoints()):
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(i)
        polydata.SetVerts(vertices)

        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetRadius(20)
        sphere_source.SetThetaResolution(5)
        sphere_source.SetPhiResolution(5)

        glyph3D = vtk.vtkGlyph3D()
        glyph3D.SetSourceConnection(sphere_source.GetOutputPort())
        glyph3D.SetInputData(polydata)

        glyph3D.SetScaleModeToDataScalingOff()
        glyph3D.Update()

        pointsNode = slicer.mrmlScene.GetFirstNodeByName("points")
        if not pointsNode:
            pointsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
            pointsNode.SetName("points")
            pointsNode.CreateDefaultDisplayNodes()

        pointsNode.SetAndObserveMesh(glyph3D.GetOutput())
        self.glyph3D = glyph3D
        self.pointsNode = pointsNode

    def begin_frame(self, sim_time):
        self.sim_time = sim_time

    def log_points(self, name, points, radii, colors, hidden):
        if math.fabs(math.floor(self.sim_time) - self.sim_time) < 0.1:
            print(f"Logged for {self.sim_time}")

        polydata = self.pointsNode.GetPolyData()
        self.vtk_points.SetData(vtk.util.numpy_support.numpy_to_vtk(toMilli * points.numpy()))
        polydata.Modified()
        self.glyph3D.Update()
        slicer.app.processEvents()

    def end_frame(self):
        pass

    def close(self):
        pass

class Example:
    def __init__(self, viewer, options):
        # setup simulation parameters first
        self.fps = 60.0
        self.frame_dt = 1.0 / self.fps

        # group related attributes by prefix
        self.sim_time = 0.0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps

        # save a reference to the viewer
        self.viewer = viewer
        builder = newton.ModelBuilder()
        sand_particles, snow_particles, mud_particles = Example.emit_particles(builder, voxel_size=options.voxel_size)

        builder.add_ground_plane()
        self.model = builder.finalize()

        sand_particles = wp.array(sand_particles, dtype=int, device=self.model.device)
        snow_particles = wp.array(snow_particles, dtype=int, device=self.model.device)
        mud_particles = wp.array(mud_particles, dtype=int, device=self.model.device)

        self.model.particle_ke = 1.0e25
        self.model.particle_kd = 0.0
        self.model.particle_mu = 0.5

        mpm_options = SolverImplicitMPM.Options()
        mpm_options.voxel_size = options.voxel_size
        mpm_options.tolerance = options.tolerance
        mpm_options.max_iterations = options.max_iterations

        # Create MPM model from Newton model
        mpm_model = SolverImplicitMPM.Model(self.model, mpm_options)

        # multi-material setup
        # some properties like elastic stiffness, damping, can be adjusted directly on the model,
        # but not all yet. here we directly adjust the MPM model's material parameters

        mpm_model.material_parameters.yield_pressure[snow_particles].fill_(1.0e10)
        mpm_model.material_parameters.yield_stress[snow_particles].fill_(1.0e2)
        mpm_model.material_parameters.tensile_yield_ratio[snow_particles].fill_(0.1)
        mpm_model.material_parameters.friction[snow_particles].fill_(0.0)
        mpm_model.material_parameters.hardening[snow_particles].fill_(1.0)

        mpm_model.material_parameters.yield_pressure[mud_particles].fill_(1.0e10)
        mpm_model.material_parameters.yield_stress[mud_particles].fill_(3.0e2)
        mpm_model.material_parameters.tensile_yield_ratio[mud_particles].fill_(1.0)
        mpm_model.material_parameters.hardening[mud_particles].fill_(2.0)
        mpm_model.material_parameters.friction[mud_particles].fill_(0.0)

        mpm_model.notify_particle_material_changed()

        # Initialize MPM solver
        self.solver = SolverImplicitMPM(mpm_model, mpm_options)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # Allocate external force buffers on the state objects
        self.state_0.body_f = wp.zeros(
            shape=self.model.particle_count, dtype=wp.vec3, device=self.model.device
        )
        self.state_1.body_f = wp.zeros(
            shape=self.model.particle_count, dtype=wp.vec3, device=self.model.device
        )
        self.solver.enrich_state(self.state_0)
        self.solver.enrich_state(self.state_1)

        # Assign different colors to each particle type
        self.particle_colors = wp.full(
            shape=self.model.particle_count, value=wp.vec3(0.1, 0.1, 0.2), device=self.model.device
        )
        self.particle_colors[sand_particles].fill_(wp.vec3(0.7, 0.6, 0.4))
        self.particle_colors[snow_particles].fill_(wp.vec3(0.75, 0.75, 0.8))
        self.particle_colors[mud_particles].fill_(wp.vec3(0.4, 0.25, 0.25))

        self.viewer.set_model(self.model)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # Apply custom attraction force
            target_point = np.array([0.0, 0.0, 1.0]) # Example target point
            attraction_strength = 100.0 # Example strength
            apply_attraction_force(
                self.state_0.particle_q.numpy(),
                target_point,
                attraction_strength,
                self.state_0.body_f.numpy(),
            )
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.solver.project_outside(self.state_1, self.state_1, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test(self):
        newton.examples.test_particle_state(
            self.state_0,
            "all particles are above the ground",
            lambda q, qd: q[2] > -0.05,
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_points(
            name="/model/particles",
            points=self.state_0.particle_q,
            radii=self.model.particle_radius,
            colors=self.particle_colors,
            hidden=False,
        )
        self.viewer.end_frame()

    @staticmethod
    def emit_particles(builder: newton.ModelBuilder, voxel_size: float):
        # inactive particles
        Example._spawn_particles(
            builder,
            voxel_size,
            bounds_lo=np.array([-0.5, -0.5, 0.0]),
            bounds_hi=np.array([0.5, 0.5, 0.25]),
            density=1000.0,
            flags=0,
        )

        # sand particles
        sand_particles = Example._spawn_particles(
            builder,
            voxel_size,
            bounds_lo=np.array([0.25, -0.5, 0.5]),
            bounds_hi=np.array([0.75, 0.5, 0.75]),
            density=2500.0,
            flags=newton.ParticleFlags.ACTIVE,
        )

        # snow particles
        snow_particles = Example._spawn_particles(
            builder,
            voxel_size,
            bounds_lo=np.array([-0.75, -0.5, 0.5]),
            bounds_hi=np.array([-0.25, 0.5, 0.75]),
            density=300,
            flags=newton.ParticleFlags.ACTIVE,
        )

        # mud particles
        mud_particles = Example._spawn_particles(
            builder,
            voxel_size,
            bounds_lo=np.array([-0.5, -0.25, 1.0]),
            bounds_hi=np.array([0.5, 0.25, 1.5]),
            density=1000.0,
            flags=newton.ParticleFlags.ACTIVE,
        )

        return sand_particles, snow_particles, mud_particles

    @staticmethod
    def _spawn_particles(builder: newton.ModelBuilder, voxel_size, bounds_lo, bounds_hi, density, flags):
        particles_per_cell = 3
        res = np.array(
            np.ceil(particles_per_cell * (bounds_hi - bounds_lo) / voxel_size),
            dtype=int,
        )

        cell_size = (bounds_hi - bounds_lo) / res
        cell_volume = np.prod(cell_size)
        radius = np.max(cell_size) * 0.5
        mass = np.prod(cell_volume) * density

        begin_id = len(builder.particle_q)
        builder.add_particle_grid(
            pos=wp.vec3(bounds_lo),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=res[0] + 1,
            dim_y=res[1] + 1,
            dim_z=res[2] + 1,
            cell_x=cell_size[0],
            cell_y=cell_size[1],
            cell_z=cell_size[2],
            mass=mass,
            jitter=2.0 * radius,
            radius_mean=radius,
            flags=flags,
        )

        if "snow" in builder.body_key: # Heuristic to identify snow particles
            builder.add_spring_grid(
                begin_id,
                dim_x=res[0] + 1,
                dim_y=res[1] + 1,
                dim_z=res[2] + 1,
                stiffness=1.0e4,
                damping=1.0e1,
                spring_rest_length=radius * 2.0,
            )


        end_id = len(builder.particle_q)
        return np.arange(begin_id, end_id, dtype=int)


print(__name__)
print("starting")
# Create parser that inherits common arguments and adds example-specific ones
parser = newton.examples.create_parser()

parser.add_argument("--max-iterations", "-it", type=int, default=250)
parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-6)
parser.add_argument("--voxel-size", "-dx", type=float, default=0.05)

# Parse arguments and initialize viewer
#viewer, args = newton.examples.init(parser)

viewer = SlicerViewer()
class Options():
    max_iterations = 25
    tolerance = 1.0e-6
    #voxel_size = 0.05
    voxel_size = 0.15
    test = False

options = Options()

# Create example and run
example = Example(viewer, options)

slicer.modules.viewer = viewer
slicer.modules.example = example
slicer.modules.options = options

# Get the initial particle positions from the model
initial_positions_wp = example.model.particle_q
initial_positions_np = initial_positions_wp.numpy()

print(f"Extracted initial particle positions with shape: {initial_positions_np.shape}")

# --- Cleanup previous run ---

for widget in slicer.util.findChildren(slicer.util.mainWindow(), "Newton Control"):
    widget.setParent(None)
    widget.delete()

# Create and show the control widget
controlWidget = qt.QWidget()
controlWidget.setLayout(qt.QVBoxLayout())

# --- Pause/Resume Button ---
pauseButton = qt.QPushButton("Pause")
def toggle_pause():
    viewer.paused = not viewer.paused
    pauseButton.text = "Resume" if viewer.paused else "Pause"
pauseButton.connect('clicked()', toggle_pause)
controlWidget.layout().addWidget(pauseButton)

# --- Stop Simulation Button ---
stopButton = qt.QPushButton("Stop Simulation")
def stop_simulation():
    """Sets the running flag to false to exit the simulation loop."""
    print("Stopping simulation...")
    viewer.running = False
stopButton.connect('clicked()', stop_simulation)
controlWidget.layout().addWidget(stopButton)
stopButton.enabled = False # Initially disabled

# --- Start Simulation Button ---
startButton = qt.QPushButton("Start Simulation")
def start_simulation():
    """Runs the simulation in the main thread. If running, it will restart."""
    if viewer.running:
        print("Restarting simulation...")
        viewer.running = False
        # Allow the current simulation loop to exit
        slicer.app.processEvents()

    # Re-create the example to reset the simulation state
    slicer.modules.example = Example(viewer, options)

    stopButton.enabled = True
    viewer.running = True # Reset the running flag
    slicer.app.processEvents() # Allow GUI to update before starting loop
    print("Simulation started...")
    newton.examples.run(slicer.modules.example, slicer.modules.options)
    print("Simulation finished.")
    stopButton.enabled = False

startButton.connect('clicked()', start_simulation)
controlWidget.layout().addWidget(startButton)

dockWidget = qt.QDockWidget("Newton Control")
dockWidget.name = "Newton Control"
dockWidget.setWidget(controlWidget)
slicer.util.mainWindow().addDockWidget(qt.Qt.LeftDockWidgetArea, dockWidget)

print("done")
