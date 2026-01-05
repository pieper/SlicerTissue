# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Catheter
#
# Demonstrates a single cable interacting with a series of fixed sphere
# obstacles, simulating a catheter moving through a simplified vessel.
# The spheres are partially submerged in the ground plane at varying
# heights to form a tortuous path. This example showcases cable-to-rigid
# contact, friction, and settling dynamics in a constrained environment.
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples


@wp.kernel
def advance_time(sim_time_array: wp.array(dtype=float), dt: float):
    """Increment the simulation time on the device (for CUDA graph compatibility)."""
    sim_time_array[0] = sim_time_array[0] + dt


@wp.kernel
def move_end_horizontally_kernel(
    body_index: int,
    sim_time_array: wp.array(dtype=float),
    start_delay: float,
    period: float,
    initial_pos: wp.vec3,
    catheter_length: float,
    motion_z_height: float,
    body_q_0: wp.array(dtype=wp.transform),
    body_q_1: wp.array(dtype=wp.transform),
):
    """Move the proximal end to push/pull the catheter through the valley.

    Motion pattern:
    1. During start_delay: Lower from initial position (x=0, z=1.2) down to motion_z_height
    2. After start_delay: Oscillate back and forth along X axis at motion_z_height
       - Push forward (+X direction): insert catheter deeper into valley
       - Pull back (-X direction): extract catheter back toward entrance

    Valley starts at x=0 and extends in +X direction with S-curve.
    """

    # Read current simulation time from device array
    sim_time = sim_time_array[0]

    # Get the original rotation to preserve it
    rot = wp.transform_get_rotation(body_q_0[body_index])

    # Phase 1: Lower to motion height during the delay period
    if sim_time < start_delay:
        # Lerp from initial Z down to motion Z
        t = sim_time / start_delay  # 0 to 1
        z = initial_pos[2] + t * (motion_z_height - initial_pos[2])
        pos = wp.vec3(initial_pos[0], initial_pos[1], z)
        transform = wp.transform(pos, rot)
        body_q_0[body_index] = transform
        body_q_1[body_index] = transform
        return

    # Phase 2: Oscillate back and forth along X axis
    time_since_start = sim_time - start_delay

    # Define X positions for the push/pull motion
    # Catheter starts at x=0 (valley entrance)
    x_start = initial_pos[0]  # x=0, valley entrance
    x_push_end = initial_pos[0] + catheter_length * 0.7  # Push 70% into valley

    # Oscillate with the given period
    cycles = time_since_start / period
    frac = cycles - wp.floor(cycles)  # 0.0 to 1.0 within current cycle
    phase_time = frac * 2.0  # 0.0 to 2.0 (two phases: push forward, pull back)

    if phase_time < 1.0:
        # Phase 1: Push forward (move in +X direction to insert into valley)
        t = phase_time  # 0 to 1
        x = x_start + t * (x_push_end - x_start)  # Lerp from entrance to deep in valley
    else:
        # Phase 2: Pull back (move in -X direction to extract from valley)
        t = phase_time - 1.0  # 0 to 1
        x = x_push_end + t * (x_start - x_push_end)  # Lerp from deep in valley back to entrance

    # Keep Y at 0 (center), Z at motion height
    pos = wp.vec3(x, 0.0, motion_z_height)
    transform = wp.transform(pos, rot)

    # Update both state buffers - VBD solver will skip kinematic bodies (inv_mass == 0)
    # but it still needs consistent positions in both buffers
    body_q_0[body_index] = transform
    body_q_1[body_index] = transform

def create_cable_geometry(
    start_pos: wp.vec3,
    num_elements: int,
    length: float,
    orientation: str = "x",
) -> tuple[list[wp.vec3], np.ndarray, list[wp.quat]]:
    """Build a straight cable polyline with parallel-transported quaternions.

    Args:
        start_pos: Starting position of the cable.
        num_elements: Number of cable segments (num_points = num_elements + 1).
        length: Total cable length.
        orientation: Cable direction ("x" for +X axis, "y" for +Y axis).

    Returns:
        Tuple of (points, edge_indices, quaternions):
        - points: List of capsule center positions (num_elements + 1).
        - edge_indices: Flattened array of edge connectivity (2*num_elements).
        - quaternions: List of capsule orientations using parallel transport (num_elements).
    """
    num_points = num_elements + 1
    if num_elements <= 0:
        raise ValueError("num_elements must be positive")

    points: list[wp.vec3] = []
    dir_vec = wp.vec3(1.0, 0.0, 0.0) if orientation == "x" else wp.vec3(0.0, 1.0, 0.0)

    for i in range(num_points):
        t = i / num_elements
        p = start_pos + dir_vec * (length * t)
        points.append(p)

    # Build edges
    edge_indices: list[int] = []
    for i in range(num_elements):
        edge_indices.extend([i, i + 1])
    edge_indices = np.array(edge_indices, dtype=np.int32)

    # Parallel-transported quaternions
    quats: list[wp.quat] = []
    if num_elements > 0:
        local_axis = wp.vec3(0.0, 0.0, 1.0)  # Capsule internal axis is +Z
        from_direction = local_axis
        for i in range(num_elements):
            p0 = points[i]
            p1 = points[i + 1]
            to_direction = wp.normalize(p1 - p0)
            dq = wp.quat_between_vectors(from_direction, to_direction)
            base_q = dq if i == 0 else wp.mul(dq, quats[i - 1])
            quats.append(base_q)
            from_direction = to_direction

    return points, edge_indices, quats


def create_valley_mesh(
    length: float,
    amplitude: float,
    valley_floor_z: float,
    valley_top_z: float,
    valley_width: float,
    resolution: int = 50,
    straight_section: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a U/V-shaped valley mesh that follows an S-curve.

    The valley:
    - Starts at x=0 (catheter base)
    - Extends in the +X direction along the catheter
    - Has a U/V-shaped cross-section in the YZ plane:
      * Bottom (center, y=0) at valley_floor_z (just below ground)
      * Walls (edges, y=±width/2) at valley_top_z (catheter starting height)
    - Initially straight for a short distance, then follows sinusoidal S-curve in Y

    Args:
        length: Total length of the valley along X axis
        amplitude: Amplitude of the S-curve in Y direction
        valley_floor_z: Z height of valley floor (bottom center)
        valley_top_z: Z height of valley walls (top edges)
        valley_width: Width of the valley opening (Y direction)
        resolution: Number of segments along the length
        straight_section: Fraction of length that's straight before S-curve starts

    Returns:
        Tuple of (vertices, triangles) for the mesh
    """
    # Create a U/V-shaped cross-section profile in the YZ plane
    # The cross-section is symmetric around y=0
    num_cross_section_points = 30
    cross_section = []

    for i in range(num_cross_section_points):
        t = i / (num_cross_section_points - 1)  # 0 to 1
        # Y coordinate: from -valley_width/2 (left wall) to +valley_width/2 (right wall)
        y_local = -valley_width / 2.0 + t * valley_width

        # Normalized distance from center: 0 at center, 1 at edges
        normalized_dist = abs(y_local / (valley_width / 2.0))

        # Z height: valley_top_z at center (floor/bottom), valley_floor_z at edges (walls)
        # Inverted: high at center, low at edges for valley trough
        z = valley_top_z + (valley_floor_z - valley_top_z) * (normalized_dist ** 1.5)

        cross_section.append((y_local, z))

    # Extrude the cross-section along X axis with S-curve
    vertices = []

    for i in range(resolution + 1):
        t = i / resolution  # 0 to 1 along the valley length
        # X position: start at 0, extend to +length
        x = t * length

        # Y offset for S-curve (sinusoidal)
        # Apply S-curve only after the straight section
        if t < straight_section:
            # Straight section: no offset
            y_offset = 0.0
        else:
            # S-curve section: smoothly transition into sinusoidal pattern
            t_curve = (t - straight_section) / (1.0 - straight_section)  # 0 to 1 in curve section
            y_offset = amplitude * np.sin(t_curve * 2.0 * np.pi)

        # Place the cross-section at this X position with Y offset
        for y_local, z_local in cross_section:
            vertices.append([x, y_local + y_offset, z_local])

    vertices = np.array(vertices, dtype=np.float32)

    # Create triangles connecting the cross-sections
    triangles = []
    for i in range(resolution):
        for j in range(num_cross_section_points - 1):
            # Current ring of vertices
            idx0 = i * num_cross_section_points + j
            idx1 = i * num_cross_section_points + j + 1
            # Next ring of vertices
            idx2 = (i + 1) * num_cross_section_points + j
            idx3 = (i + 1) * num_cross_section_points + j + 1

            # Two triangles per quad
            triangles.append([idx0, idx2, idx1])
            triangles.append([idx1, idx2, idx3])

    triangles = np.array(triangles, dtype=np.int32)

    return vertices, triangles


class Example:
    def __init__(
        self,
        viewer,
        args=None,
    ):
        # Store viewer and arguments
        self.viewer = viewer
        self.args = args

        # Simulation cadence
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_iterations = 5
        self.sim_dt = self.frame_dt / self.sim_substeps

        # Cable parameters
        self.num_elements = 100
        segment_length = 0.05
        self.cable_length = self.num_elements * segment_length
        self.cable_radius = 0.02

        builder = newton.ModelBuilder()

        # Set default material properties before adding any shapes
        builder.default_shape_cfg.ke = 1.0e6  # Contact stiffness
        builder.default_shape_cfg.kd = 1.0e-1  # Contact damping
        builder.default_shape_cfg.mu = 0.5  # Friction coefficient

        # Enable hydroelastic/SDF collision for the valley mesh
        builder.default_shape_cfg.is_hydroelastic = True
        builder.default_shape_cfg.sdf_max_resolution = 128
        builder.default_shape_cfg.sdf_narrow_band_range = (-0.05, 0.05)
        builder.default_shape_cfg.contact_margin = 0.05

        # Create a valley mesh for the catheter to move through
        # Valley starts at x=0 and extends in +X direction
        # U/V-shaped cross-section: floor just below ground, walls at catheter height
        valley_length = self.cable_length * 1.2  # Longer than catheter
        valley_floor_z = -0.1  # Just below ground plane (where catheter will rest)
        valley_top_z = 1.5  # Above catheter starting height (walls/edges of valley)
        valley_width = 2.0  # Width of valley (wider to catch catheter)
        self.valley_amplitude = 0.5  # S-curve amplitude in Y direction

        # Create the valley mesh
        vertices, triangles = create_valley_mesh(
            length=valley_length,
            amplitude=self.valley_amplitude,
            valley_floor_z=valley_floor_z,
            valley_top_z=valley_top_z,
            valley_width=valley_width,
            resolution=100,
            straight_section=0.2,  # 20% straight, then S-curve
        )

        # Create Newton mesh and add to scene
        # Debug: Print some vertex positions to understand the mesh
        print(f"Valley mesh vertices shape: {vertices.shape}")
        print(f"Sample vertices (first 5):")
        for i in range(min(5, len(vertices))):
            print(f"  v[{i}]: {vertices[i]}")
        print(f"Z range: min={vertices[:, 2].min():.3f}, max={vertices[:, 2].max():.3f}")

        # Add valley mesh, shifted up by the height of the U/V shape and forward by 4/3 catheter length
        valley_height = valley_top_z - valley_floor_z
        valley_x_offset = self.cable_length * (4.0 / 3.0)  # 1 + 1/3 of catheter length
        valley_mesh = newton.Mesh(vertices, triangles)
        valley_body = builder.add_body(xform=wp.transform(wp.vec3(valley_x_offset, 0.0, valley_height), wp.quat()))
        builder.add_shape_mesh(body=valley_body, mesh=valley_mesh)

        # Make valley kinematic (static)
        builder.body_mass[valley_body] = 0.0
        builder.body_inv_mass[valley_body] = 0.0

        # Add ground plane
        builder.add_ground_plane()

        # Build a single long cable positioned above the valley
        # Catheter starts at x=0 (valley entrance) and extends in +X direction
        # It will fall and conform to the S-shaped valley below
        # IMPORTANT: Disable hydroelastic for the cable (use standard collision for cable capsules)
        builder.default_shape_cfg.is_hydroelastic = False

        self.initial_height = 1.2  # Height at valley top (Z direction)
        self.start_pos = wp.vec3(0.0, 0.0, self.initial_height)  # Start at valley entrance (x=0)

        pts, _edges, edge_q = create_cable_geometry(
            start_pos=self.start_pos,
            num_elements=self.num_elements,
            length=self.cable_length,
            orientation="x",
        )

        rod_bodies, _ = builder.add_rod(
            positions=pts,
            quaternions=edge_q,
            radius=self.cable_radius,
            bend_stiffness=1.0e1,
            bend_damping=1.0e-1,
            stretch_stiffness=1.0e6,
            stretch_damping=1.0e-4,
            key="catheter",
        )

        # Make the proximal end of the catheter kinematic for controlled movement
        self.proximal_body_idx = rod_bodies[0]
        builder.body_mass[self.proximal_body_idx] = 0.0
        builder.body_inv_mass[self.proximal_body_idx] = 0.0

        # Store parameters for the kinematic motion
        # Proximal end starts at x=0 (valley entrance) and oscillates in +X direction
        self.motion_start_delay = 4.0  # seconds - wait for catheter to settle into valley
        self.motion_period = 20.0  # 20 second period for push/pull cycle (10s push, 10s pull)
        self.motion_z_height = 0.9  # Z height for the moving end

        # Color bodies for VBD solver
        builder.color()

        # Finalize model
        self.model = builder.finalize()

        # Create VBD solver for rigid body simulation
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.sim_iterations,
            friction_epsilon=0.1,
        )

        # Set up collision pipeline for hydroelastic contacts with the valley mesh
        from newton.geometry import SDFHydroelasticConfig

        sdf_hydroelastic_config = SDFHydroelasticConfig(
            reduce_contacts=True,
            output_contact_surface=False,
        )

        self.collision_pipeline = newton.CollisionPipelineUnified.from_model(
            self.model,
            rigid_contact_max_per_pair=100,
            broad_phase_mode=newton.BroadPhaseMode.EXPLICIT,
            sdf_hydroelastic_config=sdf_hydroelastic_config,
        )

        # Initialize states and contacts
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
        self.viewer.set_model(self.model)

        # Disable backface culling for all mesh objects so we can see inside the valley
        if hasattr(self.viewer, 'objects'):
            for obj in self.viewer.objects.values():
                if hasattr(obj, 'backface_culling'):
                    obj.backface_culling = False

        # Enable wireframe mode for better visualization of the valley
        if hasattr(self.viewer, 'renderer') and hasattr(self.viewer.renderer, 'draw_wireframe'):
            self.viewer.renderer.draw_wireframe = True

        # Create a device-side time accumulator for CUDA graph capture compatibility
        # This allows time to advance even when the simulation is captured in a CUDA graph
        self.sim_time_array = wp.zeros(1, dtype=float, device=self.solver.device)

        # Optional capture for CUDA
        self.capture()

    def capture(self):
        """Capture simulation loop into a CUDA graph for optimal GPU performance."""
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as cap:
                self.simulate()
            self.graph = cap.graph
        else:
            self.graph = None

    def simulate(self):
        """Execute all simulation substeps for one frame."""
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()

            # Apply forces to the model
            self.viewer.apply_forces(self.state_0)

            # Move the catheter end kinematically BEFORE the solver step
            # The VBD solver will skip kinematic bodies (inv_mass == 0) and preserve the position
            wp.launch(
                kernel=move_end_horizontally_kernel,
                dim=1,
                inputs=[
                    self.proximal_body_idx,
                    self.sim_time_array,
                    self.motion_start_delay,
                    self.motion_period,
                    self.start_pos,
                    self.cable_length,
                    self.motion_z_height,
                    self.state_0.body_q,
                    self.state_1.body_q,
                ],
                device=self.solver.device,
            )

            # Collide for contact detection using hydroelastic collision pipeline
            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )

            # Advance simulation time on the device for CUDA graph compatibility
            wp.launch(
                kernel=advance_time,
                dim=1,
                inputs=[self.sim_time_array, self.sim_dt],
                device=self.solver.device,
            )

            # Swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Advance simulation by one frame (either via CUDA graph or direct execution)."""
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        """Render the current simulation state to the viewer."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Test catheter simulation for stability and correctness (called after simulation)."""
        ground_tolerance = 0.1

        if self.state_0.body_q is not None and self.state_0.body_qd is not None:
            body_positions = self.state_0.body_q.numpy()
            body_velocities = self.state_0.body_qd.numpy()

            # Test 1: Check for numerical stability
            assert np.isfinite(body_positions).all(), "Non-finite positions"
            assert np.isfinite(body_velocities).all(), "Non-finite velocities"

            # Test 2: Check ground interaction
            z_positions = body_positions[:, 2]
            min_z = np.min(z_positions)
            assert min_z > -ground_tolerance, f"Cable fell through ground: min_z={min_z:.3f}"

            # Test 3: Velocity should be reasonable (cable shouldn't explode)
            assert (np.abs(body_velocities) < 5e2).all(), "Velocities too large"


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create example and run
    example = Example(viewer, args)

    newton.examples.run(example, args)
