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
def update_proximal_body_kernel(
    body_index: int,
    target_transform: wp.transform,
    body_q_0: wp.array(dtype=wp.transform),
    body_q_1: wp.array(dtype=wp.transform),
):
    """Update the proximal end position from the interactive gizmo transform.

    This kernel updates the kinematic body at the proximal end of the catheter
    to match the transform controlled by the user via the interactive gizmo.
    """
    # Update both state buffers - VBD solver will skip kinematic bodies (inv_mass == 0)
    # but it still needs consistent positions in both buffers
    body_q_0[body_index] = target_transform
    body_q_1[body_index] = target_transform

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
        # Store valley parameters for constraint checking
        self.valley_length = self.cable_length * 1.2  # Longer than catheter
        self.valley_floor_z = -0.1  # Just below ground plane (where catheter will rest)
        self.valley_top_z = 1.5  # Above catheter starting height (walls/edges of valley)
        self.valley_width = 2.0  # Width of valley (wider to catch catheter)
        self.valley_amplitude = 0.5  # S-curve amplitude in Y direction
        self.valley_x_offset = self.cable_length * (4.0 / 3.0)  # Valley position offset
        self.valley_straight_section = 0.2  # Fraction of length that's straight

        # Create the valley mesh
        vertices, triangles = create_valley_mesh(
            length=self.valley_length,
            amplitude=self.valley_amplitude,
            valley_floor_z=self.valley_floor_z,
            valley_top_z=self.valley_top_z,
            valley_width=self.valley_width,
            resolution=100,
            straight_section=self.valley_straight_section,
        )

        # Create Newton mesh and add to scene
        # Debug: Print some vertex positions to understand the mesh
        print(f"Valley mesh vertices shape: {vertices.shape}")
        print(f"Sample vertices (first 5):")
        for i in range(min(5, len(vertices))):
            print(f"  v[{i}]: {vertices[i]}")
        print(f"Z range: min={vertices[:, 2].min():.3f}, max={vertices[:, 2].max():.3f}")

        # Add valley mesh, shifted up by the height of the U/V shape and forward by 4/3 catheter length
        self.valley_height = self.valley_top_z - self.valley_floor_z
        valley_mesh = newton.Mesh(vertices, triangles)
        valley_body = builder.add_body(xform=wp.transform(wp.vec3(self.valley_x_offset, 0.0, self.valley_height), wp.quat()))
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

        # Create a persistent transform for the proximal end gizmo
        # This will be mutated by the viewer's gizmo system for interactive control
        self.proximal_tf = wp.transform(self.start_pos, wp.quat())

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

        # Set initial camera position
        # Calculate the middle of the model (center of the catheter/valley system)
        catheter_start_x = self.start_pos[0]
        catheter_end_x = catheter_start_x + self.cable_length
        valley_center_x = self.valley_x_offset + self.valley_length / 2.0

        # Model center is approximately midway between catheter and valley
        model_center_x = (catheter_start_x + valley_center_x) / 2.0
        model_center_y = 0.0  # Center of Y
        model_center_z = self.valley_height / 2.0  # Middle height

        # Camera position: 3/4 down X axis, offset in Y and up in Z
        cam_offset = self.cable_length * 0.75  # 3/4 down the length
        cam_x = model_center_x + cam_offset
        cam_y = model_center_y + cam_offset * 0.9  # Slightly less in Y
        cam_z = model_center_z + cam_offset * 0.85  # Slightly less in Z

        # Set camera position looking at model center
        # set_camera(position, pitch, yaw) where pitch/yaw are in degrees
        if hasattr(self.viewer, 'set_camera'):
            self.viewer.cam_pos = wp.vec3(cam_x, cam_y, cam_z)
            # Calculate angles to look at center
            # Pitch (up/down angle) and yaw (left/right angle)
            self.viewer.cam_pitch = -25  # Look down slightly
            self.viewer.cam_yaw = -135  # Look back toward start

        # Optional capture for CUDA
        self.capture()

    def _update_proximal_from_gizmo(self):
        """Update the proximal body state from the gizmo transform.

        This method synchronizes the kinematic body position with the interactive
        gizmo transform that has been mutated by the user's mouse/keyboard input.
        """
        # Update both state buffers with the current gizmo transform
        # The gizmo transform (self.proximal_tf) is mutated in-place by the viewer
        wp.launch(
            kernel=update_proximal_body_kernel,
            dim=1,
            inputs=[
                self.proximal_body_idx,
                self.proximal_tf,
                self.state_0.body_q,
                self.state_1.body_q,
            ],
            device=self.solver.device,
        )

    def capture(self):
        """Capture simulation loop into a CUDA graph for optimal GPU performance."""
        # NOTE: We cannot capture to CUDA graph when using interactive gizmos
        # because the gizmo transform changes each frame based on user input.
        # CUDA graphs require all inputs to be fixed at capture time.
        self.graph = None

    def simulate(self):
        """Execute all simulation substeps for one frame."""
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()

            # Apply forces to the model
            self.viewer.apply_forces(self.state_0)

            # Collide for contact detection using hydroelastic collision pipeline
            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )

            # Swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Advance simulation by one frame (either via CUDA graph or direct execution)."""
        # Update the kinematic proximal body from the gizmo transform BEFORE simulation
        self._update_proximal_from_gizmo()

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

        # Log the interactive gizmo for controlling the proximal end
        # The viewer will mutate self.proximal_tf based on user mouse/keyboard input
        self.viewer.log_gizmo("catheter_proximal", self.proximal_tf)

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
