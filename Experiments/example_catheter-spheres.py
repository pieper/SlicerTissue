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
import random

import numpy as np
import warp as wp

import newton
import newton.examples


@wp.kernel
def advance_time(sim_time_array: wp.array(dtype=float), dt: float):
    """Increment the simulation time on the device (for CUDA graph compatibility)."""
    sim_time_array[0] = sim_time_array[0] + dt


@wp.kernel
def move_end_vertically_kernel(
    body_index: int,
    sim_time_array: wp.array(dtype=float),
    start_delay: float,
    period: float,
    initial_pos: wp.vec3,
    initial_height: float,
    catheter_length: float,
    body_q_0: wp.array(dtype=wp.transform),
    body_q_1: wp.array(dtype=wp.transform),
):
    """Move the kinematic end through a complex motion pattern:
    1. Start at initial position and descend once to 3/4 way to floor (only at beginning)
    2. Then continuously: pull back 0.75x catheter length, push forward, repeat
    3. Motion at half speed (20 second period for in/out cycle)
    """

    # Read current simulation time from device array
    sim_time = sim_time_array[0]

    # Get the original rotation to preserve it
    rot = wp.transform_get_rotation(body_q_0[body_index])

    # Do nothing until the initial delay has passed - keep at initial position
    if sim_time < start_delay:
        pos = initial_pos
        transform = wp.transform(pos, rot)
        body_q_0[body_index] = transform
        body_q_1[body_index] = transform
        return

    time_since_start = sim_time - start_delay

    # Define positions
    z_high = initial_height
    z_low = initial_height * 0.75  # Only 1/4 of the way down (3/4 of the way to floor)
    x_start = initial_pos[0]
    x_pullback = initial_pos[0] + catheter_length * 0.5  # Pull back by 0.5x catheter length

    # Initial descent period (happens once at the beginning)
    descent_duration = 2.0  # 2 seconds to descend

    x = x_start
    z = z_high

    if time_since_start < descent_duration:
        # Initial descent phase (only happens once)
        t = time_since_start / descent_duration  # 0 to 1
        x = x_start
        z = z_high + t * (z_low - z_high)  # Lerp from high to low
    else:
        # After descent, stay at low height and oscillate in/out
        z = z_low
        time_in_cycle = time_since_start - descent_duration

        # Oscillate with the given period (20 seconds for full in/out cycle)
        cycles = time_in_cycle / period
        frac = cycles - wp.floor(cycles)  # 0.0 to 1.0 within current cycle
        phase_time = frac * 2.0  # 0.0 to 2.0 (two phases: pull back, push forward)

        if phase_time < 1.0:
            # Phase 1: Pull back (move in +X direction)
            t = phase_time  # 0 to 1
            x = x_start + t * (x_pullback - x_start)  # Lerp from start to pullback
        else:
            # Phase 2: Push forward (move in -X direction)
            t = phase_time - 1.0  # 0 to 1
            x = x_pullback + t * (x_start - x_pullback)  # Lerp from pullback to start

    pos = wp.vec3(x, initial_pos[1], z)
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

        # Add two parallel rows of spheres to form a valley rotated at an angle
        num_spheres_per_side = 30  # Doubled from 15 for smoother valley walls
        self.path_length = self.cable_length  # Match valley length to catheter length
        valley_width = 0.0  # Make spheres touch at the floor
        min_sphere_radius = 0.6  # Increased from 0.4 to make valley deeper
        max_sphere_radius = 0.8  # Increased from 0.5 to make valley deeper
        valley_angle_deg = 30.0
        self.valley_angle_rad = math.radians(valley_angle_deg)
        self.cos_angle = math.cos(self.valley_angle_rad)
        self.sin_angle = math.sin(self.valley_angle_rad)
        self.path_amplitude = 1.0  # Amplitude of the S-curve

        for i in range(num_spheres_per_side):
            t = i / (num_spheres_per_side - 1)
            # Create a straight path along a temporary 'u' axis
            u = -self.path_length / 2.0 + t * self.path_length

            # Add a sinusoidal curve to the path's centerline to create an S-shape
            v_center = self.path_amplitude * math.sin(t * 2.0 * math.pi)

            for side in [-1, 1]:
                sphere_radius = random.uniform(min_sphere_radius, max_sphere_radius)

                # Position spheres to touch at the centerline on the floor
                v = v_center + side * sphere_radius # type: ignore

                # Rotate the (u,v) coordinates to create the angled valley
                x = u * self.cos_angle - v * self.sin_angle
                y = u * self.sin_angle + v * self.cos_angle
                z = 0.0  # Center spheres on the ground plane

                sphere_pos = wp.vec3(x, y, z)
                sphere_body = builder.add_body(xform=wp.transform(sphere_pos, wp.quat()))
                builder.add_shape_sphere(body=sphere_body, radius=sphere_radius)
                builder.body_mass[sphere_body] = 0.0  # Make spheres kinematic
                builder.body_inv_mass[sphere_body] = 0.0

        # Add ground plane
        builder.add_ground_plane()

        # Build a single long cable positioned above the ground
        self.initial_height = 1.5
        self.start_pos = wp.vec3(-self.cable_length / 2.0, 0.0, self.initial_height)

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
        self.motion_start_delay = 2.0  # seconds
        self.motion_period = 20.0  # 20 second period for in/out cycle (10s out, 10s in)

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

        # Initialize states and contacts
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)
        self.viewer.set_model(self.model)

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
                kernel=move_end_vertically_kernel,
                dim=1,
                inputs=[
                    self.proximal_body_idx,
                    self.sim_time_array,
                    self.motion_start_delay,
                    self.motion_period,
                    self.start_pos,
                    self.initial_height,
                    self.cable_length,
                    self.state_0.body_q,
                    self.state_1.body_q,
                ],
                device=self.solver.device,
            )

            # Collide for contact detection
            self.contacts = self.model.collide(self.state_0)

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
