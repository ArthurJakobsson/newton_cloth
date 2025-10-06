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
# Example Basic Rope
#
# Builds a rope as a chain of capsules connected by ball-like joints with
# adjustable angular stiffness (rigidity). The rope is anchored to a fixed point.
# Parameters allow varying total rope length, width, rigidity, and end weight.
#
# Command: python -m newton.examples basic_rope --length 5 --width 0.1 \
#          --rigidity 2000 --end-weight 5 --segments 20
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils
from newton import Model, ModelBuilder, State, eval_fk
from newton.solvers import SolverFeatherstone, SolverXPBD


@wp.kernel
def apply_anchor_force_kernel(
    body_f: wp.array(dtype=wp.spatial_vector),
    anchor_body: int,
    force_x: float,
):
    """Simple kernel to apply force to anchor body"""
    tid = wp.tid()
    if tid == anchor_body:
        # Apply force (only in x direction)
        body_f[tid] = wp.spatial_vector(
            wp.vec3(force_x, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 0.0)
        )

@wp.kernel
def test_force_all_bodies_kernel(
    body_f: wp.array(dtype=wp.spatial_vector),
    force_x: float,
):
    """Test kernel to apply force to all bodies"""
    tid = wp.tid()
    # Apply small force to all bodies to see which one moves
    body_f[tid] = wp.spatial_vector(
        wp.vec3(force_x * 0.1, 0.0, 0.0),  # Smaller force
        wp.vec3(0.0, 0.0, 0.0)
    )


def _make_d6_ball_axes(rigidity: float, damping: float = 0.0) -> list[newton.ModelBuilder.JointDofConfig]:
    # Three angular axes with drive to zero relative angle to emulate a rotational spring.
    return [
        newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X, target=0.0, target_ke=rigidity, target_kd=damping),
        newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Y, target=0.0, target_ke=rigidity, target_kd=damping),
        newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Z, target=0.0, target_ke=rigidity, target_kd=damping),
    ]


class Example:
    def __init__(self, viewer, args=None):
        # setup simulation parameters first (same as cloth_franka)
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 15  # Same as cloth_franka
        self.sim_dt = self.frame_dt / self.sim_substeps

        # parse args and create a viewer if not provided
        if args is None or viewer is None:
            parser = newton.examples.create_parser()
            Example.add_args(parser)
            v, a = newton.examples.init(parser)
            if args is None:
                args = a
            if viewer is None:
                viewer = v

        self.args = args
        self.viewer = viewer

        # parameters
        length = float(args.length)
        width = float(args.width)
        radius = 0.5 * width
        segments = int(args.segments) if args.segments is not None else max(4, int(math.ceil(length / max(width, 1e-3))))
        rigidity = float(args.rigidity)
        damping = float(args.damping)
        end_weight = float(args.end_weight)
        
        # oscillation parameters
        self.oscillation_amplitude = 1  # Oscillate from -0.5 to +0.5 in x
        self.oscillation_period = 3.0     # 3 second period
        self.anchor_base_x = float(args.anchor_x)
        self.anchor_base_y = float(args.anchor_y)
        self.anchor_base_z = float(args.anchor_height)

        builder = newton.ModelBuilder()

        # add ground plane
        builder.add_ground_plane()

        # Set contact parameters like cloth_franka example
        builder.soft_contact_ke = 100
        builder.soft_contact_kd = 2e-3
        builder.soft_contact_mu = 0.5  # friction

        # Create anchor point and build rope
        self.anchor_body = self.create_anchor(builder)
        print(f"Created anchor body with ID: {self.anchor_body}")
        print(f"Bodies before rope creation: {builder.body_count}")
        self.create_rope(builder, length, radius, segments, rigidity, damping, end_weight, self.anchor_body)
        print(f"Bodies after rope creation: {builder.body_count}")

        # finalize model
        self.model = builder.finalize()
        self.model.ground = True  # Like the working example_rigid_force.py

        # Set contact parameters on the model like cloth_franka
        self.model.soft_contact_ke = 100
        self.model.soft_contact_kd = 2e-3
        self.model.soft_contact_mu = 0.5

        # Use XPBD solver like the working example_rigid_force.py
        self.solver = newton.solvers.SolverXPBD(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        self.viewer.set_model(self.model)

        # not required for MuJoCo, but required for other solvers
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.capture()

    def apply_anchor_oscillation(self):
        """Apply oscillation motion to the anchor body using direct force application"""
        # Calculate oscillation parameters
        omega = 2.0 * math.pi / self.oscillation_period
        oscillation_x = self.oscillation_amplitude * math.sin(omega * self.sim_time)
        
        # Calculate spring force based on target position
        target_x = self.anchor_base_x + oscillation_x
        spring_force_x = 1000.0 * oscillation_x  # Reasonable force for oscillation
        
        # Debug: print anchor body ID and oscillation info (less frequently)
        if int(self.sim_time * 2) % 20 == 0:  # Print every 10 seconds
            print(f"Anchor body ID: {self.anchor_body}, Sim time: {self.sim_time:.3f}, Omega: {omega:.3f}, Oscillation x: {oscillation_x:.3f}")
            print(f"Total bodies: {self.model.body_count}, Force being applied: {spring_force_x:.3f}")
        
        # Apply force using direct assign method (like example_rigid_force.py)
        
        # Anchor is now fixed, so no forces needed
        # The rope should hang naturally from the fixed anchor
        if int(self.sim_time * 2) % 20 == 0:  # Print every 10 seconds
            print(f"Rope hanging from fixed anchor - no forces applied")

    def create_anchor(self, builder):
        # Create a fixed anchor point for the rope (mass=0 makes it kinematic/fixed)
        anchor_body = builder.add_body(
            xform=wp.transform(
                p=wp.vec3(self.anchor_base_x, self.anchor_base_y, self.anchor_base_z),
                q=wp.quat_identity()
            ),
            mass=0.0  # Zero mass makes it kinematic (fixed in space)
        )
        
        # Add a visible sphere to the anchor so we can see it
        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.density = 1.0
        cfg.ke = 1e2
        cfg.kd = 1.5e-6
        cfg.mu = 0.5
        builder.add_shape_sphere(anchor_body, radius=0.1, cfg=cfg)
        
        return anchor_body

    def create_rope(self, builder, length, radius, segments, rigidity, damping, end_weight, anchor_body):
        # segment span and capsule geometry
        segment_span = length / segments
        # capsule total length = 2*half_height + 2*radius = segment_span
        half_height = max(0.0, 0.5 * max(segment_span - 2.0 * radius, 0.0))
        tip_offset = half_height + radius

        print(f"Creating rope with {segments} segments, length={length}, radius={radius}")

        # Create rope bodies
        bodies = []
        for i in range(segments):
            # place centers from top to bottom
            cz = float(self.args.anchor_height) - (i * segment_span + tip_offset)
            cx, cy = float(self.args.anchor_x), float(self.args.anchor_y)
            body = builder.add_body(xform=wp.transform(p=wp.vec3(cx, cy, cz), q=wp.quat_identity()))

            # Set rope material properties - reduced for stability
            cfg = newton.ModelBuilder.ShapeConfig()
            cfg.density = 0.1  # Reduced density for less bouncing
            cfg.ke = 1e1      # Reduced stiffness
            cfg.kd = 1e-3     # Increased damping for stability
            cfg.mu = 0.5      # Same friction
            builder.add_shape_capsule(body, radius=radius, half_height=half_height, cfg=cfg)

            bodies.append(body)
            print(f"Created rope segment {i} with body ID {body} at position ({cx}, {cy}, {cz})")

        # Add extra weight at the end by attaching an additional dense sphere to the last body
        if end_weight > 0.0:
            last = bodies[-1]
            weight_radius = max(radius * 1.1, (3.0 * end_weight / (4.0 * math.pi * 1000.0)) ** (1.0 / 3.0))  # heuristic
            weight_cfg = newton.ModelBuilder.ShapeConfig()
            weight_cfg.density = end_weight  # treat as density-like knob for simplicity
            builder.add_shape_sphere(last, radius=weight_radius, cfg=weight_cfg)

        # Attach first body to anchor with a ball joint (free rotation)
        if bodies:
            first = bodies[0]
            builder.add_joint_ball(
                parent=anchor_body,
                child=first,
                parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
                child_xform=wp.transform(p=wp.vec3(0.0, 0.0, +tip_offset), q=wp.quat_identity()),
                key="rope_anchor",
            )

        # Create simple revolute joints between consecutive bodies
        for i in range(1, len(bodies)):
            parent = bodies[i - 1]
            child = bodies[i]
            builder.add_joint_revolute(
                parent=parent,
                child=child,
                axis=wp.vec3(1.0, 0.0, 0.0),  # Allow rotation around X axis
                parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, -tip_offset), q=wp.quat_identity()),
                child_xform=wp.transform(p=wp.vec3(0.0, 0.0, +tip_offset), q=wp.quat_identity()),
                key=f"rope_link_{i-1}_{i}",
            )

    @staticmethod
    def add_args(parser):
        parser.add_argument("--length", type=float, default=10, help="Total rope length")
        parser.add_argument("--width", type=float, default=0.05, help="Rope width (diameter)")
        parser.add_argument("--segments", type=int, default=8, help="Number of rope segments")
        parser.add_argument("--rigidity", type=float, default=1.0, help="Angular stiffness for joints")
        parser.add_argument("--damping", type=float, default=10.0, help="Angular damping for joints")
        parser.add_argument("--end-weight", type=float, default=0.0, help="Additional weight at rope end")
        parser.add_argument("--anchor-height", type=float, default=2, help="World Z position of rope anchor")
        parser.add_argument("--anchor-x", type=float, default=0.0, help="World X position of rope anchor")
        parser.add_argument("--anchor-y", type=float, default=0.0, help="World Y position of rope anchor")

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # Apply oscillation to anchor body
            self.apply_anchor_oscillation()

            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0
            
            # Update simulation time each substep
            self.sim_time += self.sim_dt

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

    def test(self):
        pass

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    parser = newton.examples.create_parser()
    Example.add_args(parser)
    viewer, args = newton.examples.init(parser)

    # Create viewer and run
    example = Example(viewer, args=args)

    newton.examples.run(example)