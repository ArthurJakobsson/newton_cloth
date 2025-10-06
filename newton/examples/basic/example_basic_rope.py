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
# Example Basic Rope with Franka
#
# Builds a rope as a chain of capsules connected by ball-like joints with
# adjustable angular stiffness (rigidity). The rope is held by a Franka arm.
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

        builder = newton.ModelBuilder()

        # add ground plane
        builder.add_ground_plane()

        # Set contact parameters like cloth_franka example
        builder.soft_contact_ke = 100
        builder.soft_contact_kd = 2e-3
        builder.soft_contact_mu = 0.5  # friction

        # Create Franka articulation
        self.create_articulation(builder)

        # Build rope attached to Franka end effector (only if not disabled)
        if not getattr(args, 'no_rope', False):
            self.create_rope(builder, length, radius, segments, rigidity, damping, end_weight)

        # finalize model
        self.model = builder.finalize()

        # Set contact parameters on the model like cloth_franka
        self.model.soft_contact_ke = 100
        self.model.soft_contact_kd = 2e-3
        self.model.soft_contact_mu = 0.5

        # Use the same solver setup as cloth_franka
        self.solver = newton.solvers.SolverFeatherstone(self.model, update_mass_matrix_interval=self.sim_substeps)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        self.viewer.set_model(self.model)

        # not required for MuJoCo, but required for other solvers
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.capture()

    def create_articulation(self, builder):
        asset_path = newton.utils.download_asset("franka_emika_panda")

        builder.add_urdf(
            str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
            xform=wp.transform(
                (float(self.args.anchor_x), float(self.args.anchor_y), 0.0),
                wp.quat_identity(),
            ),
            floating=False,
            scale=1,  # unit: cm
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )

        # Use exact same setup as cloth_franka example
        builder.joint_q[:6] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307]

        # Find end effector body
        self.endeffector_id = builder.body_count - 3
        self.endeffector_offset = wp.transform(
            [0.0, 0.0, 0.22],
            wp.quat_identity(),
        )

    def create_rope(self, builder, length, radius, segments, rigidity, damping, end_weight):
        # segment span and capsule geometry
        segment_span = length / segments
        # capsule total length = 2*half_height + 2*radius = segment_span
        half_height = max(0.0, 0.5 * max(segment_span - 2.0 * radius, 0.0))
        tip_offset = half_height + radius

        # Create rope bodies
        bodies = []
        for i in range(segments):
            # place centers from top to bottom
            cz = float(self.args.anchor_height) - (i * segment_span + tip_offset)
            cx, cy = float(self.args.anchor_x), float(self.args.anchor_y)
            body = builder.add_body(xform=wp.transform(p=wp.vec3(cx, cy, cz), q=wp.quat_identity()))

            # Set rope material properties like cloth_franka
            cfg = newton.ModelBuilder.ShapeConfig()
            cfg.density = 0.2  # Same as cloth density in cloth_franka
            cfg.ke = 1e2      # Same stiffness as cloth_franka
            cfg.kd = 1.5e-6   # Same damping as cloth_franka
            cfg.mu = 0.5      # Same friction as cloth_franka
            builder.add_shape_capsule(body, radius=radius, half_height=half_height, cfg=cfg)

            bodies.append(body)

        # Add extra weight at the end by attaching an additional dense sphere to the last body
        if end_weight > 0.0:
            last = bodies[-1]
            weight_radius = max(radius * 1.1, (3.0 * end_weight / (4.0 * math.pi * 1000.0)) ** (1.0 / 3.0))  # heuristic
            weight_cfg = newton.ModelBuilder.ShapeConfig()
            weight_cfg.density = end_weight  # treat as density-like knob for simplicity
            builder.add_shape_sphere(last, radius=weight_radius, cfg=weight_cfg)

        # Attach first body to Franka end effector with a ball joint (free rotation)
        if bodies:
            first = bodies[0]
            builder.add_joint_ball(
                parent=self.endeffector_id,
                child=first,
                parent_xform=self.endeffector_offset,
                child_xform=wp.transform(p=wp.vec3(0.0, 0.0, +tip_offset), q=wp.quat_identity()),
                key="rope_anchor",
            )

        # Create D6 joints (ball-like) between consecutive bodies with angular stiffness
        angular_axes = _make_d6_ball_axes(rigidity=rigidity, damping=damping)
        for i in range(1, len(bodies)):
            parent = bodies[i - 1]
            child = bodies[i]
            builder.add_joint_d6(
                parent=parent,
                child=child,
                linear_axes=[],
                angular_axes=angular_axes,
                parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, -tip_offset), q=wp.quat_identity()),
                child_xform=wp.transform(p=wp.vec3(0.0, 0.0, +tip_offset), q=wp.quat_identity()),
                key=f"rope_link_{i-1}_{i}",
            )

    @staticmethod
    def add_args(parser):
        parser.add_argument("--length", type=float, default=0.3, help="Total rope length")
        parser.add_argument("--width", type=float, default=0.01, help="Rope width (diameter)")
        parser.add_argument("--segments", type=int, default=8, help="Number of rope segments")
        parser.add_argument("--rigidity", type=float, default=10.0, help="Angular stiffness for joints")
        parser.add_argument("--damping", type=float, default=1.0, help="Angular damping for joints")
        parser.add_argument("--end-weight", type=float, default=0.0, help="Additional weight at rope end")
        parser.add_argument("--anchor-height", type=float, default=0.4, help="World Z position of rope anchor")
        parser.add_argument("--anchor-x", type=float, default=0.0, help="World X position of Franka base")
        parser.add_argument("--anchor-y", type=float, default=0.0, help="World Y position of Franka base")
        parser.add_argument("--no-rope", action="store_true", help="Disable rope, show only Franka robot")

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

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

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
