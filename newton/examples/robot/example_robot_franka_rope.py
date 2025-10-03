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
# Example Franka + Rope (capsule chain)
#
# Builds a Franka arm and a parameterizable rope made of capsule links
# connected by compliant ball-like joints (D6 with angular axes).
# The rope is pinned to the arm's end-effector and can be configured via
# CLI: length, radius (width), rigidity (stiffness), damping, and end mass.
#
# Command: python -m newton.examples robot_franka_rope --num-envs 1 \
#          --rope-length 0.6 --rope-radius 0.01 --rope-segments 12 \
#          --rope-rigidity 100.0 --rope-damping 5.0 --rope-end-mass 0.2
###########################################################################

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils
import os


def _try_find_ee_body_index(builder: newton.ModelBuilder) -> int:
    """Best-effort heuristic to find Franka end-effector body index.

    Uses the body keys created by the importer and looks for a likely EE name.
    Fallback: pick the last non-world body.
    """
    candidates = [
        "hand",  # common for franka hand
        "panda_hand",
        "fr3_hand",
        "link7",
        "panda_link7",
    ]
    # World is not represented as a body; parent is -1
    for i, key in enumerate(builder.body_key):
        k = key.lower() if key is not None else ""
        if any(c in k for c in candidates):
            return i
    return builder.body_count - 1


def _add_franka(builder: newton.ModelBuilder, xform: Optional[wp.transform] = None) -> None:
    asset_path = newton.utils.download_asset("franka_emika_panda")
    builder.add_urdf(
        str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
        xform=xform or wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        floating=False,
        scale=1.0,
        enable_self_collisions=False,
        collapse_fixed_joints=True,
        force_show_colliders=False,
    )


def _add_capsule_link(
    builder: newton.ModelBuilder,
    pos,
    half_height: float,
    radius: float,
    mass: float,
    key: Optional[str] = None,
) -> int:
    body = builder.add_body(xform=wp.transform(wp.vec3(pos[0], pos[1], pos[2]), wp.quat_identity()), mass=mass, key=key)
    builder.add_shape_capsule(
        body,
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        radius=radius,
        half_height=half_height,
    )
    return body


def _add_ball_like_joint(
    builder: newton.ModelBuilder,
    parent: int,
    child: int,
    parent_xform,
    child_xform,
    stiffness: float,
    damping: float,
    key: Optional[str] = None,
) -> int:
    """Adds a 3-DoF angular joint using D6 with X,Y,Z angular axes.

    This mimics a ball joint while allowing per-axis target stiffness/damping
    to approximate compliance.
    """
    cfg = newton.ModelBuilder.JointDofConfig
    # Target around zero angle, position-control mode for compliance
    ax_x = cfg(axis=newton.Axis.X, target=0.0, target_ke=stiffness, target_kd=damping, mode=newton.JointMode.TARGET_POSITION)
    ax_y = cfg(axis=newton.Axis.Y, target=0.0, target_ke=stiffness, target_kd=damping, mode=newton.JointMode.TARGET_POSITION)
    ax_z = cfg(axis=newton.Axis.Z, target=0.0, target_ke=stiffness, target_kd=damping, mode=newton.JointMode.TARGET_POSITION)
    jid = builder.add_joint_d6(
        parent=parent,
        child=child,
        linear_axes=[],  # no translation (acts like a ball joint)
        angular_axes=[ax_x, ax_y, ax_z],
        parent_xform=parent_xform,
        child_xform=child_xform,
        key=key,
        collision_filter_parent=True,
    )
    return jid


def build_franka_with_rope(
    num_envs: int,
    rope_length: float,
    rope_radius: float,
    rope_segments: int,
    rope_rigidity: float,
    rope_damping: float,
    rope_end_mass: float,
) -> newton.ModelBuilder:
    """Creates a builder with a Franka arm and a capsule-chain rope pinned to its EE.

    - Rope is placed initially hanging below the EE.
    - Each segment is a capsule aligned along +Z in its local frame.
    """
    articulation = newton.ModelBuilder()
    articulation.default_shape_cfg.mu = 0.7
    articulation.default_shape_cfg.ke = 5.0e4
    articulation.default_shape_cfg.kd = 5.0e2
    articulation.default_joint_cfg.limit_ke = 1.0e3
    articulation.default_joint_cfg.limit_kd = 1.0e1

    _add_franka(articulation)

    # Set default control modes and gains on articulation dofs (Franka)
    for i in range(len(articulation.joint_dof_mode)):
        articulation.joint_dof_mode[i] = newton.JointMode.TARGET_POSITION
        articulation.joint_target_ke[i] = 50.0
        articulation.joint_target_kd[i] = 5.0

    # Find end-effector body to attach rope to
    ee_body = _try_find_ee_body_index(articulation)

    # Build rope bodies (in a separate builder to easily replicate per env later)
    rope_builder = newton.ModelBuilder()

    total_length = max(rope_length, 1e-4)
    nseg = max(int(rope_segments), 1)
    seg_len = total_length / nseg
    half_h = 0.5 * seg_len

    # Mass distribution: evenly over segments, with optional heavier end mass
    linear_density = 1.0  # base density scale; mass per segment set below
    rope_mass_total = linear_density * total_length * (math.pi * rope_radius * rope_radius)
    base_seg_mass = rope_mass_total / nseg if nseg > 0 else 0.0

    bodies = []
    # Layout along negative Y in Franka frame (will shift later when attaching)
    for i in range(nseg):
        m = base_seg_mass
        if i == nseg - 1:
            m += max(rope_end_mass, 0.0)
        # segments centered and spaced so that joints attach end-to-end
        pos = wp.vec3(0.0, -(i * seg_len + half_h), 0.0)
        b = _add_capsule_link(rope_builder, pos=pos, half_height=half_h, radius=rope_radius, mass=m, key=f"rope_seg_{i}")
        bodies.append(b)

    # Connect rope segments by compliant angular joints at endpoints
    for i in range(nseg - 1):
        parent = bodies[i]
        child = bodies[i + 1]
        # Parent attach at its -Z end, child at its +Z end, but our capsules are aligned along +Z.
        parent_xf = wp.transform(wp.vec3(0.0, -half_h, 0.0), wp.quat_identity())
        child_xf = wp.transform(wp.vec3(0.0, +half_h, 0.0), wp.quat_identity())
        _add_ball_like_joint(
            rope_builder,
            parent=parent,
            child=child,
            parent_xform=parent_xf,
            child_xform=child_xf,
            stiffness=rope_rigidity,
            damping=rope_damping,
            key=f"rope_joint_{i}",
        )

    # Now merge articulation and rope into scene builder with replication
    scene = newton.ModelBuilder()
    offsets = newton.examples.compute_env_offsets(num_envs)

    for env_idx in range(num_envs):
        # Add one Franka per env
        art_copy_start = scene.body_count
        off = offsets[env_idx]
        scene.add_builder(articulation, xform=wp.transform(wp.vec3(float(off[0]), float(off[1]), float(off[2])), wp.quat_identity()))
        # Determine the EE body index in the scene for this env
        ee_scene_index = art_copy_start + ee_body

        # Add one rope per env
        rope_copy_start = scene.body_count
        off = offsets[env_idx]
        scene.add_builder(rope_builder, xform=wp.transform(wp.vec3(float(off[0]), float(off[1]), float(off[2])), wp.quat_identity()))
        first_seg_scene_index = rope_copy_start + bodies[0]

        # Pin rope's first segment to Franka EE via 3-DoF angular joint
        # Offset so the first segment top meets EE frame
        parent_xf = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        child_xf = wp.transform(wp.vec3(0.0, +half_h, 0.0), wp.quat_identity())
        _add_ball_like_joint(
            scene,
            parent=ee_scene_index,
            child=first_seg_scene_index,
            parent_xform=parent_xf,
            child_xform=child_xf,
            stiffness=rope_rigidity,
            damping=rope_damping,
            key=f"rope_pin_{env_idx}",
        )

    # Ground plane (optional)
    scene.add_ground_plane()
    return scene


class Example:
    def __init__(
        self,
        viewer,
        num_envs: int = 1,
        rope_length: float = 0.6,
        rope_radius: float = 0.01,
        rope_segments: int = 12,
        rope_rigidity: float = 100.0,
        rope_damping: float = 5.0,
        rope_end_mass: float = 0.2,
        save_video: bool = False,
        video_path: str | None = None,
    ):
        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        builder = build_franka_with_rope(
            num_envs=num_envs,
            rope_length=rope_length,
            rope_radius=rope_radius,
            rope_segments=rope_segments,
            rope_rigidity=rope_rigidity,
            rope_damping=rope_damping,
            rope_end_mass=rope_end_mass,
        )

        self.model = builder.finalize()

        # Use MuJoCo solver for robust contacts
        self.solver = newton.solvers.SolverMuJoCo(self.model, iterations=200, ls_iterations=50)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        # Mild gravity to encourage swinging if robot is static
        self.model.gravity = np.array((0.0, 0.0, -9.81), dtype=np.float32)

        # Control modes and gains were set on the articulation builder

        self.swing_indices = self._pick_wrist_like_joint_indices()
        # Local buffer for joint targets
        self.joint_target_np = np.zeros(self.model.joint_dof_count, dtype=np.float32)

        self.viewer.set_model(self.model)
        # Optional offscreen renderer for video recording
        self.renderer = None
        if save_video:
            out_path = video_path
            if not out_path:
                out_path = os.path.join(os.path.dirname(__file__), "franka_rope.mp4")
            self.renderer = newton.viewer.RendererOpenGL(self.model, path=out_path, fps=self.fps)
        self._capture()

    def _pick_wrist_like_joint_indices(self) -> list[int]:
        """Heuristically pick 1-2 last arm joints to drive with a sinusoid.
        Falls back to the last available joint dofs.
        """
        dof = self.model.joint_dof_count
        if dof <= 0:
            return []
        # take last 2 dofs (often wrist and forearm roll)
        k = min(2, dof)
        return [dof - i - 1 for i in range(k)]

    def _capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self._simulate_inner()
            self.graph = capture.graph

    def _simulate_inner(self):
        self.contacts = self.model.collide(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # Simple arm motion to pump energy into the rope
            self._drive_franka(self.sim_time)

            # External forces (picking, wind) from viewer
            self.viewer.apply_forces(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _drive_franka(self, t: float):
        if not self.swing_indices:
            return
        amp = 0.5  # rad
        freq = 0.7  # Hz
        val = amp * math.sin(2.0 * math.pi * freq * t)
        # Set target for chosen dofs
        for idx in self.swing_indices:
            if 0 <= idx < self.joint_target_np.shape[0]:
                self.joint_target_np[idx] = val
        if self.control.joint_target is not None:
            wp.copy(self.control.joint_target, wp.array(self.joint_target_np, dtype=float))

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._simulate_inner()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()
        if self.renderer is not None:
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()

    def test(self):
        pass


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--num-envs", type=int, default=1, help="Total number of simulated environments.")
    parser.add_argument("--rope-length", type=float, default=0.6, help="Total rope length (meters).")
    parser.add_argument("--rope-radius", type=float, default=0.01, help="Rope radius (meters).")
    parser.add_argument("--rope-segments", type=int, default=12, help="Number of capsule segments in the rope.")
    parser.add_argument("--rope-rigidity", type=float, default=100.0, help="Angular stiffness of rope joints.")
    parser.add_argument("--rope-damping", type=float, default=5.0, help="Angular damping of rope joints.")
    parser.add_argument("--rope-end-mass", type=float, default=0.2, help="Additional mass at rope tip (kg).")
    parser.add_argument("--save-video", action="store_true", help="Record a video alongside visualization.")
    parser.add_argument("--video-path", type=str, default=None, help="Output video path (default: example folder).")

    viewer, args = newton.examples.init(parser)

    example = Example(
        viewer,
        num_envs=args.num_envs,
        rope_length=args.rope_length,
        rope_radius=args.rope_radius,
        rope_segments=args.rope_segments,
        rope_rigidity=args.rope_rigidity,
        rope_damping=args.rope_damping,
        rope_end_mass=args.rope_end_mass,
        save_video=args.save_video,
        video_path=args.video_path,
    )

    newton.examples.run(example)
    if getattr(example, "renderer", None) is not None:
        example.renderer.save()
