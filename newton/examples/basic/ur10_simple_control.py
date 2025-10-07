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
# Example Robot UR10
#
# Shows how to set up a simulation of a UR10 robot arm
# from a USD file using newton.ModelBuilder.add_usd() and
# applies a sinusoidal trajectory to the joint targets.
#
# Command: python -m newton.examples robot_ur10 --num-envs 16
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils
from newton.selection import ArticulationView

@wp.kernel
def update_joint_target_90_degree_kernel(
    joint_target: wp.array2d(dtype=wp.float32),
    time: wp.array(dtype=wp.float32),
    dt: wp.float32,
    rotation_sequence: wp.array2d(dtype=wp.float32),
    rotation_duration: wp.float32,
    # output
    updated_targets: wp.array2d(dtype=wp.float32),
):
    env_idx = wp.tid()
    t = time[env_idx]
    time[env_idx] = t + dt
    
    num_dofs = joint_target.shape[1]
    for dof in range(num_dofs):
        joint_start_time = rotation_sequence[env_idx, dof]
        joint_duration = rotation_duration
        if t >= joint_start_time and t < joint_start_time + joint_duration:
            progress = (t - joint_start_time) / joint_duration
            progress = wp.clamp(progress, 0.0, 1.0)
            
            rotation_amount = wp.pi / 2.0 * progress
            updated_targets[env_idx, dof] = rotation_amount #+ joint_target[env_idx, dof] 
        else:
            updated_targets[env_idx, dof] = joint_target[env_idx, dof]


class Example90DegreeRotation:
    def __init__(self, viewer, num_envs=4):
        self.fps = 25
        self.frame_dt = 1.0 / self.fps

        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_envs = num_envs
        self.viewer = viewer
        self.device = wp.get_device()

        ur10 = newton.ModelBuilder()

        asset_path = newton.utils.download_asset("universal_robots_ur10")
        asset_file = str(asset_path / "usd" / "ur10_instanceable.usda")
        height = 1.2
        ur10.add_usd(
            asset_file,
            xform=wp.transform(wp.vec3(0.0, 0.0, height)),
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            load_non_physics_prims=True,
            hide_collision_shapes=True,
        )
        # create a pedestal
        ur10.add_shape_cylinder(-1, xform=wp.transform(wp.vec3(0, 0, height / 2)), half_height=height / 2, radius=0.08)

        for i in range(len(ur10.joint_dof_mode)):
            ur10.joint_dof_mode[i] = newton.JointMode.TARGET_POSITION
            ur10.joint_target_ke[i] = 500
            ur10.joint_target_kd[i] = 50

        builder = newton.ModelBuilder()
        builder.replicate(ur10, self.num_envs, spacing=(2, 2, 0))

        # set random joint configurations
        rng = np.random.default_rng(42)
        joint_q = rng.uniform(-wp.pi, wp.pi, builder.joint_dof_count)
        builder.joint_q = joint_q.tolist()

        builder.add_ground_plane()

        self.model = builder.finalize()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None

        self.articulation_view = ArticulationView(
            self.model, "*ur10*", exclude_joint_types=[newton.JointType.FREE, newton.JointType.DISTANCE]
        )
        assert self.articulation_view.count == self.num_envs, (
            "Number of environments must match the number of articulations"
        )
        
        dof_count = self.articulation_view.joint_dof_count
        
        # Initialize joint targets to current positions
        self.joint_targets = self.articulation_view.get_attribute("joint_target", self.control)
        print(self.joint_targets)
        initial_positions = self.articulation_view.get_attribute("joint_q", self.state_0).numpy()
        
        # Set initial joint targets to current positions
        for env_idx in range(self.num_envs):
            for dof in range(dof_count):
                self.joint_targets.numpy()[env_idx, dof] = initial_positions[env_idx, dof]

        # Create rotation sequence - each joint rotates 90 degrees sequentially
        self.rotation_sequence = np.zeros((self.num_envs, dof_count), dtype=np.float32)
        self.rotation_duration = 5.0  # 5 seconds per rotation
        
        # Stagger the rotations so each joint rotates at a different time
        for env_idx in range(self.num_envs):
            for dof in range(dof_count):
                # Each joint starts rotating 1 second after the previous one
                self.rotation_sequence[env_idx, dof] = dof * 1.0 * self.rotation_duration
                # Add some variation per environment
                # self.rotation_sequence[env_idx, dof] += env_idx * 0.2

        self.rotation_sequence_wp = wp.array(self.rotation_sequence, dtype=wp.float32, device=self.device)
        self.time_step = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self.updated_targets = wp.zeros((self.num_envs, dof_count), dtype=wp.float32, device=self.device)

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            disable_contacts=True,
        )

        self.viewer.set_model(self.model)

        self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
                self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)

            # Update joint targets with 90-degree rotations
            wp.launch(
                update_joint_target_90_degree_kernel,
                dim=self.num_envs,
                inputs=[
                    self.joint_targets, 
                    self.time_step, 
                    self.sim_dt,
                    self.rotation_sequence_wp,
                    self.rotation_duration
                ],
                outputs=[self.updated_targets],
                device=self.device,
            )
            
            self.joint_targets.assign(self.updated_targets)
            # print("Targets", self.updated_targets)
            self.articulation_view.set_attribute("joint_target", self.control, self.joint_targets)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test(self):
        pass


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--num-envs", type=int, default=4, help="Total number of simulated environments.")

    viewer, args = newton.examples.init(parser)

    example = Example90DegreeRotation(viewer, args.num_envs)

    newton.examples.run(example, args)
