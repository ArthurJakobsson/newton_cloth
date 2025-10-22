import warp as wp
import numpy as np
import math

import newton
import newton.examples
from newton.selection import ArticulationView
import newton.utils


@wp.kernel
def _drive_single_joint_target_kernel(
    joint_target_in: wp.array2d(dtype=wp.float32),
    joint_initial: wp.array2d(dtype=wp.float32),
    time: wp.array(dtype=wp.float32),
    dt: wp.float32,
    joint_index: int,
    swivel_joint_index: int,
    rotation_duration: wp.float32,
    joint_target_out: wp.array2d(dtype=wp.float32),
):
    env_idx = wp.tid()
    t = time[env_idx]
    time[env_idx] = t + dt

    num_dofs = joint_target_in.shape[1]
    for d in range(num_dofs):
        joint_target_out[env_idx, d] = joint_target_in[env_idx, d]

    if joint_index >= 0 and joint_index < num_dofs:
        if t < rotation_duration:
            progress = t / rotation_duration
            progress = wp.clamp(progress, 0.0, 1.0)
            angle = joint_initial[env_idx, joint_index] - 0.5 * wp.pi * progress  # 90 degrees over duration
            joint_target_out[env_idx, joint_index] = angle
        else:
            joint_target_out[env_idx, joint_index] = joint_initial[env_idx, joint_index] - 0.5 * wp.pi
    time_offset = 10.0
    if swivel_joint_index >= 0 and swivel_joint_index < num_dofs:
        if t<time_offset:
            progress = t / time_offset
            progress = wp.clamp(progress, 0.0, 1.0)
            angle = joint_initial[env_idx, swivel_joint_index] - 0.4 * wp.pi * progress  # 90 degrees over duration
            joint_target_out[env_idx, swivel_joint_index] = angle
        elif (t > time_offset) and (t < rotation_duration + time_offset):
            progress = (t - time_offset) / rotation_duration
            progress = wp.clamp(progress, 0.0, 1.0)
            angle = joint_initial[env_idx, swivel_joint_index] + 0.4 * wp.pi * progress  # 90 degrees over duration
            joint_target_out[env_idx, swivel_joint_index] = angle
        else:
            joint_target_out[env_idx, swivel_joint_index] = joint_initial[env_idx, swivel_joint_index] + 0.4 * wp.pi


# (no scatter kernel; we will restrict selection via exclude_joints in ArticulationView)

class RopeRobot:
    def __init__(self, viewer, target_joint_index: int = 2, swivel_joint_index: int = 0, rotation_duration: float = 5.0):
        # setup simulation parameters first
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.device = wp.get_device()
        self.target_joint_index = int(target_joint_index)
        self.swivel_joint_index = int(swivel_joint_index)
        self.rotation_duration = float(rotation_duration)

        builder = newton.ModelBuilder()

        # add ground plane
        builder.add_ground_plane()

        # -------------------------------------------------
        # Load UR10 robot and get end-effector body index
        # -------------------------------------------------
        asset_path = newton.utils.download_asset("universal_robots_ur10")
        asset_file = str(asset_path / "usd" / "ur10_instanceable.usda")
        # mount the UR10 on a pedestal at some height
        height = 1.2
        before_bodies = builder.body_count
        before_dofs = builder.joint_dof_count
        add_info = builder.add_usd(
            asset_file,
            xform=wp.transform(wp.vec3(0.0, 0.0, height)),
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            load_non_physics_prims=True,
            hide_collision_shapes=True,
        )
        after_bodies = builder.body_count
        ur10_body_indices = list(range(before_bodies, after_bodies))
        # Heuristic: end-effector is the last body added when bodies_follow_joint_ordering=True
        ee_body = ur10_body_indices[-1]

        # Put UR10 joints into TARGET_POSITION mode with reasonable gains
        new_dofs = builder.joint_dof_count - before_dofs
        for i in range(before_dofs, before_dofs + new_dofs):
            builder.joint_dof_mode[i] = newton.JointMode.TARGET_POSITION
            builder.joint_target_ke[i] = 500.0
            builder.joint_target_kd[i] = 50.0

        # Add a cylinder pedestal centered under the robot base
        builder.add_shape_cylinder(
            -1,
            xform=wp.transform(wp.vec3(0.0, 0.0, height / 2.0)),
            half_height=height / 2.0,
            radius=0.08,
        )

        # the pole
        builder.add_shape_cylinder(
            -1,
            xform=wp.transform(wp.vec3(1.5, 0.0, height / 2.0)),
            half_height=height,
            radius=0.02,
        )

        # optional: small sphere visual at the end-effector for clarity
        vis_cfg = newton.ModelBuilder.ShapeConfig()
        vis_cfg.density = 0.0
        builder.add_shape_sphere(ee_body, radius=0.03, cfg=vis_cfg)

        # -------------------------------------------------
        # Build a rope and attach its first segment to UR10 EE
        # -------------------------------------------------
        length = 1.4
        width = 0.02
        radius = 0.5 * width
        segments = 25

        segment_span = length / segments
        half_height = max(0.0, 0.5 * max(segment_span - 2.0 * radius, 0.0))
        tip_offset = half_height + radius

        # create rope bodies below the end-effector
        bodies = []
        # estimate ee world pose at build time (we don't have dynamics yet, use identity relative)
        # place the first segment just below the EE
        ee_xform = wp.transform(p=wp.vec3(0.0, 0.0, height), q=wp.quat_identity())
        for i in range(segments):
            cz = float(height) - (i * segment_span + tip_offset)
            cx, cy = 0.0, 0.0
            body = builder.add_body(xform=wp.transform(p=wp.vec3(cx, cy, cz), q=wp.quat_identity()))

            cfg = newton.ModelBuilder.ShapeConfig()
            cfg.density = 0.2
            cfg.ke = 1e6      # Very high stiffness for rigid segments
            cfg.kd = 1e4      # Very high damping for stability
            cfg.mu = 0.6
            builder.add_shape_capsule(body, radius=radius, half_height=half_height, cfg=cfg)
            bodies.append(body)

        # attach first rope body to UR10 end-effector via a ball joint
        if bodies:
            first = bodies[0]
            builder.add_joint_ball(
                parent=ee_body,
                child=first,
                parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
                child_xform=wp.transform(p=wp.vec3(0.0, 0.0, +tip_offset), q=wp.quat_identity()),
                key="rope_anchor_ur10",
            )

        # D6 joints along the rope: two bending axes, locked twist
        for i in range(1, len(bodies)):
            parent = bodies[i - 1]
            child = bodies[i]

            builder.add_joint_d6(
                parent=parent,
                child=child,
                parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, -tip_offset), q=wp.quat_identity()),
                child_xform=wp.transform(p=wp.vec3(0.0, 0.0, +tip_offset), q=wp.quat_identity()),
                angular_axes=[
                    newton.ModelBuilder.JointDofConfig(axis=wp.vec3(1.0, 0.0, 0.0), limit_lower=-0.3, limit_upper=0.3),
                    newton.ModelBuilder.JointDofConfig(axis=wp.vec3(0.0, 1.0, 0.0), limit_lower=-0.3, limit_upper=0.3),
                    newton.ModelBuilder.JointDofConfig(axis=wp.vec3(0.0, 0.0, 1.0), limit_lower=0.0, limit_upper=0.0),  # lock twist
                ],
                key=f"rope_link_{i-1}_{i}",
            )

        self.model = builder.finalize()

        self.solver = newton.solvers.SolverXPBD(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        self.viewer.set_model(self.model)

        # not required for MuJoCo, but required for other solvers
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Create an articulation view for the UR10 only and exclude rope joints by name pattern
        # Rope joints were created with keys like "rope_link_*" and the ball joint "rope_anchor_ur10"
        self.ur10 = ArticulationView(
            self.model,
            pattern="*ur10*",
            exclude_joints=["rope_link_*", "rope_anchor_ur10"],
            exclude_joint_types=[newton.JointType.DISTANCE],
        )
        self.joint_target = self.ur10.get_attribute("joint_target", self.control)
        initial_q_np = self.ur10.get_attribute("joint_q", self.state_0).numpy()
        self.joint_initial = wp.array(initial_q_np, dtype=wp.float32, device=self.device)
        self.time_step = wp.zeros(1, dtype=wp.float32, device=self.device)
        self.joint_target_out = wp.zeros((1, self.ur10.joint_dof_count), dtype=wp.float32, device=self.device)

        self.capture()

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

            # Drive a single UR10 joint by updating joint_target (PD control)
            wp.launch(
                _drive_single_joint_target_kernel,
                dim=1,
                inputs=[
                    self.joint_target,
                    self.joint_initial,
                    self.time_step,
                    self.sim_dt,
                    self.target_joint_index,
                    self.swivel_joint_index,
                    self.rotation_duration,
                ],
                outputs=[self.joint_target_out],
                device=self.device,
            )
            self.joint_target.assign(self.joint_target_out)
            # Write PD targets; solver will move the robot accordingly
            self.ur10.set_attribute("joint_target", self.control, self.joint_target)
            # step with contacts
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
    viewer, args = newton.examples.init()

    # Create viewer and run
    example = RopeRobot(viewer)

    newton.examples.run(example, args)
