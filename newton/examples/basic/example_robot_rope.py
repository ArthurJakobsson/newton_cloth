import warp as wp
import numpy as np
import math

import newton
import newton.examples
import newton.utils


class RopeRobot:
    def __init__(self, viewer):
        # setup simulation parameters first
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

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

        # Add a cylinder pedestal centered under the robot base
        builder.add_shape_cylinder(
            -1,
            xform=wp.transform(wp.vec3(0.0, 0.0, height / 2.0)),
            half_height=height / 2.0,
            radius=0.08,
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

        # revolute joints along the rope
        for i in range(1, len(bodies)):
            parent = bodies[i - 1]
            child = bodies[i]
            
            builder.add_joint_revolute(
                parent=parent,
                child=child,
                axis=wp.vec3(1.0, 0.0, 0.0), #try alternating the revolute joints
                parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, -tip_offset), q=wp.quat_identity()),
                child_xform=wp.transform(p=wp.vec3(0.0, 0.0, +tip_offset), q=wp.quat_identity()),
                limit_lower=-0.005,  # Even smaller limit range (±3°)
                limit_upper=0.005,
                limit_ke=1e10,       # Much higher stiffness for limits
                limit_kd=1e6,       # Much higher damping for limits
                target_ke=0.0,      # No target control
                target_kd=0.0,
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
