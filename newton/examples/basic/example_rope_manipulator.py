import warp as wp
import numpy as np

import newton
import newton.examples


class Example:
    def __init__(self, viewer):
        # setup simulation parameters first
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        builder = newton.ModelBuilder()

        builder.add_articulation(key="pendulum")

        hx = 0.2
        hy = 0.05
        hz = 0.05

        # Create fixed manipulator base link (link 0) that is 3x longer than other links
        hx_arm = 3.0 * hx
        base_link = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 6.0), q=wp.quat_identity()),
            mass=0.0,  # fixed/kinematic
        )
        builder.add_shape_box(base_link, hx=hx_arm, hy=hy, hz=hz)
        # Explicitly fix base to world with a fixed joint
        # Use a very stiff D6 joint at 90 degrees (pi/2 about Z)
        builder.add_joint_d6(
            parent=-1,
            child=base_link,
            # No linear DOF, only angular, very stiff
            angular_axes=[
                builder.JointDofConfig(axis=newton.Axis.X, target_ke=1e4, target_kd=1e4),
                builder.JointDofConfig(axis=newton.Axis.Y, target_ke=1e4, target_kd=1e4),
                builder.JointDofConfig(axis=newton.Axis.Z, target_ke=1e4, target_kd=1e4),
            ],
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 6.0), q=wp.quat_identity()),
            # Rotate base link 90 degrees about Z
            child_xform=wp.transform(
                p=wp.vec3(0.0, 0.0, 0.0),
                q=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi / 2),
            ),
            key="fixed_base_link",
        )

        num_links = 12
        links = []
        
        # Create all the links
        for i in range(num_links):
            link = builder.add_body()
            builder.add_shape_box(link, hx=hx, hy=hy, hz=hz)
            links.append(link)

        for i in range(num_links):
            if i == 0:
                # Use D6 joint with only angular DOF for 3D rotation with damping only
                angular_axes = [
                    builder.JointDofConfig(axis=newton.Axis.X, target_ke=1.0, target_kd=50.0),
                    builder.JointDofConfig(axis=newton.Axis.Y, target_ke=1.0, target_kd=50.0),
                    builder.JointDofConfig(axis=newton.Axis.Z, target_ke=1.0, target_kd=50.0),
                ]
                builder.add_joint_d6(
                    parent=base_link,
                    child=links[i],
                    angular_axes=angular_axes,
                    # attach at the tip of the base link along +X
                    parent_xform=wp.transform(p=wp.vec3(hx_arm, 0.0, 0.0), q=wp.quat_identity()),
                    child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
                )
            else:
                # Use D6 joint with only angular DOF for 3D rotation with stiffness
                angular_axes = [
                    builder.JointDofConfig(axis=newton.Axis.X, target_ke=500.0, target_kd=50.0),
                    builder.JointDofConfig(axis=newton.Axis.Y, target_ke=500.0, target_kd=50.0),
                    builder.JointDofConfig(axis=newton.Axis.Z, target_ke=500.0, target_kd=50.0),
                ]
                builder.add_joint_d6(
                    parent=links[i-1],
                    child=links[i],
                    angular_axes=angular_axes,
                    parent_xform=wp.transform(p=wp.vec3(hx, 0.0, 0.0), q=wp.quat_identity()),
                    child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
                )

        builder.add_ground_plane()

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
    example = Example(viewer)

    newton.examples.run(example, args)
