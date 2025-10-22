import warp as wp
import numpy as np
import math

import newton
import newton.examples
from newton.selection import ArticulationView
import newton.utils


@wp.kernel
def _trajectory_interpolation_kernel(
    start_config: wp.array2d(dtype=wp.float32),
    end_config: wp.array2d(dtype=wp.float32),
    current_timestep: wp.array(dtype=wp.int32),
    total_timesteps: wp.int32,
    joint_target_out: wp.array2d(dtype=wp.float32),
):
    env_idx = wp.tid()
    t = wp.float32(current_timestep[env_idx])
    total_t = wp.float32(total_timesteps)
    
    progress = wp.clamp(t / total_t, 0.0, 1.0)
    
    num_dofs = start_config.shape[1]
    for d in range(num_dofs):
        start_val = start_config[env_idx, d]
        end_val = end_config[env_idx, d]
        joint_target_out[env_idx, d] = start_val + (end_val - start_val) * progress


@wp.kernel
def _update_min_distance_kernel(
    body_transforms: wp.array(dtype=wp.transform),
    rope_start_idx: wp.array(dtype=wp.int32),
    rope_segments: wp.array(dtype=wp.int32),
    target_point: wp.vec3,
    min_distances: wp.array(dtype=wp.float32),
):
    env_idx = wp.tid()
    start_idx = rope_start_idx[env_idx]
    num_segments = rope_segments[env_idx]
    
    current_min = min_distances[env_idx]
    
    for i in range(num_segments):
        body_idx = start_idx + i
        transform = body_transforms[body_idx]
        pos = wp.transform_get_translation(transform)
        dist = wp.length(pos - target_point)
        current_min = wp.min(current_min, dist)
    
    min_distances[env_idx] = current_min


# (no scatter kernel; we will restrict selection via exclude_joints in ArticulationView)

class RopeRobotGym:
    def __init__(self, trajectory, total_time, render=False):
        """
        Initialize the gym environment.
        
        Args:
            trajectory: shape (num_ur10_dofs, 2) tensor - start and end joint configurations
            total_time: float - duration to interpolate from start to end
            render: bool - whether to enable rendering (affects simulation speed)
        """
        # Convert trajectory to warp array if it's not already
        if not isinstance(trajectory, wp.array):
            trajectory = wp.array(trajectory, dtype=wp.float32, device=wp.get_device())
        
        self.trajectory = trajectory  # (num_dofs, 2)
        self.total_time = total_time
        self.render_flag = render
        
        # Setup simulation parameters
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.num_timesteps = int(total_time / self.frame_dt)
        
        self.device = wp.get_device()
        self.model = None
        self.solver = None
        self.state_0 = None
        self.state_1 = None
        self.control = None
        self.contacts = None
        self.ur10 = None
        self.graph = None
        
        # Will be set in reset()
        self.n_envs = 0
        self.rope_start_indices = None
        self.rope_segments = None
        self.current_timestep = None
        self.target_point = wp.vec3(1.5, 0.0, 0.6)
        self.min_distances = None  # Track minimum distances in real-time

    def reset(self, rope_configs):
        """
        Reset the environment with n parallel robot-rope systems.
        
        Args:
            rope_configs: list of n dictionaries with rope attributes:
                - length, width, segments, density, ke, kd, mu
        """
        self.n_envs = len(rope_configs)
        
        # Initialize viewer if rendering is enabled
        if self.render_flag:
            self.viewer, _ = newton.examples.init()
        
        # Create a single robot-rope environment template
        robot_rope = newton.ModelBuilder()
        
        # Load UR10 asset
        asset_path = newton.utils.download_asset("universal_robots_ur10")
        asset_file = str(asset_path / "usd" / "ur10_instanceable.usda")
        height = 1.2
        
        # Add UR10 robot
        robot_rope.add_usd(
            asset_file,
            xform=wp.transform(wp.vec3(0.0, 0.0, height)),
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            load_non_physics_prims=True,
            hide_collision_shapes=True,
        )
        
        # Configure UR10 joints
        for i in range(len(robot_rope.joint_dof_mode)):
            robot_rope.joint_dof_mode[i] = newton.JointMode.TARGET_POSITION
            robot_rope.joint_target_ke[i] = 500.0
            robot_rope.joint_target_kd[i] = 50.0
        
        # Add pedestal
        robot_rope.add_shape_cylinder(
            -1,
            xform=wp.transform(wp.vec3(0.0, 0.0, height / 2.0)),
            half_height=height / 2.0,
            radius=0.08,
        )
        
        # Add pole
        robot_rope.add_shape_cylinder(
            -1,
            xform=wp.transform(wp.vec3(1.5, 0.0, height / 2.0)),
            half_height=height,
            radius=0.02,
        )
        
        # Get end-effector body (last body added)
        ee_body = robot_rope.body_count - 1
        
        # Add visual sphere at end-effector
        vis_cfg = newton.ModelBuilder.ShapeConfig()
        vis_cfg.density = 0.0
        robot_rope.add_shape_sphere(ee_body, radius=0.03, cfg=vis_cfg)
        
        # Build rope (using first rope config as template)
        rope_config = rope_configs[0]  # Use first config as template
        length = rope_config.get('length', 1.4)
        width = rope_config.get('width', 0.02)
        segments = rope_config.get('segments', 25)
        density = rope_config.get('density', 0.2)
        ke = rope_config.get('ke', 1e6)
        kd = rope_config.get('kd', 1e4)
        mu = rope_config.get('mu', 0.6)
        
        radius = 0.5 * width
        segment_span = length / segments
        half_height = max(0.0, 0.5 * max(segment_span - 2.0 * radius, 0.0))
        tip_offset = half_height + radius
        
        # Create rope bodies
        bodies = []
        for i in range(segments):
            cz = float(height) - (i * segment_span + tip_offset)
            body = robot_rope.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, cz), q=wp.quat_identity()))
            
            cfg = newton.ModelBuilder.ShapeConfig()
            cfg.density = density
            cfg.ke = ke
            cfg.kd = kd
            cfg.mu = mu
            robot_rope.add_shape_capsule(body, radius=radius, half_height=half_height, cfg=cfg)
            bodies.append(body)
        
        # Attach first rope body to UR10 end-effector
        if bodies:
            first = bodies[0]
            robot_rope.add_joint_ball(
                parent=ee_body,
                child=first,
                child_xform=wp.transform(p=wp.vec3(0.0, 0.0, -tip_offset), q=wp.quat_identity()),
                key="rope_anchor_ur10",
            )
        
        # D6 joints along the rope
        for i in range(1, len(bodies)):
            parent = bodies[i - 1]
            child = bodies[i]
            
            robot_rope.add_joint_d6(
                parent=parent,
                child=child,
                child_xform=wp.transform(p=wp.vec3(0.0, 0.0, -segment_span), q=wp.quat_identity()),
                angular_axes=[
                    newton.ModelBuilder.JointDofConfig(axis=wp.vec3(1.0, 0.0, 0.0), limit_lower=-0.3, limit_upper=0.3),
                    newton.ModelBuilder.JointDofConfig(axis=wp.vec3(0.0, 1.0, 0.0), limit_lower=-0.3, limit_upper=0.3),
                    newton.ModelBuilder.JointDofConfig(axis=wp.vec3(0.0, 0.0, 1.0), limit_lower=0.0, limit_upper=0.0),
                ],
                key=f"rope_link_{i-1}_{i}",
            )
        
        # Create main builder and replicate the robot-rope template
        builder = newton.ModelBuilder()
        builder.replicate(robot_rope, self.n_envs, spacing=(3.0, 3.0, 0))
        builder.add_ground_plane()
        
        # Finalize model
        self.model = builder.finalize()
        self.solver = newton.solvers.SolverXPBD(self.model)
        
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)
        
        if self.render_flag and self.viewer:
            self.viewer.set_model(self.model)
        
        # Initialize forward kinematics
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        
        # Create articulation view for all UR10s
        self.ur10 = ArticulationView(
            self.model,
            pattern="*ur10*",
            exclude_joints=["rope_anchor_ur10", "rope_link_*"],
            exclude_joint_types=[newton.JointType.DISTANCE],
        )
        
        # Verify we have the right number of articulations
        assert self.ur10.count == self.n_envs, f"Expected {self.n_envs} articulations, got {self.ur10.count}"
        
        # Initialize control arrays
        self.joint_target = self.ur10.get_attribute("joint_target", self.control)
        self.joint_target_out = wp.zeros((self.n_envs, self.ur10.joint_dof_count), dtype=wp.float32, device=self.device)
        self.current_timestep = wp.zeros(self.n_envs, dtype=wp.int32, device=self.device)
        
        # Pre-compute trajectory arrays for all environments (avoid numpy conversion in simulation loop)
        start_vals = self.trajectory[:, 0].numpy()  # (num_dofs,)
        end_vals = self.trajectory[:, 1].numpy()    # (num_dofs,)
        
        # Use numpy broadcasting to create arrays for all environments
        start_config_np = np.tile(start_vals, (self.n_envs, 1))  # (n_envs, num_dofs)
        end_config_np = np.tile(end_vals, (self.n_envs, 1))      # (n_envs, num_dofs)
        
        self.start_config = wp.array(start_config_np, dtype=wp.float32, device=self.device)
        self.end_config = wp.array(end_config_np, dtype=wp.float32, device=self.device)
        
        # Initialize real-time minimum distance tracking
        self.min_distances = wp.full(self.n_envs, 1e6, dtype=wp.float32, device=self.device)
        
        # Calculate rope start indices (each environment has the same structure)
        bodies_per_env = robot_rope.body_count
        self.rope_start_indices = wp.array([i * bodies_per_env + ee_body + 1 for i in range(self.n_envs)], dtype=wp.int32, device=self.device)
        self.rope_segments = wp.array([segments] * self.n_envs, dtype=wp.int32, device=self.device)
        
        # Capture for GPU optimization
        self.capture()
        
        return self._get_observation()

    def capture(self):
        """Capture GPU graph for optimization"""
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate(0)  # Use timestep 0 for capture
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self, current_timestep):
        """Run one simulation step"""
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            
            # Apply forces if rendering
            if self.render_flag and self.viewer:
                self.viewer.apply_forces(self.state_0)
            
            # Fill current timestep array for all environments
            self.current_timestep.fill_(current_timestep)
            
            # Interpolate joint targets using pre-computed arrays
            wp.launch(
                _trajectory_interpolation_kernel,
                dim=self.n_envs,
                inputs=[
                    self.start_config,
                    self.end_config,
                    self.current_timestep,
                    self.num_timesteps,
                ],
                outputs=[self.joint_target_out],
                device=self.device,
            )
            
            # Set joint targets
            self.ur10.set_attribute("joint_target", self.control, self.joint_target_out)
            
            # Step simulation
            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            
            # Update minimum distances in real-time
            wp.launch(
                _update_min_distance_kernel,
                dim=self.n_envs,
                inputs=[
                    self.state_0.body_q,  # This contains transform data (pos + quat)
                    self.rope_start_indices,
                    self.rope_segments,
                    self.target_point,
                    self.min_distances,
                ],
                device=self.device,
            )
            
            # Swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self, action=None):
        """
        Execute one step of the simulation.
        
        Args:
            action: shape (n, num_ur10_dofs, 2) - new trajectory for each environment
        """
        # Reset minimum distances for new episode
        self.min_distances.fill_(1e6)
        
        if action is not None:
            # Update trajectory if provided
            self.trajectory = wp.array(action, dtype=wp.float32, device=self.device)
            
            # Recompute trajectory arrays for all environments
            start_vals = self.trajectory[:, 0].numpy()  # (num_dofs,)
            end_vals = self.trajectory[:, 1].numpy()    # (num_dofs,)
            
            # Use numpy broadcasting to create arrays for all environments
            start_config_np = np.tile(start_vals, (self.n_envs, 1))  # (n_envs, num_dofs)
            end_config_np = np.tile(end_vals, (self.n_envs, 1))      # (n_envs, num_dofs)
            
            self.start_config = wp.array(start_config_np, dtype=wp.float32, device=self.device)
            self.end_config = wp.array(end_config_np, dtype=wp.float32, device=self.device)
        
        # Run simulation for all timesteps
        for timestep in range(self.num_timesteps):
            # Update current timestep for all environments
            self.current_timestep.fill_(timestep)
            
            # Run simulation step
            if self.graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate(timestep)
            
        
        # Compute scores
        scores = self.compute_score()
        
        return self._get_observation(), scores, True, {}

    def compute_score(self):
        """Return minimum distance scores for all environments"""
        return self.min_distances.numpy()

    def _get_observation(self):
        """Get current observation (placeholder for now)"""
        return {}

    def render(self):
        """Render the current state"""
        if self.render_flag and self.viewer:
            self.viewer.begin_frame(0.0)
            self.viewer.log_state(self.state_0)
            self.viewer.log_contacts(self.contacts, self.state_0)
            self.viewer.end_frame()


if __name__ == "__main__":
    # Example usage of the gym environment
    import numpy as np
    
    # Create example trajectory (6 DOF UR10: start and end configurations)
    trajectory = np.array([
        [0.0, 0.0],      # joint 0
        [-1.57, -1.57],  # joint 1  
        [0.0, 0.0],      # joint 2
        [-1.57, -1.57],  # joint 3
        [0.0, 0.0],      # joint 4
        [0.0, 0.0],      # joint 5
    ])
    
    # Create gym environment
    env = RopeRobotGym(trajectory, total_time=2.0, render=True)
    
    # Example rope configurations
    rope_configs = [
        {'length': 1.4, 'width': 0.02, 'segments': 25, 'density': 0.2, 'ke': 1e6, 'kd': 1e4, 'mu': 0.6},
        {'length': 1.2, 'width': 0.015, 'segments': 20, 'density': 0.3, 'ke': 1.5e6, 'kd': 1.5e4, 'mu': 0.5},
    ]
    
    # Reset environment
    obs = env.reset(rope_configs)
    
    # Run simulation
    obs, scores, done, info = env.step()
    
    print(f"Scores: {scores}")
    print(f"Minimum distances to target point: {scores}")
    
    # Render if enabled
    if env.render_flag:
        print("Rendering is enabled - simulation will be slower but visual")
        env.render()  # Render the final state
    else:
        print("Rendering is disabled - simulation will be faster")