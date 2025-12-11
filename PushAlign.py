
import numpy as np
from collections import OrderedDict
from scipy.spatial.transform import Rotation as R

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial, new_site
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.utils.transform_utils import convert_quat


class PushAlign(ManipulationEnv):
    """
    A custom environment to push a block to a target 2D pose (position + orientation).
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=True,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        # --- OUR CUSTOM GOAL DEFINITION ---
        # The 2D (x, y) target position on the table
        self.target_pos = np.array([0.0, 0.2])
        # The target z-angle (in radians)
        self.target_angle = 0.0
        # --- END CUSTOM GOAL ---

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action=None):
        # --- 1. Get State ---
        push_obj_pos = np.array(self.sim.data.xpos[self.push_obj_body_id])
        push_obj_quat = np.array(self.sim.data.xquat[self.push_obj_body_id])
        eef_id = list(self.robots[0].eef_site_id.values())[0]
        gripper_loc = np.array(self.sim.data.site_xpos[eef_id])

        # --- 2. Reaching Reward (Shaped) ---
        # We use tanh so the max reward is 1.0 when touching, 0.0 when far.
        # We strictly limit how much this contributes so it doesn't overpower pushing.
        dist_to_obj = np.linalg.norm(gripper_loc - push_obj_pos)
        reach_reward = 1 - np.tanh(dist_to_obj)

        # --- 3. Position Reward (Pushing) ---
        # Reward based on how close object is to target
        dist_to_target = np.linalg.norm(push_obj_pos[:2] - self.target_pos)
        push_reward = 1 - np.tanh(dist_to_target)

        # --- 4. Orientation Reward (Cosine Similarity) ---
        # Convert quat to Z-rotation safely
        rotation = R.from_quat(push_obj_quat[[1, 2, 3, 0]])
        obj_z_angle = rotation.as_euler('xyz', degrees=False)[2]
        
        # Cosine distance: 1.0 if aligned, -1.0 if opposite
        # We map it to [0, 1] range: (cos(diff) + 1) / 2
        angle_diff = obj_z_angle - self.target_angle
        ori_reward = np.cos(angle_diff)

        # --- 5. Total Reward Combination ---
        # STAGE 1: Reach the object (Max 0.1)
        # STAGE 2: Push the object (Max 0.5)
        # STAGE 3: Align the object (Max 0.4)
        
        # We encourage pushing ONLY if we are somewhat close to the object
        # We encourage aligning ONLY if we are somewhat close to the target

        total_reward = (0.2 * reach_reward) + (0.5 * push_reward) + (0.3 * ori_reward)

        # Sparse success bonus (Small enough to not break SAC, big enough to matter)
        if self._check_success():
            total_reward += 5.0 

        return total_reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # --- ADD A VISUAL GOAL SITE ---
        goal_pos_3d = np.array([self.target_pos[0], self.target_pos[1], self.table_offset[2] + 0.001])
        
        # Create the new site element
        goal_site = new_site(
            name="goal_site",
            type="box",
            size=[0.02, 0.06, 0.001],
            pos=goal_pos_3d,
            euler=[0, 0, self.target_angle],
            rgba=[0, 1, 0, 0.5],
            group=1,
        )
        # Append the new element to the arena's worldbody
        mujoco_arena.worldbody.append(goal_site)
        # --- END VISUAL GOAL ---

        # Create material for our object
        tex_attrib = {"type": "cube"}
        mat_attrib = {"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        # --- Create the object to be pushed ---
        self.object_to_push = BoxObject(
            name="object_to_push",
            size=[0.02, 0.06, 0.02],
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        
        # --- Create placement sampler ---
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="ObjectPushSampler",
                mujoco_objects=self.object_to_push,
                x_range=[0.1, 0.1],
                y_range=[-0.1, -0.1],
                rotation=[np.pi/2, np.pi/2],
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )
        )

        # task includes arena, robot, and our single object
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.object_to_push],
        )

    def _setup_references(self):
        """
        Sets up references to important components.
        """
        super()._setup_references()

        # --- Object References ---
        self.push_obj_body_id = self.sim.model.body_name2id(self.object_to_push.root_body)

        # --- Gripper Joint References (for locking) ---
        default_gripper = list(self.robots[0].robot_model.grippers.values())[0]
        gripper_joint_names = default_gripper.joints
        self.g_joint_ids = [self.sim.model.joint_name2id(jnt) for jnt in gripper_joint_names]
        self.gripper_qpos_addr = [self.sim.model.jnt_qposadr[idx] for idx in self.g_joint_ids]
        
    def _setup_observables(self):
        observables = super()._setup_observables()

        if self.use_object_obs:
            modality = "object"

            # 1. Absolute Object Position
            @sensor(modality=modality)
            def push_obj_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.push_obj_body_id])

            # 2. Continuous Orientation (Sin/Cos) - Better than Euler
            @sensor(modality=modality)
            def push_obj_trig(obs_cache):
                tmp = convert_quat(np.array(self.sim.data.body_xquat[self.push_obj_body_id]), to="xyzw")
                rot = R.from_quat(tmp)
                z_angle = rot.as_euler('xyz', degrees=False)[2]
                return np.array([np.sin(z_angle), np.cos(z_angle)])

            # 3. Key Vector: Gripper to Object (Crucial for Reaching)
            @sensor(modality=modality)
            def gripper_to_obj(obs_cache):
                eef_id = list(self.robots[0].eef_site_id.values())[0]
                gripper_loc = np.array(self.sim.data.site_xpos[eef_id])
                obj_loc = np.array(self.sim.data.body_xpos[self.push_obj_body_id])
                return obj_loc - gripper_loc

            # 4. Key Vector: Object to Target (Crucial for Pushing)
            @sensor(modality=modality)
            def obj_to_target(obs_cache):
                obj_loc = np.array(self.sim.data.body_xpos[self.push_obj_body_id])
                # We only care about XY for the target vector usually
                return self.target_pos - obj_loc[:2]

            sensors = [push_obj_pos, push_obj_trig, gripper_to_obj, obj_to_target]
            names = [s.__name__ for s in sensors]

            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def _check_success(self):
        """
        Check if the object is at the target pose.
        SYNCHRONIZED with improved reward logic.
        """
        # --- 1. Get object's current pose from self.sim ---
        push_obj_pos = np.array(self.sim.data.xpos[self.push_obj_body_id])
        push_obj_quat = np.array(self.sim.data.xquat[self.push_obj_body_id])

        # --- 2. Calculate Position Error (2D distance) ---
        pos_error = np.linalg.norm(push_obj_pos[:2] - self.target_pos)

        # --- 3. Calculate Orientation Alignment (Cosine Similarity) ---
        rotation = R.from_quat(push_obj_quat[[1, 2, 3, 0]])
        z_angle = rotation.as_euler('xyz', degrees=False)[2]

        angle_diff = z_angle - self.target_angle
        # Alignment is np.cos(angle_diff). 1.0 is perfect alignment.
        alignment = np.cos(angle_diff)

        # --- 4. Define Success Thresholds ---
        POS_THRESHOLD = 0.03  # 2 cm distance
        ORI_THRESHOLD = 0.98 

        pos_aligned = (pos_error <= POS_THRESHOLD)
        ang_aligned = (alignment >= ORI_THRESHOLD)

        return pos_aligned and ang_aligned

    # --- Gripper Locking Helper Functions ---

    def _lock_gripper_closed(self):
        """
        Uses the pre-found joint addresses to lock the gripper closed.
        """
        self.sim.data.qpos[self.gripper_qpos_addr] = [0.0, 0.0]

    def reset(self):
        """
        Overrides the parent reset method to force gripper to be closed.
        """
        obs = super().reset()
        self._lock_gripper_closed()
        return obs

    def step(self, action):
        """
        Overrides the parent step method to intercept actions and force gripper closed.
        """
        obs, reward, done, info = super().step(action)
        self._lock_gripper_closed()
        return obs, reward, done, info
    
