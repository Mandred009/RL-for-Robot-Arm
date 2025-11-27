
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
        """
        Reward function for the Push-to-Pose task.
        Reads directly from the simulation state.
        """
        # --- 1. Get object's current pose from self.sim ---
        push_obj_pos = np.array(self.sim.data.xpos[self.push_obj_body_id])
        push_obj_quat = np.array(self.sim.data.xquat[self.push_obj_body_id])

        # --- 2. Calculate Position Reward (2D distance) ---
        pos_error = np.linalg.norm(push_obj_pos[:2] - self.target_pos)
        pos_reward = -pos_error  # Negative error is a good reward

        # --- 3. Calculate Orientation Reward (Z-angle) ---
        rotation = R.from_quat(push_obj_quat[[1, 2, 3, 0]]) # Convert MuJoCo (w,x,y,z) to Scipy (x,y,z,w)
        z_angle = rotation.as_euler('xyz', degrees=False)[2]

        angle_error = np.sin(z_angle)-np.sin(self.target_angle)
        orientation_reward = -np.abs(angle_error) # Negative error
        
        eef_id=list(self.robots[0].eef_site_id.values())[0]
        gripper_loc=np.array(self.sim.data.site_xpos[eef_id])

        
        gripper_error=-np.linalg.norm(gripper_loc-push_obj_pos)
        
        # --- 4. Combine Rewards ---
        reward = (0.5 * pos_reward) + (0.3 * orientation_reward) + (0.2*gripper_error)
        # print(f"pos: {pos_reward} || ori: {orientation_reward} || grip: {gripper_error}")
        
        if self._check_success():
            reward += 100.0

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale
            
        return reward

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
                x_range=[0.0, 0.1],
                y_range=[-0.1, 0.0],
                rotation=[0.1, np.pi],
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
        # self.grip_site_id=self.sim.model.site_name2id(self.robots[0].eef_site_name)
        
    def _setup_observables(self):
        """
        Sets up observables to be used for this environment.
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def push_obj_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.push_obj_body_id])

            @sensor(modality=modality)
            def push_obj_quat(obs_cache):
                tmp=convert_quat(np.array(self.sim.data.body_xquat[self.push_obj_body_id]), to="xyzw")
                rot = R.from_quat(tmp)
                rot_euler = rot.as_euler('xyz', degrees=True)
                return rot_euler
            
            @sensor(modality=modality)
            def target_obj(obs_cache):
                return np.array(self.target_pos)
            
            @sensor(modality=modality)
            def target_obj_rot(obs_cache):
                return np.array(self.target_angle)

            sensors = [push_obj_pos, push_obj_quat]
            
            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])

            # sensors += [
            #     self._get_obj_eef_sensor(full_pf, "push_obj_pos", f"{arm_pf}gripper_to_push_obj_pos", modality)
            #     for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
            # ]
            sensors.append(target_obj)
            sensors.append(target_obj_rot)

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
        """
        # --- 1. Get object's current pose from self.sim ---
        push_obj_pos = np.array(self.sim.data.body_xpos[self.push_obj_body_id])
        push_obj_quat = np.array(self.sim.data.body_xquat[self.push_obj_body_id])

        # --- 2. Check Position ---
        pos_error = np.linalg.norm(push_obj_pos[:2] - self.target_pos)
        pos_aligned = (pos_error < 0.03)  # Within 3 cm

        # --- 3. Check Orientation ---
        rotation = R.from_quat(push_obj_quat[[1, 2, 3, 0]]) # Convert MuJoCo (w,x,y,z) to Scipy (x,y,z,w)        z_angle = rotation.as_euler('xyz', degrees=False)[2]
        z_angle = rotation.as_euler('xyz', degrees=False)[2]    
        angle_error = np.abs(np.arctan2(np.sin(z_angle - self.target_angle), np.cos(z_angle - self.target_angle)))
        orientation_aligned = (angle_error < 0.1) # Within ~5.7 degrees

        return pos_aligned and orientation_aligned

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