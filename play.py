import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC
# from stable_baselines3.common.vec_env import DummyVecEnv

from PushAlign import Lift as PushAlignLift

from robosuite.controllers import load_composite_controller_config



env = PushAlignLift(
    robots="Panda",
    controller_configs = suite.controllers.load_composite_controller_config(controller="WHOLE_BODY_IK"),             
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=True,
    use_object_obs=False,
    camera_names="agentview",
    camera_heights=84,
    camera_widths=84,
    horizon=200,
    control_freq=20,
)

# Wrap in Gym Wrapper
env = GymWrapper(env, keys=["image"])
# env = DummyVecEnv([lambda: env])

# create RL model
model = SAC("CnnPolicy", env, verbose=1)    

# train the model
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
    
env.close()
