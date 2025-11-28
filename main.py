
import numpy as np
import robosuite as suite
from PushAlign import PushAlign 

print("Loading environment...")

# --- 1. Load the controller config ---
controller_config = suite.controllers.load_composite_controller_config(controller="WHOLE_BODY_IK")


env = PushAlign(
    robots="Panda",
    controller_configs=controller_config,
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,       
    use_object_obs=True,
    # Available "camera" names = ('frontview', 'birdview', 'agentview', 
#     # 'sideview', 'robot0_robotview', 'robot0_eye_in_hand')
    render_camera="sideview",  
    horizon=500,
    control_freq=120,
)


env.reset()
env.render()
r_tot=0

for i in range(1000):
    
    action_dim = env.action_spec[0].shape[0]

    action = np.random.randn(action_dim)
    # action = np.zeros(action_dim)
    
    obs, reward, done, info = env.step(action)
    # print(obs)
    # print(i)
    # print(reward)
    r_tot+=reward

    env.render()

    if done:
        print(f"Episode finished. Resetting. Reward={r_tot}")
        env.reset()

env.close()
print("Demo complete!")