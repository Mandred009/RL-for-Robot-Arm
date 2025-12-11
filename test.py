import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import collections
import time
import os
from PushAlign import PushAlign 
import robosuite as suite
from ppo import PPO_ACTOR 


# --- Normalization Utilities (Required for Testing) ---
class RunningMeanStd:
    """A minimal class to hold and apply the loaded mean/variance."""
    def __init__(self, shape):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = 1e-4

# Using a simplified NormalizedEnv for testing (no update logic)
class NormalizedEnv:
    def __init__(self, env, rms_obj):
        self.env = env
        self.rms = rms_obj
        self.epsilon = 1e-8
        self.clip = 10.0

    def _normalize(self, obs):
        obs_norm = (obs - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon)
        return np.clip(obs_norm, -self.clip, self.clip)

    def _flatten(self, state_dict):
        return np.concatenate([
            state_dict['robot0_proprio-state'].flatten(),
            state_dict['object-state'].flatten() 
        ])

    def reset(self):
        state = self.env.reset()
        state_vec = self._flatten(state)
        return self._normalize(state_vec)

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        next_state_vec = self._flatten(next_state)
        return self._normalize(next_state_vec), reward, done, info


# --- Test Agent  ---
@torch.no_grad()
def test_agent(env, actor_net: nn.Module):
    total_r = 0
    actor_net.eval()
    
    state = env.reset()
    
    for t in range(NO_OF_TESTS):
        print(f"test:{t}")
        state = env.reset() 
        current_episode_reward = 0
        
        while True:

            s_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            mu, _ = actor_net(s_tensor) 
            
            # Apply Tanh to squashed the action to [-1, 1]
            action = torch.tanh(mu).squeeze(0).cpu().numpy()
            action = action * JOINT_LIMITS
            
            if action[3] > 0: action[3] *= -1
            if action[5] <= 0: action[5] *= -1 

            next_state, reward, is_done, _ = env.step(action)

            if is_done:
                break
            state = next_state

            current_episode_reward += reward

        total_r += current_episode_reward

    return total_r / NO_OF_TESTS


if __name__=="__main__":

    NO_OF_TESTS=3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on {DEVICE}")
    JOINT_LIMITS = np.array([2.5, 1.57, 2.5, 3.14, 2.5, 3.14, 2.5])
    
    CHECKPOINT_PATH = "/home/kuka/Documents/frl_proj/Learning-to-Push---Deep-Reinforcement-Learning-main/saves3/best_reward_-180.28.pt"
    
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
    except FileNotFoundError:
        print(f"ERROR: Checkpoint file not found at {CHECKPOINT_PATH}")
        exit()

    OBS_SPACE_SHAPE = checkpoint['rms_mean'].shape[0]
    
    rms = RunningMeanStd(OBS_SPACE_SHAPE)
    rms.mean = checkpoint['rms_mean']
    rms.var = checkpoint['rms_var']
    rms.count = checkpoint['rms_count']
    print(f"RMS Stats Loaded. OBS Shape: {OBS_SPACE_SHAPE}")

    controller_config = suite.controllers.load_composite_controller_config(controller="WHOLE_BODY_IK")
    raw_env = PushAlign(
            robots="Panda",
            controller_configs=controller_config,
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,       
            use_object_obs=True,
            render_camera="frontview",  
            horizon=500,
            control_freq=20,
        )

    env = NormalizedEnv(raw_env, rms_obj=rms)
    
    N_ACTIONS = 7
    ppo_agent = PPO_ACTOR(OBS_SPACE_SHAPE, N_ACTIONS).to(DEVICE)
    
    ppo_agent.load_state_dict(checkpoint['actor_state'])
    
    avg_reward = test_agent(env, ppo_agent)
    print(f"\nAverage Test Reward over {NO_OF_TESTS} episodes: {avg_reward:.2f}")
