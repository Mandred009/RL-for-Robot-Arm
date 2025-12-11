import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter
import collections
import time
from PushAlign import PushAlign 
import robosuite as suite
import os

# === Normalization Utilities  ===
class RunningMeanStd:
    """Tracks the mean and variance of a data stream."""
    def __init__(self, shape):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

class NormalizedEnv:
    """Wraps the environment to normalize observations and flatten state."""
    def __init__(self, env, obs_shape, training=True, rms_obj=None):
        self.env = env
        self.training = training
        if rms_obj is None:
            self.rms = RunningMeanStd(shape=obs_shape)
        else:
            self.rms = rms_obj
        self.epsilon = 1e-8
        self.clip = 10.0

    def _normalize(self, obs):
        if self.training:
            self.rms.update(obs[None]) 
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

# === Networks ===
class PPO_ACTOR(nn.Module):
    def __init__(self, input_size, n_actions):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU()
        )
        self.mu_head = nn.Linear(128, n_actions)
        self.log_std = nn.Parameter(torch.zeros(n_actions))

    def forward(self, x):
        x = self.base(x)
        mu = self.mu_head(x)
        std = torch.exp(self.log_std)
        return mu, std

class PPO_CRITIC(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.model(x)

# === PPO Agent ===
class PPO:
    def __init__(self, env, act_net:PPO_ACTOR, crt_net:PPO_CRITIC):
        self.state = []
        self.action = [] # Will store raw gaussian actions
        self.reward = []
        self.log_prob = []
        self.done = []   # Added to track episode boundaries
        self.env = env
        self.act_net = act_net
        self.crt_net = crt_net
        self.episode_no = 0
        self.trajectory_len = 0

    def reset_buffer(self):
        self.state = []
        self.action = []
        self.reward = []
        self.log_prob = []
        self.done = []
        self.trajectory_len = 0

    def calculate_advantage_reference_vals(self):
        states_v = torch.FloatTensor(np.array(self.state)).to(DEVICE)
        # Actions stored are RAW (pre-tanh)
        action_v = torch.FloatTensor(np.array(self.action)).to(DEVICE) 
        log_probs_v = torch.FloatTensor(np.array(self.log_prob)).to(DEVICE)
        
        # We need values for all states + value of the very last state
        with torch.no_grad():
            value_v = self.crt_net(states_v)
            values = value_v.squeeze().cpu().numpy()
            values = np.append(values, 0.0) 

        gae = 0.0
        adv_list = []
        ref_list = []

        # Loop backward using masks for correct GAE at episode boundaries
        for i in range(len(self.state) - 1, -1, -1):
            mask = 1.0 - self.done[i]
            
            delta = self.reward[i] + (GAMMA * values[i+1] * mask) - values[i]
            gae = delta + (GAMMA * GAE_LAMBDA * mask * gae)

            adv_list.append(gae)
            ref_list.append(gae + values[i])

        adv_v = torch.FloatTensor(list(reversed(adv_list))).to(DEVICE)
        ref_v = torch.FloatTensor(list(reversed(ref_list))).to(DEVICE)

        return states_v, action_v, adv_v, ref_v, log_probs_v

    @torch.no_grad
    def play_episode(self):
        state = self.env.reset() 
        
        total_r = 0
        while True:
            s_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            mu, std = self.act_net(s_tensor)
            dist = torch.distributions.Normal(mu, std)
            
            # Sample Raw (Gaussian) Action
            raw_action = dist.sample()
            
            # Jacobian correction for Tanh Log Prob
            log_prob = dist.log_prob(raw_action).sum(dim=1)
            log_prob -= (2 * (np.log(2) - raw_action - F.softplus(-2 * raw_action))).sum(dim=1)

            # Apply tanh for env action
            action_tanh = torch.tanh(raw_action).squeeze(0).cpu().numpy()
            action_env = action_tanh * JOINT_LIMITS

            self.state.append(state)
            self.action.append(raw_action.squeeze(0).cpu().numpy()) # Store RAW action
            self.log_prob.append(log_prob.item())

            next_state, reward, terminated, _ = self.env.step(action_env)
            
            self.reward.append(reward)
            self.done.append(terminated) # Store done flag
            total_r += reward

            self.trajectory_len += 1

            if terminated:
                break
            state = next_state

        self.episode_no += 1
        return self.episode_no, self.trajectory_len, total_r

@torch.no_grad()
def test_agent(env, actor_net: PPO_ACTOR):
    total_r = 0
    actor_net.eval()

    for t in range(NO_OF_TEST_EPI):
        state = env.reset()
        
        while True:
            s_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            mu, std = actor_net(s_tensor)

            action = torch.tanh(mu).squeeze(0).cpu().numpy()
            action = action * JOINT_LIMITS

            next_state, reward, is_done, _ = env.step(action)
            
            if is_done:
                break
            state = next_state
            total_r += reward

    return total_r / NO_OF_TEST_EPI

# === MAIN ===
if __name__ == "__main__":
    # Hyperparameters
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    LEARNING_RATE_ACTOR = 0.0001
    LEARNING_RATE_CRITIC = 0.001
    TRAJECTORY_SIZE = 8192
    PPO_EPSILON = 0.2
    PPO_EPOCHS = 10
    PPO_BATCH_SIZE = 64
    TEST_EPS = 20
    NO_OF_TEST_EPI = 3
    REWARD_LIMIT = 100000
    ENTROPY_COEF = 0.01 # Added for exploration

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    JOINT_LIMITS = np.array([2.5, 1.57, 2.5, 3.14, 2.5, 3.14, 2.5])
    
    print(f"Training on {DEVICE}")

    save_path = os.path.join("saves3")
    os.makedirs(save_path, exist_ok=True)

    # Initialize Controller config
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
    
    temp_state = raw_env.reset()
    obs_dim = np.concatenate([
        temp_state['robot0_proprio-state'].flatten(),
        temp_state['object-state'].flatten() 
    ]).shape[0]
    N_ACTIONS=7
    
    # Wrap Training Env
    env = NormalizedEnv(raw_env, obs_shape=obs_dim, training=True)
    
    # Setup Test Env 
    raw_test_env = PushAlign(
            robots="Panda",
            controller_configs=controller_config,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,       
            use_object_obs=True,
            render_camera="agentview",  
            horizon=500,
            control_freq=20,
        )
    test_env = NormalizedEnv(raw_test_env, obs_shape=obs_dim, training=False, rms_obj=env.rms)

    print(f"Observation Dimension: {obs_dim}")

    # Initialize networks
    actor_net = PPO_ACTOR(obs_dim , N_ACTIONS).to(DEVICE)
    critic_net = PPO_CRITIC(obs_dim).to(DEVICE)
    
    actor_optimizer = optim.AdamW(actor_net.parameters(), lr=LEARNING_RATE_ACTOR)
    critic_optimizer = optim.AdamW(critic_net.parameters(), lr=LEARNING_RATE_CRITIC)

    ppo_agent = PPO(env, actor_net, critic_net)
    writer = SummaryWriter(comment="-PPO_ALGO")

    best_reward = -np.inf
    start_time = time.time()

    # Main training loop
    while True:
        actor_net.train()
        eps, traj_len, eps_reward = ppo_agent.play_episode()
        writer.add_scalar("train reward", eps_reward, eps)
        print(f"EPISODE NO: {eps} || REWARD: {eps_reward} || TRAJ: {traj_len}")

        # Periodic testing
        if eps % TEST_EPS == 0:
            test_reward = test_agent(test_env, actor_net)
            print(f"TESTING REWARD {test_reward}")
            writer.add_scalar("test reward", test_reward, eps)

            if test_reward > best_reward:
                best_reward = test_reward
                # Save Model AND Normalization Stats
                checkpoint = {
                    'actor_state': actor_net.state_dict(),
                    'critic_state': critic_net.state_dict(),
                    'rms_mean': env.rms.mean,
                    'rms_var': env.rms.var,
                    'rms_count': env.rms.count
                }
                torch.save(checkpoint, os.path.join(save_path, f"best_reward_{best_reward:.2f}.pt"))

            if best_reward > REWARD_LIMIT:
                print(f"SOLVED AT Total Episodes: {eps}")
                break

        if traj_len < TRAJECTORY_SIZE:
            continue

        # Compute GAE and reference values
        states_v, actions_v, adv_v, ref_v, old_log_probs_v = ppo_agent.calculate_advantage_reference_vals()
        adv_v = (adv_v - adv_v.mean()) / (adv_v.std() + 1e-8)

        sum_loss_policy = 0.0
        sum_loss_value = 0.0
        sum_entropy = 0.0
        steps = 0

        print("Training...")
        for epoch in range(PPO_EPOCHS):
            indices = np.random.permutation(len(states_v))
            for batch in range(0, len(states_v), PPO_BATCH_SIZE):
                idx = indices[batch:batch+PPO_BATCH_SIZE]
                
                b_states = states_v[idx]
                b_actions = actions_v[idx] # Raw Gaussian actions
                b_adv = adv_v[idx]
                b_ref = ref_v[idx]
                b_old_logp = old_log_probs_v[idx]

                # === Critic Update ===
                critic_optimizer.zero_grad()
                values = critic_net(b_states).squeeze(-1)
                loss_value = F.mse_loss(values, b_ref)
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(critic_net.parameters(), 1.0)
                critic_optimizer.step()

                # === Actor Update ===
                actor_optimizer.zero_grad()
                mu, std = actor_net(b_states)
                dist = torch.distributions.Normal(mu, std)
                
                # Tanh Correction for NEW log_prob
                new_logp = dist.log_prob(b_actions).sum(dim=1)
                new_logp -= (2 * (np.log(2) - b_actions - F.softplus(-2 * b_actions))).sum(dim=1)
                
                ratio = torch.exp(new_logp - b_old_logp)

                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - PPO_EPSILON, 1.0 + PPO_EPSILON) * b_adv
                
                entropy = dist.entropy().mean()
                loss_policy = -torch.min(surr1, surr2).mean() - (ENTROPY_COEF * entropy)
                
                loss_policy.backward()
                torch.nn.utils.clip_grad_norm_(actor_net.parameters(), 0.5)
                actor_optimizer.step()

                sum_loss_policy += loss_policy.item()
                sum_loss_value += loss_value.item()
                sum_entropy += entropy.item()
                steps += 1

        ppo_agent.reset_buffer()
        writer.add_scalar("advantage", adv_v.mean().item(), eps)
        writer.add_scalar("values", ref_v.mean().item(), eps)
        writer.add_scalar("loss_policy", sum_loss_policy / steps, eps)
        writer.add_scalar("loss_value", sum_loss_value / steps, eps)
        writer.add_scalar("entropy", sum_entropy / steps, eps)

    end_time = time.time()
    elapsed_time = (end_time - start_time)/60
    print(f"Time taken for complete training: {elapsed_time:.2f} min")
    writer.close()
