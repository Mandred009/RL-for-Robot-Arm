# === Imports ===
import numpy as np
from PushAlign import PushAlign 
import robosuite as suite
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter
from dataclasses import dataclass
import collections
import time
import os
import random

# === Data structure to store a single transition in the replay buffer ===
@dataclass
class Experience:
    state: np.array
    action: np.array
    reward: float
    next_state: np.array
    done: bool

# === Experience Replay Buffer ===
class ExperienceReplay:
    def __init__(self, max_buffer_size):
        self.buffer = collections.deque(maxlen=max_buffer_size)
    
    def __len__(self):
        return len(self.buffer)

    def add_experience(self, experience: Experience):
        self.buffer.append(experience)

    def sample_experiences(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]

# === SAC Actor Network===
class SAC_ACTOR(nn.Module):
    def __init__(self, input_size, n_actions, action_bounds):
        super().__init__()
        self.action_low = torch.tensor(action_bounds['low'], dtype=torch.float32)
        self.action_high = torch.tensor(action_bounds['high'], dtype=torch.float32)
        
        self.base = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU()
        )
        self.mu_head = nn.Linear(64, n_actions)
        self.log_std_head = nn.Linear(64, n_actions)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base(x)
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, -20, 2) 
        std = torch.exp(log_std)
        return mu, std
    
    def sample(self, state):
        mu, std = self.forward(state)
        dist = torch.distributions.Normal(mu, std)
        raw_action = dist.rsample()  # Reparameterization trick
        action = torch.tanh(raw_action)
        
        # Scale to actual action bounds
        self.action_low = self.action_low.to(state.device)
        self.action_high = self.action_high.to(state.device)
        scaled_action = self.action_low + (action + 1) * 0.5 * (self.action_high - self.action_low)
        
        # Compute log probability with tanh correction
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        
        return scaled_action, log_prob, raw_action

    @torch.no_grad()
    def get_deterministic_action(self, state):
        """Returns the mean action (mu) for deterministic testing."""
        mu, _ = self.forward(state)
        action = torch.tanh(mu)
        
        # Scale to actual action bounds
        self.action_low = self.action_low.to(state.device)
        self.action_high = self.action_high.to(state.device)
        scaled_action = self.action_low + (action + 1) * 0.5 * (self.action_high - self.action_low)
        return scaled_action

# === SAC Critic Network ===
class SAC_CRITIC(nn.Module):
    def __init__(self, input_size, n_actions):
        super().__init__()
        self.obs_net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU()
        )
        self.out_net = nn.Sequential(
            nn.Linear(64 + n_actions, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, a):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))

# === Learnable Entropy Temperature ===
class EntropyAlpha:
    def __init__(self, action_dim, initial_value=0.2, lr=3e-4, device='cpu'):
        self.target_entropy = -action_dim 
        self.log_alpha = torch.tensor(np.log(initial_value), requires_grad=True, device=device)
        self.optimizer = optim.Adam([self.log_alpha], lr=lr)
        self.device = device
    
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def update(self, log_prob):
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.optimizer.zero_grad()
        alpha_loss.backward()
        self.optimizer.step()
        return alpha_loss.item()

# === SAC Agent with Exploration ===
class SAC:
    def __init__(self, env, net: SAC_ACTOR, buffer: ExperienceReplay, action_bounds, device='cpu'):
        self.env = env
        self.net = net
        self.buffer = buffer
        self.action_bounds = action_bounds
        self.device = device
        self.total_r = 0
        self.steps = 0
        self.total_steps = 0 

    def add_exploration_noise(self, action, exploration_steps=50000):
        """Add exploration noise that decays over time"""
        noise_scale = max(0.1, 1.0 - self.total_steps / exploration_steps)
        noise = np.random.normal(0, noise_scale * 0.1, size=action.shape)
        return np.clip(action + noise, self.action_bounds['low'], self.action_bounds['high'])

    @torch.no_grad()
    def play_episode(self, add_noise=True):
        self.net.eval()
        self.total_r = 0
        self.steps = 0

        state = self.env.reset()
        state = np.concatenate([
            state['robot0_proprio-state'].flatten(),
            state['object-state'].flatten() 
        ])
        
        while True:
            s_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

            action, _, _ = self.net.sample(s_tensor) 
            action = action.squeeze(0).cpu().numpy()
            
            next_state, reward, is_done, _ = self.env.step(action)
            next_state = np.concatenate([
                next_state['robot0_proprio-state'].flatten(),
                next_state['object-state'].flatten() 
            ])

            self.total_r += reward
            exp = Experience(state, action, reward, next_state, is_done)
            self.buffer.add_experience(exp)

            if is_done:
                break

            state = next_state
            self.steps += 1
            self.total_steps += 1

        return self.total_r, self.steps

# === Batch Processing ===
def calculate_batch(batch):
    states, actions, rewards, next_states, dones = zip(*[(e.state, e.action, e.reward, e.next_state, e.done) for e in batch])
    
    states_t = torch.from_numpy(np.array(states, dtype=np.float32)).to(DEVICE)
    actions_t = torch.from_numpy(np.array(actions, dtype=np.float32)).to(DEVICE)
    rewards_t = torch.from_numpy(np.array(rewards, dtype=np.float32)).unsqueeze(dim=-1).to(DEVICE)
    next_states_t = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(DEVICE)
    dones_t = torch.from_numpy(np.array(dones, dtype=np.uint8)).unsqueeze(dim=-1).to(DEVICE).bool()

    return states_t, actions_t, rewards_t, next_states_t, dones_t

# === Soft Update ===
def soft_update(target_net: nn.Module, source_net: nn.Module, tau=0.005):
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

# === Test Agent ===
@torch.no_grad()
def test_agent(env, actor_net: SAC_ACTOR, num_episodes=3):
    total_r = 0
    actor_net.eval()

    for t in range(num_episodes):
        print(f"Test episode: {t+1}/{num_episodes}")
        state = env.reset()
        state = np.concatenate([
            state['robot0_proprio-state'].flatten(),
            state['object-state'].flatten() 
        ])
        episode_reward = 0
        
        while True:
            s_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            action = actor_net.get_deterministic_action(s_tensor) 
            
            action = action.squeeze(0).cpu().numpy()

            next_state, reward, is_done, _ = env.step(action)
            next_state = np.concatenate([
                next_state['robot0_proprio-state'].flatten(),
                next_state['object-state'].flatten() 
            ])
            
            episode_reward += reward
            if is_done:
                break
            state = next_state

        total_r += episode_reward
        print(f"  Episode reward: {episode_reward:.2f}")

    return total_r / num_episodes

# === Learning Rate Scheduler ===
class LRScheduler:
    def __init__(self, optimizer, initial_lr, min_lr=1e-5, decay_steps=100000):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.decay_steps = decay_steps
        self.step_count = 0
    
    def step(self):
        self.step_count += 1
        lr = max(self.min_lr, self.initial_lr * (1 - self.step_count / self.decay_steps))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

# === MAIN TRAINING LOOP ===
if __name__ == "__main__":
    # Hyperparameters
    GAMMA = 0.98
    LEARNING_RATE_ACTOR = 3e-4
    LEARNING_RATE_CRITIC = 3e-4
    LEARNING_RATE_ALPHA = 3e-4
    MAX_BUFFER = 1000000
    MIN_BUFFER_TRAIN = 20000
    BATCH_SIZE = 512
    TAU_SOFT_UP = 0.005
    TEST_ITER = 20
    NOT_OF_TEST_EPI = 3
    REWARD_LIMIT = 100000
    EXPLORATION_STEPS = 50000
    UPDATE_EVERY = 2  # Update networks every N steps
    GRADIENT_STEPS = 2  # Number of gradient steps per update

    # Define proper action bounds
    ACTION_BOUNDS = {
        'low': np.array([-2.5, -1.57, -2.5, -3.14, -2.5, 0.0, -2.5]),
        'high': np.array([2.5, 1.57, 2.5, 0.0, 2.5, 3.14, 2.5])
    }

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {DEVICE}")

    save_path = os.path.join("saves_sac_improved")
    os.makedirs(save_path, exist_ok=True)

    # === Environment setup ===
    controller_config = suite.controllers.load_composite_controller_config(controller="WHOLE_BODY_IK")
    env = PushAlign(
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
    test_env = PushAlign(
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,       
        use_object_obs=True,
        render_camera="frontview",  
        horizon=500,
        control_freq=20,
    )
    
    N_ACTIONS = 7

    temp_state = env.reset()
    OBS_SPACE_SHAPE = np.concatenate([
        temp_state['robot0_proprio-state'].flatten(),
        temp_state['object-state'].flatten() 
    ]).shape[0]

    # === Network initialization ===
    actor_net = SAC_ACTOR(OBS_SPACE_SHAPE, N_ACTIONS, ACTION_BOUNDS).to(DEVICE)
    critic_net_q1 = SAC_CRITIC(OBS_SPACE_SHAPE, N_ACTIONS).to(DEVICE)
    critic_net_q2 = SAC_CRITIC(OBS_SPACE_SHAPE, N_ACTIONS).to(DEVICE)
    
    print(f"The actor net architecture: {actor_net}")
    print(f"The critic net architecture: {critic_net_q1}")

    target_critic_q1 = SAC_CRITIC(OBS_SPACE_SHAPE, N_ACTIONS).to(DEVICE)
    target_critic_q2 = SAC_CRITIC(OBS_SPACE_SHAPE, N_ACTIONS).to(DEVICE)

    # Synchronize target networks
    target_critic_q1.load_state_dict(critic_net_q1.state_dict())
    target_critic_q2.load_state_dict(critic_net_q2.state_dict())

    # === Optimizers and Schedulers ===
    actor_optimizer = optim.AdamW(actor_net.parameters(), lr=LEARNING_RATE_ACTOR)
    critic_optimizer_q1 = optim.AdamW(critic_net_q1.parameters(), lr=LEARNING_RATE_CRITIC)
    critic_optimizer_q2 = optim.AdamW (critic_net_q2.parameters(), lr=LEARNING_RATE_CRITIC)
    
    actor_scheduler = LRScheduler(actor_optimizer, LEARNING_RATE_ACTOR)
    critic_scheduler_q1 = LRScheduler(critic_optimizer_q1, LEARNING_RATE_CRITIC)
    critic_scheduler_q2 = LRScheduler(critic_optimizer_q2, LEARNING_RATE_CRITIC)

    # === Entropy temperature ===
    entropy_alpha = EntropyAlpha(N_ACTIONS, initial_value=0.2, lr=LEARNING_RATE_ALPHA, device=DEVICE)

    # === Replay buffer and agent setup ===
    exp_buffer = ExperienceReplay(MAX_BUFFER)
    sac_agent = SAC(env, actor_net, exp_buffer, ACTION_BOUNDS, device=DEVICE)

    # === TensorBoard logging ===
    writer = SummaryWriter(comment="-SAC-Improved")
    steps, episodes = 0, 0
    best_reward = -np.inf
    start_time = time.time()
    update_count = 0

    # === Main training loop ===
    while True:
        reward, stp = sac_agent.play_episode(add_noise=True)
        steps += stp
        episodes += 1

        print(f"Training EP: {episodes} || Steps: {steps} || Reward: {reward:.2f}")
        writer.add_scalar("train/episodes", episodes, steps)
        writer.add_scalar("train/reward", reward, steps)
        writer.add_scalar("train/episode_length", stp, steps)

        # Wait until buffer has enough samples
        if len(exp_buffer) < MIN_BUFFER_TRAIN:
            continue

        # Perform multiple gradient steps
        for _ in range(GRADIENT_STEPS):
            update_count += 1
            batch = exp_buffer.sample_experiences(BATCH_SIZE)
            states_t, actions_t, rewards_t, next_states_t, dones_t = calculate_batch(batch)

            # === Update Critics ===
            with torch.no_grad():
                next_actions, next_log_probs, _ = actor_net.sample(next_states_t)
                next_q1 = target_critic_q1(next_states_t, next_actions)
                next_q2 = target_critic_q2(next_states_t, next_actions)
                next_q = torch.min(next_q1, next_q2)
                target_q = rewards_t + GAMMA * (~dones_t).float() * (next_q - entropy_alpha.alpha * next_log_probs.unsqueeze(-1))

            # Update Q1
            critic_optimizer_q1.zero_grad()
            current_q1 = critic_net_q1(states_t, actions_t)
            critic_loss_q1 = F.smooth_l1_loss(current_q1, target_q)
            critic_loss_q1.backward()
            torch.nn.utils.clip_grad_norm_(critic_net_q1.parameters(), 1.0)
            critic_optimizer_q1.step()

            # Update Q2
            critic_optimizer_q2.zero_grad()
            current_q2 = critic_net_q2(states_t, actions_t)
            critic_loss_q2 = F.smooth_l1_loss(current_q2, target_q)
            critic_loss_q2.backward()
            torch.nn.utils.clip_grad_norm_(critic_net_q2.parameters(), 1.0)
            critic_optimizer_q2.step()

            # === Update Actor ===
            actor_optimizer.zero_grad()
            new_actions, log_probs, _ = actor_net.sample(states_t)
            q1_new = critic_net_q1(states_t, new_actions)
            q2_new = critic_net_q2(states_t, new_actions)
            q_new = torch.min(q1_new, q2_new)
            actor_loss = (entropy_alpha.alpha * log_probs.unsqueeze(-1) - q_new).mean()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_net.parameters(), 1.0)
            actor_optimizer.step()

            # === Update Entropy Temperature ===
            alpha_loss = entropy_alpha.update(log_probs)

            # === Soft update target networks ===
            soft_update(target_critic_q1, critic_net_q1, TAU_SOFT_UP)
            soft_update(target_critic_q2, critic_net_q2, TAU_SOFT_UP)

            # === Update learning rates ===
            if update_count % 1000 == 0:
                actor_lr = actor_scheduler.step()
                critic_scheduler_q1.step()
                critic_scheduler_q2.step()
                writer.add_scalar("train/actor_lr", actor_lr, steps)

            # === Logging ===
            if update_count % 100 == 0:
                writer.add_scalar("loss/critic_q1", critic_loss_q1.item(), update_count)
                writer.add_scalar("loss/critic_q2", critic_loss_q2.item(), update_count)
                writer.add_scalar("loss/actor", actor_loss.item(), update_count)
                writer.add_scalar("loss/alpha", alpha_loss, update_count)
                writer.add_scalar("train/alpha", entropy_alpha.alpha.item(), update_count)
                writer.add_scalar("train/q1_mean", current_q1.mean().item(), update_count)
                writer.add_scalar("train/q2_mean", current_q2.mean().item(), update_count)
                
                for name, param in actor_net.named_parameters():
                    if param.grad is not None:
                        writer.add_histogram(f'gradients/actor_{name}', param.grad, update_count)
                        writer.add_scalar(f'gradient_norm/actor_{name}', param.grad.norm().item(), update_count)

        # === Periodic testing and saving ===
        if episodes % TEST_ITER == 0:
            print("=" * 50)
            print("TESTING...")
            test_reward = test_agent(test_env, actor_net, NOT_OF_TEST_EPI)
            print(f"Average test reward: {test_reward:.2f}")
            print("=" * 50)
            
            writer.add_scalar("test/reward", test_reward, steps)

            if test_reward > best_reward:
                best_reward = test_reward
                # Save all networks
                checkpoint = {
                    'actor_state_dict': actor_net.state_dict(),
                    'critic_q1_state_dict': critic_net_q1.state_dict(),
                    'critic_q2_state_dict': critic_net_q2.state_dict(),
                    'actor_optimizer': actor_optimizer.state_dict(),
                    'critic_q1_optimizer': critic_optimizer_q1.state_dict(),
                    'critic_q2_optimizer': critic_optimizer_q2.state_dict(),
                    'alpha': entropy_alpha.alpha.item(),
                    'episodes': episodes,
                    'steps': steps,
                    'best_reward': best_reward
                }
                torch.save(checkpoint, os.path.join(save_path, f"sac_best_reward_{best_reward:.2f}.pth"))
                print(f"New best model saved with reward: {best_reward:.2f}")

            if best_reward >= REWARD_LIMIT:
                print(f"SOLVED! Steps: {steps} || Episodes: {episodes}")
                break

    # === Final summary ===
    elapsed_time = (time.time() - start_time) / 60
    print(f"Training completed in {elapsed_time:.2f} minutes")
    print(f"Final best reward: {best_reward:.2f}")
    writer.close()
