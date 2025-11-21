"""
Custom PPO Implementation for PX4 Drone RL Training
Pure PyTorch implementation without Stable-Baselines3
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    Supports both vision (CNN) and kinematic-only inputs.
    """
    
    def __init__(self, obs_dim, action_dim, use_vision=False):
        super(ActorCritic, self).__init__()
        
        self.use_vision = use_vision
        
        if use_vision:
            # Vision encoder (CNN for depth images)
            # Input: 128 visual features (16x8 downsampled depth)
            self.vision_encoder = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU()
            )
            
            # Kinematic encoder (13 dimensional)
            self.kinematic_encoder = nn.Sequential(
                nn.Linear(13, 32),
                nn.ReLU()
            )
            
            # Combined features: 32 (vision) + 32 (kinematic) = 64
            combined_dim = 64
            
        else:
            # Kinematic-only encoder
            self.kinematic_encoder = nn.Sequential(
                nn.Linear(obs_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU()
            )
            combined_dim = 64
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor_mean = nn.Linear(64, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (value function)
        self.critic = nn.Linear(64, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        
        # Special initialization for actor output
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
    
    def forward(self, obs):
        """Forward pass through network"""
        if self.use_vision:
            # Split observation into visual and kinematic
            visual_features = obs[:, :128]
            kinematic_features = obs[:, 128:]
            
            # Encode separately
            visual_encoded = self.vision_encoder(visual_features)
            kinematic_encoded = self.kinematic_encoder(kinematic_features)
            
            # Combine
            combined = torch.cat([visual_encoded, kinematic_encoded], dim=1)
        else:
            # Kinematic only
            combined = self.kinematic_encoder(obs)
        
        # Shared processing
        shared_features = self.shared(combined)
        
        # Actor: mean and std for action distribution
        action_mean = torch.tanh(self.actor_mean(shared_features))
        action_std = torch.exp(self.actor_log_std)
        
        # Critic: state value
        value = self.critic(shared_features)
        
        return action_mean, action_std, value
    
    def get_action(self, obs, deterministic=False):
        """
        Sample action from policy.
        
        Args:
            obs: Observation tensor
            deterministic: If True, return mean action
        
        Returns:
            action, log_prob, value
        """
        action_mean, action_std, value = self.forward(obs)
        
        if deterministic:
            return action_mean, None, value
        
        # Create normal distribution
        dist = Normal(action_mean, action_std)
        
        # Sample action
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, value
    
    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions for PPO update.
        
        Returns:
            log_probs, values, entropy
        """
        action_mean, action_std, value = self.forward(obs)
        
        # Create distribution
        dist = Normal(action_mean, action_std)
        
        # Evaluate
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_probs, value, entropy


class RolloutBuffer:
    """
    Buffer for storing experience during rollouts.
    """
    
    def __init__(self, buffer_size, obs_dim, action_dim, device):
        self.buffer_size = buffer_size
        self.device = device
        
        # Storage
        self.observations = torch.zeros((buffer_size, obs_dim)).to(device)
        self.actions = torch.zeros((buffer_size, action_dim)).to(device)
        self.rewards = torch.zeros(buffer_size).to(device)
        self.dones = torch.zeros(buffer_size).to(device)
        self.log_probs = torch.zeros(buffer_size).to(device)
        self.values = torch.zeros(buffer_size).to(device)
        
        self.ptr = 0
        self.full = False
    
    def add(self, obs, action, reward, done, log_prob, value):
        """Add experience to buffer"""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        
        self.ptr += 1
        if self.ptr == self.buffer_size:
            self.full = True
            self.ptr = 0
    
    def get(self):
        """Get all data and compute advantages"""
        size = self.buffer_size if self.full else self.ptr
        
        return {
            'observations': self.observations[:size],
            'actions': self.actions[:size],
            'rewards': self.rewards[:size],
            'dones': self.dones[:size],
            'log_probs': self.log_probs[:size],
            'values': self.values[:size]
        }
    
    def clear(self):
        """Clear buffer"""
        self.ptr = 0
        self.full = False


class PPO:
    """
    Proximal Policy Optimization (PPO) agent.
    """
    
    def __init__(
        self,
        obs_dim,
        action_dim,
        use_vision=False,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        n_epochs=10,
        batch_size=64,
        buffer_size=2048,
        device='cuda'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        # Network
        self.policy = ActorCritic(obs_dim, action_dim, use_vision).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Buffer
        self.buffer = RolloutBuffer(buffer_size, obs_dim, action_dim, self.device)
        
        # Training stats
        self.n_updates = 0
    
    def select_action(self, obs, deterministic=False):
        """
        Select action from policy.
        
        Args:
            obs: Observation (numpy array)
            deterministic: If True, return mean action
        
        Returns:
            action (numpy), log_prob, value
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(obs_tensor, deterministic)
        
        return (
            action.cpu().numpy()[0],
            log_prob.cpu().item() if log_prob is not None else None,
            value.cpu().item()
        )
    
    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
        
        returns = advantages + values
        
        return advantages, returns
    
    def update(self):
        """Perform PPO update"""
        # Get buffer data
        data = self.buffer.get()
        
        obs = data['observations']
        actions = data['actions']
        old_log_probs = data['log_probs']
        rewards = data['rewards']
        dones = data['dones']
        values = data['values']
        
        # Compute advantages
        advantages, returns = self.compute_gae(rewards, values, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training stats
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_batches = 0
        
        # Multiple epochs
        for epoch in range(self.n_epochs):
            # Create mini-batches
            indices = torch.randperm(len(obs))
            
            for start in range(0, len(obs), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Get batch
                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate actions
                log_probs, values, entropy = self.policy.evaluate_actions(batch_obs, batch_actions)
                values = values.squeeze()
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(values, batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Stats
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_batches += 1
        
        self.n_updates += 1
        
        # Clear buffer
        self.buffer.clear()
        
        # Return training stats
        return {
            'loss': total_loss / n_batches,
            'policy_loss': total_policy_loss / n_batches,
            'value_loss': total_value_loss / n_batches,
            'entropy': total_entropy / n_batches
        }
    
    def save(self, path):
        """Save model"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'n_updates': self.n_updates
        }, path)
    
    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.n_updates = checkpoint['n_updates']


# For testing
if __name__ == "__main__":
    print("Testing PPO implementation...")
    
    # Test without vision
    agent = PPO(obs_dim=13, action_dim=4, use_vision=False, device='cpu')
    print(f"✓ Created agent (device: {agent.device})")
    
    # Test action selection
    obs = np.random.randn(13)
    action, log_prob, value = agent.select_action(obs)
    print(f"✓ Action shape: {action.shape}, value: {value:.3f}")
    
    # Test with vision
    agent_vision = PPO(obs_dim=141, action_dim=4, use_vision=True, device='cpu')
    obs_vision = np.random.randn(141)
    action, log_prob, value = agent_vision.select_action(obs_vision)
    print(f"✓ Vision action shape: {action.shape}")
    
    print("✓ All tests passed!")