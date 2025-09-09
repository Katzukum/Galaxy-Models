import os
import yaml
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Dict, Any, List, Tuple
from collections import deque
import gym
from gym import spaces
import random
from Utilities.data_utils import prepare_delta_features

# ####################################################################
# --- 1. Custom Trading Environment for PPO ---
# ####################################################################

class TradingEnvironment(gym.Env):
    """
    Custom trading environment for PPO training.
    Actions: 0=Hold, 1=Buy, 2=Sell
    """
    
    def __init__(self, data, features, lookback_window=60, initial_balance=50000):
        super(TradingEnvironment, self).__init__()
        
        self.data = data
        self.features = features
        self.lookback_window = lookback_window
        self.initial_balance = initial_balance
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: normalized features + account state
        # Features + balance + position + unrealized_pnl
        n_features = data.shape[1] + 3  # data features + 3 for balance, position, unrealized_pnl
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(lookback_window, n_features), 
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0  # 0=no position, 1=long, -1=short
        self.position_size = 0
        self.entry_price = 0
        self.unrealized_pnl = 0
        self.trades = []
        self.equity_history = [self.initial_balance]
        
        return self._get_observation()
    
    def _get_observation(self):
        """Get current observation including features and account state"""
        if self.current_step < self.lookback_window:
            # Pad with zeros if not enough history
            start_idx = max(0, self.current_step - self.lookback_window)
            end_idx = self.current_step
            pad_size = self.lookback_window - (end_idx - start_idx)
            
            feature_data = self.data[start_idx:end_idx, :]
            if pad_size > 0:
                feature_data = np.vstack([np.zeros((pad_size, feature_data.shape[1])), feature_data])
        else:
            start_idx = self.current_step - self.lookback_window
            end_idx = self.current_step
            feature_data = self.data[start_idx:end_idx, :]
        
        # Add account state to each timestep
        account_state = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.position,  # Position (-1, 0, 1)
            self.unrealized_pnl / self.initial_balance  # Normalized unrealized PnL
        ])
        
        # Broadcast account state to all timesteps
        account_state_broadcast = np.tile(account_state, (self.lookback_window, 1))
        
        # Combine features with account state
        observation = np.concatenate([feature_data, account_state_broadcast], axis=1)
        
        return observation.astype(np.float32)
    
    def step(self, action):
        """Execute action and return next observation, reward, done, info"""
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, {}
        
        current_price = self.data[self.current_step, 0]  # Assuming close price is first column
        reward = 0
        
        # Execute action
        if action == 1:  # Buy
            if self.position <= 0:  # Not already long
                # Close short position if exists
                if self.position < 0:
                    pnl = (self.entry_price - current_price) * self.position_size
                    self.balance += pnl
                    self.trades.append(pnl)
                
                # Open long position
                self.position = 1
                self.position_size = self.balance * 0.1 / current_price  # Use 10% of balance
                self.entry_price = current_price
                self.balance -= self.position_size * current_price
        
        elif action == 2:  # Sell
            if self.position >= 0:  # Not already short
                # Close long position if exists
                if self.position > 0:
                    pnl = (current_price - self.entry_price) * self.position_size
                    self.balance += pnl
                    self.trades.append(pnl)
                
                # Open short position
                self.position = -1
                self.position_size = self.balance * 0.1 / current_price  # Use 10% of balance
                self.entry_price = current_price
                self.balance += self.position_size * current_price
        
        # Calculate unrealized PnL
        if self.position != 0:
            if self.position > 0:  # Long
                self.unrealized_pnl = (current_price - self.entry_price) * self.position_size
            else:  # Short
                self.unrealized_pnl = (self.entry_price - current_price) * self.position_size
        else:
            self.unrealized_pnl = 0
        
        # Calculate reward
        total_equity = self.balance + self.unrealized_pnl
        self.equity_history.append(total_equity)
        
        # Reward based on equity change
        if len(self.equity_history) > 1:
            equity_change = (total_equity - self.equity_history[-2]) / self.initial_balance
            reward = equity_change * 100  # Scale reward
        
        # Add small penalty for holding to encourage trading
        if action == 0:
            reward -= 0.001
        
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1
        
        info = {
            'balance': self.balance,
            'position': self.position,
            'unrealized_pnl': self.unrealized_pnl,
            'total_equity': total_equity
        }
        
        return self._get_observation(), reward, done, info

# ####################################################################
# --- 2. PPO Actor-Critic Network ---
# ####################################################################

class PPONetwork(nn.Module):
    """
    PPO Actor-Critic network for trading decisions.
    """
    
    def __init__(self, input_dim, hidden_dim=128, num_actions=3):
        super(PPONetwork, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        """Forward pass through the network"""
        # Flatten the sequence dimension for processing
        batch_size, seq_len, feature_dim = x.shape
        x_flat = x.view(batch_size * seq_len, feature_dim)
        
        # Extract features
        features = self.feature_extractor(x_flat)
        
        # Reshape back to sequence
        features = features.view(batch_size, seq_len, -1)
        
        # Use the last timestep for decision making
        last_features = features[:, -1, :]
        
        # Get action probabilities and value
        action_logits = self.actor(last_features)
        value = self.critic(last_features)
        
        return action_logits, value
    
    def get_action(self, x):
        """Get action and log probability for given observation"""
        action_logits, value = self.forward(x)
        action_probs = torch.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action, log_prob, value

# ####################################################################
# --- 3. PPO Trainer Class ---
# ####################################################################

class PPOTrainer:
    """
    PPO trainer for trading agent.
    """
    
    def __init__(self, model_name: str, config: Dict[str, Any], output_path: str = '/models'):
        self.model_name = model_name
        self.config = config
        self.output_path = output_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        # Get model parameters
        model_params = config.get('model_params', {})
        self.data_input_dim = model_params.get('input_dim', 10)  # Raw data features
        self.input_dim = self.data_input_dim + 3  # +3 for account state
        self.hidden_dim = model_params.get('hidden_dim', 128)
        self.num_actions = model_params.get('num_actions', 3)
        self.lookback_window = model_params.get('lookback_window', 60)
        
        # Get training parameters
        train_params = config.get('train_params', {})
        self.learning_rate = train_params.get('learning_rate', 3e-4)
        self.epochs = train_params.get('epochs', 100)
        self.batch_size = train_params.get('batch_size', 64)
        self.ppo_epochs = train_params.get('ppo_epochs', 4)
        self.clip_ratio = train_params.get('clip_ratio', 0.2)
        self.value_coef = train_params.get('value_coef', 0.5)
        self.entropy_coef = train_params.get('entropy_coef', 0.01)
        
        # Initialize network
        # input_dim already includes +3 for account state from the config processing
        self.network = PPONetwork(
            input_dim=self.input_dim,  # This is already data_features + 3
            hidden_dim=self.hidden_dim,
            num_actions=self.num_actions
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
    
    def prepare_data(self, data: np.ndarray, features: List[str]):
        """Prepare data for training"""
        print("Preparing data for PPO training...")
        
        # Scale the data (only the market features, not account state)
        # The scaler should be fitted on the raw data features only
        scaled_data = self.scaler.fit_transform(data)
        
        # Create environment
        env = TradingEnvironment(
            data=scaled_data,
            features=features,
            lookback_window=self.lookback_window,
            initial_balance=50000
        )
        
        return env
    
    def collect_rollouts(self, env, num_episodes=10):
        """Collect rollouts for PPO training"""
        print(f"Collecting rollouts for {num_episodes} episodes...")
        
        observations = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        dones = []
        
        for episode in range(num_episodes):
            obs = env.reset()
            episode_rewards = []
            episode_obs = []
            episode_actions = []
            episode_log_probs = []
            episode_values = []
            episode_dones = []
            
            done = False
            while not done:
                # Convert observation to tensor
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                
                # Get action from network
                with torch.no_grad():
                    action, log_prob, value = self.network.get_action(obs_tensor)
                
                # Take action in environment
                next_obs, reward, done, info = env.step(action.item())
                
                # Store experience
                episode_obs.append(obs)
                episode_actions.append(action.item())
                episode_rewards.append(reward)
                episode_log_probs.append(log_prob.item())
                episode_values.append(value.item())
                episode_dones.append(done)
                
                obs = next_obs
            
            # Store episode data
            observations.extend(episode_obs)
            actions.extend(episode_actions)
            rewards.extend(episode_rewards)
            log_probs.extend(episode_log_probs)
            values.extend(episode_values)
            dones.extend(episode_dones)
            
            print(f"Episode {episode + 1}: Total reward = {sum(episode_rewards):.2f}")
        
        return {
            'observations': np.array(observations),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'log_probs': np.array(log_probs),
            'values': np.array(values),
            'dones': np.array(dones)
        }
    
    def compute_returns(self, rewards, values, dones, gamma=0.99):
        """Compute returns and advantages"""
        returns = []
        advantages = []
        
        # Compute returns
        running_return = 0
        for i in reversed(range(len(rewards))):
            if dones[i]:
                running_return = 0
            running_return = rewards[i] + gamma * running_return
            returns.insert(0, running_return)
        
        returns = np.array(returns)
        
        # Compute advantages
        advantages = returns - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def train(self, env):
        """Train the PPO agent"""
        print(f"Starting PPO training for model: {self.model_name}")
        
        best_reward = float('-inf')
        
        for epoch in range(self.epochs):
            # Collect rollouts
            rollouts = self.collect_rollouts(env, num_episodes=5)
            
            # Compute returns and advantages
            returns, advantages = self.compute_returns(
                rollouts['rewards'], 
                rollouts['values'], 
                rollouts['dones']
            )
            
            # Convert to tensors
            observations = torch.FloatTensor(rollouts['observations']).to(self.device)
            actions = torch.LongTensor(rollouts['actions']).to(self.device)
            old_log_probs = torch.FloatTensor(rollouts['log_probs']).to(self.device)
            returns = torch.FloatTensor(returns).to(self.device)
            advantages = torch.FloatTensor(advantages).to(self.device)
            
            # PPO training
            for ppo_epoch in range(self.ppo_epochs):
                # Shuffle data
                indices = torch.randperm(len(observations))
                
                for i in range(0, len(observations), self.batch_size):
                    batch_indices = indices[i:i + self.batch_size]
                    batch_obs = observations[batch_indices]
                    batch_actions = actions[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]
                    batch_returns = returns[batch_indices]
                    batch_advantages = advantages[batch_indices]
                    
                    # Forward pass
                    action_logits, values = self.network(batch_obs)
                    action_probs = torch.softmax(action_logits, dim=-1)
                    dist = Categorical(action_probs)
                    
                    # Compute new log probabilities
                    new_log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy().mean()
                    
                    # Compute ratios
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    
                    # Compute clipped surrogate loss
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()
                    
                    # Compute value loss
                    value_loss = nn.MSELoss()(values.squeeze(), batch_returns)
                    
                    # Total loss
                    total_loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                    self.optimizer.step()
            
            # Evaluate performance
            avg_reward = np.mean(rollouts['rewards'])
            print(f"Epoch {epoch + 1}/{self.epochs}: Average reward = {avg_reward:.2f}")
            
            # Save best model
            if avg_reward > best_reward:
                best_reward = avg_reward
                self.save()
                print(f"New best model saved with reward: {best_reward:.2f}")
        
        print("PPO training completed.")
    
    def save(self):
        """Save the trained model and configuration"""
        os.makedirs(self.output_path, exist_ok=True)
        print(f"Output directory '{self.output_path}' is ready.")
        
        scaler_filename = f"{self.model_name}_scaler.pkl"
        model_filename = f"{self.model_name}_model.pt"
        config_filename = f"{self.model_name}_config.yaml"
        
        scaler_path = os.path.join(self.output_path, scaler_filename)
        model_path = os.path.join(self.output_path, model_filename)
        config_path = os.path.join(self.output_path, config_filename)
        
        # Save scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved to: '{scaler_path}'")
        
        # Save model
        torch.save(self.network.state_dict(), model_path)
        print(f"Model saved to: '{model_path}'")
        
        # Save configuration
        final_config = {
            'model_name': self.model_name,
            'Type': 'PPO Agent',
            'artifact_paths': {
                'scaler': scaler_filename,
                'model_state_dict': model_filename
            },
            'Config': self.config.copy()
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(final_config, f, default_flow_style=False, sort_keys=False)
        print(f"Configuration saved to: '{config_path}'")
        print("Save process completed.")

# ####################################################################
# --- 4. Training Pipeline Function ---
# ####################################################################

def run_training_pipeline(model_name: str, output_dir: str, training_config: Dict, data: np.ndarray, features: List[str]):
    """Execute the full PPO training pipeline"""
    
    # Convert raw data numpy array to a DataFrame for processing
    feature_df_raw = pd.DataFrame(data, columns=features)
    feature_df_delta = prepare_delta_features(feature_df_raw)

    # Use the delta data for training
    delta_data = feature_df_delta.values

    # Split delta data for training
    train_size = int(len(delta_data) * 0.8)
    train_data = delta_data[:train_size]
    
    # Initialize trainer
    trainer = PPOTrainer(
        model_name=model_name,
        config=training_config,
        output_path=output_dir
    )
    
    # Prepare environment
    env = trainer.prepare_data(train_data, features)
    
    # Train the agent
    trainer.train(env)
    
    print("\n" + "="*50)
    print("--- PPO Training Pipeline Completed ---")
    print("="*50)
    
    # Verify saved files
    print(f"\nGenerated files in '{output_dir}':")
    for filename in os.listdir(output_dir):
        print(f"- {filename}")

if __name__ == '__main__':
    # --- 1. Load Data and Define Parameters ---
    data = pd.read_csv('sample.csv')
    data.columns = data.columns.str.lower()
    
    # --- 2. Prepare feature data (excluding date/time columns) ---
    columns_to_exclude = ['date', 'time']
    existing_columns_to_exclude = [col for col in columns_to_exclude if col in data.columns]
    feature_data = data.drop(columns=existing_columns_to_exclude, errors='ignore')
    feature_names = feature_data.columns.tolist()
    X_sample = feature_data.values
    
    print(f"Data prepared with {X_sample.shape[0]} samples and {X_sample.shape[1]} features.")
    print(f"Feature names: {feature_names[:10]}...")  # Show first 10 features
    
    # --- 3. Define the training configuration for PPO ---
    example_config = {
        'model_params': {
            'input_dim': X_sample.shape[1],  # Number of market features
            'hidden_dim': 128,
            'num_actions': 3,  # Hold, Buy, Sell
            'lookback_window': 60
        },
        'train_params': {
            'learning_rate': 3e-4,
            'epochs': 50,
            'batch_size': 64,
            'ppo_epochs': 4,
            'clip_ratio': 0.2,
            'value_coef': 0.5,
            'entropy_coef': 0.01
        },
        'data_params': {
            'features': feature_names
        }
    }
    
    # --- 4. Define a model name and output path ---
    MODEL_NAME = "nq_trading_agent_v1"
    OUTPUT_DIR = f"./Models/PPO_{MODEL_NAME}"
    
    # --- 5. Call the main training pipeline function ---
    print(f"\nStarting PPO training pipeline for model: {MODEL_NAME}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    run_training_pipeline(
        model_name=MODEL_NAME,
        output_dir=OUTPUT_DIR,
        training_config=example_config,
        data=X_sample,
        features=feature_names
    )