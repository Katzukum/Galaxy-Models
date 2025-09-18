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
from typing import Dict, List, Any, Tuple, Union
from datetime import datetime
from collections import deque
import gym
from gym import spaces
import random
from Utilities.data_utils import prepare_delta_features


class PPOEnsembleEnvironment(gym.Env):
    """
    Custom trading environment for PPO ensemble training.
    Uses individual model predictions as features to learn optimal trading strategies.
    Actions: 0=Hold, 1=Buy, 2=Sell
    """
    
    def __init__(self, data, model_predictions, features, lookback_window=60, 
                 initial_balance=50000, position_size=0.1, transaction_cost=0.001):
        super(PPOEnsembleEnvironment, self).__init__()
        
        self.data = data  # Market data (price, volume, technical indicators)
        self.model_predictions = model_predictions  # Individual model predictions
        self.features = features
        self.lookback_window = lookback_window
        self.initial_balance = initial_balance
        self.position_size = position_size
        self.transaction_cost = transaction_cost
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: model predictions + market data + portfolio state
        n_model_features = model_predictions.shape[1]  # Number of individual models
        n_market_features = data.shape[1]  # Market data features
        n_portfolio_features = 3  # balance, position, unrealized_pnl
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(lookback_window, n_model_features + n_market_features + n_portfolio_features), 
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0  # 0=no position, 1=long, -1=short
        self.position_size_actual = 0
        self.entry_price = 0
        self.unrealized_pnl = 0
        self.trades = []
        self.equity_history = [self.initial_balance]
        
        return self._get_observation()
    
    def _get_observation(self):
        """Get current observation including model predictions, market data, and portfolio state"""
        if self.current_step < self.lookback_window:
            # Pad with zeros if not enough history
            start_idx = max(0, self.current_step - self.lookback_window)
            end_idx = self.current_step
            pad_size = self.lookback_window - (end_idx - start_idx)
            
            model_data = self.model_predictions[start_idx:end_idx, :]
            market_data = self.data[start_idx:end_idx, :]
            
            if pad_size > 0:
                model_data = np.vstack([np.zeros((pad_size, model_data.shape[1])), model_data])
                market_data = np.vstack([np.zeros((pad_size, market_data.shape[1])), market_data])
        else:
            start_idx = self.current_step - self.lookback_window
            end_idx = self.current_step
            model_data = self.model_predictions[start_idx:end_idx, :]
            market_data = self.data[start_idx:end_idx, :]
        
        # Add portfolio state to each timestep
        portfolio_state = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.position,  # Position (-1, 0, 1)
            self.unrealized_pnl / self.initial_balance  # Normalized unrealized PnL
        ])
        
        # Broadcast portfolio state to all timesteps
        portfolio_state_broadcast = np.tile(portfolio_state, (self.lookback_window, 1))
        
        # Combine model predictions, market data, and portfolio state
        observation = np.concatenate([model_data, market_data, portfolio_state_broadcast], axis=1)
        
        return observation.astype(np.float32)
    
    def step(self, action):
        """Execute action and return next observation, reward, done, info"""
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, {}
        
        current_price = self.data[self.current_step, 0]  # Assuming close price is first column
        reward = 0
        info = {}
        
        # Execute trading action
        if action == 1:  # Buy
            if self.position <= 0:  # Can only buy if not already long
                # Close short position if exists
                if self.position < 0:
                    self._close_position(current_price)
                
                # Open long position
                self._open_position(current_price, 1)
                
        elif action == 2:  # Sell
            if self.position >= 0:  # Can only sell if not already short
                # Close long position if exists
                if self.position > 0:
                    self._close_position(current_price)
                
                # Open short position
                self._open_position(current_price, -1)
        
        # Update unrealized PnL
        if self.position != 0:
            if self.position > 0:  # Long position
                self.unrealized_pnl = (current_price - self.entry_price) * self.position_size_actual
            else:  # Short position
                self.unrealized_pnl = (self.entry_price - current_price) * abs(self.position_size_actual)
        else:
            self.unrealized_pnl = 0
        
        # Calculate reward based on portfolio performance
        current_equity = self.balance + self.unrealized_pnl
        self.equity_history.append(current_equity)
        
        # Reward calculation
        if len(self.equity_history) > 1:
            equity_change = (current_equity - self.equity_history[-2]) / self.equity_history[-2]
            reward = equity_change * 100  # Scale reward
        else:
            reward = 0
        
        # Add small penalty for holding to encourage trading
        if action == 0:  # Hold
            reward -= 0.001
        
        # Update step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1
        
        info = {
            'balance': self.balance,
            'position': self.position,
            'unrealized_pnl': self.unrealized_pnl,
            'equity': current_equity,
            'trades': len(self.trades)
        }
        
        return self._get_observation(), reward, done, info
    
    def _open_position(self, price, direction):
        """Open a new position"""
        if self.position == 0:  # Only open if no current position
            self.position = direction
            self.entry_price = price
            self.position_size_actual = self.balance * self.position_size
            self.unrealized_pnl = 0
    
    def _close_position(self, price):
        """Close current position"""
        if self.position != 0:
            # Calculate PnL
            if self.position > 0:  # Long position
                pnl = (price - self.entry_price) * self.position_size_actual
            else:  # Short position
                pnl = (self.entry_price - price) * abs(self.position_size_actual)
            
            # Apply transaction cost
            transaction_cost_amount = self.position_size_actual * self.transaction_cost
            pnl -= transaction_cost_amount
            
            # Update balance
            self.balance += pnl
            
            # Record trade
            self.trades.append({
                'entry_price': self.entry_price,
                'exit_price': price,
                'position': self.position,
                'pnl': pnl,
                'transaction_cost': transaction_cost_amount
            })
            
            # Reset position
            self.position = 0
            self.position_size_actual = 0
            self.entry_price = 0
            self.unrealized_pnl = 0


class PPOEnsembleNetwork(nn.Module):
    """
    PPO network architecture for ensemble trading.
    Takes model predictions and market data as input.
    """
    
    def __init__(self, input_size, hidden_size=128, num_actions=3):
        super(PPOEnsembleNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_actions)
        )
        
        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        """Forward pass through the network"""
        batch_size, seq_len, _ = x.shape
        
        # Reshape for feature extraction
        x_flat = x.view(-1, self.input_size)
        features = self.feature_extractor(x_flat)
        features = features.view(batch_size, seq_len, self.hidden_size)
        
        # LSTM processing
        lstm_out, _ = self.lstm(features)
        
        # Use last timestep for action and value
        last_output = lstm_out[:, -1, :]
        
        # Actor and critic outputs
        action_logits = self.actor(last_output)
        value = self.critic(last_output)
        
        return action_logits, value
    
    def get_action(self, x):
        """Get action and log probability for given observation"""
        action_logits, value = self.forward(x)
        action_probs = torch.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action, log_prob, value


class PPOEnsembleTrainer:
    """
    A class to train a PPO-based ensemble model that learns to combine
    individual model predictions for optimal trading strategies.
    """
    
    def __init__(self, 
                 model_name: str, 
                 config: Dict[str, Any], 
                 output_path: str = '/models'):
        """
        Initializes the PPOEnsembleTrainer.

        Args:
            model_name (str): A unique name for the ensemble model.
            config (dict): A dictionary containing ensemble configuration.
            output_path (str): The directory path to save the output files.
        """
        if not model_name:
            raise ValueError("A 'model_name' must be provided.")
            
        self.model_name = model_name
        self.config = config
        self.output_path = output_path
        
        # Extract ensemble configuration
        self.ensemble_type = config.get('ensemble_type', 'ppo')
        self.selected_models = config.get('selected_models', [])
        self.ppo_params = config.get('ppo_params', {})
        self.trading_params = config.get('trading_params', {})
        self.features = config.get('features', [])
        
        # PPO parameters with defaults
        self.learning_rate = self.ppo_params.get('learning_rate', 0.0003)
        self.epochs = self.ppo_params.get('epochs', 100)
        self.batch_size = self.ppo_params.get('batch_size', 64)
        self.sequence_length = self.ppo_params.get('sequence_length', 60)
        self.gamma = self.ppo_params.get('gamma', 0.99)
        self.clip_ratio = self.ppo_params.get('clip_ratio', 0.2)
        
        # Trading parameters with defaults
        self.initial_balance = self.trading_params.get('initial_balance', 50000)
        self.position_size = self.trading_params.get('position_size', 0.1)
        self.transaction_cost = self.trading_params.get('transaction_cost', 0.001)
        
        # Initialize model loaders
        self.model_loaders = {}
        self.ppo_model = None
        self.scaler = None
        
        # Training data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model_predictions_train = None
        self.model_predictions_test = None
        
        # Results
        self.training_results = {}
        
    def load_models(self):
        """Load all selected models for prediction generation"""
        print(f"Loading {len(self.selected_models)} models for PPO ensemble...")
        
        # Import model loaders
        try:
            from NetworkConfigs.NN_loader import NNModelLoader
            from NetworkConfigs.Transformer_loader import TransformerModelLoader
            from NetworkConfigs.XGBoost_loader import XGBoostModelLoader
            from NetworkConfigs.PPO_loader import PPOModelLoader
        except ImportError:
            # Try importing with relative path if running from NetworkConfigs directory
            from NN_loader import NNModelLoader
            from Transformer_loader import TransformerModelLoader
            from XGBoost_loader import XGBoostModelLoader
            from PPO_loader import PPOModelLoader
        
        for model_info in self.selected_models:
            model_name = model_info['name']
            model_type = model_info['type']
            model_dir = os.path.dirname(model_info['configPath'])
            
            print(f"Loading model: {model_name} ({model_type})")
            
            try:
                # Load model based on type
                model_type_lower = model_type.lower()
                if 'neural network' in model_type_lower or model_type_lower in ['nn', 'neural network (regression)']:
                    loader = NNModelLoader(model_dir)
                elif 'transformer' in model_type_lower or 'time-series transformer' in model_type_lower:
                    loader = TransformerModelLoader(model_dir)
                elif 'xgboost' in model_type_lower or 'xgboostclassifier' in model_type_lower:
                    loader = XGBoostModelLoader(model_dir)
                elif 'ppo' in model_type_lower or 'ppo agent' in model_type_lower:
                    loader = PPOModelLoader(model_dir)
                else:
                    print(f"Warning: Unknown model type {model_type} for {model_name}, skipping...")
                    continue
                
                self.model_loaders[model_name] = {
                    'loader': loader,
                    'type': model_type,
                    'config_path': model_info['configPath']
                }
                print(f"Successfully loaded {model_name}")
                
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
                continue
        
        print(f"Successfully loaded {len(self.model_loaders)} models")
    
    def prepare_ensemble_data(self, csv_path: str):
        """
        Load and prepare data for PPO ensemble training.
        Applies same preprocessing as current ensemble system.
        """
        print(f"Loading data from: {csv_path}")
        
        # Load CSV data
        data = pd.read_csv(csv_path)
        
        # Normalize column names to lowercase
        data.columns = data.columns.str.lower()
        
        # Apply delta features preprocessing
        data = prepare_delta_features(data)
        
        # Select features
        if self.features:
            available_features = [col for col in self.features if col in data.columns]
            if not available_features:
                print("Warning: No specified features found in data, using all numeric columns")
                available_features = data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            available_features = data.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"Using features: {available_features}")
        
        # Prepare feature matrix
        X = data[available_features].values
        
        # Create target (for compatibility, though PPO doesn't use it directly)
        # Use price change as target (same as current ensemble)
        if 'close' in data.columns:
            y = data['close'].pct_change().fillna(0).values
        else:
            y = np.zeros(len(data))
        
        # Split data
        split_idx = int(len(X) * 0.8)
        self.X_train = X[:split_idx]
        self.X_test = X[split_idx:]
        self.y_train = y[:split_idx]
        self.y_test = y[split_idx:]
        
        # Normalize features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"Data prepared: Train={len(self.X_train)}, Test={len(self.X_test)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def collect_model_predictions(self, X_data):
        """
        Generate predictions from all individual models.
        Returns matrix of shape (n_samples, n_models)
        """
        print("Collecting predictions from individual models...")
        
        predictions = []
        model_names = []
        
        for model_name, model_info in self.model_loaders.items():
            try:
                loader = model_info['loader']
                model_type = model_info['type']
                
                print(f"Generating predictions from {model_name} ({model_type})")
                
                # Generate predictions based on model type
                if 'neural network' in model_type.lower() or 'nn' in model_type.lower():
                    pred = loader.predict(X_data)
                elif 'transformer' in model_type.lower():
                    pred = loader.predict(X_data)
                elif 'xgboost' in model_type.lower():
                    pred = loader.predict(X_data)
                elif 'ppo' in model_type.lower():
                    # For PPO models, we need to handle action-based predictions
                    pred = loader.predict(X_data)
                else:
                    print(f"Warning: Unknown model type {model_type} for {model_name}")
                    continue
                
                # Ensure predictions are 1D
                if pred.ndim > 1:
                    pred = pred.flatten()
                
                predictions.append(pred)
                model_names.append(model_name)
                
            except Exception as e:
                print(f"Error generating predictions from {model_name}: {e}")
                continue
        
        if not predictions:
            raise ValueError("No valid predictions generated from any model")
        
        # Stack predictions into matrix
        predictions_matrix = np.column_stack(predictions)
        
        print(f"Generated predictions matrix: {predictions_matrix.shape}")
        print(f"Model names: {model_names}")
        
        return predictions_matrix, model_names
    
    def create_ppo_dataset(self, X_data, model_predictions):
        """
        Create dataset for PPO training by combining model predictions and market data.
        """
        print("Creating PPO training dataset...")
        
        # Ensure data lengths match
        min_length = min(len(X_data), len(model_predictions))
        X_data = X_data[:min_length]
        model_predictions = model_predictions[:min_length]
        
        # Create sequences for time-series PPO training
        sequences = []
        targets = []
        
        for i in range(self.sequence_length, len(X_data)):
            # Get sequence of model predictions and market data
            seq_model_preds = model_predictions[i-self.sequence_length:i]
            seq_market_data = X_data[i-self.sequence_length:i]
            
            # Combine into single sequence
            sequence = np.concatenate([seq_model_preds, seq_market_data], axis=1)
            sequences.append(sequence)
            
            # Target is next period's return (for reward calculation)
            if i < len(X_data) - 1:
                target = X_data[i+1, 0] - X_data[i, 0]  # Price change
            else:
                target = 0
            targets.append(target)
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        print(f"Created PPO dataset: {sequences.shape}")
        
        return sequences, targets
    
    def train(self):
        """
        Train the PPO ensemble model.
        """
        print("Starting PPO ensemble training...")
        
        # Load individual models
        self.load_models()
        
        # Prepare data
        self.prepare_ensemble_data(self.config.get('csv_path', 'sample.csv'))
        
        # Collect model predictions
        self.model_predictions_train, model_names = self.collect_model_predictions(self.X_train)
        self.model_predictions_test, _ = self.collect_model_predictions(self.X_test)
        
        # Create PPO training dataset
        train_sequences, train_targets = self.create_ppo_dataset(self.X_train, self.model_predictions_train)
        
        # Create PPO environment
        env = PPOEnsembleEnvironment(
            data=self.X_train,
            model_predictions=self.model_predictions_train,
            features=self.features,
            lookback_window=self.sequence_length,
            initial_balance=self.initial_balance,
            position_size=self.position_size,
            transaction_cost=self.transaction_cost
        )
        
        # Initialize PPO model
        input_size = train_sequences.shape[2]  # Features per timestep
        self.ppo_model = PPOEnsembleNetwork(
            input_size=input_size,
            hidden_size=128,
            num_actions=3
        )
        
        # Training loop
        optimizer = optim.Adam(self.ppo_model.parameters(), lr=self.learning_rate)
        
        print(f"Starting PPO training for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            # Collect rollouts
            observations, actions, rewards, log_probs, values = self._collect_rollouts(env)
            
            # Compute returns and advantages
            returns, advantages = self._compute_returns_and_advantages(rewards)
            
            # Update policy
            policy_loss, value_loss = self._update_policy(
                observations, actions, log_probs, returns, advantages, values
            )
            
            if epoch % 10 == 0:
                avg_reward = np.mean(rewards)
                print(f"Epoch {epoch}: Avg Reward={avg_reward:.4f}, Policy Loss={policy_loss:.4f}, Value Loss={value_loss:.4f}")
        
        print("PPO ensemble training completed!")
        
        # Save model
        self.save_model()
        
        return self.training_results
    
    def _collect_rollouts(self, env, num_rollouts=10):
        """Collect rollouts from the environment"""
        observations = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        
        for _ in range(num_rollouts):
            obs = env.reset()
            done = False
            
            while not done:
                # Convert observation to tensor
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                
                # Get action from model
                with torch.no_grad():
                    action, log_prob, value = self.ppo_model.get_action(obs_tensor)
                
                # Take action
                next_obs, reward, done, info = env.step(action.item())
                
                # Store experience
                observations.append(obs)
                actions.append(action.item())
                rewards.append(reward)
                log_probs.append(log_prob.item())
                values.append(value.item())
                
                obs = next_obs
        
        return (np.array(observations), np.array(actions), np.array(rewards), 
                np.array(log_probs), np.array(values))
    
    def _compute_returns_and_advantages(self, rewards, gamma=0.99):
        """Compute returns and advantages for PPO"""
        returns = []
        advantages = []
        
        # Compute returns
        running_return = 0
        for reward in reversed(rewards):
            running_return = reward + gamma * running_return
            returns.insert(0, running_return)
        
        returns = np.array(returns)
        
        # Compute advantages (simplified)
        advantages = returns - np.mean(returns)
        
        return returns, advantages
    
    def _update_policy(self, observations, actions, old_log_probs, returns, advantages, old_values):
        """Update PPO policy"""
        # Convert to tensors
        obs_tensor = torch.FloatTensor(observations)
        action_tensor = torch.LongTensor(actions)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs)
        returns_tensor = torch.FloatTensor(returns)
        advantages_tensor = torch.FloatTensor(advantages)
        old_values_tensor = torch.FloatTensor(old_values)
        
        # Get current policy outputs
        action_logits, values = self.ppo_model(obs_tensor)
        action_probs = torch.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        
        # Compute new log probabilities
        new_log_probs = dist.log_prob(action_tensor)
        
        # Compute policy loss (PPO)
        ratio = torch.exp(new_log_probs - old_log_probs_tensor)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        policy_loss = -torch.min(ratio * advantages_tensor, clipped_ratio * advantages_tensor).mean()
        
        # Compute value loss
        value_loss = nn.MSELoss()(values.squeeze(), returns_tensor)
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss
        
        # Update model
        optimizer = optim.Adam(self.ppo_model.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return policy_loss.item(), value_loss.item()
    
    def save_model(self):
        """Save the trained PPO ensemble model and configuration"""
        print("Saving PPO ensemble model...")
        
        # Create output directory
        model_dir = os.path.join(self.output_path, f"{self.model_name}_ppo_ensemble")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save PPO model
        model_path = os.path.join(model_dir, f"{self.model_name}_ppo_model.pth")
        torch.save(self.ppo_model.state_dict(), model_path)
        
        # Save scaler
        scaler_path = os.path.join(model_dir, f"{self.model_name}_scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save model references
        model_refs = {}
        for model_name, model_info in self.model_loaders.items():
            model_refs[model_name] = {
                'type': model_info['type'],
                'config_path': model_info['config_path']
            }
        
        model_refs_path = os.path.join(model_dir, f"{self.model_name}_model_refs.yaml")
        with open(model_refs_path, 'w') as f:
            yaml.dump(model_refs, f)
        
        # Create main configuration file
        config_data = {
            'model_name': self.model_name,
            'model_type': 'PPOEnsemble',
            'Config': {
                'ensemble_type': self.ensemble_type,
                'selected_models': self.selected_models,
                'ppo_params': self.ppo_params,
                'trading_params': self.trading_params,
                'features': self.features,
                'sequence_length': self.sequence_length,
                'input_size': self.ppo_model.input_size if self.ppo_model else None
            }
        }
        
        config_path = os.path.join(model_dir, f"{self.model_name}_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        print(f"Model saved to: {model_dir}")
        
        return model_dir


def run_ppo_ensemble_training(model_name, config, output_path='/models'):
    """
    Main function to run PPO ensemble training.
    Compatible with existing training pipeline.
    """
    try:
        trainer = PPOEnsembleTrainer(model_name, config, output_path)
        results = trainer.train()
        return {'success': True, 'results': results, 'model_dir': trainer.output_path}
    except Exception as e:
        print(f"Error in PPO ensemble training: {e}")
        return {'success': False, 'error': str(e)}