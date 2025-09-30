# --- START OF FILE PPOEnsembleTrainer.py ---

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
import gymnasium as gym
from gymnasium import spaces
import random
from Utilities.data_utils import prepare_delta_features
import time


class PPOEnsembleEnvironment(gym.Env):
    """
    Custom trading environment for PPO ensemble training.
    Uses individual model predictions as features to learn optimal trading strategies.
    Actions: 0=Hold, 1=Buy, 2=Sell
    """
    
    # --- FIX 3c: Modified __init__ to accept the fitted scaler ---
    def __init__(self, data, model_predictions, features, scaler, lookback_window=60, 
                 initial_balance=50000, position_size=0.1, transaction_cost=0.001):
        super(PPOEnsembleEnvironment, self).__init__()
        
        self.data = data  # Market data (price, volume, technical indicators)
        self.model_predictions = model_predictions  # Individual model predictions
        self.features = features
        self.scaler = scaler # Store the PPO's meta-scaler
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
        
        # The observation space must reflect the total number of features after concatenation
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(lookback_window, n_market_features + n_model_features + n_portfolio_features), 
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
        
        # Portfolio state is NOT scaled
        portfolio_state = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.position,  # Position (-1, 0, 1)
            self.unrealized_pnl / self.initial_balance  # Normalized unrealized PnL
        ])
        
        portfolio_state_broadcast = np.tile(portfolio_state, (self.lookback_window, 1))
        
        # --- FIX 3d: Apply PPO's scaling within the environment ---
        # 1. Combine raw market data and model predictions in the correct order.
        features_to_scale = np.concatenate([market_data, model_data], axis=1)
        
        # 2. Apply the PPO's pre-fitted scaler to this combined feature set.
        scaled_features = self.scaler.transform(features_to_scale)
        
        # 3. Combine the SCALED features with the UNSCALED portfolio state.
        observation = np.concatenate([scaled_features, portfolio_state_broadcast], axis=1)
        
        return observation.astype(np.float32)
    
    # In PPOEnsembleEnvironment.step()

    def step(self, action):
        """Execute action and return next observation, reward, done, info"""
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, {}
        
        current_price = self.data[self.current_step, 0]

        if np.isnan(current_price):
            print(f"Warning: NaN price detected at step {self.current_step}. Ending episode.")
            return self._get_observation(), 0, True, {}

        reward = 0
        info = {}
        
        # ... (trading action logic remains the same)
        if action == 1:
            if self.position <= 0:
                if self.position < 0:
                    self._close_position(current_price)
                self._open_position(current_price, 1)
        elif action == 2:
            if self.position >= 0:
                if self.position > 0:
                    self._close_position(current_price)
                self._open_position(current_price, -1)
        
        # Update unrealized PnL
        if self.position != 0:
            if self.position > 0:
                self.unrealized_pnl = (current_price - self.entry_price) * self.position_size_actual
            else:
                self.unrealized_pnl = (self.entry_price - current_price) * abs(self.position_size_actual)
        else:
            self.unrealized_pnl = 0
        
        # Calculate current portfolio equity
        current_equity = self.balance + self.unrealized_pnl
        self.equity_history.append(current_equity)
        
        # --- START OF FIX: Robust Reward Calculation and Termination on Ruin ---
        
        # 1. Safe Reward Calculation
        previous_equity = self.equity_history[-2]
        if previous_equity > 1e-6:  # Check if previous equity is not zero
            equity_change = (current_equity - previous_equity) / previous_equity
            reward = equity_change * 100
        else:
            # If previous equity was zero, we can't calculate a percentage change.
            # Reward is simply the absolute change.
            reward = (current_equity - previous_equity)

        # 2. Add penalty for holding
        if action == 0:
            reward -= 0.001
        
        self.current_step += 1
        
        # 3. Check for financial ruin to end the episode
        # End if equity falls below 20% of the starting balance.
        done = self.current_step >= len(self.data) - 1
        if current_equity < (self.initial_balance * 0.05):
            print(f"Agent financially ruined at step {self.current_step}. Ending episode.")
            done = True
            reward = -200 # Apply a large penalty for going bankrupt
            
        # --- END OF FIX ---

        info = {
            'balance': self.balance,
            'position': self.position,
            'unrealized_pnl': self.unrealized_pnl,
            'equity': current_equity,
            'trades': len(self.trades)
        }
        
        return self._get_observation(), reward, done, info
    def _open_position(self, price, direction):
        if self.position == 0:
            self.position = direction
            self.entry_price = price
            self.position_size_actual = self.balance * self.position_size
            self.unrealized_pnl = 0
    
    def _close_position(self, price):
        if self.position != 0:
            if self.position > 0:
                pnl = (price - self.entry_price) * self.position_size_actual
            else:
                pnl = (self.entry_price - price) * abs(self.position_size_actual)
            
            transaction_cost_amount = self.position_size_actual * self.transaction_cost
            pnl -= transaction_cost_amount
            
            self.balance += pnl
            
            self.trades.append({
                'entry_price': self.entry_price,
                'exit_price': price,
                'position': self.position,
                'pnl': pnl,
                'transaction_cost': transaction_cost_amount
            })
            
            self.position = 0
            self.position_size_actual = 0
            self.entry_price = 0
            self.unrealized_pnl = 0


class PPOEnsembleNetwork(nn.Module):
    """
    PPO network architecture for ensemble trading.
    Takes model predictions and market data as input.
    """
    
    def __init__(self, input_size, hidden_size=64, num_actions=3):
        super(PPOEnsembleNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        
        print(f"Initializing PPO network with input_size: {input_size}, hidden_size: {hidden_size}")
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_actions)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        batch_size, seq_len, input_features = x.shape
        
        # This warning logic is no longer needed if input_size is calculated correctly
        if input_features != self.input_size:
            print(f"CRITICAL Error: Input feature size {input_features} doesn't match network's expected size {self.input_size}")
            # This will likely cause a runtime error, which is better than silently failing.
        
        x_flat = x.view(-1, self.input_size)
        features = self.feature_extractor(x_flat)
        features = features.view(batch_size, seq_len, self.hidden_size)
        
        if torch.isnan(features).any():
            print("Warning: NaN values detected in features after feature extraction")
        
        lstm_out, _ = self.lstm(features)
        
        if torch.isnan(lstm_out).any():
            print("Warning: NaN values detected in LSTM output")
        
        last_output = lstm_out[:, -1, :]
        
        action_logits = self.actor(last_output)
        value = self.critic(last_output)
        
        if torch.isnan(action_logits).any():
            print("Warning: NaN values detected in action_logits")
        if torch.isnan(value).any():
            print("Warning: NaN values detected in value")
        
        return action_logits, value
    
    def get_action(self, x):
        action_logits, value = self.forward(x)
        
        if torch.isnan(action_logits).any():
            print("Warning: NaN values in action_logits before softmax, replacing with zeros.")
            action_logits = torch.nan_to_num(action_logits, 0.0)
        
        action_probs = torch.softmax(action_logits, dim=-1)
        
        if torch.isnan(action_probs).any():
            print("Warning: NaN values in action_probs after softmax, using uniform distribution.")
            action_probs = torch.ones_like(action_probs) / action_probs.shape[-1]
        
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
        if not model_name:
            raise ValueError("A 'model_name' must be provided.")
            
        self.model_name = model_name
        self.config = config
        self.output_path = output_path
        
        self.ensemble_type = config.get('ensemble_type', 'ppo')
        self.selected_models = config.get('selected_models', [])
        self.ppo_params = config.get('ppo_params', {})
        self.trading_params = config.get('trading_params', {})
        self.features = config.get('features', [])
        
        self.learning_rate = self.ppo_params.get('learning_rate', 0.0003)
        self.epochs = self.ppo_params.get('epochs', 1000)
        self.batch_size = self.ppo_params.get('batch_size', 64)
        self.sequence_length = self.ppo_params.get('sequence_length', 60)
        self.gamma = self.ppo_params.get('gamma', 0.99)
        self.clip_ratio = self.ppo_params.get('clip_ratio', 0.2)
        
        self.initial_balance = self.trading_params.get('initial_balance', 50000)
        self.position_size = self.trading_params.get('position_size', 0.1)
        self.transaction_cost = self.trading_params.get('transaction_cost', 0.001)
        
        self.model_loaders = {}
        self.ppo_model = None
        self.scaler = None
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model_predictions_train = None
        self.model_predictions_test = None
        
        self.training_results = {}
        
    def load_models(self):
        """Load all selected models for prediction generation"""
        print(f"Loading {len(self.selected_models)} models for PPO ensemble...")
        
        try:
            from NetworkConfigs.NN_loader import NNModelLoader
            from NetworkConfigs.Transformer_loader import TransformerModelLoader
            from NetworkConfigs.XGBoost_loader import XGBoostModelLoader
            from NetworkConfigs.PPO_loader import PPOModelLoader
        except ImportError:
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
        """
        print(f"Loading data from: {csv_path}")
        data = pd.read_csv(csv_path)
        data.columns = data.columns.str.lower()
        self.original_data = data.copy()
        
        print("Using raw data without delta features preprocessing")
        
        basic_features = ['open', 'high', 'low', 'close', 'volume']
        available_features = [col for col in basic_features if col in data.columns]
        
        numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
        
        excluded_columns = ['index', 'id', 'timestamp', 'date', 'datetime']
        for feature in numeric_features:
            if feature not in available_features:
                if (feature.lower() == 'time' or 
                    (feature.lower().startswith('time') and 'plot' not in feature.lower() and 'day' not in feature.lower() and 'hour' not in feature.lower() and 'qrt' not in feature.lower())):
                    continue
                elif data[feature].isna().all():
                    continue
                else:
                    available_features.append(feature)
        
        basic_features_found = [f for f in basic_features if f in available_features]
        other_features = sorted([f for f in available_features if f not in basic_features])
        available_features = basic_features_found + other_features
        
        print(f"Using features: {available_features}")
        self.features = available_features
        
        X = data[available_features].values
        
        split_idx = int(len(X) * 0.8)
        self.X_train = X[:split_idx]
        self.X_test = X[split_idx:]
        self.y_train = None
        self.y_test = None
        
        self.scaler = StandardScaler()
        
        print(f"Data prepared: Train={len(self.X_train)}, Test={len(self.X_test)}")
        
        return self.X_train, self.X_test
    
    def collect_model_predictions(self, X_data):
        print("Collecting predictions from individual models...")
        all_model_preds_list = []
        model_names = []
        
        raw_features_df = pd.DataFrame(X_data, columns=self.features)
        
        for model_name, model_info in self.model_loaders.items():
            try:
                loader = model_info['loader']
                model_type = model_info['type']
                
                print(f"Generating predictions from {model_name} ({model_type})")
                
                feature_dicts = raw_features_df.to_dict('records')
                individual_predictions = []
                is_classifier = 'classifier' in model_type.lower()
                
                if is_classifier:
                    sorted_labels = sorted(loader.label_mapping.keys(), key=loader.label_mapping.get)

                for row_dict in feature_dicts:
                    try:
                        if is_classifier:
                            prob_dict = loader.predict_proba(row_dict)
                            prediction = [prob_dict[label] for label in sorted_labels]
                        else:
                            prediction = [loader.predict(row_dict)]
                        individual_predictions.append(prediction)
                    except ValueError:
                        pass
                
                if not individual_predictions:
                    raise ValueError(f"No successful predictions were generated for {model_name}.")

                num_missing = len(feature_dicts) - len(individual_predictions)
                
                if num_missing > 0:
                    padding_value = individual_predictions[0]
                    padding = [padding_value] * num_missing
                    individual_predictions = padding + individual_predictions

                pred_array = np.array(individual_predictions, dtype=np.float32)
                
                if pred_array.ndim == 1:
                    pred_array = pred_array.reshape(-1, 1)

                all_model_preds_list.append(pred_array)
                model_names.append(model_name)
                
            except Exception as e:
                raise RuntimeError(f"Failed to get predictions from model '{model_name}'. Reason: {e}") from e
        
        if not all_model_preds_list:
            raise ValueError("No valid predictions generated from any model")
        
        predictions_matrix = np.hstack(all_model_preds_list)
        return predictions_matrix, model_names 
        
    def create_ppo_dataset(self, X_data, model_predictions):
        """This function is kept for consistency but scaling is now handled in the environment."""
        print("Creating PPO dataset structure (scaling is now deferred to environment)...")
        min_length = min(len(X_data), len(model_predictions))
        X_data = X_data[:min_length]
        model_predictions = model_predictions[:min_length]
        
        combined_features = np.concatenate([X_data, model_predictions], axis=1)
        
        sequences = []
        targets = []
        for i in range(self.sequence_length, len(combined_features)):
            sequence = combined_features[i-self.sequence_length:i]
            sequences.append(sequence)
            if i < len(X_data) - 1:
                target = X_data[i+1, 0] - X_data[i, 0]
            else:
                target = 0
            targets.append(target)
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        return sequences, targets
    
    def train(self):
        """
        Train the PPO ensemble model.
        """
        print("Starting PPO ensemble training...")
        
        self.load_models()
        self.prepare_ensemble_data(self.config.get('csv_path', 'sample.csv'))
        
        self.model_predictions_train, model_names = self.collect_model_predictions(self.X_train)
        self.model_predictions_test, _ = self.collect_model_predictions(self.X_test)
        
        print("Fitting PPO's meta-scaler on combined features...")
        combined_features = np.concatenate([self.X_train, self.model_predictions_train], axis=1)
        
        # --- FIX 1: Handle potential NaNs before fitting the PPO's scaler ---
        if np.isnan(combined_features).any():
            print(f"Warning: NaNs detected in combined features. Count: {np.isnan(combined_features).sum()}. Replacing with 0.")
            combined_features = np.nan_to_num(combined_features)
        
        self.scaler.fit(combined_features)
        print(f"Scaler fitted on {combined_features.shape[1]} combined features.")
        
        # --- FIX 3a: Pass the fitted scaler to the environment ---
        env = PPOEnsembleEnvironment(
            data=self.X_train,
            model_predictions=self.model_predictions_train,
            features=self.features,
            scaler=self.scaler, # Pass the fitted PPO scaler
            lookback_window=self.sequence_length,
            initial_balance=self.initial_balance,
            position_size=self.position_size,
            transaction_cost=self.transaction_cost
        )
        
        # --- FIX 3b: Calculate the correct input size for the network ---
        num_raw_features = self.X_train.shape[1]
        num_model_predictions = self.model_predictions_train.shape[1]
        num_portfolio_features = 3  # balance, position, unrealized_pnl
        input_size = num_raw_features + num_model_predictions + num_portfolio_features
        
        print(f"Initializing PPO model with correct input_size: {input_size}")
        
        self.ppo_model = PPOEnsembleNetwork(
            input_size=input_size,
            hidden_size=64,
            num_actions=3
        )
        
        print(f"Starting PPO training for {self.epochs} epochs...")
        all_rewards = []
        all_policy_losses = []
        all_value_losses = []

        for epoch in range(self.epochs):
            observations, actions, rewards, log_probs, values = self._collect_rollouts(env, num_rollouts=1)
            returns, advantages = self._compute_returns_and_advantages(rewards, self.gamma)
            policy_loss, value_loss = self._update_policy(observations, actions, log_probs, returns, advantages, values)
            
            avg_reward = np.mean(rewards)
            all_rewards.append(avg_reward)
            all_policy_losses.append(policy_loss)
            all_value_losses.append(value_loss)
            
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch}/{self.epochs}: Avg Reward={avg_reward:.4f}, Policy Loss={policy_loss:.4f}, Value Loss={value_loss:.4f}")

        self.training_results = {
            'rewards': all_rewards,
            'policy_losses': all_policy_losses,
            'value_losses': all_value_losses,
            'final_avg_reward': np.mean(all_rewards[-10:])
        }
        
        print("PPO ensemble training completed!")
        self.save_model()
        
        return self.training_results
    
    def _collect_rollouts(self, env, num_rollouts=10):
        observations, actions, rewards, log_probs, values = [], [], [], [], []
        
        for _ in range(num_rollouts):
            obs = env.reset()
            done = False
            while not done:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    action, log_prob, value = self.ppo_model.get_action(obs_tensor)
                
                next_obs, reward, done, info = env.step(action.item())
                
                observations.append(obs)
                actions.append(action.item())
                rewards.append(reward)
                log_probs.append(log_prob.item())
                values.append(value.item())
                
                obs = next_obs
        
        return (np.array(observations), np.array(actions), np.array(rewards), 
                np.array(log_probs), np.array(values))
    
    def _compute_returns_and_advantages(self, rewards, gamma=0.99):
        returns = []
        running_return = 0
        for reward in reversed(rewards):
            running_return = reward + gamma * running_return
            returns.insert(0, running_return)
        
        returns = np.array(returns)
        advantages = returns - np.mean(returns)
        
        return returns, advantages
    
    def _update_policy(self, observations, actions, old_log_probs, returns, advantages, old_values):
        obs_tensor = torch.FloatTensor(observations)
        action_tensor = torch.LongTensor(actions)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs)
        returns_tensor = torch.FloatTensor(returns)
        advantages_tensor = torch.FloatTensor(advantages)
        
        action_logits, values = self.ppo_model(obs_tensor)
        dist = Categorical(torch.softmax(action_logits, dim=-1))
        new_log_probs = dist.log_prob(action_tensor)
        
        ratio = torch.exp(new_log_probs - old_log_probs_tensor)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        policy_loss = -torch.min(ratio * advantages_tensor, clipped_ratio * advantages_tensor).mean()
        
        value_loss = nn.MSELoss()(values.squeeze(), returns_tensor)
        total_loss = policy_loss + 0.5 * value_loss
        
        optimizer = optim.Adam(self.ppo_model.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return policy_loss.item(), value_loss.item()
    
    def save_model(self):
        print("Saving PPO ensemble model...")
        model_dir = os.path.join(self.output_path, f"{self.model_name}_ppo_ensemble")
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, f"{self.model_name}_ppo_model.pth")
        torch.save(self.ppo_model.state_dict(), model_path)
        
        scaler_path = os.path.join(model_dir, f"{self.model_name}_scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        model_refs = {}
        for model_name, model_info in self.model_loaders.items():
            model_refs[model_name] = {'type': model_info['type'], 'config_path': model_info['config_path']}
        
        model_refs_path = os.path.join(model_dir, f"{self.model_name}_model_refs.yaml")
        with open(model_refs_path, 'w') as f:
            yaml.dump(model_refs, f)
        
        total_features = len(self.features) if self.features else 0
        if hasattr(self, 'model_predictions_train') and self.model_predictions_train is not None:
            total_features += self.model_predictions_train.shape[1]
        
        config_data = {
            'model_name': self.model_name,
            'Type': 'PPO Ensemble',
            'model_type': 'PPOEnsemble',
            'Config': {
                'ensemble_type': self.ensemble_type,
                'selected_models': self.selected_models,
                'ppo_params': self.ppo_params,
                'trading_params': self.trading_params,
                'features': self.features,
                'sequence_length': self.sequence_length,
                'input_size': self.ppo_model.input_size if self.ppo_model else None,
                'hidden_size': self.ppo_model.hidden_size if self.ppo_model else 64,
                'num_raw_features': len(self.features) if self.features else 0,
                'num_model_predictions': self.model_predictions_train.shape[1] if hasattr(self, 'model_predictions_train') and self.model_predictions_train is not None else 0,
                'total_features': total_features,
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
        # The model directory is now determined within the save_model method
        saved_model_dir = os.path.join(output_path, f"{model_name}_ppo_ensemble")
        return {'success': True, 'results': results, 'model_dir': saved_model_dir}
    except Exception as e:
        print(f"Error in PPO ensemble training: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}