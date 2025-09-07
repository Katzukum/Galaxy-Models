import os
import yaml
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
from collections import deque

# Import the PPO network from PPOTrainer
from NetworkConfigs.PPOTrainer import PPONetwork

class PPOBacktester:
    """
    Specialized backtester for PPO agents.
    """
    
    def __init__(self, config_path: str, data_path: str):
        self.config_path = config_path
        self.data_path = data_path
        self.model_dir = os.path.dirname(config_path)
        
        self.model = None
        self.scaler = None
        self.config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history = deque(maxlen=60)  # Store last 60 observations
        
    def load_artifacts(self):
        """Load the PPO model and its artifacts"""
        print("--- Loading PPO Artifacts ---")
        
        # Load config
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print(f"Model Type: {self.config.get('Type')}")
        
        # Load scaler
        artifact_paths = self.config['artifact_paths']
        scaler_path = os.path.join(self.model_dir, artifact_paths['scaler'])
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print("Scaler loaded.")
        
        # Load model
        model_path = os.path.join(self.model_dir, artifact_paths['model_state_dict'])
        model_params = self.config['Config']['model_params']
        
        self.model = PPONetwork(
            input_dim=model_params['input_dim'] + 3,  # +3 for account state
            hidden_dim=model_params['hidden_dim'],
            num_actions=model_params['num_actions']
        )
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print("PPO model loaded successfully.")
    
    def run_with_results(self, initial_capital=50000, take_profit_pips=50, stop_loss_pips=25, tick_size=0.25, tick_value=5):
        """Execute the PPO backtest and return results"""
        print("\n--- Starting PPO Backtest ---")
        print(f"Loading CSV from: {self.data_path}")
        
        # Load and prepare data
        try:
            df = pd.read_csv(self.data_path)
            df.columns = df.columns.str.lower()
            print(f"CSV loaded successfully. Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
        except Exception as e:
            print(f"Error loading CSV: {e}")
            raise Exception(f"Failed to load CSV file: {e}")
        
        # Get features from config
        if 'data_params' in self.config['Config'] and 'features' in self.config['Config']['data_params']:
            model_features = self.config['Config']['data_params']['features']
        elif 'features' in self.config['Config']:
            model_features = self.config['Config']['features']
        else:
            raise KeyError("Could not find features in configuration")
        
        # Verify features
        if not all(feature in df.columns for feature in model_features):
            missing_features = [f for f in model_features if f not in df.columns]
            raise ValueError(f"Dataframe is missing required features: {missing_features}")
        
        features_df = df[model_features].copy()
        
        # Get sequence length
        seq_length = self.config['Config']['model_params'].get('lookback_window', 60)
        
        # Backtest loop
        capital = initial_capital
        position = 0  # 1 for long, -1 for short, 0 for flat
        entry_price = 0
        trades = []
        equity_history = [initial_capital]
        
        for i in range(seq_length, len(features_df)):
            current_price = df.loc[i, 'close']
            
            # Check for take-profit or stop-loss
            if position != 0:
                pnl = 0
                trade_closed = False
                if position == 1:  # Long
                    if current_price >= entry_price + (take_profit_pips * tick_size):
                        pnl = (current_price - entry_price) * tick_value
                        trade_closed = True
                    elif current_price <= entry_price - (stop_loss_pips * tick_size):
                        pnl = (current_price - entry_price) * tick_value
                        trade_closed = True
                elif position == -1:  # Short
                    if current_price <= entry_price - (take_profit_pips * tick_size):
                        pnl = (entry_price - current_price) * tick_value
                        trade_closed = True
                    elif current_price >= entry_price + (stop_loss_pips * tick_size):
                        pnl = (entry_price - current_price) * tick_value
                        trade_closed = True
                
                if trade_closed:
                    capital += pnl
                    trades.append(pnl)
                    position = 0
            
            # Prepare input data for PPO prediction
            input_data = features_df.iloc[i-seq_length:i].values
            scaled_input = self.scaler.transform(input_data)
            
            # Add account state to the input
            account_state = np.array([
                capital / initial_capital,  # Normalized balance
                position,  # Position (-1, 0, 1)
                0.0  # Unrealized PnL (simplified)
            ])
            account_state_broadcast = np.tile(account_state, (scaled_input.shape[0], 1))
            input_with_state = np.concatenate([scaled_input, account_state_broadcast], axis=1)
            
            # Make PPO prediction
            input_tensor = torch.FloatTensor(input_with_state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action_logits, value = self.model(input_tensor)
                action_probs = torch.softmax(action_logits, dim=-1)
                action = torch.argmax(action_probs, dim=-1).item()
            
            # Execute action based on PPO prediction
            if position == 0:  # No current position
                if action == 1:  # Buy
                    position = 1
                    entry_price = current_price
                elif action == 2:  # Sell
                    position = -1
                    entry_price = current_price
                # action == 0 means Hold, so no action taken
            
            elif position != 0:  # Have a position
                # Check for opposite signals
                if position == 1 and action == 2:  # Long position, Sell signal
                    pnl = (current_price - entry_price) * tick_value
                    capital += pnl
                    trades.append(pnl)
                    position = 0
                elif position == -1 and action == 1:  # Short position, Buy signal
                    pnl = (entry_price - current_price) * tick_value
                    capital += pnl
                    trades.append(pnl)
                    position = 0
            
            equity_history.append(capital)
        
        # Calculate metrics
        metrics = self.calculate_metrics(trades, initial_capital, np.array(equity_history))
        
        return {
            'trades': trades,
            'equity_history': equity_history,
            'metrics': metrics
        }
    
    def calculate_metrics(self, trades, initial_capital, equity_curve):
        """Calculate performance metrics"""
        if not trades:
            return {
                'net_profit': 0,
                'max_drawdown': 0,
                'profit_factor': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'total_trades': 0,
                'final_capital': initial_capital
            }
        
        trades = np.array(trades)
        net_profit = np.sum(trades)
        gross_profit = np.sum(trades[trades > 0])
        gross_loss = np.abs(np.sum(trades[trades < 0]))
        
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf
        win_rate = np.sum(trades > 0) / len(trades) * 100 if len(trades) > 0 else 0
        
        avg_win = np.mean(trades[trades > 0]) if np.sum(trades > 0) > 0 else 0
        avg_loss = np.abs(np.mean(trades[trades < 0])) if np.sum(trades < 0) > 0 else 0
        
        # Calculate max drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_drawdown = np.max(drawdown) * 100
        
        final_capital = initial_capital + net_profit
        
        return {
            'net_profit': net_profit,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor if profit_factor != np.inf else 0,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': len(trades),
            'final_capital': final_capital
        }

if __name__ == '__main__':
    # Example usage
    print("PPO Backtester - Example usage")
    pass