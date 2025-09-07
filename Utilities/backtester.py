import os
import yaml
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ####################################################################
# --- Re-defining Model Architectures ---
# ####################################################################

# Import PPO network for backtesting
from NetworkConfigs.PPOTrainer import PPONetwork

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int, nhead: int,
                 num_encoder_layers: int, dim_feedforward: int, dropout: float = 0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.d_model = d_model
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 5000, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.input_embedding(src) * np.sqrt(self.d_model)
        src = src + self.pos_encoder[:, :src.size(1), :]
        output = self.transformer_encoder(src)
        output = self.output_layer(output[:, -1, :])
        return output

def build_nn_from_config(architecture: list) -> nn.Module:
    """Builds a PyTorch Sequential model from an architecture configuration."""
    layers = []
    for layer_config in architecture:
        layer_type_str = layer_config.get('type')
        if not layer_type_str:
            continue
        params = layer_config.copy()
        del params['type']
        if hasattr(nn, layer_type_str):
            layer_class = getattr(nn, layer_type_str)
            layers.append(layer_class(**params))
        else:
            raise ValueError(f"Unknown layer type: {layer_type_str}")
    return nn.Sequential(*layers)


# ####################################################################
# --- Unified Backtester Class ---
# ####################################################################

class Backtester:
    def __init__(self, config_path: str, data_path: str):
        self.config_path = config_path
        self.data_path = data_path
        self.model_dir = os.path.dirname(config_path)
        
        self.model = None
        self.scaler = None
        self.config = None
        self.model_type = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_artifacts(self):
        """Loads model, scaler, and config from the YAML file."""
        print("--- Loading Artifacts ---")
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_type = self.config.get('Type')
        print(f"Model Type: {self.model_type}")

        artifact_paths = self.config['artifact_paths']
        original_config = self.config['Config']

        # Load scaler
        scaler_path = os.path.join(self.model_dir, artifact_paths['scaler'])
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print("Scaler loaded.")

        # Load model based on type
        model_key = 'model' if 'model' in artifact_paths else 'model_state_dict'
        model_path = os.path.join(self.model_dir, artifact_paths[model_key])

        if self.model_type == 'Time-Series Transformer':
            self.model = TimeSeriesTransformer(**original_config['model_params'])
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        elif self.model_type == 'XGBoostClassifier':
            self.model = xgb.XGBClassifier()
            self.model.load_model(model_path)
        elif self.model_type == 'Neural Network (Regression)':
            self.model = build_nn_from_config(original_config['architecture'])
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        elif self.model_type == 'PPO Agent':
            # Load PPO model
            model_params = original_config['model_params']
            self.model = PPONetwork(
                input_dim=model_params['input_dim'] + 3,  # +3 for account state
                hidden_dim=model_params['hidden_dim'],
                num_actions=model_params['num_actions']
            )
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        if isinstance(self.model, nn.Module):
            self.model.to(self.device)
            self.model.eval()
        
        print("Model loaded successfully.")

    def run(self, initial_capital=50000, take_profit_pips=50, stop_loss_pips=25, tick_size=0.25, tick_value=5):
        """Executes the backtest."""
        print("\n--- Starting Backtest ---")

        # 1. Load and prepare data
        df = pd.read_csv(self.data_path)
        
        # Gracefully handle different config structures for features
        model_features = None
        if 'data_params' in self.config['Config'] and 'features' in self.config['Config']['data_params']:
            # Path for Time-Series Transformer config
            model_features = self.config['Config']['data_params']['features']
        elif 'features' in self.config['Config']:
            # Path for XGBoost and NN configs
            model_features = self.config['Config']['features']
        else:
            raise KeyError("Could not find the 'features' list in the configuration YAML.")

        # Verify features
        if not all(feature in df.columns for feature in model_features):
            missing_features = [f for f in model_features if f not in df.columns]
            raise ValueError(f"Dataframe is missing required features: {missing_features}")

        features_df = df[model_features].copy()

        # 2. Backtest loop
        capital = initial_capital
        position = 0  # 1 for long, -1 for short, 0 for flat
        entry_price = 0
        trades = []
        equity_history = [initial_capital]

        # Determine sequence length for sequence-based models
        seq_length = 1
        if self.model_type == 'Time-Series Transformer':
            seq_length = self.config['Config']['data_params']['sequence_length']

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
            
            # Prepare input data for prediction (needed for both position checking and new signals)
            if self.model_type == 'Time-Series Transformer':
                input_data = features_df.iloc[i-seq_length:i].values
                scaled_input = self.scaler.transform(input_data)
            elif self.model_type == 'PPO Agent':
                # PPO needs sequence data
                input_data = features_df.iloc[i-seq_length:i].values
                scaled_input = self.scaler.transform(input_data)
            else:  # XGBoost and NN
                input_data = features_df.iloc[[i]].values
                scaled_input = self.scaler.transform(input_data)

            # Generate prediction based on model type
            prediction = None
            if self.model_type == 'XGBoostClassifier':
                prediction = self.model.predict(scaled_input)
            
            elif isinstance(self.model, nn.Module):
                input_tensor = torch.FloatTensor(scaled_input).to(self.device)
                if self.model_type == 'Time-Series Transformer':
                     input_tensor = input_tensor.unsqueeze(0)
                     
                with torch.no_grad():
                    prediction = self.model(input_tensor)

            # Check for opposite signals if still in position
            if position != 0:
                opposite_signal = False
                if self.model_type == 'Time-Series Transformer':
                    pred_price_scaled = prediction.cpu().numpy().flatten()[0]
                    dummy_array = np.zeros((1, len(model_features)))
                    dummy_array[0, 0] = pred_price_scaled
                    predicted_price = self.scaler.inverse_transform(dummy_array)[0, 0]

                    if position == 1 and predicted_price < current_price - (2 * tick_size):  # Long position, sell signal
                        opposite_signal = True
                    elif position == -1 and predicted_price > current_price + (2 * tick_size):  # Short position, buy signal
                        opposite_signal = True

                elif self.model_type == 'XGBoostClassifier':
                    action = int(prediction[0])
                    if position == 1 and action == 0:  # Long position, Strong Sell signal
                        opposite_signal = True
                    elif position == -1 and action == 4:  # Short position, Strong Buy signal
                        opposite_signal = True

                elif self.model_type == 'Neural Network (Regression)':
                    predicted_tick_change = prediction.cpu().numpy().flatten()[0]
                    if position == 1 and predicted_tick_change < -2:  # Long position, negative tick change
                        opposite_signal = True
                    elif position == -1 and predicted_tick_change > 2:  # Short position, positive tick change
                        opposite_signal = True

                if opposite_signal:
                    # Calculate PnL for opposite signal exit
                    if position == 1:  # Long
                        pnl = (current_price - entry_price) * tick_value
                    else:  # Short
                        pnl = (entry_price - current_price) * tick_value
                    capital += pnl
                    trades.append(pnl)
                    position = 0
            
            # If flat, look for a new signal
            if position == 0:
                # Take action based on model type and prediction
                if self.model_type == 'Time-Series Transformer':
                    pred_price_scaled = prediction.cpu().numpy().flatten()[0]
                    dummy_array = np.zeros((1, len(model_features)))
                    dummy_array[0, 0] = pred_price_scaled
                    predicted_price = self.scaler.inverse_transform(dummy_array)[0, 0]

                    if predicted_price > current_price + (2 * tick_size):
                        position = 1
                        entry_price = current_price
                    elif predicted_price < current_price - (2 * tick_size):
                        position = -1
                        entry_price = current_price

                elif self.model_type == 'XGBoostClassifier':
                    action = int(prediction[0])
                    if action == 4 or action == 3:  # Strong Buy
                        position = 1
                        entry_price = current_price
                    elif action == 0 or action == 1:  # Strong Sell
                        position = -1
                        entry_price = current_price

                elif self.model_type == 'Neural Network (Regression)':
                    predicted_tick_change = prediction.cpu().numpy().flatten()[0]
                    if predicted_tick_change > 2:
                        position = 1
                        entry_price = current_price
                    elif predicted_tick_change < -2:
                        position = -1
                        entry_price = current_price

            equity_history.append(capital)

        # 3. Calculate and print performance metrics
        self.calculate_metrics(trades, initial_capital, np.array(equity_history))

    def run_with_results(self, initial_capital=50000, take_profit_pips=50, stop_loss_pips=25, tick_size=0.25, tick_value=5):
        """Executes the backtest and returns results instead of printing them."""
        print("\n--- Starting Backtest ---")
        print(f"Loading CSV from: {self.data_path}")

        # 1. Load and prepare data
        try:
            df = pd.read_csv(self.data_path)
            df.columns = df.columns.str.lower()
            print(f"CSV loaded successfully. Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
        except Exception as e:
            print(f"Error loading CSV: {e}")
            raise Exception(f"Failed to load CSV file: {e}")
        
        # Gracefully handle different config structures for features
        model_features = None
        if 'data_params' in self.config['Config'] and 'features' in self.config['Config']['data_params']:
            # Path for Time-Series Transformer config
            model_features = self.config['Config']['data_params']['features']
        elif 'features' in self.config['Config']:
            # Path for XGBoost and NN configs
            model_features = self.config['Config']['features']
        else:
            raise KeyError("Could not find the 'features' list in the configuration YAML.")

        # Verify features
        if not all(feature in df.columns for feature in model_features):
            missing_features = [f for f in model_features if f not in df.columns]
            raise ValueError(f"Dataframe is missing required features: {missing_features}")

        features_df = df[model_features].copy()

        # 2. Backtest loop
        capital = initial_capital
        position = 0  # 1 for long, -1 for short, 0 for flat
        entry_price = 0
        trades = []
        equity_history = [initial_capital]

        # Determine sequence length for sequence-based models
        seq_length = 1
        if self.model_type == 'Time-Series Transformer':
            seq_length = self.config['Config']['data_params']['sequence_length']

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
            
            # Prepare input data for prediction
            if self.model_type == 'Time-Series Transformer':
                input_data = features_df.iloc[i-seq_length:i].values
                scaled_input = self.scaler.transform(input_data)
            else:  # XGBoost and NN
                input_data = features_df.iloc[[i]].values
                scaled_input = self.scaler.transform(input_data)

            # Generate prediction based on model type
            prediction = None
            if self.model_type == 'XGBoostClassifier':
                prediction = self.model.predict(scaled_input)
            
            elif isinstance(self.model, nn.Module):
                input_tensor = torch.FloatTensor(scaled_input).to(self.device)
                if self.model_type == 'Time-Series Transformer':
                     input_tensor = input_tensor.unsqueeze(0)
                     
                with torch.no_grad():
                    prediction = self.model(input_tensor)

            # Check for opposite signals if still in position
            if position != 0:
                opposite_signal = False
                if self.model_type == 'Time-Series Transformer':
                    pred_price_scaled = prediction.cpu().numpy().flatten()[0]
                    dummy_array = np.zeros((1, len(model_features)))
                    dummy_array[0, 0] = pred_price_scaled
                    predicted_price = self.scaler.inverse_transform(dummy_array)[0, 0]

                    if position == 1 and predicted_price < current_price - (2 * tick_size):  # Long position, sell signal
                        opposite_signal = True
                    elif position == -1 and predicted_price > current_price + (2 * tick_size):  # Short position, buy signal
                        opposite_signal = True

                elif self.model_type == 'XGBoostClassifier':
                    action = int(prediction[0])
                    if position == 1 and action == 0:  # Long position, Strong Sell signal
                        opposite_signal = True
                    elif position == -1 and action == 4:  # Short position, Strong Buy signal
                        opposite_signal = True

                elif self.model_type == 'Neural Network (Regression)':
                    predicted_tick_change = prediction.cpu().numpy().flatten()[0]
                    if position == 1 and predicted_tick_change < -2:  # Long position, negative tick change
                        opposite_signal = True
                    elif position == -1 and predicted_tick_change > 2:  # Short position, positive tick change
                        opposite_signal = True

                if opposite_signal:
                    # Calculate PnL for opposite signal exit
                    if position == 1:  # Long
                        pnl = (current_price - entry_price) * tick_value
                    else:  # Short
                        pnl = (entry_price - current_price) * tick_value
                    capital += pnl
                    trades.append(pnl)
                    position = 0
            
            # If flat, look for a new signal
            if position == 0:
                # Take action based on model type and prediction
                if self.model_type == 'Time-Series Transformer':
                    pred_price_scaled = prediction.cpu().numpy().flatten()[0]
                    dummy_array = np.zeros((1, len(model_features)))
                    dummy_array[0, 0] = pred_price_scaled
                    predicted_price = self.scaler.inverse_transform(dummy_array)[0, 0]

                    if predicted_price > current_price + (2 * tick_size):
                        position = 1
                        entry_price = current_price
                    elif predicted_price < current_price - (2 * tick_size):
                        position = -1
                        entry_price = current_price

                elif self.model_type == 'XGBoostClassifier':
                    action = int(prediction[0])
                    if action == 4 or action == 3:  # Strong Buy
                        position = 1
                        entry_price = current_price
                    elif action == 0 or action == 1:  # Strong Sell
                        position = -1
                        entry_price = current_price

                elif self.model_type == 'Neural Network (Regression)':
                    predicted_tick_change = prediction.cpu().numpy().flatten()[0]
                    if predicted_tick_change > 2:
                        position = 1
                        entry_price = current_price
                    elif predicted_tick_change < -2:
                        position = -1
                        entry_price = current_price

            equity_history.append(capital)

        # 3. Calculate metrics and return results
        metrics = self.calculate_metrics_with_results(trades, initial_capital, np.array(equity_history))
        
        return {
            'trades': trades,
            'equity_history': equity_history,
            'metrics': metrics
        }

    def calculate_metrics(self, trades, initial_capital, equity_curve):
        """Calculates and prints performance metrics."""
        print("\n--- Backtest Results ---")
        if not trades:
            print("No trades were made.")
            return

        trades = np.array(trades)
        net_profit = np.sum(trades)
        gross_profit = np.sum(trades[trades > 0])
        gross_loss = np.abs(np.sum(trades[trades < 0]))
        
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf
        win_rate = np.sum(trades > 0) / len(trades) * 100 if len(trades) > 0 else 0
        
        avg_win = np.mean(trades[trades > 0]) if np.sum(trades > 0) > 0 else 0
        avg_loss = np.abs(np.mean(trades[trades < 0])) if np.sum(trades < 0) > 0 else 0

        # Calculate max drawdown from the equity curve
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_drawdown = np.max(drawdown) * 100

        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Net Profit: ${net_profit:,.2f}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Average Win: ${avg_win:,.2f}")
        print(f"Average Loss: ${avg_loss:,.2f}")
        print(f"Total Trades: {len(trades)}")

    def calculate_metrics_with_results(self, trades, initial_capital, equity_curve):
        """Calculates performance metrics and returns them as a dictionary."""
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

        # Calculate max drawdown from the equity curve
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
    # --- Select the model to backtest by providing the path to its YAML config ---
    
    # Example for Transformer model
    #config_file_path = './Models/test/test_config.yaml' # Assuming you have a 'test' folder
    
    # Example for XGBoost model
    config_file_path = './Models/XGBoost_test/test_config.yaml'
    
    # Example for Neural Network model
    #config_file_path = './Models/NN_test/test_config.yaml'

    # Path to the data file
    csv_data_path = 'sample.csv' # Make sure this CSV has all the required columns

    if not os.path.exists(config_file_path):
        print(f"Error: Configuration file not found at '{config_file_path}'")
    elif not os.path.exists(csv_data_path):
        print(f"Error: Data file not found at '{csv_data_path}'")
    else:
        backtester = Backtester(config_path=config_file_path, data_path=csv_data_path)
        backtester.load_artifacts()
        # For NQ futures, tick_value=5. For ES futures, tick_value=12.5
        backtester.run(tick_value=5)