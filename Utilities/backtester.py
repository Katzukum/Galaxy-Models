import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from Utilities.yaml_utils import YAMLConfig, load_yaml_config

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

    def _get_timestamp(self, df, index):
        """Get timestamp from dataframe, handling different column formats"""
        if 'datetime' in df.columns:
            return str(df.loc[index, 'datetime'])
        elif 'date' in df.columns and 'time' in df.columns:
            date_str = str(df.loc[index, 'date'])
            time_str = str(df.loc[index, 'time'])
            return f"{date_str} {time_str}"
        else:
            return str(index)

    def load_artifacts(self):
        """Loads model, scaler, and config from the YAML file."""
        print("--- Loading Artifacts ---")
        self.config = load_yaml_config(self.config_path)
        
        # Use recursive key finding to get model type
        self.model_type = self.config.find_key('Type')
        print(f"Model Type: {self.model_type}")

        # Use recursive key finding to get artifact paths and config
        artifact_paths = self.config.find_key('artifact_paths')
        original_config = self.config.find_key('Config')

        # Handle ensemble models differently
        if self.model_type == 'Ensemble Model':
            self._load_ensemble_artifacts(artifact_paths, original_config)
        else:
            # Load scaler for individual models
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

    def _load_ensemble_artifacts(self, artifact_paths, original_config):
        """Load artifacts for ensemble models."""
        from NetworkConfigs.Ensemble_loader import EnsembleModelLoader
        
        # Load the ensemble model using the EnsembleModelLoader
        self.ensemble_loader = EnsembleModelLoader(self.model_dir)
        
        # For ensemble models, we'll use the first component model's scaler
        # as a reference for data preprocessing
        first_model = self.ensemble_loader.selected_models[0]
        first_model_dir = os.path.dirname(first_model['configPath'])
        
        # Load the first model's scaler as reference
        first_model_config = load_yaml_config(first_model['configPath'])
        first_model_artifact_paths = first_model_config.find_key('artifact_paths')
        scaler_path = os.path.join(first_model_dir, first_model_artifact_paths['scaler'])
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print("Ensemble scaler (from first model) loaded.")
        
        # Set model to None since we'll use ensemble_loader for predictions
        self.model = None
        print("Ensemble model loaded successfully.")
        
        # Initialize ensemble prediction method
        self._ensemble_prediction_count = 0

    def run(self, initial_capital=50000, take_profit_pips=50, stop_loss_pips=25, tick_size=0.25, tick_value=5):
        """Executes the backtest."""
        print("\n--- Starting Backtest ---")

        # 1. Load and prepare data
        df = pd.read_csv(self.data_path)
        
        # Convert column names to lowercase to match model expectations
        df.columns = df.columns.str.lower()
        
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
            print(f"Available columns: {list(df.columns)}")
            print(f"Required features: {model_features}")
            print(f"Missing features: {missing_features}")
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
        elif self.model_type == 'PPO Agent':
            seq_length = self.config['Config']['model_params'].get('lookback_window', 60)
        elif self.model_type == 'Ensemble Model':
            # For ensemble models, use the maximum sequence length of component models
            seq_length = 60  # Transformer needs 60, others need 1

        # For ensemble models, skip the first 60 rows to allow history buffer to build up
        start_index = 60 if self.model_type == 'Ensemble Model' else 0

        for i in range(max(seq_length, start_index), len(features_df)):
            current_price = df.loc[i, 'close']

            # Check for take-profit or stop-loss
            if position != 0:
                pnl = 0
                trade_closed = False
                if position == 1:  # Long
                    if current_price >= entry_price + (take_profit_pips * tick_size):
                        price_difference = current_price - entry_price
                        number_of_ticks = price_difference / tick_size
                        pnl = number_of_ticks * tick_value
                        trade_closed = True
                    elif current_price <= entry_price - (stop_loss_pips * tick_size):
                        price_difference = current_price - entry_price
                        number_of_ticks = price_difference / tick_size
                        pnl = number_of_ticks * tick_value
                        trade_closed = True
                elif position == -1:  # Short
                    if current_price <= entry_price - (take_profit_pips * tick_size):
                        price_difference = entry_price - current_price
                        number_of_ticks = price_difference / tick_size
                        pnl = number_of_ticks * tick_value
                        trade_closed = True
                    elif current_price >= entry_price + (stop_loss_pips * tick_size):
                        price_difference = entry_price - current_price
                        number_of_ticks = price_difference / tick_size
                        pnl = number_of_ticks * tick_value
                        trade_closed = True
                
                if trade_closed:
                    capital += pnl
                    trades.append(pnl)
                    position = 0
            
            # Prepare input data for prediction (needed for both position checking and new signals)
            if self.model_type == 'Ensemble Model':
                # For ensemble models, we need to prepare data for each component model
                input_data = features_df.iloc[[i]].values  # Use single row for ensemble
                scaled_input = self.scaler.transform(input_data)
            elif self.model_type == 'Time-Series Transformer':
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
            if self.model_type == 'Ensemble Model':
                # Convert numpy array to feature dictionary for ensemble prediction
                feature_dict = dict(zip(model_features, scaled_input[0]))
                try:
                    prediction = self.ensemble_loader.predict(feature_dict)
                    if i < 5:  # Debug first few predictions
                        print(f"Debug - Row {i}: prediction = {prediction}, current_price = {current_price}")
                except Exception as e:
                    if i < 5:  # Debug first few predictions
                        print(f"Debug - Row {i}: prediction failed - {e}")
                    # For ensemble models, we might not have enough data for all models initially
                    # Set prediction to 0 and continue with the loop
                    prediction = 0.0
            elif self.model_type == 'XGBoostClassifier':
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
                if self.model_type == 'Ensemble Model':
                    # For ensemble models, treat prediction as a delta (price change)
                    predicted_delta = prediction
                    if position == 1 and predicted_delta < -(2 * tick_size):  # Long position, sell signal
                        opposite_signal = True
                    elif position == -1 and predicted_delta > (2 * tick_size):  # Short position, buy signal
                        opposite_signal = True
                elif self.model_type == 'Time-Series Transformer':
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
                    elif position == -1 and action == 2:  # Short position, Strong Buy signal
                        opposite_signal = True

                elif self.model_type == 'Neural Network (Regression)':
                    predicted_tick_change = prediction.cpu().numpy().flatten()[0]
                    if position == 1 and predicted_tick_change < -2:  # Long position, negative tick change
                        opposite_signal = True
                    elif position == -1 and predicted_tick_change > 2:  # Short position, positive tick change
                        opposite_signal = True

                elif self.model_type == 'PPO Agent':
                    action = prediction  # prediction is already the action (0: hold, 1: buy, 2: sell)
                    if position == 1 and action == 2:  # Long position, Sell signal
                        opposite_signal = True
                    elif position == -1 and action == 1:  # Short position, Buy signal
                        opposite_signal = True

                if opposite_signal:
                    # Calculate PnL for opposite signal exit
                    if position == 1:  # Long
                        price_difference = current_price - entry_price
                        number_of_ticks = price_difference / tick_size
                        pnl = number_of_ticks * tick_value
                    else:  # Short
                        price_difference = entry_price - current_price
                        number_of_ticks = price_difference / tick_size
                        pnl = number_of_ticks * tick_value
                    capital += pnl
                    trades.append(pnl)
                    position = 0
            
            # If flat, look for a new signal
            if position == 0:
                # Take action based on model type and prediction
                if self.model_type == 'Ensemble Model':
                    # For ensemble models, treat prediction as a delta (price change)
                    predicted_delta = prediction
                    predicted_price = current_price + predicted_delta
                    price_diff = predicted_delta
                    threshold = 1 * tick_size  # Much more sensitive threshold for delta predictions
                    if i < 5:  # Debug first few predictions
                        print(f"Debug - Signal check: predicted_delta={predicted_delta:.6f}, current_price={current_price}, predicted_price={predicted_price:.2f}, threshold={threshold}")
                    if predicted_delta > threshold:
                        position = 1
                        entry_price = current_price
                        print(f"Debug - BUY signal triggered at row {i}")
                    elif predicted_delta < -threshold:
                        position = -1
                        entry_price = current_price
                        print(f"Debug - SELL signal triggered at row {i}")
                elif self.model_type == 'Time-Series Transformer':
                    pred_price_scaled = prediction.cpu().numpy().flatten()[0]
                    dummy_array = np.zeros((1, len(model_features)))
                    dummy_array[0, 0] = pred_price_scaled
                    predicted_price = self.scaler.inverse_transform(dummy_array)[0, 0]

                    if predicted_price > current_price + (20 * tick_size):
                        position = 1
                        entry_price = current_price
                    elif predicted_price < current_price - (20 * tick_size):
                        position = -1
                        entry_price = current_price

                elif self.model_type == 'XGBoostClassifier':
                    action = int(prediction[0])
                    if action == 2:  # Strong Buy
                        position = 1
                        entry_price = current_price
                    elif action == 0:  # Strong Sell
                        position = -1
                        entry_price = current_price

                elif self.model_type == 'Neural Network (Regression)':
                    predicted_tick_change = prediction.cpu().numpy().flatten()[0]
                    if predicted_tick_change > 20:
                        position = 1
                        entry_price = current_price
                    elif predicted_tick_change < -20:
                        position = -1
                        entry_price = current_price

            equity_history.append(capital)

        # 3. Calculate and print performance metrics
        self.calculate_metrics(trades, initial_capital, np.array(equity_history))

    def run_with_results(self, initial_capital=50000, take_profit_pips=50, stop_loss_pips=25, tick_size=0.25, tick_value=5):
        """Executes the backtest and returns results instead of printing them."""
        print("\n--- Starting Unified Backtest ---")
        print(f"Loading CSV from: {self.data_path}")

        # 1. Load and prepare data
        try:
            df = pd.read_csv(self.data_path)
            df.columns = df.columns.str.lower()
        except Exception as e:
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
            print(f"Available columns: {list(df.columns)}")
            print(f"Required features: {model_features}")
            print(f"Missing features: {missing_features}")
            raise ValueError(f"Dataframe is missing required features: {missing_features}")

        features_df = df[model_features].copy()

        # 2. Backtest loop setup
        capital = initial_capital
        position = 0  # 1 for long, -1 for short, 0 for flat
        entry_price = 0
        trades = [] # This will now store dictionaries with full trade details
        equity_history = [initial_capital]
        
        # Determine sequence length for sequence-based models
        seq_length = 1
        if self.model_type == 'Time-Series Transformer':
            seq_length = self.config['Config']['data_params']['sequence_length']
        elif self.model_type == 'PPO Agent':
            seq_length = self.config['Config']['model_params'].get('lookback_window', 60)
        elif self.model_type == 'Ensemble Model':
            # For ensemble models, use the maximum sequence length of component models
            seq_length = 60  # Transformer needs 60, others need 1

        # For ensemble models, skip the first 60 rows to allow history buffer to build up
        start_index = 60 if self.model_type == 'Ensemble Model' else 0
        
        # Debug statistics
        price_diffs = []
        buy_signals = 0
        sell_signals = 0
        long_trades = 0
        short_trades = 0
        total_bars_analyzed = 0

        for i in range(max(seq_length, start_index), len(features_df)):
            current_price = df.loc[i, 'close']
            trade_closed_this_bar = False
            total_bars_analyzed += 1

            # --- Part 1: Check for an EXIT on the current bar ---
            if position != 0:
                pnl = 0
                exit_reason = None

                # Condition 1: Take-Profit or Stop-Loss hit
                if position == 1: # Long Position
                    if current_price >= entry_price + (take_profit_pips * tick_size):
                        exit_reason = "Take Profit"
                    elif current_price <= entry_price - (stop_loss_pips * tick_size):
                        exit_reason = "Stop Loss"
                elif position == -1: # Short Position
                    if current_price <= entry_price - (take_profit_pips * tick_size):
                        exit_reason = "Take Profit"
                    elif current_price >= entry_price + (stop_loss_pips * tick_size):
                        exit_reason = "Stop Loss"
                
                # If TP/SL not hit, check for model-based exit signal (but only if we are still in a position)
                if not exit_reason:
                    # Prepare input data for prediction
                    prediction_action = 0 # 0=hold, 1=buy, 2=sell
                    
                    # --- PPO Agent Prediction ---
                    if self.model_type == 'PPO Agent':
                        input_data = features_df.iloc[i-seq_length:i].values
                        scaled_input = self.scaler.transform(input_data)
                        account_state = np.array([capital / initial_capital, position, 0.0])
                        account_state_broadcast = np.tile(account_state, (scaled_input.shape[0], 1))
                        input_with_state = np.concatenate([scaled_input, account_state_broadcast], axis=1)
                        input_tensor = torch.FloatTensor(input_with_state).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            action_logits, _ = self.model(input_tensor)
                            prediction_action = torch.argmax(torch.softmax(action_logits, dim=-1)).item()
                    
                    # --- Other Model Types Prediction ---
                    else:
                        # Prepare input data for prediction
                        if self.model_type == 'Ensemble Model':
                            input_data = features_df.iloc[[i]].values
                            scaled_input = self.scaler.transform(input_data)
                        elif self.model_type == 'Time-Series Transformer':
                            input_data = features_df.iloc[i-seq_length:i].values
                            scaled_input = self.scaler.transform(input_data)
                        else:  # XGBoost and NN
                            input_data = features_df.iloc[[i]].values
                            scaled_input = self.scaler.transform(input_data)

                        # Generate prediction based on model type
                        prediction = None
                        if self.model_type == 'Ensemble Model':
                            # Convert numpy array to feature dictionary for ensemble prediction
                            feature_dict = dict(zip(model_features, scaled_input[0]))
                            try:
                                prediction = self.ensemble_loader.predict(feature_dict)
                            except Exception as e:
                                # Skip this prediction if not enough data
                                prediction = 0.0
                        elif self.model_type == 'XGBoostClassifier':
                            prediction = self.model.predict(scaled_input)
                        
                        elif isinstance(self.model, nn.Module):
                            input_tensor = torch.FloatTensor(scaled_input).to(self.device)
                            if self.model_type == 'Time-Series Transformer':
                                 input_tensor = input_tensor.unsqueeze(0)
                                 
                            with torch.no_grad():
                                prediction = self.model(input_tensor)

                        # Convert prediction to action based on model type
                        if self.model_type == 'Ensemble Model':
                            # For ensemble models, treat prediction as a delta (price change)
                            predicted_delta = prediction
                            if position == 1 and predicted_delta < -(1 * tick_size):  # Long position, sell signal
                                prediction_action = 2
                            elif position == -1 and predicted_delta > (1 * tick_size):  # Short position, buy signal
                                prediction_action = 1
                        elif self.model_type == 'Time-Series Transformer':
                            pred_price_scaled = prediction.cpu().numpy().flatten()[0]
                            dummy_array = np.zeros((1, len(model_features)))
                            dummy_array[0, 0] = pred_price_scaled
                            predicted_price = self.scaler.inverse_transform(dummy_array)[0, 0]

                            if position == 1 and predicted_price < current_price - (2 * tick_size):  # Long position, sell signal
                                prediction_action = 2
                            elif position == -1 and predicted_price > current_price + (2 * tick_size):  # Short position, buy signal
                                prediction_action = 1

                        elif self.model_type == 'XGBoostClassifier':
                            action = int(prediction[0])
                            if position == 1 and action == 0:  # Long position, Strong Sell signal
                                prediction_action = 2
                            elif position == -1 and action == 2:  # Short position, Strong Buy signal
                                prediction_action = 1

                        elif self.model_type == 'Neural Network (Regression)':
                            predicted_tick_change = prediction.cpu().numpy().flatten()[0]
                            if position == 1 and predicted_tick_change < -2:  # Long position, negative tick change
                                prediction_action = 2
                            elif position == -1 and predicted_tick_change > 2:  # Short position, positive tick change
                                prediction_action = 1
                    
                    # Check for opposite signal
                    if (position == 1 and prediction_action == 2): # Long position, but model says Sell
                        exit_reason = "Opposite Signal"
                    elif (position == -1 and prediction_action == 1): # Short position, but model says Buy
                        exit_reason = "Opposite Signal"

                # If any exit condition was met, close the trade
                if exit_reason:
                    if position == 1: # Closing a long
                        price_difference = current_price - entry_price
                        pnl = (price_difference / tick_size) * tick_value
                    else: # Closing a short
                        price_difference = entry_price - current_price
                        pnl = (price_difference / tick_size) * tick_value
                    
                    capital += pnl
                    trade_type = 'LONG' if position == 1 else 'SHORT'
                    trades.append({
                        'pnl': pnl,
                        'type': trade_type,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'exit_reason': exit_reason,
                        'timestamp': self._get_timestamp(df, i) # Use a real timestamp if available
                    })
                    
                    # Track trade direction
                    if trade_type == 'LONG':
                        long_trades += 1
                    else:
                        short_trades += 1
                    
                    position = 0
                    entry_price = 0
                    trade_closed_this_bar = True

            # --- Part 2: Check for a new ENTRY if we are flat ---
            if position == 0:
                # Get model prediction. If a trade was just closed, we can reuse the prediction.
                # Otherwise, we need to generate a new one.
                # For simplicity, we'll just predict again. The performance impact is negligible.
                prediction_action = 0 # 0=hold, 1=buy, 2=sell

                # --- PPO Agent Prediction ---
                if self.model_type == 'PPO Agent':
                    input_data = features_df.iloc[i-seq_length:i].values
                    scaled_input = self.scaler.transform(input_data)
                    account_state = np.array([capital / initial_capital, position, 0.0])
                    account_state_broadcast = np.tile(account_state, (scaled_input.shape[0], 1))
                    input_with_state = np.concatenate([scaled_input, account_state_broadcast], axis=1)
                    input_tensor = torch.FloatTensor(input_with_state).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        action_logits, _ = self.model(input_tensor)
                        prediction_action = torch.argmax(torch.softmax(action_logits, dim=-1)).item()
                    
                    # Track signals for PPO
                    if prediction_action == 1:  # Buy
                        buy_signals += 1
                    elif prediction_action == 2:  # Sell
                        sell_signals += 1

                # --- Other Model Types Prediction ---
                else:
                    # Prepare input data for prediction
                    if self.model_type == 'Ensemble Model':
                        input_data = features_df.iloc[[i]].values
                        scaled_input = self.scaler.transform(input_data)
                    elif self.model_type == 'Time-Series Transformer':
                        input_data = features_df.iloc[i-seq_length:i].values
                        scaled_input = self.scaler.transform(input_data)
                    elif self.model_type == 'PPO Agent':
                        # Already handled above
                        pass
                    else:  # XGBoost and NN
                        input_data = features_df.iloc[[i]].values
                        scaled_input = self.scaler.transform(input_data)

                    # Generate prediction based on model type
                    prediction = None
                    if self.model_type == 'Ensemble Model':
                        # Convert numpy array to feature dictionary for ensemble prediction
                        feature_dict = dict(zip(model_features, scaled_input[0]))
                        try:
                            prediction = self.ensemble_loader.predict(feature_dict)
                        except Exception as e:
                            # Skip this prediction if not enough data
                            prediction = 0.0
                    elif self.model_type == 'XGBoostClassifier':
                        prediction = self.model.predict(scaled_input)
                    
                    elif isinstance(self.model, nn.Module):
                        input_tensor = torch.FloatTensor(scaled_input).to(self.device)
                        if self.model_type == 'Time-Series Transformer':
                             input_tensor = input_tensor.unsqueeze(0)
                             
                        with torch.no_grad():
                            prediction = self.model(input_tensor)

                    # Convert prediction to action based on model type
                    if self.model_type == 'Ensemble Model':
                        # For ensemble models, treat prediction as a delta (price change)
                        predicted_delta = prediction
                        predicted_price = current_price + predicted_delta
                        # Calculate price difference and threshold
                        price_diff = predicted_delta
                        threshold = 1 * tick_size  # Much more sensitive threshold for delta predictions
                        price_diffs.append(price_diff)
                    elif self.model_type == 'Time-Series Transformer':
                        pred_price_scaled = prediction.cpu().numpy().flatten()[0]
                        dummy_array = np.zeros((1, len(model_features)))
                        dummy_array[0, 0] = pred_price_scaled
                        predicted_price = self.scaler.inverse_transform(dummy_array)[0, 0]

                        # Calculate price difference and threshold
                        price_diff = predicted_price - current_price
                        threshold = 5 * tick_size  # Reduced from 20 to 5 for more sensitive signals
                        price_diffs.append(price_diff)

                        if predicted_price > current_price + threshold:
                            prediction_action = 1  # Buy
                            buy_signals += 1
                        elif predicted_price < current_price - threshold:
                            prediction_action = 2  # Sell
                            sell_signals += 1

                    elif self.model_type == 'XGBoostClassifier':
                        action = int(prediction[0])
                        if action == 2:  # Strong Buy
                            prediction_action = 1
                            buy_signals += 1
                        elif action == 0:  # Strong Sell
                            prediction_action = 2
                            sell_signals += 1

                    elif self.model_type == 'Neural Network (Regression)':
                        predicted_tick_change = prediction.cpu().numpy().flatten()[0]
                        if predicted_tick_change > 20:
                            prediction_action = 1  # Buy
                            buy_signals += 1
                        elif predicted_tick_change < -20:
                            prediction_action = 2  # Sell
                            sell_signals += 1

                if prediction_action == 1: # Buy signal
                    position = 1
                    entry_price = current_price
                elif prediction_action == 2: # Sell signal
                    position = -1
                    entry_price = current_price
            
            # --- Part 3: Update equity curve at the end of each bar ---
            equity_history.append(capital)

        # 3. Calculate final metrics and return all results
        metrics = self.calculate_metrics_with_results([t['pnl'] for t in trades], initial_capital, np.array(equity_history))
        
        # Signal and Trade Summary
        print(f"\n=== Model Signal Analysis ===")
        print(f"📊 Total bars analyzed: {total_bars_analyzed:,}")
        print(f"🟢 Bullish signals: {buy_signals:,} ({buy_signals/total_bars_analyzed*100:.1f}%)" if total_bars_analyzed > 0 else "🟢 Bullish signals: 0")
        print(f"🔴 Bearish signals: {sell_signals:,} ({sell_signals/total_bars_analyzed*100:.1f}%)" if total_bars_analyzed > 0 else "🔴 Bearish signals: 0")
        print(f"⚪ No signal: {total_bars_analyzed - buy_signals - sell_signals:,} ({(total_bars_analyzed - buy_signals - sell_signals)/total_bars_analyzed*100:.1f}%)" if total_bars_analyzed > 0 else "⚪ No signal: 0")
        print(f"🎯 Signal threshold: ±{5 * tick_size:.2f} points")
        
        print(f"\n=== Trade Execution Summary ===")
        print(f"📈 Total trades executed: {len(trades):,}")
        print(f"🟢 LONG trades: {long_trades:,} ({long_trades/len(trades)*100:.1f}%)" if trades else "🟢 LONG trades: 0")
        print(f"🔴 SHORT trades: {short_trades:,} ({short_trades/len(trades)*100:.1f}%)" if trades else "🔴 SHORT trades: 0")
        
        if trades and total_bars_analyzed > 0:
            print(f"📊 Signal-to-Trade ratio: {(buy_signals + sell_signals)/len(trades):.1f}:1")
        
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
    config_file_path = './Models/XGBoost_30Sec_NQ_SuperCCI/30Sec_NQ_SuperCCI_config.yaml'
    
    # Example for Neural Network model
    #config_file_path = './Models/NN_test/test_config.yaml'

    # Path to the data file
    csv_data_path = 'uploads/16ebb85c-3cb5-4d58-ae59-9b8e709f4fb7_NQ_30sec_SuperCCI_Val.csv' # Make sure this CSV has all the required columns

    if not os.path.exists(config_file_path):
        print(f"Error: Configuration file not found at '{config_file_path}'")
    elif not os.path.exists(csv_data_path):
        print(f"Error: Data file not found at '{csv_data_path}'")
    else:
        backtester = Backtester(config_path=config_file_path, data_path=csv_data_path)
        backtester.load_artifacts()
        # For NQ futures, tick_value=5. For ES futures, tick_value=12.5
        backtester.run(tick_value=5)