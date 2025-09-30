# File: Transformer_loader.py

import os
import yaml
import pickle
import numpy as np
import torch
import torch.nn as nn
import math
from typing import Dict, List
from collections import deque
from pydantic import BaseModel

class TransformerPredictionResponse(BaseModel):
    """Response model for Transformer predictions"""
    model_config = {"protected_namespaces": ()}
    model_name: str
    model_type: str = "Time-Series Transformer"
    forecasted_value: float
    sequence_length: int

# --- 1. Positional Encoding Module ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.permute(1, 0, 2)) # Shape: [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# --- 2. Define the Core PyTorch Model Architecture ---
# This section defines the actual Transformer model. This code should be
# identical to the model definition used during training to ensure the loaded
# state dictionary keys match perfectly.

class TimeSeriesTransformer(nn.Module):
    """
    A custom Transformer model for time-series forecasting.
    This implementation must match exactly with the training code.
    """
    def __init__(self, input_dim: int, d_model: int, nhead: int, 
                 num_encoder_layers: int, dim_feedforward: int, dropout: float = 0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.d_model = d_model
        
        # Input embedding layer to project input features to d_model
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Output layer to project back to a single prediction value
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src shape: [batch_size, seq_len, input_dim]
        
        # Embed input and add positional encoding
        src = self.input_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # Pass through transformer encoder
        # output shape: [batch_size, seq_len, d_model]
        output = self.transformer_encoder(src)
        
        # Take the output of the last time step and pass to the output layer
        output = self.output_layer(output[:, -1, :])
        
        return output


# --- 2. Create the Model Loader Class ---
# This class follows the required interface for the Api_Loader.

class TransformerModelLoader:
    """
    Loads and serves a Time-Series Transformer model.

    This loader is stateful. It maintains a history of the last `sequence_length` 
    data points to form the required input sequence for the model. A prediction
    is only made when this history buffer is full.
    """

    def __init__(self, model_dir: str):
        """
        Initializes the loader by loading all necessary model artifacts.
        """
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        print(f"Initializing Time-Series Transformer from: {model_dir}")
        
        # --- Load Configuration ---
        config_path = self._find_file_by_extension(model_dir, '.yaml')
        if not config_path:
            raise FileNotFoundError(f"No .yaml config file found in {model_dir}")

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model_name: str = self.config['model_name']
        
        # Extract data and model parameters
        data_params = self.config['Config']['data_params']
        model_params = self.config['Config']['model_params']
        
        self.features: List[str] = data_params['features']
        self.sequence_length: int = data_params['sequence_length']
        self.delta_features: List[str] = data_params.get('delta_feature_list', ['close', 'open', 'high', 'low']) # With a fallback
        
        # --- Initialize the history buffer ---
        self.history = deque(maxlen=self.sequence_length)
        print(f"Initialized history buffer with sequence length: {self.sequence_length}")

        artifact_paths = self.config['artifact_paths']
        feature_scaler_path = os.path.join(model_dir, artifact_paths['feature_scaler'])
        target_scaler_path = os.path.join(model_dir, artifact_paths['target_scaler'])
        model_path = os.path.join(model_dir, artifact_paths['model_state_dict'])

        print(f"Loaded config for model '{self.model_name}'. Expecting {len(self.features)} features.")

        # --- Load the Scalers ---
        with open(feature_scaler_path, 'rb') as f:
            self.feature_scaler = pickle.load(f)
        with open(target_scaler_path, 'rb') as f:
            self.target_scaler = pickle.load(f)
        print(f"Feature scaler loaded from: '{feature_scaler_path}'")
        print(f"Target scaler loaded from: '{target_scaler_path}'")

        # --- Build and Load the PyTorch Model ---
        # The model architecture is reconstructed using parameters from the config
        self.model = TimeSeriesTransformer(**model_params)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()  # Set model to evaluation mode
        print(f"PyTorch model state loaded from: '{model_path}'")
        print("Transformer model loader is ready.")

    @staticmethod
    def _find_file_by_extension(directory: str, extension: str) -> str:
        """Finds the first file with a given extension in a directory."""
        for filename in os.listdir(directory):
            if filename.endswith(extension):
                return os.path.join(directory, filename)
        return ""

    def _convert_to_deltas(self, data: np.ndarray) -> np.ndarray:
        """
        Converts absolute price data to deltas (price changes) to match training data format.
        This method replicates the prepare_delta_data function used during training.
        
        Args:
            data (np.ndarray): Input data with shape [sequence_length, n_features]
            
        Returns:
            np.ndarray: Delta data with the same shape, first row will be NaN values
        """
        # Calculate deltas for specified features (same logic as in training)
        delta_data = data.copy()
        
        # Get the integer indices of the columns that need differencing
        delta_indices = [self.features.index(col) for col in self.delta_features if col in self.features]

        # Calculate differences for the identified columns
        for i in delta_indices:
            delta_data[1:, i] = data[1:, i] - data[:-1, i]
        
        # The first row will contain NaN values after diff operation, but we keep it
        # to maintain the same sequence length as the input
        return delta_data

    def predict(self, feature_dict: Dict[str, float]) -> float:
        """
        Accepts a single time-step of features, adds it to the sequence history,
        and performs a prediction if the sequence is full.

        Args:
            feature_dict (Dict[str, float]): A dictionary of features for the LATEST time step.

        Returns:
            float: The regression model's prediction.
        
        Raises:
            ValueError: If the input features are incorrect or if the history buffer
                        is not yet full enough to make a prediction.
        """
        # --- 1. Validate and Order Features, then update history ---
        if sorted(feature_dict.keys()) != sorted(self.features):
            raise ValueError(
                "Input features do not match model's expected features."
                f"\nExpected: {self.features}"
                f"\nGot: {list(feature_dict.keys())}"
            )
        
        ordered_values = [feature_dict[feature] for feature in self.features]
        self.history.append(ordered_values)
        
        # --- 2. Check if the history buffer is full ---
        if len(self.history) < self.sequence_length:
            raise ValueError(
                f"Not enough historical data to predict. "
                f"Need {self.sequence_length} data points, but only have {len(self.history)}."
            )
            
        # --- 3. Prepare Data Sequence for PyTorch ---
        # Convert the full sequence history to a NumPy array
        input_sequence = np.array(list(self.history))
        
        # Convert absolute prices to deltas (price changes) to match training data
        # The scaler was fitted on delta data during training
        delta_sequence = self._convert_to_deltas(input_sequence)
        
        # Scale the delta sequence using the feature scaler fitted on delta data
        scaled_sequence = self.feature_scaler.transform(delta_sequence)
        
        # Convert to a PyTorch tensor and add the batch dimension
        # Required shape: [batch_size, seq_len, n_features] for the training model
        input_tensor = torch.FloatTensor(scaled_sequence).unsqueeze(0)  # Add batch dimension at the beginning
        
        # --- 4. Perform Inference ---
        with torch.no_grad():
            scaled_prediction_tensor = self.model(input_tensor)
        
        scaled_prediction = scaled_prediction_tensor.item() # The raw model output, e.g., -0.8635

        # --- 5. INVERSE TRANSFORM THE PREDICTED DELTA ---
        scaled_predicted_delta = scaled_prediction_tensor.item()

        # Inverse transform the delta using the dedicated target scaler
        unscaled_predicted_delta = self.target_scaler.inverse_transform(np.array([[scaled_predicted_delta]]))[0, 0]

        # --- 6. CALCULATE THE FINAL PRICE FORECAST ---
        # Retrieve the last *actual* close price from the original, unscaled input.
        # This is the last value passed into the function via feature_dict.
        last_actual_close_price = feature_dict['close']

        # The final forecast is the last actual price plus the predicted change.
        final_price_forecast = last_actual_close_price + unscaled_predicted_delta

        return final_price_forecast

    def create_prediction_response(self, prediction: float) -> TransformerPredictionResponse:
        """Creates a properly formatted prediction response for the API"""
        return TransformerPredictionResponse(
            model_name=self.model_name,
            forecasted_value=prediction,
            sequence_length=self.sequence_length
        )