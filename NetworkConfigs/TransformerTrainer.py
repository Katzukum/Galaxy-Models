import os
import yaml
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from typing import Dict, Any, List, Tuple

# ####################################################################
# --- 1. Custom Time-Series Transformer Model Definition ---
# ####################################################################
class TimeSeriesTransformer(nn.Module):
    """
    A custom Transformer model for time-series forecasting.
    """
    def __init__(self, input_dim: int, d_model: int, nhead: int, 
                 num_encoder_layers: int, dim_feedforward: int, dropout: float = 0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.d_model = d_model
        
        # Input embedding layer to project input features to d_model
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional Encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, 5000, d_model)) # Max sequence length 5000

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
        src = self.input_embedding(src) * np.sqrt(self.d_model)
        src = src + self.pos_encoder[:, :src.size(1), :]
        
        # Pass through transformer encoder
        # output shape: [batch_size, seq_len, d_model]
        output = self.transformer_encoder(src)
        
        # Take the output of the last time step and pass to the output layer
        output = self.output_layer(output[:, -1, :])
        
        return output

# ####################################################################
# --- 2. Trainer Class for the Time-Series Transformer ---
# ####################################################################
class TransformerTrainer:
    """
    A class to train a custom Transformer model for time-series forecasting.
    It saves the model's state, data scaler, and a primary YAML config file.
    """
    def __init__(self, 
                 model_name: str, 
                 config: Dict[str, Any], 
                 output_path: str = '/models'):
        """
        Initializes the TransformerTrainer.

        Args:
            model_name (str): A unique name for the model (e.g., 'nq_futures_forecaster_v1').
            config (dict): A dictionary containing 'model_params', 'data_params',
                           and 'train_params'.
            output_path (str): The directory path to save the output files.
        """
        if not model_name:
            raise ValueError("A 'model_name' must be provided.")
        
        self.model_name = model_name
        self.config = config
        self.output_path = output_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- Build Model from Config ---
        print("Building custom TimeSeriesTransformer model...")
        self.model = TimeSeriesTransformer(**self.config.get('model_params', {}))
        self.model.to(self.device)
        print(f"Model built and moved to device: {self.device}")

        # --- Initialize Scaler ---
        # Using MinMaxScaler as it's common for financial time-series
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

    def create_sequences(self, data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Creates sequences and corresponding targets from time-series data."""
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:(i + seq_length)]
            y = data[i + seq_length, 0] # Assuming target is the first feature (e.g., 'close')
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def train(self, X_train: np.ndarray):
        """
        Trains the scaler and the Transformer model.

        Args:
            X_train (np.ndarray): Training data with shape [num_samples, num_features].
                                  The target variable must be the first column.
        """
        print(f"Starting training for model: {self.model_name}")
        
        # --- 1. Scale and Prepare Data ---
        data_params = self.config.get('data_params', {})
        seq_length = data_params.get('sequence_length', 60)

        print("Fitting scaler and transforming data...")
        scaled_data = self.scaler.fit_transform(X_train)
        
        print(f"Creating sequences with length: {seq_length}...")
        X_sequences, y_sequences = self.create_sequences(scaled_data, seq_length)

        # --- 2. Create DataLoader ---
        train_params = self.config.get('train_params', {})
        batch_size = train_params.get('batch_size', 32)
        
        dataset = TensorDataset(torch.FloatTensor(X_sequences), torch.FloatTensor(y_sequences))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # --- 3. Set up Optimizer and Loss Function ---
        epochs = train_params.get('epochs', 25)
        learning_rate = train_params.get('learning_rate', 0.0005)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # --- 4. Training Loop ---
        self.model.train() # Set model to training mode
        for epoch in range(epochs):
            epoch_loss = 0
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device).view(-1, 1)

                optimizer.zero_grad()
                
                # Forward pass
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        print("\nModel training has been completed.")

    def save(self):
        """
        Saves the model state_dict, data scaler, and a primary YAML config file
        with the specified nested structure.
        """
        os.makedirs(self.output_path, exist_ok=True)
        print(f"Output directory '{self.output_path}' is ready.")

        scaler_filename = f"{self.model_name}_scaler.pkl"
        model_filename = f"{self.model_name}_model.pt"
        config_filename = f"{self.model_name}_config.yaml"

        scaler_path = os.path.join(self.output_path, scaler_filename)
        model_path = os.path.join(self.output_path, model_filename)
        config_path = os.path.join(self.output_path, config_filename)

        # Save the scaler and model state_dict
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler has been saved to: '{scaler_path}'")
        
        torch.save(self.model.state_dict(), model_path)
        print(f"Model state_dict has been saved to: '{model_path}'")

        # --- Create and save the final configuration YAML with the specified format ---
        final_config = {
            'model_name': self.model_name,
            'Type': 'Time-Series Transformer',
            'artifact_paths': {
                'scaler': scaler_filename,
                'model_state_dict': model_filename
            },
            'Config': self.config.copy()
        }
        
        with open(config_path, 'w') as yaml_file:
            yaml.dump(final_config, yaml_file, default_flow_style=False, sort_keys=False)
        print(f"Primary configuration has been saved to: '{config_path}'")
        print("\nSave process completed.")

# ####################################################################
# --- 3. Main Pipeline Execution ---
# ####################################################################
def run_training_pipeline(model_name: str, output_dir: str, training_config: Dict, data: np.ndarray):
    """
    Executes the full model training, saving, and verification process.
    """
    # --- 1. Split data ---
    train_size = int(len(data) * 0.80)
    train_data, test_data = data[0:train_size, :], data[train_size:len(data), :]

    # --- 2. Initialize and run the trainer ---
    # The trainer ONLY sees the training data to prevent data leakage
    trainer = TransformerTrainer(
        model_name=model_name,
        config=training_config,
        output_path=output_dir
    )
    trainer.train(train_data)
    trainer.save()

    # ####################################################################
    # --- 3. Verification: Load and use the saved artifacts ---
    # ####################################################################
    print("\n" + "="*50)
    print("--- Verifying artifacts by loading from YAML config ---")
    print("="*50)

    config_yaml_path = os.path.join(output_dir, f"{model_name}_config.yaml")
    with open(config_yaml_path, 'r') as f:
        loaded_config = yaml.safe_load(f)

    print(f"Loaded config for model: {loaded_config['model_name']} (Type: {loaded_config['Type']})")

    # --- 4. Load scaler and model using the new nested structure ---
    artifact_paths = loaded_config['artifact_paths']
    original_config = loaded_config['Config'] # Access the nested config
    
    scaler_path = os.path.join(output_dir, artifact_paths['scaler'])
    model_path = os.path.join(output_dir, artifact_paths['model_state_dict'])

    with open(scaler_path, 'rb') as f:
        loaded_scaler = pickle.load(f)

    loaded_model = TimeSeriesTransformer(**original_config['model_params'])
    loaded_model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model.to(device)
    loaded_model.eval()
    print("\nScaler and Model loaded successfully.")

    # --- 5. Make predictions and evaluate ---
    seq_length = original_config['data_params']['sequence_length']
    
    # Scale test data using the *already fitted* scaler
    scaled_test_data = loaded_scaler.transform(test_data)
    X_test, y_test = trainer.create_sequences(scaled_test_data, seq_length)
    
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        predictions_scaled = loaded_model(X_test_tensor).cpu().numpy()

    # Inverse transform predictions and actuals to get meaningful error
    # Create a dummy array to inverse transform only the first column
    dummy_pred = np.zeros((len(predictions_scaled), loaded_scaler.n_features_in_))
    dummy_pred[:, 0] = predictions_scaled.flatten()
    predictions_unscaled = loaded_scaler.inverse_transform(dummy_pred)[:, 0]

    dummy_y = np.zeros((len(y_test), loaded_scaler.n_features_in_))
    dummy_y[:, 0] = y_test.flatten()
    y_test_unscaled = loaded_scaler.inverse_transform(dummy_y)[:, 0]
    
    rmse = np.sqrt(mean_squared_error(y_test_unscaled, predictions_unscaled))
    print(f"\nRoot Mean Squared Error (RMSE) on test set (unscaled): {rmse:.4f}")

    # Verify output files
    print(f"\nGenerated files in '{output_dir}':")
    for item in os.listdir(output_dir):
        print(f"- {item}")


if __name__ == '__main__':
    # --- 1. Create a sample NQ futures dataset for the example ---
    print("Generating synthetic NQ futures data...")
    # This simulates daily data: [close, Open, High, Low, Volume]
    # In a real scenario, you would load this from a CSV or API.
    num_samples = 1500
    noise = np.random.randn(num_samples) * 20
    trend = np.linspace(15000, 18000, num_samples)
    seasonality = 150 * np.sin(np.linspace(0, 4 * np.pi, num_samples))
    
    close_price = trend + seasonality + noise
    open_price = close_price - np.random.uniform(-15, 15, num_samples)
    high_price = np.maximum(close_price, open_price) + np.random.uniform(0, 25, num_samples)
    low_price = np.minimum(close_price, open_price) - np.random.uniform(0, 25, num_samples)
    volume = np.random.uniform(300000, 700000, num_samples)
    
    # IMPORTANT: The target variable ('close') must be the first column
    nq_data = np.stack([close_price, open_price, high_price, low_price, volume], axis=1)
    feature_names = ['close', 'hpen', 'high', 'low', 'volume']
    
    # --- 2. Define the training configuration ---
    example_config = {
        'model_params': {
            'input_dim': len(feature_names), # Number of input features
            'd_model': 64,          # Internal dimension of the model
            'nhead': 4,             # Number of attention heads
            'num_encoder_layers': 3,
            'dim_feedforward': 128, # Dimension of the feedforward network
            'dropout': 0.1
        },
        'data_params': {
            'sequence_length': 60, # Use 60 days of data to predict the next day
            'features': feature_names
        },
        'train_params': {
            'learning_rate': 0.0005,
            'epochs': 25,
            'batch_size': 32,
        }
    }
    
    # --- 3. Define a model name and output path ---
    MODEL_NAME = "nq_forecaster_transformer_v1"
    OUTPUT_DIR = f"./Models/{MODEL_NAME}"

    # --- 4. Call the main training pipeline function ---
    run_training_pipeline(
        model_name=MODEL_NAME,
        output_dir=OUTPUT_DIR,
        training_config=example_config,
        data=nq_data
    )