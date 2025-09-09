import os
import yaml
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from Utilities.data_utils import prepare_delta_features

class NNTrainer:
    """
    A class to train a PyTorch neural network for regression, save its configuration, 
    the data scaler, and the model itself.

    Includes a static method to prepare data for a price-change regression task.
    """

    def __init__(self, 
                 model_name: str, 
                 config: Dict[str, Any], 
                 output_path: str = '/models'):
        """
        Initializes the NNTrainer.
        """
        if not model_name:
            raise ValueError("A 'model_name' must be provided.")
            
        self.model_name = model_name
        self.config = config
        self.output_path = output_path
        self.model = self._build_model()
        self.scaler = StandardScaler()

    @staticmethod
    def prepare_regression_data(
        data: pd.DataFrame,
        look_ahead_period: int,
        tick_size: float,
        columns_to_exclude: List[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepares data for a regression task to predict future price change in ticks.

        Args:
            data (pd.DataFrame): The input dataframe with at least a 'close' column.
            look_ahead_period (int): The number of bars to look into the future for the target.
            tick_size (float): The value of a single tick (e.g., 0.25 for NQ).
            columns_to_exclude (List[str], optional): A list of columns to exclude from features. 
                                                     Defaults to ['Date', 'Time', 'target'].

        Returns:
            Tuple[np.ndarray, np.ndarray, List[str]]: 
                - X (features), 
                - y (target), 
                - list of feature names.
        """
        print("Preparing data for regression...")
        
        if columns_to_exclude is None:
            columns_to_exclude = ['date', 'time', 'target']

        # Normalize column names to lowercase for consistent processing
        data.columns = data.columns.str.lower()
        
        # Update columns_to_exclude to lowercase as well
        columns_to_exclude = [col.lower() for col in columns_to_exclude]

        # --- 1. Create the Target Variable (Price Change in Ticks) ---
        future_close = data['close'].shift(-look_ahead_period)
        data['target'] = (future_close - data['close']) / tick_size
        
        # Drop rows with NaN values resulting from the shift
        processed_data = data.dropna().copy()

        # --- 2. Prepare X and y for the model ---
        # Only drop columns that actually exist in the dataframe
        existing_columns_to_exclude = [col for col in columns_to_exclude if col in processed_data.columns]
        X_sample = processed_data.drop(columns=existing_columns_to_exclude).values 
        feature_names = processed_data.drop(columns=existing_columns_to_exclude).columns.tolist()
        y_sample = processed_data['target'].values
        
        print(f"Data prepared with {X_sample.shape[0]} samples and {X_sample.shape[1]} features.")
        return X_sample, y_sample, feature_names

    @staticmethod
    def _build_model_from_config(architecture: list) -> nn.Module:
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
        
        model = nn.Sequential(*layers)
        print("PyTorch model has been built successfully.")
        print(model)
        return model

    def _build_model(self) -> nn.Module:
        """Builds the PyTorch model for the trainer instance."""
        return self._build_model_from_config(self.config.get('architecture', []))

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Trains the StandardScaler and the PyTorch model for regression."""
        print(f"\nStarting training for model: {self.model_name}")
        
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)

        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)

        train_params = self.config.get('train_params', {})
        learning_rate = train_params.get('learning_rate', 0.001)
        epochs = train_params.get('epochs', 10)
        
        loss_function_class = getattr(nn, train_params.get('loss', 'MSELoss'))
        optimizer_class = getattr(optim, train_params.get('optimizer', 'Adam'))
        
        criterion = loss_function_class()
        optimizer = optimizer_class(self.model.parameters(), lr=learning_rate)
        
        self.model.train()
        for epoch in range(epochs):
            outputs = self.model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
        print("Model training has been completed.")

    def save(self):
        """Saves the scaler, model state_dict, and a primary YAML configuration file."""
        os.makedirs(self.output_path, exist_ok=True)
        print(f"Output directory '{self.output_path}' is ready.")

        scaler_filename = f"{self.model_name}_scaler.pkl"
        model_filename = f"{self.model_name}_model.pt"
        config_filename = f"{self.model_name}_config.yaml"

        scaler_path = os.path.join(self.output_path, scaler_filename)
        model_path = os.path.join(self.output_path, model_filename)
        config_path = os.path.join(self.output_path, config_filename)

        with open(scaler_path, 'wb') as scaler_file:
            pickle.dump(self.scaler, scaler_file)
        print(f"Scaler has been saved to: '{scaler_path}'")

        torch.save(self.model.state_dict(), model_path)
        print(f"Trained PyTorch model state_dict has been saved to: '{model_path}'")

        final_config = {
            'model_name': self.model_name,
            'Type':'Neural Network (Regression)',
            'artifact_paths': {
                'scaler': scaler_filename,
                'model': model_filename
            },
            'Config': self.config.copy(),
        }

        with open(config_path, 'w') as yaml_file:
            yaml.dump(final_config, yaml_file, default_flow_style=False, sort_keys=False)
        print(f"Primary configuration has been saved to: '{config_path}'")
        print("\nSave process completed.")


def run_training_pipeline(model_name: str, output_dir: str, training_config: Dict, X: np.ndarray, y: np.ndarray):
    """Executes the full model training, saving, and verification process."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=False)

    trainer = NNTrainer(
        model_name=model_name,
        config=training_config,
        output_path=output_dir
    )
    trainer.train(X_train, y_train)
    trainer.save()

    print("\n" + "="*50)
    print("--- Verifying artifacts by loading from YAML config ---")
    print("="*50)

    config_yaml_path = os.path.join(output_dir, f"{model_name}_config.yaml")
    with open(config_yaml_path, 'r') as f:
        loaded_config = yaml.safe_load(f)

    print(f"Loaded config for model: {loaded_config['model_name']}")
    
    artifact_paths = loaded_config['artifact_paths']
    scaler_path = os.path.join(output_dir, artifact_paths['scaler'])
    model_path = os.path.join(output_dir, artifact_paths['model'])

    with open(scaler_path, 'rb') as f:
        loaded_scaler = pickle.load(f)

    loaded_model = NNTrainer._build_model_from_config(loaded_config['Config']['architecture'])
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()
    print("\nScaler and Model loaded successfully.")

    X_test_scaled = loaded_scaler.transform(X_test)
    X_test_tensor = torch.FloatTensor(X_test_scaled)

    with torch.no_grad():
        predictions_tensor = loaded_model(X_test_tensor)
    
    predictions = predictions_tensor.numpy().flatten()
    
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"\nMean Squared Error on test set: {mse:.4f}")
    print(f"Mean Absolute Error on test set: {mae:.4f} (ticks)")

    print(f"\nGenerated files in '{output_dir}':")
    for filename in os.listdir(output_dir):
        print(f"- {filename}")


if __name__ == '__main__':
    # --- 1. Load Data and Define Parameters ---
    data = pd.read_csv('sample.csv')
    data.columns = data.columns.str.lower()
    
    # --- 2. Prepare Data using the Class Method ---
    X_sample_raw, y_sample_raw, feature_names = NNTrainer.prepare_regression_data(
        data=data,
        look_ahead_period=5,
        tick_size=0.25,
        columns_to_exclude=['date', 'time', 'target']  # Updated to lowercase after header lowercasing
    )

    feature_df_raw = pd.DataFrame(X_sample_raw, columns=feature_names)
    feature_df_delta = prepare_delta_features(feature_df_raw)

    X_sample = feature_df_delta.values
    y_sample = y_sample_raw[1:] # Align labels
    N_FEATURES = X_sample.shape[1]

    # --- 3. Define the training configuration for regression ---
    example_config = {
        'architecture': [
            {'type': 'Linear', 'in_features': N_FEATURES, 'out_features': 64},
            {'type': 'ReLU'},
            {'type': 'Dropout', 'p': 0.3},
            {'type': 'Linear', 'in_features': 64, 'out_features': 32},
            {'type': 'ReLU'},
            {'type': 'Linear', 'in_features': 32, 'out_features': 1}
        ],
        'train_params': {
            'optimizer': 'Adam',
            'loss': 'MSELoss',
            'learning_rate': 0.001,
            'epochs': 100,
        },
        'features': feature_names
    }
    
    # --- 4. Define a model name and output path ---
    MODEL_NAME = "nq_price_change_predictor_v2"
    OUTPUT_DIR = f"./Models/NN_{MODEL_NAME}"

    # --- 5. Call the main training pipeline function ---
    run_training_pipeline(
        model_name=MODEL_NAME,
        output_dir=OUTPUT_DIR,
        training_config=example_config,
        X=X_sample,
        y=y_sample
    )