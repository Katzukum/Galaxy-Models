# File: neural_network_loader.py

import os
import yaml
import pickle
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List
from pydantic import BaseModel

class NNPredictionResponse(BaseModel):
    """Response model for Neural Network predictions"""
    model_config = {"protected_namespaces": ()}
    model_name: str
    model_type: str = "Neural Network (Regression)"
    predicted_price_change_ticks: float

class NNModelLoader:
    """
    Loads and serves a neural network model trained by the NNTrainer class.
    
    This class encapsulates all the logic required to load the artifacts 
    (config, scaler, model weights) and perform predictions.
    """

    def __init__(self, model_dir: str):
        """
        Initializes the loader by loading all necessary model artifacts.

        Args:
            model_dir (str): The path to the directory containing the model files 
                             (_config.yaml, _scaler.pkl, _model.pt).
        """
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        print(f"Initializing model from directory: {model_dir}")
        
        # --- 1. Load Configuration from YAML ---
        config_path = self._find_file_by_extension(model_dir, '.yaml')
        if not config_path:
            raise FileNotFoundError(f"Could not find a .yaml config file in {model_dir}")

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model_name: str = self.config['model_name']
        self.features: List[str] = self.config['Config']['features']
        self.architecture: List[Dict] = self.config['Config']['architecture']
        
        artifact_paths = self.config['artifact_paths']
        scaler_path = os.path.join(model_dir, artifact_paths['scaler'])
        model_path = os.path.join(model_dir, artifact_paths['model'])

        print(f"Successfully loaded configuration for model '{self.model_name}'.")
        print(f"Expecting {len(self.features)} features.")

        # --- 2. Load the Scaler ---
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"Scaler loaded from: '{scaler_path}'")

        # --- 3. Build and Load the PyTorch Model ---
        self.model = self._build_model_from_config(self.architecture)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()  # Set model to evaluation mode
        print(f"PyTorch model state loaded from: '{model_path}'")
        print("Model loader is ready.")

    @staticmethod
    def _find_file_by_extension(directory: str, extension: str) -> str:
        """Finds the first file with a given extension in a directory."""
        for filename in os.listdir(directory):
            if filename.endswith(extension):
                return os.path.join(directory, filename)
        return ""
        
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
        
        return nn.Sequential(*layers)

    def predict(self, feature_dict: Dict[str, float]) -> float:
        """
        Performs a prediction for a single sample.

        Args:
            feature_dict (Dict[str, float]): A dictionary where keys are feature 
                                             names and values are the feature values.

        Returns:
            float: The regression model's prediction.
        """
        # --- 1. Validate and Order the Input Features ---
        if sorted(feature_dict.keys()) != sorted(self.features):
            raise ValueError(
                "Input features do not match the features the model was trained on."
                f"\nExpected: {self.features}"
                f"\nGot: {list(feature_dict.keys())}"
            )
        
        # Create a list of feature values in the correct order
        ordered_values = [feature_dict[feature] for feature in self.features]

        # --- 2. Prepare Data for PyTorch (Scale and Convert to Tensor) ---
        # Reshape to (1, n_features) for the scaler and model
        input_array = np.array(ordered_values).reshape(1, -1)
        
        # Scale the features
        scaled_array = self.scaler.transform(input_array)
        
        # Convert to a PyTorch tensor
        input_tensor = torch.FloatTensor(scaled_array)
        
        # --- 3. Perform Inference ---
        with torch.no_grad():
            prediction_tensor = self.model(input_tensor)
        
        # Extract the single float value from the output tensor
        prediction = prediction_tensor.item()
        
        return prediction

    def create_prediction_response(self, prediction: float) -> NNPredictionResponse:
        """Creates a properly formatted prediction response for the API"""
        return NNPredictionResponse(
            model_name=self.model_name,
            predicted_price_change_ticks=prediction
        )