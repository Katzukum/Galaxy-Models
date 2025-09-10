# File: XGBoost_loader.py

import os
import yaml
import pickle
import numpy as np
import xgboost as xgb
from typing import Dict, Any, List
from pydantic import BaseModel

class XGBoostPredictionResponse(BaseModel):
    """Response model for XGBoost predictions"""
    model_config = {"protected_namespaces": ()}
    model_name: str
    model_type: str = "XGBoost Classifier"
    predicted_action: str
    confidence: float
    probabilities: Dict[str, float]

class XGBoostModelLoader:
    """
    Loads and serves an XGBoost classification model trained by the XGBoostTrainer class.
    
    This class encapsulates all the logic required to load the artifacts 
    (config, scaler, model) and perform predictions for trading action classification.
    """

    def __init__(self, model_dir: str):
        """
        Initializes the loader by loading all necessary model artifacts.

        Args:
            model_dir (str): The path to the directory containing the model files 
                             (_config.yaml, _scaler.pkl, _model.json).
        """
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        print(f"Initializing XGBoost model from directory: {model_dir}")
        
        # --- 1. Load Configuration from YAML ---
        config_path = self._find_file_by_extension(model_dir, '.yaml')
        if not config_path:
            raise FileNotFoundError(f"Could not find a .yaml config file in {model_dir}")

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model_name: str = self.config['model_name']
        self.model_type: str = self.config['Type']
        self.features: List[str] = self.config['Config']['features']
        self.label_mapping: Dict[str, int] = self.config['Config']['label_mapping']
        self.model_params: Dict[str, Any] = self.config['Config']['model_params']
        
        # Create reverse mapping for converting predictions back to action names
        self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
        
        # Add state for delta calculation
        self.previous_feature_dict = None
        
        artifact_paths = self.config['artifact_paths']
        scaler_path = os.path.join(model_dir, artifact_paths['scaler'])
        model_path = os.path.join(model_dir, artifact_paths['model'])

        print(f"Successfully loaded configuration for model '{self.model_name}'.")
        print(f"Model type: {self.model_type}")
        print(f"Expecting {len(self.features)} features.")
        print(f"Classification labels: {list(self.label_mapping.keys())}")

        # --- 2. Load the Scaler ---
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"Scaler loaded from: '{scaler_path}'")

        # --- 3. Load the XGBoost Model ---
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        print(f"XGBoost model loaded from: '{model_path}'")
        print("XGBoost model loader is ready.")

    @staticmethod
    def _find_file_by_extension(directory: str, extension: str) -> str:
        """Finds the first file with a given extension in a directory."""
        for filename in os.listdir(directory):
            if filename.endswith(extension):
                return os.path.join(directory, filename)
        return ""

    def predict(self, feature_dict: Dict[str, float]) -> str:
        """
        Performs a prediction for a single sample and returns the trading action.

        Args:
            feature_dict (Dict[str, float]): A dictionary where keys are feature 
                                             names and values are the feature values.

        Returns:
            str: The predicted trading action (e.g., 'Strong Buy', 'Hold', 'Weak Sell').
        """
        if self.previous_feature_dict is None:
            self.previous_feature_dict = feature_dict
            raise ValueError("Not enough historical data to calculate deltas. Received first data point.")

        # Calculate deltas for price-related features
        delta_feature_dict = feature_dict.copy()
        for col in ['close', 'open', 'high', 'low']:
            if col in delta_feature_dict:
                delta_feature_dict[col] = feature_dict[col] - self.previous_feature_dict.get(col, feature_dict[col])

        # Update the history
        self.previous_feature_dict = feature_dict

        # Now, proceed with the original prediction logic using the delta_feature_dict
        ordered_values = [delta_feature_dict[feature] for feature in self.features]
        input_array = np.array(ordered_values).reshape(1, -1)
        scaled_array = self.scaler.transform(input_array)
        prediction_class = self.model.predict(scaled_array)[0]
        predicted_action = self.reverse_label_mapping[prediction_class]
        
        return predicted_action

    def predict_proba(self, feature_dict: Dict[str, float]) -> Dict[str, float]:
        """
        Performs a prediction for a single sample and returns class probabilities.

        Args:
            feature_dict (Dict[str, float]): A dictionary where keys are feature 
                                             names and values are the feature values.

        Returns:
            Dict[str, float]: A dictionary mapping action names to their probabilities.
        """
        if self.previous_feature_dict is None:
            self.previous_feature_dict = feature_dict
            raise ValueError("Not enough historical data to calculate deltas. Received first data point.")

        # Calculate deltas for price-related features
        delta_feature_dict = feature_dict.copy()
        for col in ['close', 'open', 'high', 'low']:
            if col in delta_feature_dict:
                delta_feature_dict[col] = feature_dict[col] - self.previous_feature_dict.get(col, feature_dict[col])

        # Update the history
        self.previous_feature_dict = feature_dict

        # Now, proceed with the original prediction logic using the delta_feature_dict
        ordered_values = [delta_feature_dict[feature] for feature in self.features]
        input_array = np.array(ordered_values).reshape(1, -1)
        scaled_array = self.scaler.transform(input_array)
        prediction_probabilities = self.model.predict_proba(scaled_array)[0]
        
        # Create a dictionary mapping action names to probabilities
        prob_dict = {}
        for class_idx, probability in enumerate(prediction_probabilities):
            action_name = self.reverse_label_mapping[class_idx]
            prob_dict[action_name] = float(probability)
        
        return prob_dict

    def get_model_info(self) -> Dict[str, Any]:
        """
        Returns information about the loaded model.

        Returns:
            Dict[str, Any]: A dictionary containing model metadata.
        """
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'features': self.features,
            'num_features': len(self.features),
            'label_mapping': self.label_mapping,
            'available_actions': list(self.label_mapping.keys()),
            'model_params': self.model_params
        }

    def create_prediction_response(self, feature_dict: Dict[str, float]) -> XGBoostPredictionResponse:
        """Creates a properly formatted prediction response for the API"""
        # Get both the prediction and probabilities
        predicted_action = self.predict(feature_dict)
        probabilities = self.predict_proba(feature_dict)
        
        # The confidence is the probability of the predicted action
        confidence = probabilities[predicted_action]
        
        return XGBoostPredictionResponse(
            model_name=self.model_name,
            predicted_action=predicted_action,
            confidence=confidence,
            probabilities=probabilities
        )
