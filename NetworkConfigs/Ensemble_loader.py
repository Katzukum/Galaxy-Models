# File: Ensemble_loader.py

import os
import yaml
import pickle
import numpy as np
from typing import Dict, Any, List, Union
from pydantic import BaseModel

class EnsemblePredictionResponse(BaseModel):
    """Response model for Ensemble predictions"""
    model_config = {"protected_namespaces": ()}
    model_name: str
    model_type: str = "Ensemble Model"
    predicted_value: float
    individual_predictions: Dict[str, Union[float, str]]
    ensemble_type: str

class EnsembleModelLoader:
    """
    Loads and serves an ensemble model trained by the EnsembleTrainer class.
    
    This class encapsulates all the logic required to load the artifacts 
    (config, ensemble model, model references) and perform predictions.
    """

    def __init__(self, model_dir: str):
        """
        Initializes the loader by loading all necessary model artifacts.

        Args:
            model_dir (str): The path to the directory containing the model files 
                             (_config.yaml, _ensemble.pkl, _model_refs.yaml).
        """
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        print(f"Initializing ensemble model from directory: {model_dir}")
        
        # --- 1. Load Configuration from YAML ---
        config_path = self._find_file_by_extension(model_dir, '.yaml')
        if not config_path:
            raise FileNotFoundError(f"Could not find a .yaml config file in {model_dir}")

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model_name: str = self.config['model_name']
        self.ensemble_type: str = self.config['Config']['ensemble_type']
        self.selected_models: List[Dict] = self.config['Config']['selected_models']
        self.weights: Dict = self.config['Config'].get('weights', {})
        self.advanced_options: Dict = self.config['Config'].get('advanced_options', {})
        
        # Load model references
        artifact_paths = self.config['artifact_paths']
        refs_path = os.path.join(model_dir, artifact_paths['model_refs'])
        
        with open(refs_path, 'r') as f:
            self.model_refs = yaml.safe_load(f)
        
        print(f"Successfully loaded configuration for ensemble '{self.model_name}'.")
        print(f"Ensemble type: {self.ensemble_type}")
        print(f"Number of models: {len(self.selected_models)}")

        # --- 2. Load the Ensemble Model ---
        ensemble_path = os.path.join(model_dir, artifact_paths['ensemble_model'])
        if os.path.exists(ensemble_path):
            with open(ensemble_path, 'rb') as f:
                self.ensemble_model = pickle.load(f)
            print(f"Ensemble model loaded from: '{ensemble_path}'")
        else:
            self.ensemble_model = None
            print("No ensemble model file found - using simple ensemble methods")

        # --- 3. Load Individual Model Loaders ---
        self.model_loaders = {}
        self._load_individual_models()

        print("Ensemble model loader is ready.")

    @staticmethod
    def _find_file_by_extension(directory: str, extension: str) -> str:
        """Finds the first file with a given extension in a directory."""
        for filename in os.listdir(directory):
            if filename.endswith(extension):
                return os.path.join(directory, filename)
        return ""

    def _load_individual_models(self):
        """Load all individual models that make up the ensemble"""
        print("Loading individual models for ensemble...")
        
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
                    print(f"Warning: Unknown model type {model_type} for {model_name}")
                    continue
                
                self.model_loaders[model_name] = loader
                print(f"✓ Loaded {model_name} ({model_type})")
                
            except Exception as e:
                print(f"✗ Failed to load {model_name}: {e}")
                continue
        
        if not self.model_loaders:
            raise ValueError("No individual models could be loaded successfully")
        
        print(f"Successfully loaded {len(self.model_loaders)} individual models")

    def predict(self, feature_dict: Dict[str, float]) -> float:
        """
        Performs a prediction for a single sample using the ensemble.

        Args:
            feature_dict (Dict[str, float]): A dictionary where keys are feature 
                                             names and values are the feature values.

        Returns:
            float: The ensemble model's prediction.
        """
        # Collect predictions from all individual models
        individual_predictions = {}
        
        for model_name, loader in self.model_loaders.items():
            try:
                pred = loader.predict(feature_dict)
                
                # Handle different return types
                if isinstance(pred, tuple):
                    # PPO returns (action, confidence, value) - use action
                    pred = pred[0]
                elif isinstance(pred, str):
                    # XGBoost returns string labels - convert to numeric
                    if hasattr(loader, 'label_mapping'):
                        pred = loader.label_mapping.get(pred, 0)
                    else:
                        # Try to map common action strings to numbers
                        action_map = {
                            'strong sell': 0, 'weak sell': 1, 'hold': 2, 
                            'weak buy': 3, 'strong buy': 4
                        }
                        pred = action_map.get(pred.lower(), 2)  # Default to hold
                
                individual_predictions[model_name] = pred
                
            except Exception as e:
                print(f"Error getting prediction from {model_name}: {e}")
                # Use default prediction for this model
                individual_predictions[model_name] = 0.0
        
        # Create ensemble prediction based on type
        if self.ensemble_type == 'voting':
            ensemble_pred = self._create_voting_prediction(individual_predictions)
        elif self.ensemble_type == 'averaging':
            ensemble_pred = self._create_averaging_prediction(individual_predictions)
        elif self.ensemble_type == 'weighted':
            ensemble_pred = self._create_weighted_prediction(individual_predictions)
        elif self.ensemble_type == 'stacking':
            ensemble_pred = self._create_stacking_prediction(individual_predictions)
        else:
            # Default to averaging
            ensemble_pred = self._create_averaging_prediction(individual_predictions)
        
        return ensemble_pred

    def _create_voting_prediction(self, individual_predictions: Dict[str, float]) -> float:
        """Create voting ensemble prediction (majority vote for classification)"""
        predictions = list(individual_predictions.values())
        # For regression, use mean; for classification, use mode
        return np.mean(predictions)

    def _create_averaging_prediction(self, individual_predictions: Dict[str, float]) -> float:
        """Create averaging ensemble prediction (mean prediction for regression)"""
        predictions = list(individual_predictions.values())
        return np.mean(predictions)

    def _create_weighted_prediction(self, individual_predictions: Dict[str, float]) -> float:
        """Create weighted ensemble prediction"""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for model_name, pred in individual_predictions.items():
            weight = self.weights.get(model_name, 1.0)
            weighted_sum += weight * pred
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else np.mean(list(individual_predictions.values()))

    def _create_stacking_prediction(self, individual_predictions: Dict[str, float]) -> float:
        """Create stacking ensemble prediction using meta-learner"""
        if self.ensemble_model is None:
            # Fallback to averaging if no meta-learner
            return self._create_averaging_prediction(individual_predictions)
        
        # Prepare meta-features (predictions from base models)
        meta_features = np.array(list(individual_predictions.values())).reshape(1, -1)
        
        # Use the trained meta-learner
        ensemble_pred = self.ensemble_model.predict(meta_features)[0]
        
        return ensemble_pred

    def get_individual_predictions(self, feature_dict: Dict[str, float]) -> Dict[str, Union[float, str]]:
        """
        Get predictions from all individual models.

        Args:
            feature_dict (Dict[str, float]): A dictionary where keys are feature 
                                             names and values are the feature values.

        Returns:
            Dict[str, Union[float, str]]: A dictionary mapping model names to their predictions.
        """
        individual_predictions = {}
        
        for model_name, loader in self.model_loaders.items():
            try:
                pred = loader.predict(feature_dict)
                
                # Handle different return types
                if isinstance(pred, tuple):
                    # PPO returns (action, confidence, value) - use action
                    pred = pred[0]
                elif isinstance(pred, str):
                    # Keep string predictions as-is for display
                    pass
                
                individual_predictions[model_name] = pred
                
            except Exception as e:
                print(f"Error getting prediction from {model_name}: {e}")
                individual_predictions[model_name] = "Error"
        
        return individual_predictions

    def get_model_info(self) -> Dict[str, Any]:
        """
        Returns information about the loaded ensemble model.

        Returns:
            Dict[str, Any]: A dictionary containing model metadata.
        """
        return {
            'model_name': self.model_name,
            'model_type': 'Ensemble Model',
            'ensemble_type': self.ensemble_type,
            'num_models': len(self.selected_models),
            'selected_models': self.selected_models,
            'weights': self.weights,
            'advanced_options': self.advanced_options,
            'individual_models': list(self.model_loaders.keys())
        }

    def create_prediction_response(self, feature_dict: Dict[str, float]) -> EnsemblePredictionResponse:
        """Creates a properly formatted prediction response for the API"""
        # Get both the ensemble prediction and individual predictions
        ensemble_prediction = self.predict(feature_dict)
        individual_predictions = self.get_individual_predictions(feature_dict)
        
        return EnsemblePredictionResponse(
            model_name=self.model_name,
            predicted_value=ensemble_prediction,
            individual_predictions=individual_predictions,
            ensemble_type=self.ensemble_type
        )
