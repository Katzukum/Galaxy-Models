# File: PPOEnsemble_loader.py

import os
import yaml
import pickle
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Union
from pydantic import BaseModel
from NetworkConfigs.PPOEnsembleTrainer import PPOEnsembleNetwork, PPOEnsembleEnvironment


class PPOEnsemblePredictionResponse(BaseModel):
    """Response model for PPO Ensemble predictions"""
    model_config = {"protected_namespaces": ()}
    model_name: str
    model_type: str = "PPO Ensemble Model"
    predicted_action: int
    action_probabilities: List[float]
    individual_predictions: Dict[str, Union[float, str]]
    market_features: List[float]
    portfolio_state: Dict[str, float]


class PPOEnsembleModelLoader:
    """
    Loads and serves a PPO ensemble model trained by the PPOEnsembleTrainer class.
    
    This class encapsulates all the logic required to load the artifacts 
    (config, PPO model, individual model references, scalers) and perform predictions.
    """

    def __init__(self, model_dir: str):
        """
        Initializes the loader by loading all necessary model artifacts.

        Args:
            model_dir (str): The path to the directory containing the model files 
                             (_config.yaml, _ppo_model.pth, _model_refs.yaml, _scaler.pkl).
        """
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        print(f"Initializing PPO ensemble model from directory: {model_dir}")
        
        # --- 1. Load Configuration from YAML ---
        config_path = self._find_file_by_extension(model_dir, '.yaml')
        if not config_path:
            raise FileNotFoundError(f"Could not find a .yaml config file in {model_dir}")

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model_name: str = self.config['model_name']
        self.ensemble_type: str = self.config['Config']['ensemble_type']
        self.selected_models: List[Dict] = self.config['Config']['selected_models']
        self.ppo_params: Dict = self.config['Config'].get('ppo_params', {})
        self.trading_params: Dict = self.config['Config'].get('trading_params', {})
        self.features: List[str] = self.config['Config'].get('features', [])
        self.sequence_length: int = self.config['Config'].get('sequence_length', 60)
        self.input_size: int = self.config['Config'].get('input_size', 0)
        
        # --- 2. Load Scaler ---
        scaler_path = self._find_file_by_extension(model_dir, '.pkl')
        if not scaler_path:
            raise FileNotFoundError(f"Could not find a .pkl scaler file in {model_dir}")
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # --- 3. Load Model References ---
        model_refs_path = self._find_file_by_extension(model_dir, '_model_refs.yaml')
        if not model_refs_path:
            raise FileNotFoundError(f"Could not find a _model_refs.yaml file in {model_dir}")
        
        with open(model_refs_path, 'r') as f:
            self.model_refs = yaml.safe_load(f)
        
        # --- 4. Load Individual Model Loaders ---
        self.individual_loaders = {}
        self._load_individual_models()
        
        # --- 5. Load PPO Model ---
        self._load_ppo_model(model_dir)
        
        # --- 6. Initialize Environment ---
        self._initialize_environment()
        
        print(f"Successfully loaded PPO ensemble model: {self.model_name}")
        print(f"Individual models: {list(self.individual_loaders.keys())}")
        print(f"Sequence length: {self.sequence_length}")
        print(f"Input size: {self.input_size}")

    def _find_file_by_extension(self, directory: str, extension: str) -> str:
        """Find a file with the given extension in the directory"""
        for file in os.listdir(directory):
            if file.endswith(extension):
                return os.path.join(directory, file)
        return None

    def _load_individual_models(self):
        """Load all individual model loaders"""
        print("Loading individual model loaders...")
        
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
        
        for model_name, model_ref in self.model_refs.items():
            try:
                model_type = model_ref['type']
                model_dir = os.path.dirname(model_ref['config_path'])
                
                print(f"Loading individual model: {model_name} ({model_type})")
                
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
                    print(f"Warning: Unknown model type {model_type} for {model_name}, skipping...")
                    continue
                
                self.individual_loaders[model_name] = {
                    'loader': loader,
                    'type': model_type
                }
                print(f"Successfully loaded individual model: {model_name}")
                
            except Exception as e:
                print(f"Error loading individual model {model_name}: {e}")
                continue
        
        print(f"Successfully loaded {len(self.individual_loaders)} individual models")

    def _load_ppo_model(self, model_dir: str):
        """Load the trained PPO model"""
        print("Loading PPO model...")
        
        # Find PPO model file
        ppo_model_path = None
        for file in os.listdir(model_dir):
            if file.endswith('_ppo_model.pth'):
                ppo_model_path = os.path.join(model_dir, file)
                break
        
        if not ppo_model_path:
            raise FileNotFoundError(f"Could not find PPO model file in {model_dir}")
        
        # Load the state dict to determine the correct architecture
        state_dict = torch.load(ppo_model_path, map_location='cpu')
        
        # Determine hidden_size from the saved model
        # The first linear layer in feature_extractor should tell us the hidden_size
        if 'feature_extractor.0.weight' in state_dict:
            hidden_size = state_dict['feature_extractor.0.weight'].shape[0]
        else:
            # Fallback to config or default
            hidden_size = self.config['Config'].get('hidden_size', 64)
            print(f"Using hidden_size from config: {hidden_size}")
        
        print(f"Detected hidden_size from saved model: {hidden_size}")
        
        # Initialize PPO network with correct architecture
        self.ppo_model = PPOEnsembleNetwork(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_actions=3
        )
        
        # Load trained weights
        self.ppo_model.load_state_dict(state_dict)
        self.ppo_model.eval()
        
        print("Successfully loaded PPO model")

    def _initialize_environment(self):
        """Initialize the trading environment for inference"""
        # Create dummy data for environment initialization
        dummy_data = np.zeros((100, len(self.features)))
        dummy_predictions = np.zeros((100, len(self.individual_loaders)))
        
        self.environment = PPOEnsembleEnvironment(
            data=dummy_data,
            model_predictions=dummy_predictions,
            features=self.features,
            lookback_window=self.sequence_length,
            initial_balance=self.trading_params.get('initial_balance', 50000),
            position_size=self.trading_params.get('position_size', 0.1),
            transaction_cost=self.trading_params.get('transaction_cost', 0.001)
        )

    def predict(self, X: np.ndarray) -> PPOEnsemblePredictionResponse:
        """
        Make predictions using the PPO ensemble model.
        
        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features)
            
        Returns:
            PPOEnsemblePredictionResponse: Prediction response with action and probabilities
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Normalize input features
        X_scaled = self.scaler.transform(X)
        
        # Get individual model predictions
        individual_predictions = {}
        model_predictions = []
        
        for model_name, model_info in self.individual_loaders.items():
            try:
                loader = model_info['loader']
                pred = loader.predict(X_scaled)
                
                # Ensure prediction is scalar
                if hasattr(pred, 'item'):
                    pred_value = pred.item()
                elif isinstance(pred, (list, np.ndarray)) and len(pred) > 0:
                    pred_value = float(pred[0])
                else:
                    pred_value = float(pred)
                
                individual_predictions[model_name] = pred_value
                model_predictions.append(pred_value)
                
            except Exception as e:
                print(f"Error getting prediction from {model_name}: {e}")
                individual_predictions[model_name] = 0.0
                model_predictions.append(0.0)
        
        # Convert to numpy array
        model_predictions = np.array(model_predictions).reshape(1, -1)
        
        # Create sequence for PPO model
        if len(X_scaled) >= self.sequence_length:
            # Use last sequence_length samples
            sequence_data = X_scaled[-self.sequence_length:]
            sequence_predictions = np.tile(model_predictions, (self.sequence_length, 1))
        else:
            # Pad with zeros
            sequence_data = np.zeros((self.sequence_length, X_scaled.shape[1]))
            sequence_data[-len(X_scaled):] = X_scaled
            sequence_predictions = np.zeros((self.sequence_length, len(model_predictions)))
            sequence_predictions[-1] = model_predictions[0]
        
        # Combine model predictions and market data
        sequence = np.concatenate([sequence_predictions, sequence_data], axis=1)
        
        # Add portfolio state (dummy values for inference)
        portfolio_state = np.array([1.0, 0.0, 0.0])  # Normalized balance, position, unrealized_pnl
        portfolio_state_broadcast = np.tile(portfolio_state, (self.sequence_length, 1))
        sequence = np.concatenate([sequence, portfolio_state_broadcast], axis=1)
        
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
        
        # Get PPO prediction
        with torch.no_grad():
            action_logits, value = self.ppo_model(sequence_tensor)
            action_probs = torch.softmax(action_logits, dim=-1)
            predicted_action = torch.argmax(action_probs, dim=-1).item()
            action_probabilities = action_probs.squeeze().tolist()
        
        # Create response
        response = PPOEnsemblePredictionResponse(
            model_name=self.model_name,
            model_type="PPO Ensemble Model",
            predicted_action=predicted_action,
            action_probabilities=action_probabilities,
            individual_predictions=individual_predictions,
            market_features=X_scaled[-1].tolist() if len(X_scaled) > 0 else [0.0] * len(self.features),
            portfolio_state={
                'balance': 1.0,  # Normalized
                'position': 0.0,
                'unrealized_pnl': 0.0
            }
        )
        
        return response

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'model_name': self.model_name,
            'model_type': 'PPO Ensemble Model',
            'ensemble_type': self.ensemble_type,
            'individual_models': list(self.individual_loaders.keys()),
            'sequence_length': self.sequence_length,
            'input_size': self.input_size,
            'features': self.features,
            'ppo_params': self.ppo_params,
            'trading_params': self.trading_params
        }

    def get_individual_predictions(self, X: np.ndarray) -> Dict[str, float]:
        """Get predictions from all individual models"""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        X_scaled = self.scaler.transform(X)
        predictions = {}
        
        for model_name, model_info in self.individual_loaders.items():
            try:
                loader = model_info['loader']
                pred = loader.predict(X_scaled)
                
                if hasattr(pred, 'item'):
                    pred_value = pred.item()
                elif isinstance(pred, (list, np.ndarray)) and len(pred) > 0:
                    pred_value = float(pred[0])
                else:
                    pred_value = float(pred)
                
                predictions[model_name] = pred_value
                
            except Exception as e:
                print(f"Error getting prediction from {model_name}: {e}")
                predictions[model_name] = 0.0
        
        return predictions

    def get_action_probabilities(self, X: np.ndarray) -> List[float]:
        """Get action probabilities from PPO model"""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        X_scaled = self.scaler.transform(X)
        
        # Get individual model predictions
        model_predictions = []
        for model_name, model_info in self.individual_loaders.items():
            try:
                loader = model_info['loader']
                pred = loader.predict(X_scaled)
                
                if hasattr(pred, 'item'):
                    pred_value = pred.item()
                elif isinstance(pred, (list, np.ndarray)) and len(pred) > 0:
                    pred_value = float(pred[0])
                else:
                    pred_value = float(pred)
                
                model_predictions.append(pred_value)
                
            except Exception as e:
                model_predictions.append(0.0)
        
        # Create sequence for PPO model
        model_predictions = np.array(model_predictions).reshape(1, -1)
        
        if len(X_scaled) >= self.sequence_length:
            sequence_data = X_scaled[-self.sequence_length:]
            sequence_predictions = np.tile(model_predictions, (self.sequence_length, 1))
        else:
            sequence_data = np.zeros((self.sequence_length, X_scaled.shape[1]))
            sequence_data[-len(X_scaled):] = X_scaled
            sequence_predictions = np.zeros((self.sequence_length, len(model_predictions)))
            sequence_predictions[-1] = model_predictions[0]
        
        # Combine and add portfolio state
        sequence = np.concatenate([sequence_predictions, sequence_data], axis=1)
        portfolio_state = np.array([1.0, 0.0, 0.0])
        portfolio_state_broadcast = np.tile(portfolio_state, (self.sequence_length, 1))
        sequence = np.concatenate([sequence, portfolio_state_broadcast], axis=1)
        
        # Get PPO prediction
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
        
        with torch.no_grad():
            action_logits, _ = self.ppo_model(sequence_tensor)
            action_probs = torch.softmax(action_logits, dim=-1)
            return action_probs.squeeze().tolist()