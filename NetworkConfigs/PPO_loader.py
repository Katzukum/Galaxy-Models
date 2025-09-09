import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from typing import Dict, Any, List
from pydantic import BaseModel
from Utilities.yaml_utils import YAMLConfig, load_yaml_config

# Import the PPO network from PPOTrainer
from NetworkConfigs.PPOTrainer import PPONetwork, TradingEnvironment

class PPOPredictionResponse(BaseModel):
    """Response model for PPO predictions"""
    prediction: int
    action_name: str
    confidence: float
    value_estimate: float

class PPOModelLoader:
    """
    Model loader for PPO agents.
    Handles loading and prediction for trained PPO models.
    """
    
    def __init__(self, model_dir: str):
        """
        Initialize the PPO model loader.
        
        Args:
            model_dir (str): Path to the model directory containing artifacts
        """
        self.model_dir = model_dir
        self.model_name = None
        self.config = None
        self.features = None
        self.scaler = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history = deque(maxlen=60)  # Store last 60 observations for sequence models
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the PPO model and its artifacts"""
        try:
            # Find and load config file
            config_files = [f for f in os.listdir(self.model_dir) if f.endswith('.yaml')]
            if not config_files:
                raise FileNotFoundError(f"No YAML config file found in {self.model_dir}")
            
            config_path = os.path.join(self.model_dir, config_files[0])
            self.config = load_yaml_config(config_path)
            
            # Extract model name using recursive key finding
            self.model_name = self.config.find_key('model_name')
            if not self.model_name:
                raise ValueError("No model_name found in config")
            
            # Extract features using recursive key finding
            self.features = self.config.find_key('features')
            if not self.features:
                raise ValueError("No features found in config")
            
            # Load scaler
            artifact_paths = self.config.find_key('artifact_paths')
            if not artifact_paths:
                raise ValueError("No artifact_paths found in config")
            
            scaler_path = os.path.join(self.model_dir, artifact_paths['scaler'])
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load model
            model_path = os.path.join(self.model_dir, artifact_paths['model_state_dict'])
            
            # Get model parameters using recursive key finding
            data_input_dim = self.config.find_key('input_dim', len(self.features))  # Raw data features
            input_dim = data_input_dim + 3  # +3 for account state (balance, position, unrealized_pnl)
            hidden_dim = self.config.find_key('hidden_dim', 128)
            num_actions = self.config.find_key('num_actions', 3)
            lookback_window = self.config.find_key('lookback_window', 60)
            
            # Initialize and load model
            self.model = PPONetwork(
                input_dim=input_dim,  # Already includes +3 for account state
                hidden_dim=hidden_dim,
                num_actions=num_actions
            )
            
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            print(f"PPO model loaded successfully: {self.model_name}")
            
        except Exception as e:
            print(f"Error loading PPO model: {e}")
            raise
    
    def predict(self, feature_dict: Dict[str, float]) -> int:
        """
        Make a prediction using the PPO model.
        
        Args:
            feature_dict (Dict[str, float]): Dictionary of feature values
            
        Returns:
            int: Predicted action (0=Hold, 1=Buy, 2=Sell)
        """
        try:
            # Validate features
            if not all(feature in feature_dict for feature in self.features):
                missing_features = [f for f in self.features if f not in feature_dict]
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Extract features in the correct order
            feature_values = [feature_dict[feature] for feature in self.features]
            
            # Add to history
            self.history.append(feature_values)
            
            # Check if we have enough history
            if len(self.history) < self.model.lookback_window:
                # Not enough history, return hold action
                return 0
            
            # Prepare observation
            observation = self._prepare_observation()
            
            # Make prediction
            with torch.no_grad():
                action_logits, value = self.model(observation)
                action_probs = torch.softmax(action_logits, dim=-1)
                action = torch.argmax(action_probs, dim=-1).item()
                confidence = torch.max(action_probs).item()
                value_estimate = value.item()
            
            return action, confidence, value_estimate
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return 0, 0.0, 0.0  # Return hold action on error
    
    def _prepare_observation(self):
        """Prepare observation tensor for the model"""
        # Convert history to numpy array
        history_array = np.array(list(self.history))
        
        # Scale features
        scaled_features = self.scaler.transform(history_array)
        
        # Add dummy account state (balance, position, unrealized_pnl)
        # In a real implementation, these would come from the trading system
        account_state = np.array([1.0, 0.0, 0.0])  # Normalized balance=1, no position, no unrealized PnL
        account_state_broadcast = np.tile(account_state, (len(scaled_features), 1))
        
        # Combine features with account state
        observation = np.concatenate([scaled_features, account_state_broadcast], axis=1)
        
        # Convert to tensor and add batch dimension
        observation_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        return observation_tensor
    
    def create_prediction_response(self, prediction_data) -> PPOPredictionResponse:
        """
        Create a prediction response object.
        
        Args:
            prediction_data: Tuple of (action, confidence, value_estimate) or just action
            
        Returns:
            PPOPredictionResponse: Formatted response object
        """
        if isinstance(prediction_data, tuple):
            action, confidence, value_estimate = prediction_data
        else:
            action = prediction_data
            confidence = 0.0
            value_estimate = 0.0
        
        # Map action to name
        action_names = {0: "Hold", 1: "Buy", 2: "Sell"}
        action_name = action_names.get(action, "Unknown")
        
        return PPOPredictionResponse(
            prediction=action,
            action_name=action_name,
            confidence=confidence,
            value_estimate=value_estimate
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_name': self.model_name,
            'model_type': 'PPO Agent',
            'features': self.features,
            'num_actions': 3,
            'action_names': ['Hold', 'Buy', 'Sell']
        }

# For backward compatibility with the existing API structure
def load_ppo_model(model_dir: str) -> PPOModelLoader:
    """Load a PPO model from the specified directory"""
    return PPOModelLoader(model_dir)

if __name__ == '__main__':
    # Example usage
    print("PPO Model Loader - Example usage")
    
    # This would be used by the API loader
    pass