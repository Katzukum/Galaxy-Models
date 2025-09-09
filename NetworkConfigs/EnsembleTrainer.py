#!/usr/bin/env python3
"""
Ensemble Model Trainer for Galaxy Models
Combines multiple trained models to create ensemble predictions
"""

import os
import pickle
import numpy as np
import pandas as pd
import yaml
from typing import Dict, List, Any, Tuple, Union
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import xgboost as xgb
from Utilities.yaml_utils import YAMLConfig, load_yaml_config

# Import model loaders
from NetworkConfigs.NN_loader import NNModelLoader
from NetworkConfigs.Transformer_loader import TransformerModelLoader
from NetworkConfigs.XGBoost_loader import XGBoostModelLoader
from NetworkConfigs.PPO_loader import PPOModelLoader


class EnsembleTrainer:
    """
    Ensemble model trainer that combines multiple trained models
    """
    
    def __init__(self, ensemble_name: str, ensemble_type: str, selected_models: List[Dict], 
                 weights: Dict = None, advanced_options: Dict = None):
        """
        Initialize ensemble trainer
        
        Args:
            ensemble_name: Name of the ensemble model
            ensemble_type: Type of ensemble (voting, averaging, weighted, stacking)
            selected_models: List of selected models with their configurations
            weights: Weights for weighted ensemble
            advanced_options: Advanced configuration options
        """
        self.ensemble_name = ensemble_name
        self.ensemble_type = ensemble_type
        self.selected_models = selected_models
        self.weights = weights or {}
        self.advanced_options = advanced_options or {}
        
        # Initialize model loaders
        self.model_loaders = {}
        self.model_predictions = {}
        self.ensemble_model = None
        
        # Training data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.features = None
        
        # Results
        self.training_results = {}
        self.ensemble_accuracy = None
        self.ensemble_mse = None
        
    def load_models(self):
        """Load all selected models"""
        print(f"Loading {len(self.selected_models)} models for ensemble...")
        
        for model_info in self.selected_models:
            model_name = model_info['name']
            model_type = model_info['type']
            model_dir = os.path.dirname(model_info['configPath'])
            
            print(f"[DEBUG] Processing model: {model_name}")
            print(f"[DEBUG] Model type: {model_type}")
            print(f"[DEBUG] Config path: {model_info['configPath']}")
            print(f"[DEBUG] Model directory: {model_dir}")
            
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
            raise ValueError("No models could be loaded successfully")
        
        print(f"Successfully loaded {len(self.model_loaders)} models")
    
    def prepare_data(self, csv_path: str):
        """Prepare training data"""
        print("Preparing ensemble training data...")
        
        # Load CSV data
        data = pd.read_csv(csv_path)
        print(f"Loaded data shape: {data.shape}")
        
        # Get features from the first model (assuming all models use same features)
        first_model = list(self.model_loaders.values())[0]
        self.features = first_model.features
        
        # Prepare feature data
        X = data[self.features].values
        y = data.iloc[:, -1].values  # Assuming target is last column
        
        # Split data
        test_size = self.advanced_options.get('validationSplit', 0.2)
        random_state = self.advanced_options.get('randomState', 42)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        print(f"Features: {len(self.features)}")
    
    def collect_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Collect predictions from all models"""
        predictions = {}
        
        for model_name, loader in self.model_loaders.items():
            try:
                # Get predictions from model
                if hasattr(loader, 'predict'):
                    pred = loader.predict(X)
                else:
                    # For models without predict method, use the model directly
                    pred = loader.model.predict(X)
                
                predictions[model_name] = pred
                print(f"✓ Collected predictions from {model_name}")
                
            except Exception as e:
                print(f"✗ Failed to get predictions from {model_name}: {e}")
                continue
        
        return predictions
    
    def create_voting_ensemble(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Create voting ensemble (majority vote for classification)"""
        print("Creating voting ensemble...")
        
        # Convert predictions to numpy array
        pred_array = np.array(list(predictions.values()))
        
        # For classification, use majority vote
        if self.ensemble_type == 'voting':
            # Get most common prediction for each sample
            ensemble_pred = []
            for i in range(pred_array.shape[1]):
                votes = pred_array[:, i]
                ensemble_pred.append(np.bincount(votes.astype(int)).argmax())
            
            return np.array(ensemble_pred)
        
        return None
    
    def create_averaging_ensemble(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Create averaging ensemble (mean prediction for regression)"""
        print("Creating averaging ensemble...")
        
        # Convert predictions to numpy array
        pred_array = np.array(list(predictions.values()))
        
        # Calculate mean prediction
        ensemble_pred = np.mean(pred_array, axis=0)
        
        return ensemble_pred
    
    def create_weighted_ensemble(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Create weighted ensemble"""
        print("Creating weighted ensemble...")
        
        # Initialize weighted sum
        ensemble_pred = None
        
        for model_name, pred in predictions.items():
            weight = self.weights.get(model_name, 1.0 / len(predictions))
            
            if ensemble_pred is None:
                ensemble_pred = weight * pred
            else:
                ensemble_pred += weight * pred
        
        return ensemble_pred
    
    def create_stacking_ensemble(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Create stacking ensemble with meta-learner"""
        print("Creating stacking ensemble...")
        
        # Prepare meta-features (predictions from base models)
        meta_features = np.column_stack(list(predictions.values()))
        
        # Get meta-learner type
        meta_learner_type = self.advanced_options.get('metaLearner', 'linear')
        
        # Create meta-learner
        if meta_learner_type == 'linear':
            meta_learner = LinearRegression()
        elif meta_learner_type == 'ridge':
            meta_learner = Ridge(alpha=1.0)
        elif meta_learner_type == 'lasso':
            meta_learner = Lasso(alpha=0.1)
        elif meta_learner_type == 'elastic':
            meta_learner = ElasticNet(alpha=0.1, l1_ratio=0.5)
        elif meta_learner_type == 'xgboost':
            meta_learner = xgb.XGBRegressor(n_estimators=100, random_state=42)
        else:
            meta_learner = LinearRegression()
        
        # Train meta-learner
        meta_learner.fit(meta_features, self.y_train)
        
        # Store meta-learner for later use
        self.ensemble_model = meta_learner
        
        # Make predictions
        ensemble_pred = meta_learner.predict(meta_features)
        
        return ensemble_pred
    
    def train_ensemble(self):
        """Train the ensemble model"""
        print(f"Training {self.ensemble_type} ensemble...")
        
        # Collect predictions from all models on training data
        train_predictions = self.collect_predictions(self.X_train)
        
        if not train_predictions:
            raise ValueError("No predictions could be collected from models")
        
        # Create ensemble based on type
        if self.ensemble_type == 'voting':
            ensemble_pred = self.create_voting_ensemble(train_predictions)
        elif self.ensemble_type == 'averaging':
            ensemble_pred = self.create_averaging_ensemble(train_predictions)
        elif self.ensemble_type == 'weighted':
            ensemble_pred = self.create_weighted_ensemble(train_predictions)
        elif self.ensemble_type == 'stacking':
            ensemble_pred = self.create_stacking_ensemble(train_predictions)
        else:
            raise ValueError(f"Unknown ensemble type: {self.ensemble_type}")
        
        # Evaluate ensemble on training data
        self.evaluate_ensemble(ensemble_pred, self.y_train, "training")
        
        print("Ensemble training completed successfully")
    
    def evaluate_ensemble(self, predictions: np.ndarray, true_values: np.ndarray, dataset_name: str):
        """Evaluate ensemble performance"""
        print(f"Evaluating ensemble on {dataset_name} data...")
        
        # Calculate metrics
        if self.ensemble_type == 'voting':
            # Classification metrics
            accuracy = accuracy_score(true_values, predictions)
            self.training_results[f'{dataset_name}_accuracy'] = accuracy
            print(f"{dataset_name.capitalize()} Accuracy: {accuracy:.4f}")
            
        else:
            # Regression metrics
            mse = mean_squared_error(true_values, predictions)
            r2 = r2_score(true_values, predictions)
            self.training_results[f'{dataset_name}_mse'] = mse
            self.training_results[f'{dataset_name}_r2'] = r2
            print(f"{dataset_name.capitalize()} MSE: {mse:.4f}")
            print(f"{dataset_name.capitalize()} R²: {r2:.4f}")
    
    def test_ensemble(self):
        """Test ensemble on test data"""
        print("Testing ensemble on test data...")
        
        # Collect predictions from all models on test data
        test_predictions = self.collect_predictions(self.X_test)
        
        if not test_predictions:
            raise ValueError("No predictions could be collected from models")
        
        # Create ensemble predictions
        if self.ensemble_type == 'voting':
            ensemble_pred = self.create_voting_ensemble(test_predictions)
        elif self.ensemble_type == 'averaging':
            ensemble_pred = self.create_averaging_ensemble(test_predictions)
        elif self.ensemble_type == 'weighted':
            ensemble_pred = self.create_weighted_ensemble(test_predictions)
        elif self.ensemble_type == 'stacking':
            # For stacking, use the trained meta-learner
            meta_features = np.column_stack(list(test_predictions.values()))
            ensemble_pred = self.ensemble_model.predict(meta_features)
        else:
            raise ValueError(f"Unknown ensemble type: {self.ensemble_type}")
        
        # Evaluate ensemble on test data
        self.evaluate_ensemble(ensemble_pred, self.y_test, "test")
        
        # Store final metrics
        if self.ensemble_type == 'voting':
            self.ensemble_accuracy = self.training_results.get('test_accuracy', 0)
        else:
            self.ensemble_mse = self.training_results.get('test_mse', 0)
    
    def save_ensemble(self, output_dir: str):
        """Save ensemble model and configuration"""
        print(f"Saving ensemble model to {output_dir}...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save ensemble configuration
        config = {
            'ensemble_name': self.ensemble_name,
            'ensemble_type': self.ensemble_type,
            'selected_models': self.selected_models,
            'weights': self.weights,
            'features': self.features,
            'training_results': self.training_results,
            'created_at': datetime.now().isoformat(),
            'advanced_options': self.advanced_options
        }
        
        config_path = os.path.join(output_dir, f"{self.ensemble_name}_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        # Save ensemble model (if applicable)
        if self.ensemble_model is not None:
            model_path = os.path.join(output_dir, f"{self.ensemble_name}_ensemble.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(self.ensemble_model, f)
        
        # Save model loaders reference
        model_refs = {}
        for model_name, loader in self.model_loaders.items():
            model_refs[model_name] = {
                'type': loader.__class__.__name__,
                'config_path': getattr(loader, 'config_path', None)
            }
        
        refs_path = os.path.join(output_dir, f"{self.ensemble_name}_model_refs.yaml")
        with open(refs_path, 'w') as f:
            yaml.dump(model_refs, f, default_flow_style=False, sort_keys=False)
        
        print(f"Ensemble model saved successfully")
        print(f"Configuration: {config_path}")
        if self.ensemble_model is not None:
            print(f"Model: {model_path}")
        print(f"Model references: {refs_path}")


def run_ensemble_training(ensemble_name: str, ensemble_type: str, selected_models: List[Dict], 
                         csv_path: str, weights: Dict = None, advanced_options: Dict = None,
                         output_dir: str = "Models/Ensemble"):
    """
    Run ensemble training pipeline
    
    Args:
        ensemble_name: Name of the ensemble model
        ensemble_type: Type of ensemble (voting, averaging, weighted, stacking)
        selected_models: List of selected models
        csv_path: Path to CSV training data
        weights: Weights for weighted ensemble
        advanced_options: Advanced configuration options
        output_dir: Output directory for ensemble model
    """
    print(f"Starting ensemble training: {ensemble_name}")
    print(f"Ensemble type: {ensemble_type}")
    print(f"Selected models: {len(selected_models)}")
    print(f"CSV path: {csv_path}")
    
    try:
        # Create ensemble trainer
        trainer = EnsembleTrainer(
            ensemble_name=ensemble_name,
            ensemble_type=ensemble_type,
            selected_models=selected_models,
            weights=weights,
            advanced_options=advanced_options
        )
        
        # Load models
        trainer.load_models()
        
        # Prepare data
        trainer.prepare_data(csv_path)
        
        # Train ensemble
        trainer.train_ensemble()
        
        # Test ensemble
        trainer.test_ensemble()
        
        # Save ensemble
        ensemble_output_dir = os.path.join(output_dir, ensemble_name)
        trainer.save_ensemble(ensemble_output_dir)
        
        print("Ensemble training completed successfully!")
        return True
        
    except Exception as e:
        print(f"Ensemble training failed: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    selected_models = [
        {'name': 'model1', 'type': 'XGBoost', 'configPath': '/path/to/model1/config.yaml'},
        {'name': 'model2', 'type': 'Neural Network', 'configPath': '/path/to/model2/config.yaml'}
    ]
    
    run_ensemble_training(
        ensemble_name="test_ensemble",
        ensemble_type="averaging",
        selected_models=selected_models,
        csv_path="data.csv"
    )