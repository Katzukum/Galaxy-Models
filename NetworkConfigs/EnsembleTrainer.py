#!/usr/bin/env python3
"""
Ensemble Model Trainer for Galaxy Models
Combines multiple trained models to create ensemble predictions
"""

import os
import sys
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

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

try:
    from Utilities.yaml_utils import YAMLConfig, load_yaml_config
except ImportError:
    # Fallback if yaml_utils is not available
    print("Warning: yaml_utils not found, using basic YAML loading")
    
    class YAMLConfig:
        def __init__(self, data):
            self.data = data
        
        def find_key(self, key, default=None):
            def _search_dict(d, search_key):
                if isinstance(d, dict):
                    if search_key in d:
                        return d[search_key]
                    for value in d.values():
                        result = _search_dict(value, search_key)
                        if result is not None:
                            return result
                return None
            
            result = _search_dict(self.data, key)
            return result if result is not None else default
    
    def load_yaml_config(file_path):
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        return YAMLConfig(data)

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
        data.columns = data.columns.str.lower()
        print(f"Loaded data shape: {data.shape}")
        
        # Find common features across all models and available in data
        print(f"Data columns available: {list(data.columns)}")
        
        # Get features from all models
        all_model_features = []
        for model_name, loader in self.model_loaders.items():
            model_features = loader.features
            print(f"Model '{model_name}' expects features: {model_features}")
            all_model_features.append(set(model_features))
        
        # Find intersection of features (common to all models)
        common_features = set.intersection(*all_model_features) if all_model_features else set()
        print(f"Common features across all models: {list(common_features)}")
        
        # Filter to only features available in data
        available_common_features = [f for f in common_features if f in data.columns]
        
        if not available_common_features:
            print("No common features found, using features from first model...")
            # Fallback: use first model's features that are available in data
            first_model = list(self.model_loaders.values())[0]
            first_model_features = first_model.features
            available_common_features = [f for f in first_model_features if f in data.columns]
            
            if not available_common_features:
                raise ValueError("No matching features found between any model and data")
        
        self.features = available_common_features
        print(f"Using features for ensemble: {self.features}")
        print(f"Number of features: {len(self.features)}")
        
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
                model_predictions = []
                
                for i in range(X.shape[0]):
                    # Convert row to feature dictionary
                    base_feature_dict = {feature: float(X[i, j]) for j, feature in enumerate(self.features)}
                    
                    # Extend feature dict with missing features (set to 0 or use defaults)
                    model_features = loader.features
                    full_feature_dict = base_feature_dict.copy()
                    
                    for required_feature in model_features:
                        if required_feature not in full_feature_dict:
                            # Use 0 as default for missing features
                            full_feature_dict[required_feature] = 0.0
                            if i == 0:  # Log only once
                                print(f"  Warning: Setting missing feature '{required_feature}' to 0.0 for model {model_name}")
                    
                    # Get prediction from model
                    if hasattr(loader, 'predict'):
                        pred = loader.predict(full_feature_dict)
                        
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
                        
                        model_predictions.append(pred)
                    else:
                        # Fallback for models without predict method
                        pred = loader.model.predict(X[i:i+1])
                        model_predictions.append(pred[0] if hasattr(pred, '__len__') else pred)
                
                predictions[model_name] = np.array(model_predictions)
                print(f"✓ Collected predictions from {model_name}: shape {predictions[model_name].shape}")
                
            except Exception as e:
                print(f"✗ Failed to get predictions from {model_name}: {e}")
                import traceback
                traceback.print_exc()
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
    # --- 1. Find and select random models from Models folder ---
    import random
    import glob
    from pathlib import Path
    
    # Try to find Models folder - could be in current dir or parent dir
    models_folder = Path("Models")
    if not models_folder.exists():
        models_folder = Path("../Models")
        if not models_folder.exists():
            print("Error: Models folder not found in current directory or parent directory!")
            print(f"Current working directory: {os.getcwd()}")
            exit(1)
    
    # Find all YAML config files in Models folder
    yaml_files = list(models_folder.glob("**/*_config.yaml"))
    
    if len(yaml_files) < 2:
        print(f"Error: Need at least 2 models, found {len(yaml_files)} YAML files")
        print("Available files:", [str(f) for f in yaml_files])
        exit(1)
    
    # Select 2 random models
    selected_yaml_files = random.sample(yaml_files, 2)
    print(f"Found {len(yaml_files)} models, selected 2 random models:")
    
    selected_models = []
    for yaml_file in selected_yaml_files:
        print(f"Loading config from: {yaml_file}")
        
        try:
            # Load YAML config to get model info
            config = load_yaml_config(str(yaml_file))
            model_name = config.find_key('model_name', f'model_{yaml_file.stem}')
            model_type = config.find_key('Type', 'Unknown')
            
            selected_models.append({
                'name': model_name,
                'type': model_type,
                'configPath': str(yaml_file)
            })
            
            print(f"  - {model_name} ({model_type})")
            
        except Exception as e:
            print(f"Error loading {yaml_file}: {e}")
            continue
    
    if len(selected_models) < 2:
        print("Error: Could not load at least 2 valid models")
        exit(1)
    
    # --- 2. Prepare CSV data ---
    csv_path = "sample.csv"
    if not os.path.exists(csv_path):
        csv_path = "../sample.csv"
        if not os.path.exists(csv_path):
            print(f"Error: sample.csv not found in current directory or parent directory!")
            print(f"Current working directory: {os.getcwd()}")
            exit(1)
    
    # --- 3. Set ensemble configuration ---
    ensemble_name = f"auto_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ensemble_type = "averaging"  # Use averaging as default for mixed model types
    
    # Advanced options
    advanced_options = {
        'validationSplit': 0.2,
        'randomState': 42,
        'metaLearner': 'linear'  # For stacking ensemble
    }
    
    print(f"\n--- Starting Ensemble Training ---")
    print(f"Ensemble name: {ensemble_name}")
    print(f"Ensemble type: {ensemble_type}")
    print(f"CSV data: {csv_path}")
    print(f"Selected models: {len(selected_models)}")
    
    # --- 4. Run ensemble training ---
    try:
        success = run_ensemble_training(
            ensemble_name=ensemble_name,
            ensemble_type=ensemble_type,
            selected_models=selected_models,
            csv_path=csv_path,
            weights=None,  # Equal weights
            advanced_options=advanced_options,
            output_dir="Models/Ensemble"
        )
        
        if success:
            print(f"\n✅ Ensemble training completed successfully!")
            print(f"Ensemble saved in: Models/Ensemble/{ensemble_name}")
        else:
            print(f"\n❌ Ensemble training failed!")
            
    except Exception as e:
        print(f"\n❌ Ensemble training error: {e}")
        import traceback
        traceback.print_exc()