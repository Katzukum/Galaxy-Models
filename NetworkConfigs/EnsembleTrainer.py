import os
import yaml
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Union
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import xgboost as xgb
from Utilities.data_utils import prepare_delta_features


class EnsembleTrainer:
    """
    A class to train an ensemble model that combines multiple trained models.
    It saves the ensemble configuration, model references, and a primary YAML config file.
    """

    def __init__(self, 
                 model_name: str, 
                 config: Dict[str, Any], 
                 output_path: str = '/models'):
        """
        Initializes the EnsembleTrainer.

        Args:
            model_name (str): A unique name for the ensemble model.
            config (dict): A dictionary containing ensemble configuration.
            output_path (str): The directory path to save the output files.
        """
        if not model_name:
            raise ValueError("A 'model_name' must be provided.")
            
        self.model_name = model_name
        self.config = config
        self.output_path = output_path
        
        # Extract ensemble configuration
        self.ensemble_type = config.get('ensemble_type', 'averaging')
        self.selected_models = config.get('selected_models', [])
        self.weights = config.get('weights', {})
        self.advanced_options = config.get('advanced_options', {})
        
        # Initialize model loaders
        self.model_loaders = {}
        self.ensemble_model = None
        
        # Training data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.features = None
        
        # Results
        self.training_results = {}
        
    def load_models(self):
        """Load all selected models"""
        print(f"Loading {len(self.selected_models)} models for ensemble...")
        
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
                    # Try to determine model type from directory name or config file
                    if 'nn_' in model_dir.lower() or 'neural' in model_dir.lower():
                        loader = NNModelLoader(model_dir)
                    elif 'transformer_' in model_dir.lower():
                        loader = TransformerModelLoader(model_dir)
                    elif 'xgboost_' in model_dir.lower():
                        loader = XGBoostModelLoader(model_dir)
                    elif 'ppo_' in model_dir.lower():
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
    
    @staticmethod
    def prepare_ensemble_data(
        data: pd.DataFrame,
        look_ahead_period: int = 5,
        tick_size: float = 0.25,
        columns_to_exclude: List[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepares data for ensemble training by creating a target variable and features.

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
        print("Preparing data for ensemble training...")
        
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

    def prepare_data(self, csv_path: str):
        """Prepare training data for ensemble"""
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
                print(f"Collecting predictions from {model_name}...")
                
                # Check if this is a stateful model that needs sequential data
                is_stateful = hasattr(loader, 'history') or hasattr(loader, 'previous_feature_dict')
                
                if is_stateful:
                    # For stateful models, we need to process data sequentially
                    model_predictions = self._collect_sequential_predictions(loader, X, model_name)
                else:
                    # For stateless models, we can process in batch
                    model_predictions = self._collect_batch_predictions(loader, X, model_name)
                
                predictions[model_name] = np.array(model_predictions)
                print(f"✓ Collected predictions from {model_name}: shape {predictions[model_name].shape}")
                
            except Exception as e:
                print(f"✗ Failed to get predictions from {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return predictions
    
    def _collect_sequential_predictions(self, loader, X: np.ndarray, model_name: str) -> list:
        """Collect predictions from stateful models that need sequential data"""
        model_predictions = []
        
        # Reset the loader's state for clean prediction
        if hasattr(loader, 'history'):
            loader.history.clear()
        if hasattr(loader, 'previous_feature_dict'):
            loader.previous_feature_dict = None
        
        # Determine sequence length for this model
        sequence_length = 1  # Default for most models
        if hasattr(loader, 'sequence_length'):
            sequence_length = loader.sequence_length
        elif hasattr(loader, 'config') and 'Config' in loader.config:
            # Try to get sequence length from model config
            config = loader.config['Config']
            if 'data_params' in config and 'sequence_length' in config['data_params']:
                sequence_length = config['data_params']['sequence_length']
        
        print(f"  Model {model_name} requires {sequence_length} data points for prediction")
        
        # Process data sequentially
        for i in range(X.shape[0]):
            try:
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
                pred = loader.predict(full_feature_dict)
                
                # Handle None predictions
                if pred is None:
                    print(f"  Warning: Model {model_name} returned None prediction at data point {i}")
                    model_predictions.append(0.0)
                    continue
                
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
                
            except ValueError as e:
                # Handle cases where not enough historical data is available
                if "Not enough historical data" in str(e) or "first data point" in str(e):
                    # For the first data points, use a default prediction
                    model_predictions.append(0.0)  # Default prediction
                    if i == 0:  # Log only once
                        print(f"  Using default predictions for first data point (not enough historical data)")
                    continue
                raise e
            except Exception as e:
                print(f"  Error at data point {i}: {e}")
                # Use default prediction for this data point
                model_predictions.append(0.0)
        
        return model_predictions
    
    def _collect_batch_predictions(self, loader, X: np.ndarray, model_name: str) -> list:
        """Collect predictions from stateless models"""
        model_predictions = []
        
        for i in range(X.shape[0]):
            try:
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
                    
                    # Handle None predictions
                    if pred is None:
                        print(f"  Warning: Model {model_name} returned None prediction at data point {i}")
                        model_predictions.append(0.0)
                        continue
                    
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
                    if pred is None:
                        print(f"  Warning: Model {model_name} returned None prediction at data point {i}")
                        model_predictions.append(0.0)
                    else:
                        model_predictions.append(pred[0] if hasattr(pred, '__len__') else pred)
                    
            except ValueError as e:
                # Handle cases where not enough historical data is available
                if "Not enough historical data" in str(e) or "first data point" in str(e):
                    model_predictions.append(0.0)  # Default prediction
                    if i == 0:  # Log only once
                        print(f"  Using default predictions for first data point (not enough historical data)")
                    continue
                raise e
            except Exception as e:
                print(f"  Error at data point {i}: {e}")
                # Use default prediction for this data point
                model_predictions.append(0.0)
        
        return model_predictions
    
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
            
            # Store ensemble configuration for saving
            self.ensemble_model = {
                'type': 'voting',
                'model_names': list(predictions.keys())
            }
            
            return np.array(ensemble_pred)
        
        return None
    
    def create_averaging_ensemble(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Create averaging ensemble (mean prediction for regression)"""
        print("Creating averaging ensemble...")
        
        # Convert predictions to numpy array
        pred_array = np.array(list(predictions.values()))
        
        # Calculate mean prediction
        ensemble_pred = np.mean(pred_array, axis=0)
        
        # Store ensemble configuration for saving
        self.ensemble_model = {
            'type': 'averaging',
            'model_names': list(predictions.keys())
        }
        
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
        
        # Store ensemble configuration for saving
        self.ensemble_model = {
            'type': 'weighted',
            'weights': self.weights,
            'model_names': list(predictions.keys())
        }
        
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
    
    def save(self):
        """Saves the ensemble configuration, model references, and a primary YAML config file."""
        os.makedirs(self.output_path, exist_ok=True)
        print(f"Output directory '{self.output_path}' is ready.")

        # Save ensemble model (if applicable)
        ensemble_filename = f"{self.model_name}_ensemble.pkl"
        config_filename = f"{self.model_name}_config.yaml"
        refs_filename = f"{self.model_name}_model_refs.yaml"

        ensemble_path = os.path.join(self.output_path, ensemble_filename)
        config_path = os.path.join(self.output_path, config_filename)
        refs_path = os.path.join(self.output_path, refs_filename)

        # Save ensemble model
        if self.ensemble_model is not None:
            with open(ensemble_path, 'wb') as f:
                pickle.dump(self.ensemble_model, f)
            print(f"Ensemble model has been saved to: '{ensemble_path}'")

        # Create final configuration with the same structure as other trainers
        final_config = {
            'model_name': self.model_name,
            'Type': 'Ensemble Model',
            'artifact_paths': {
                'ensemble_model': ensemble_filename,
                'model_refs': refs_filename
            },
            'Config': self.config.copy()
        }

        with open(config_path, 'w') as yaml_file:
            yaml.dump(final_config, yaml_file, default_flow_style=False, sort_keys=False)
        print(f"Primary configuration has been saved to: '{config_path}'")

        # Save model loaders reference
        model_refs = {}
        for model_name, loader in self.model_loaders.items():
            model_refs[model_name] = {
                'type': loader.__class__.__name__,
                'config_path': getattr(loader, 'config_path', None)
            }
        
        with open(refs_path, 'w') as f:
            yaml.dump(model_refs, f, default_flow_style=False, sort_keys=False)
        print(f"Model references have been saved to: '{refs_path}'")
        print("\nSave process completed.")


def run_training_pipeline(model_name: str, output_dir: str, training_config: Dict, csv_path: str):
    """Executes the full ensemble training, saving, and verification process."""
    trainer = EnsembleTrainer(
        model_name=model_name,
        config=training_config,
        output_path=output_dir
    )
    
    # Load models
    trainer.load_models()
    
    # Prepare data (this sets self.features and splits the data)
    trainer.prepare_data(csv_path)
    
    # Train ensemble
    trainer.train_ensemble()
    
    # Test ensemble
    trainer.test_ensemble()
    
    # Save ensemble
    trainer.save()

    print("\n" + "="*50)
    print("--- Verifying artifacts by loading from YAML config ---")
    print("="*50)

    config_yaml_path = os.path.join(output_dir, f"{model_name}_config.yaml")
    with open(config_yaml_path, 'r') as f:
        loaded_config = yaml.safe_load(f)

    print(f"Loaded config for model: {loaded_config['model_name']}")
    
    # Verify output files
    print(f"\nGenerated files in '{output_dir}':")
    for filename in os.listdir(output_dir):
        print(f"- {filename}")


if __name__ == '__main__':
    # --- 1. Load Data and Define Parameters ---
    data = pd.read_csv('sample.csv')
    data.columns = data.columns.str.lower()
    
    # --- 2. Prepare Data using the Class Method ---
    X_sample_raw, y_sample_raw, feature_names = EnsembleTrainer.prepare_ensemble_data(
        data=data,
        look_ahead_period=5,
        tick_size=0.25,
        columns_to_exclude=['date', 'time', 'target']
    )

    # Convert features to deltas
    feature_df_raw = pd.DataFrame(X_sample_raw, columns=feature_names)
    feature_df_delta = prepare_delta_features(feature_df_raw)

    X_sample = feature_df_delta.values
    y_sample = y_sample_raw[1:] # Align labels

    # --- 3. Define the training configuration for ensemble ---
    example_config = {
        'ensemble_type': 'averaging',
        'selected_models': [
            {
                'name': 'test_model_1',
                'type': 'Neural Network (Regression)',
                'configPath': 'Models/test/test_config.yaml'
            },
            {
                'name': 'test_model_2', 
                'type': 'XGBoostClassifier',
                'configPath': 'Models/test/test_config.yaml'
            }
        ],
        'weights': {},
        'advanced_options': {
            'validationSplit': 0.2,
            'randomState': 42,
            'metaLearner': 'linear'
        },
        'features': feature_names
    }
    
    # --- 4. Define a model name and output path ---
    MODEL_NAME = "ensemble_test_model"
    OUTPUT_DIR = f"./Models/Ensemble_{MODEL_NAME}"

    # --- 5. Call the main training pipeline function ---
    run_training_pipeline(
        model_name=MODEL_NAME,
        output_dir=OUTPUT_DIR,
        training_config=example_config,
        X=X_sample,
        y=y_sample
    )


def run_ensemble_training(ensemble_name: str, ensemble_type: str, selected_models: List[Dict], 
                         csv_path: str, weights: Dict = None, advanced_options: Dict = None,
                         output_dir: str = "Models"):
    """
    Backward compatibility wrapper for the old ensemble training interface.
    
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
        # Load CSV data
        data = pd.read_csv(csv_path)
        data.columns = data.columns.str.lower()
        
        # Prepare data using the static method
        X_sample_raw, y_sample_raw, feature_names = EnsembleTrainer.prepare_ensemble_data(
            data=data,
            look_ahead_period=5,
            tick_size=0.25,
            columns_to_exclude=['date', 'time', 'target']
        )
        
        # Convert features to deltas
        feature_df_raw = pd.DataFrame(X_sample_raw, columns=feature_names)
        feature_df_delta = prepare_delta_features(feature_df_raw)
        
        X_sample = feature_df_delta.values
        y_sample = y_sample_raw[1:] # Align labels
        
        # Create ensemble configuration
        ensemble_config = {
            'ensemble_type': ensemble_type,
            'selected_models': selected_models,
            'weights': weights or {},
            'advanced_options': advanced_options or {},
            'features': feature_names
        }
        
        # Create output directory
        ensemble_output_dir = os.path.join(output_dir, f"Ensemble_{ensemble_name}")
        
        # Run training pipeline
        run_training_pipeline(
            model_name=ensemble_name,
            output_dir=ensemble_output_dir,
            training_config=ensemble_config,
            csv_path=csv_path
        )
        
        print("Ensemble training completed successfully!")
        return True
        
    except Exception as e:
        print(f"Ensemble training failed: {e}")
        import traceback
        traceback.print_exc()
        return False