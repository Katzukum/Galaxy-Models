import os
import yaml
import pickle
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from Utilities.data_utils import prepare_delta_features

class XGBoostTrainer:
    """
    A class to train an XGBoost classification model, save its configuration, 
    the data scaler, and the model itself to a specified directory.

    Includes a static method to generate 5 trading labels (Strong/Weak Buy/Sell, Hold).
    """

    def __init__(self, 
                 model_name: str, 
                 config: Dict[str, Any], 
                 output_path: str = '/models'):
        """
        Initializes the XGBoostTrainer.
        """
        if not model_name:
            raise ValueError("A 'model_name' must be provided.")
            
        self.model_name = model_name
        self.config = config
        self.output_path = output_path
        self.model = xgb.XGBClassifier(**self.config.get('model_params', {}))
        self.scaler = MinMaxScaler()

    @staticmethod
    def generate_labels(
        data: pd.DataFrame, 
        look_ahead_periods: List[int],
        min_tick_change: int,
        strong_tick_change: int, # <-- ADD THIS
        tick_size: float,
        use_3_class: bool = True  # New parameter to control 3-class vs 5-class
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Generates labels based on the MAGNITUDE of future price changes.
        
        Args:
            use_3_class: If True, combines weak signals and hold into 'Neutral' (3-class)
                        If False, uses original 5-class system
        
        3-class system:
        - Strong Sell: Significant downward movement
        - Neutral: Weak movements or holds (combines Weak Sell, Hold, Weak Buy)
        - Strong Buy: Significant upward movement
        
        5-class system (original):
        - Strong Signal: Price moves by at least strong_tick_change
        - Weak Signal: Price moves by at least min_tick_change but less than strong_tick_change
        """
        class_type = "3-class" if use_3_class else "5-class"
        print(f"Generating {class_type} Buy/Hold/Sell labels based on MAGNITUDE...")
        
        if 'close' not in data.columns:
            raise ValueError("'close' column not found in the dataframe.")

        weak_price_threshold = min_tick_change * tick_size
        strong_price_threshold = strong_tick_change * tick_size

        # Find the max high and min low across all future periods
        future_highs = [data['high'].shift(-p) for p in look_ahead_periods]
        future_lows = [data['low'].shift(-p) for p in look_ahead_periods]
        
        max_future_high = pd.concat(future_highs, axis=1).max(axis=1)
        min_future_low = pd.concat(future_lows, axis=1).min(axis=1)
        
        # --- Define conditions based on the magnitude of the future move ---
        strong_buy_condition = (max_future_high >= data['close'] + strong_price_threshold)
        weak_buy_condition = (max_future_high >= data['close'] + weak_price_threshold)

        strong_sell_condition = (min_future_low <= data['close'] - strong_price_threshold)
        weak_sell_condition = (min_future_low <= data['close'] - weak_price_threshold)
        
        if use_3_class:
            # 3-class system: Strong Sell, Neutral, Strong Buy
            data['action'] = 'Neutral'  # Default to neutral
            data.loc[strong_sell_condition, 'action'] = 'Strong Sell'
            data.loc[strong_buy_condition, 'action'] = 'Strong Buy'
            
            # Define the 3-class mapping
            label_mapping = {
                'Strong Sell': 0,
                'Neutral': 1, 
                'Strong Buy': 2
            }
        else:
            # Original 5-class system
            data['action'] = 'Hold'
            data.loc[weak_sell_condition, 'action'] = 'Weak Sell'
            data.loc[weak_buy_condition, 'action'] = 'Weak Buy'
            data.loc[strong_sell_condition, 'action'] = 'Strong Sell'
            data.loc[strong_buy_condition, 'action'] = 'Strong Buy'
            
            # Define the 5-class mapping
            label_mapping = {
                'Strong Sell': 0, 
                'Weak Sell': 1, 
                'Hold': 2, 
                'Weak Buy': 3, 
                'Strong Buy': 4
            }
        
        # Drop rows with NaN values created by the shift operation
        data.dropna(inplace=True)
        
        data['target'] = data['action'].map(label_mapping)
        
        print(f"\n{class_type} Label Distribution:")
        print(data['action'].value_counts().sort_index())

        return data, label_mapping

    def _remap_labels_to_consecutive(self, y: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
        """
        Remaps non-consecutive class labels to consecutive integers starting from 0.
        
        Args:
            y: Original labels array
            
        Returns:
            Tuple of (remapped_labels, mapping_dict)
        """
        unique_labels = np.unique(y)
        print(f"Original unique labels: {unique_labels}")
        
        # Create mapping from original labels to consecutive labels
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        print(f"Label remapping: {label_mapping}")
        
        # Apply mapping
        y_remapped = np.array([label_mapping[label] for label in y])
        print(f"Remapped unique labels: {np.unique(y_remapped)}")
        
        return y_remapped, label_mapping

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Trains the MinMaxScaler and the XGBoost model."""
        print(f"\nStarting training for model: {self.model_name}")
        
        # Debug prints to understand data format
        print(f"X_train type: {type(X_train)}")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_train dtype: {X_train.dtype}")
        
        # Safe NaN checking for mixed data types
        try:
            if X_train.dtype == 'object' or X_train.dtype.kind in ['U', 'S']:  # String/object types
                # Convert to DataFrame for mixed type handling
                X_df = pd.DataFrame(X_train)
                has_nan = X_df.isnull().any().any()
                print(f"Has NaN values: {has_nan}")
                print("Note: Data contains non-numeric types, skipping infinite value check")
            else:
                # Numeric data - can use np.isnan safely
                print(f"Has NaN values: {np.isnan(X_train).any()}")
                print(f"Has infinite values: {np.isinf(X_train).any()}")
        except Exception as e:
            print(f"Error checking for NaN/inf values: {e}")
            
        print(f"First few values of X_train:\n{X_train[:3, :5] if X_train.shape[1] >= 5 else X_train[:3, :]}")
        
        # Check if X_train contains non-numeric data
        try:
            X_train_float = X_train.astype(float)
            print("Successfully converted X_train to float")
        except (ValueError, TypeError) as e:
            print(f"Error converting X_train to float: {e}")
            print("X_train contains non-numeric data!")
            return
        
        # Clean the data before fitting
        if np.isnan(X_train_float).any() or np.isinf(X_train_float).any():
            print("Cleaning NaN and infinite values...")
            X_train_float = np.nan_to_num(X_train_float, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Fix non-consecutive label issue for XGBoost
        y_train_remapped, self.label_remapping = self._remap_labels_to_consecutive(y_train)
        
        self.scaler.fit(X_train_float)
        X_train_scaled = self.scaler.transform(X_train_float)
        
        # Calculate balanced class weights to handle imbalanced datasets
        sample_weights = class_weight.compute_sample_weight(
            class_weight='balanced',
            y=y_train_remapped  # Use remapped labels for weight calculation
        )
        
        self.model.fit(X_train_scaled, y_train_remapped, sample_weight=sample_weights)
        print("Model training has been completed.")

    def save(self):
        """Saves the scaler, model, and a primary YAML configuration file."""
        os.makedirs(self.output_path, exist_ok=True)
        print(f"Output directory '{self.output_path}' is ready.")

        scaler_filename = f"{self.model_name}_scaler.pkl"
        model_filename = f"{self.model_name}_model.json"
        config_filename = f"{self.model_name}_config.yaml"

        scaler_path = os.path.join(self.output_path, scaler_filename)
        model_path = os.path.join(self.output_path, model_filename)
        config_path = os.path.join(self.output_path, config_filename)

        with open(scaler_path, 'wb') as scaler_file:
            pickle.dump(self.scaler, scaler_file)
        print(f"Scaler has been saved to: '{scaler_path}'")

        self.model.save_model(model_path)
        print(f"Trained XGBoost model has been saved to: '{model_path}'")

        # Clean the config to ensure YAML serialization compatibility
        clean_config = self.config.copy()
        if 'label_mapping' in clean_config:
            # Convert any numpy types in the original label mapping
            original_mapping = clean_config['label_mapping']
            clean_mapping = {}
            for k, v in original_mapping.items():
                clean_k = str(k) if not isinstance(k, (str, int, float)) else k
                clean_v = int(v) if hasattr(v, 'item') else v
                clean_mapping[clean_k] = clean_v
            clean_config['label_mapping'] = clean_mapping
        
        final_config = {
            'model_name': self.model_name,
            'Type': 'XGBoostClassifier',
            'artifact_paths': {
                'scaler': scaler_filename,
                'model': model_filename
            },
            'Config': clean_config,
        }
        
        # Add label remapping if it exists (convert numpy types to native Python types)
        if hasattr(self, 'label_remapping'):
            # Convert numpy types to native Python types for YAML serialization
            python_label_remapping = {}
            for k, v in self.label_remapping.items():
                python_k = int(k) if hasattr(k, 'item') else k
                python_v = int(v) if hasattr(v, 'item') else v
                python_label_remapping[python_k] = python_v
            final_config['Config']['consecutive_label_remapping'] = python_label_remapping

        with open(config_path, 'w') as yaml_file:
            yaml.dump(final_config, yaml_file, default_flow_style=False, sort_keys=False)
        print(f"Primary configuration has been saved to: '{config_path}'")
        print("\nSave process completed.")


def run_training_pipeline(model_name: str, output_dir: str, training_config: Dict, X: np.ndarray, y: np.ndarray):
    """Executes the full model training, saving, and verification process."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=False)
    
    trainer = XGBoostTrainer(model_name=model_name, config=training_config, output_path=output_dir)
    trainer.train(X_train, y_train)
    trainer.save()
    
    print("\n" + "="*50)
    print("--- Verifying artifacts by loading from YAML config ---")
    print("="*50)

    config_yaml_path = os.path.join(output_dir, f"{model_name}_config.yaml")
    with open(config_yaml_path, 'r') as f:
        loaded_config = yaml.safe_load(f)

    print(f"Loaded config for model: {loaded_config['model_name']}")
    print(f"Label mapping: {loaded_config['Config']['label_mapping']}")

    artifact_paths = loaded_config['artifact_paths']
    scaler_path = os.path.join(output_dir, artifact_paths['scaler'])
    model_path = os.path.join(output_dir, artifact_paths['model'])

    with open(scaler_path, 'rb') as f:
        loaded_scaler = pickle.load(f)

    loaded_model = xgb.XGBClassifier()
    loaded_model.load_model(model_path)
    print("\nScaler and Model loaded successfully.")

    X_test_scaled = loaded_scaler.transform(X_test)
    predictions = loaded_model.predict(X_test_scaled)
    
    # Handle label remapping for evaluation
    y_test_for_eval = y_test
    if 'consecutive_label_remapping' in loaded_config['Config']:
        consecutive_remapping = loaded_config['Config']['consecutive_label_remapping']
        print(f"Consecutive label remapping found: {consecutive_remapping}")
        
        # The consecutive_remapping maps original_label -> consecutive_label
        # We need to apply the same mapping to y_test
        try:
            y_test_for_eval = np.array([consecutive_remapping[int(label)] for label in y_test])
            print(f"Applied consecutive label remapping for evaluation")
        except KeyError as e:
            print(f"Warning: Label {e} not found in consecutive mapping. Using original labels for evaluation.")
            print(f"y_test unique values: {np.unique(y_test)}")
            print(f"Available mapping keys: {list(consecutive_remapping.keys())}")
            # Fall back to original labels if there's a mismatch
            y_test_for_eval = y_test
    
    accuracy = accuracy_score(y_test_for_eval, predictions)
    print(f"\nPrediction Accuracy on test set: {accuracy:.4f}")
    
    print("\nClassification Report:")
    target_names = sorted(loaded_config['Config']['label_mapping'], key=loaded_config['Config']['label_mapping'].get)
    print(classification_report(y_test_for_eval, predictions, target_names=target_names, zero_division=0))

    print(f"\nGenerated files in '{output_dir}':")
    for filename in os.listdir(output_dir):
        print(f"- {filename}")


if __name__ == '__main__':
    # --- 1. Load NQ Futures Data ---
    data = pd.read_csv('sample.csv')
    data.columns = data.columns.str.lower()

    # --- 2. Generate 3-Class Target Variable ---
    processed_data, label_mapping = XGBoostTrainer.generate_labels(
        data=data.copy(), # Pass a copy to avoid modifying original dataframe in place
        look_ahead_periods=[3, 5],
        min_tick_change=20, # Threshold for a "Weak" signal
        strong_tick_change=40, # Threshold for a "Strong" signal
        tick_size=0.25,
        use_3_class=True  # Use 3-class system for better balance
    )
    
    # --- 3. Prepare X and y for the model ---
    # Define columns to exclude from features (using lowercase names after header lowercasing)
    columns_to_exclude = [col for col in processed_data.columns if 'ahead' in str(col)]
    columns_to_exclude.extend(['date', 'time', 'target', 'action'])
    
    feature_df_raw = processed_data.drop(columns=columns_to_exclude, errors='ignore')
    feature_names = feature_df_raw.columns.tolist()
    y_sample_raw = processed_data['target'].values

    # Convert features to deltas
    feature_df_delta = prepare_delta_features(feature_df_raw)

    # Align labels with the features (the first label is now invalid)
    y_sample = y_sample_raw[1:]
    X_sample = feature_df_delta.values
    
    # --- 4. Update Training Configuration for Dynamic Classification ---
    num_classes = len(label_mapping)
    example_config = {
        'model_params': {
            'objective': 'multi:softmax',
            'num_class': num_classes,  # Dynamic based on actual classes
            'eval_metric': 'mlogloss',
            'n_estimators': 150,
            'learning_rate': 0.1,
            'max_depth': 4,
            'use_label_encoder': False
        },
        'features': feature_names,
        'label_mapping': label_mapping
    }
    
    # --- 5. Define Model Name and Output Path ---
    class_type = "3_class" if num_classes == 3 else f"{num_classes}_class"
    MODEL_NAME = f"nq_futures_action_classifier_v3_{class_type}"
    OUTPUT_DIR = f"./Models/XGBoost_{MODEL_NAME}"

    # --- 6. Run the Training Pipeline ---
    run_training_pipeline(
        model_name=MODEL_NAME,
        output_dir=OUTPUT_DIR,
        training_config=example_config,
        X=X_sample,
        y=y_sample
    )