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
        tick_size: float
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Generates 5 labels based on the MAGNITUDE of future price changes.
        - Strong Signal: Price moves by at least strong_tick_change.
        - Weak Signal: Price moves by at least min_tick_change but less than strong_tick_change.
        """
        print("Generating 5-class Buy/Hold/Sell labels based on MAGNITUDE...")
        
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
        
        # Apply conditions sequentially. Order matters to ensure strong signals overwrite weak ones.
        data['action'] = 'Hold'
        data.loc[weak_sell_condition, 'action'] = 'Weak Sell'
        data.loc[weak_buy_condition, 'action'] = 'Weak Buy'
        data.loc[strong_sell_condition, 'action'] = 'Strong Sell'
        data.loc[strong_buy_condition, 'action'] = 'Strong Buy'
        
        # Drop rows with NaN values created by the shift operation
        data.dropna(inplace=True)
        
        # Define the 5-class mapping
        label_mapping = {
            'Strong Sell': 0, 
            'Weak Sell': 1, 
            'Hold': 2, 
            'Weak Buy': 3, 
            'Strong Buy': 4
        }
        data['target'] = data['action'].map(label_mapping)
        
        print("\nLabel Distribution:")
        print(data['action'].value_counts().sort_index())

        return data, label_mapping

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Trains the MinMaxScaler and the XGBoost model."""
        print(f"\nStarting training for model: {self.model_name}")
        
        # Debug prints to understand data format
        print(f"X_train type: {type(X_train)}")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_train dtype: {X_train.dtype}")
        print(f"Has NaN values: {np.isnan(X_train).any()}")
        print(f"Has infinite values: {np.isinf(X_train).any()}")
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
        
        self.scaler.fit(X_train_float)
        X_train_scaled = self.scaler.transform(X_train_float)
        
        # Calculate balanced class weights to handle imbalanced datasets
        sample_weights = class_weight.compute_sample_weight(
            class_weight='balanced',
            y=y_train
        )
        
        self.model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
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

        final_config = {
            'model_name': self.model_name,
            'Type': 'XGBoostClassifier',
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
    
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nPrediction Accuracy on test set: {accuracy:.4f}")
    
    print("\nClassification Report:")
    target_names = sorted(loaded_config['Config']['label_mapping'], key=loaded_config['Config']['label_mapping'].get)
    print(classification_report(y_test, predictions, target_names=target_names, zero_division=0))

    print(f"\nGenerated files in '{output_dir}':")
    for filename in os.listdir(output_dir):
        print(f"- {filename}")


if __name__ == '__main__':
    # --- 1. Load NQ Futures Data ---
    data = pd.read_csv('sample.csv')

    # --- 2. Generate 5-Class Target Variable ---
    processed_data, label_mapping = XGBoostTrainer.generate_labels(
        data=data.copy(), # Pass a copy to avoid modifying original dataframe in place
        look_ahead_periods=[3, 5],
        min_tick_change=20, # Threshold for a "Weak" signal
        strong_tick_change=40, # Threshold for a "Strong" signal
        tick_size=0.25
    )
    
    # --- 3. Prepare X and y for the model ---
    # Define columns to exclude from features
    columns_to_exclude = [col for col in processed_data.columns if 'ahead' in str(col)]
    columns_to_exclude.extend(['Date', 'Time', 'target', 'action'])
    
    X_sample = processed_data.drop(columns=columns_to_exclude, errors='ignore').values
    feature_names = processed_data.drop(columns=columns_to_exclude, errors='ignore').columns.tolist()
    y_sample = processed_data['target'].values
    
    # --- 4. Update Training Configuration for 5-Class Classification ---
    example_config = {
        'model_params': {
            'objective': 'multi:softmax',
            'num_class': 5,  # *** CRITICAL: Changed from 3 to 5 ***
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
    MODEL_NAME = "nq_futures_action_classifier_v3_5_actions"
    OUTPUT_DIR = f"./Models/XGBoost_{MODEL_NAME}"

    # --- 6. Run the Training Pipeline ---
    run_training_pipeline(
        model_name=MODEL_NAME,
        output_dir=OUTPUT_DIR,
        training_config=example_config,
        X=X_sample,
        y=y_sample
    )