# run_training.py

import argparse
import os
import sys
import json
import pandas as pd
import numpy as np

# Add the parent directory to the Python path to import NetworkConfigs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Import the training pipelines from the NetworkConfigs package ---
from NetworkConfigs.TransformerTrainer import run_training_pipeline as run_transformer_pipeline
from NetworkConfigs.NNTrainer import run_training_pipeline as run_nn_pipeline, NNTrainer
from NetworkConfigs.XGboostTrainer import run_training_pipeline as run_xgb_pipeline, XGBoostTrainer
from NetworkConfigs.PPOTrainer import run_training_pipeline as run_ppo_pipeline, PPOTrainer


def main():
    """
    Main function to parse arguments and launch the selected model training pipeline.
    """
    parser = argparse.ArgumentParser(
        description="A unified script to load a CSV and train one of three different network models.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the input CSV data file."
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=['transformer', 'nn', 'xgboost', 'ppo'],
        help="The type of model to train.\n"
             "  - 'transformer': Time-series forecasting model.\n"
             "  - 'nn': Neural network for price-change regression.\n"
             "  - 'xgboost': XGBoost for 5-class action classification.\n"
             "  - 'ppo': PPO reinforcement learning agent for trading."
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        help="Custom name for the trained model. If not provided, a default name will be generated."
    )
    
    parser.add_argument(
        "--training_params",
        type=str,
        help="JSON string containing custom training parameters."
    )

    args = parser.parse_args()

    # --- Parse Training Parameters ---
    training_params = None
    if args.training_params:
        try:
            training_params = json.loads(args.training_params)
            print(f"Using custom training parameters: {training_params}")
        except json.JSONDecodeError as e:
            print(f"Error parsing training parameters: {e}")
            return

    # --- 1. Load Data ---
    print(f"Loading data from '{args.csv_path}'...")
    if not os.path.exists(args.csv_path):
        print(f"Error: The file '{args.csv_path}' was not found.")
        return
    
    data = pd.read_csv(args.csv_path)
    #update the header to be all lowercase
    data.columns = data.columns.str.lower()

    print("Data loaded successfully.")

    # --- 2. Select and Launch Model Training ---
    if args.model == 'transformer':
        # --- A. Configure and run the Transformer pipeline ---
        print("\nConfiguring Time-Series Transformer training pipeline...")
        
        # The Transformer requires the target variable to be the first column.
        # Ensure 'close' is the first column if it exists, and exclude non-numeric columns.
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if 'close' in numeric_cols:
            # Put close first, then other numeric columns
            feature_cols = ['close'] + [col for col in numeric_cols if col != 'close']
            model_data = data[feature_cols].values
            feature_names = feature_cols
            print("Using 'close' as the target (first column).")
        else:
            print("Warning: 'close' column not found in numeric columns. Using the first numeric column as the target.")
            model_data = data[numeric_cols].values
            feature_names = numeric_cols
        
        # Validate that we have numeric data
        if model_data.size == 0:
            print("Error: No numeric data found in the CSV file.")
            return
        
        # Check for any NaN or infinite values
        if np.any(np.isnan(model_data)) or np.any(np.isinf(model_data)):
            print("Warning: Found NaN or infinite values in data. Cleaning...")
            model_data = np.nan_to_num(model_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"Data shape: {model_data.shape}, Features: {len(feature_names)}")

        # Use custom parameters if provided, otherwise use defaults
        if training_params and 'model_params' in training_params:
            model_params = training_params['model_params'].copy()
            model_params['input_dim'] = len(feature_names)
        else:
            model_params = {
                'input_dim': len(feature_names),
                'd_model': 64, 'nhead': 4, 'num_encoder_layers': 3,
                'dim_feedforward': 128, 'dropout': 0.1
            }
        
        if training_params and 'data_params' in training_params:
            data_params = training_params['data_params'].copy()
            data_params['features'] = feature_names
        else:
            data_params = {'sequence_length': 60, 'features': feature_names}
        
        if training_params and 'train_params' in training_params:
            train_params = training_params['train_params']
        else:
            train_params = {'learning_rate': 0.0005, 'epochs': 25, 'batch_size': 32}
        
        config = {
            'model_params': model_params,
            'data_params': data_params,
            'train_params': train_params
        }
        
        MODEL_NAME = args.model_name if args.model_name else "nq_forecaster_transformer_cli"
        OUTPUT_DIR = f"./Models/{MODEL_NAME}"

        run_transformer_pipeline(
            model_name=MODEL_NAME, output_dir=OUTPUT_DIR,
            training_config=config, data=model_data
        )

    elif args.model == 'nn':
        # --- B. Configure and run the Neural Network pipeline ---
        print("\nConfiguring Neural Network (Regression) training pipeline...")
        
        # Validate data before processing
        if 'close' not in data.columns.str.lower():
            print("Error: 'close' column not found in the CSV file. Required for Neural Network training.")
            return
        
        X_sample, y_sample, feature_names = NNTrainer.prepare_regression_data(
            data=data, look_ahead_period=5, tick_size=0.25,
            columns_to_exclude=['date', 'time', 'target']
        )
        
        # Validate processed data
        if X_sample.size == 0 or y_sample.size == 0:
            print("Error: No valid data after processing for Neural Network training.")
            return
        
        # Use custom architecture if provided, otherwise use defaults
        if training_params and 'architecture' in training_params:
            architecture = training_params['architecture'].copy()
            # Update the first layer's in_features to match the actual data
            if architecture and architecture[0]['type'] == 'Linear':
                architecture[0]['in_features'] = X_sample.shape[1]
        else:
            architecture = [
                {'type': 'Linear', 'in_features': X_sample.shape[1], 'out_features': 64},
                {'type': 'ReLU'}, {'type': 'Dropout', 'p': 0.3},
                {'type': 'Linear', 'in_features': 64, 'out_features': 32},
                {'type': 'ReLU'},
                {'type': 'Linear', 'in_features': 32, 'out_features': 1}
            ]
        
        if training_params and 'train_params' in training_params:
            train_params = training_params['train_params']
        else:
            train_params = {'optimizer': 'Adam', 'loss': 'MSELoss', 'learning_rate': 0.001, 'epochs': 100}
        
        config = {
            'architecture': architecture,
            'train_params': train_params,
            'features': feature_names
        }
        
        MODEL_NAME = args.model_name if args.model_name else "nq_price_change_predictor_cli"
        OUTPUT_DIR = f"./Models/NN_{MODEL_NAME}"
        
        run_nn_pipeline(
            model_name=MODEL_NAME, output_dir=OUTPUT_DIR,
            training_config=config, X=X_sample, y=y_sample
        )

    elif args.model == 'xgboost':
        # --- C. Configure and run the XGBoost pipeline ---
        print("\nConfiguring XGBoost (Classification) training pipeline...")
        
        # Use custom label parameters if provided
        if training_params and 'label_params' in training_params:
            label_params = training_params['label_params']
            look_ahead_periods = label_params.get('look_ahead_periods', [3, 5])
            min_tick_change = label_params.get('min_tick_change', 20)
        else:
            look_ahead_periods = [3, 5]
            min_tick_change = 20
        
        # Validate data before processing
        if 'close' not in data.columns:
            print("Error: 'close' column not found in the CSV file. Required for XGBoost training.")
            return
        
        processed_data, label_mapping = XGBoostTrainer.generate_labels(
            data=data.copy(), look_ahead_periods=look_ahead_periods,
            min_tick_change=min_tick_change, tick_size=0.25
        )
        
        # Validate processed data
        if processed_data.empty:
            print("Error: No valid data after processing for XGBoost training.")
            return
        
        # Prepare X and y for the model
        cols_to_exclude = [col for col in processed_data.columns if 'ahead' in str(col)]
        cols_to_exclude.extend(['Date', 'Time', 'target', 'action'])
        
        # Debug: Show what columns we're working with
        print(f"Original columns: {processed_data.columns.tolist()}")
        print(f"Columns to exclude: {cols_to_exclude}")
        
        features_df = processed_data.drop(columns=cols_to_exclude, errors='ignore')
        print(f"Remaining feature columns: {features_df.columns.tolist()}")
        print(f"Feature data types:\n{features_df.dtypes}")
        
        # Check for non-numeric columns
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        non_numeric_cols = features_df.select_dtypes(exclude=[np.number]).columns
        
        if len(non_numeric_cols) > 0:
            print(f"Warning: Found non-numeric columns: {non_numeric_cols.tolist()}")
            print("Dropping non-numeric columns...")
            features_df = features_df[numeric_cols]
        
        X_sample = features_df.values
        feature_names = features_df.columns.tolist()
        y_sample = processed_data['target'].values
        
        print(f"Final X_sample shape: {X_sample.shape}")
        print(f"Final feature names: {feature_names}")
        
        # Use custom model parameters if provided
        if training_params and 'model_params' in training_params:
            model_params = training_params['model_params'].copy()
            model_params['num_class'] = 5  # Always 5 classes for this implementation
            model_params['use_label_encoder'] = False
        else:
            model_params = {
                'objective': 'multi:softmax', 'num_class': 5, 'eval_metric': 'mlogloss',
                'n_estimators': 150, 'learning_rate': 0.1, 'max_depth': 4, 'use_label_encoder': False
            }
        
        config = {
            'model_params': model_params,
            'features': feature_names, 
            'label_mapping': label_mapping
        }
        
        MODEL_NAME = args.model_name if args.model_name else "nq_futures_action_classifier_cli"
        OUTPUT_DIR = f"./Models/XGBoost_{MODEL_NAME}"
        
        run_xgb_pipeline(
            model_name=MODEL_NAME, output_dir=OUTPUT_DIR,
            training_config=config, X=X_sample, y=y_sample
        )

    elif args.model == 'ppo':
        # --- D. Configure and run the PPO pipeline ---
        print("\nConfiguring PPO (Reinforcement Learning) training pipeline...")
        
        # Validate data before processing
        if 'close' not in data.columns.str.lower():
            print("Error: 'close' column not found in the CSV file. Required for PPO training.")
            return
        
        # Prepare features for PPO (exclude non-numeric columns)
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in ['date', 'time', 'target']]
        
        if not feature_cols:
            print("Error: No numeric features found for PPO training.")
            return
        
        # Prepare data
        model_data = data[feature_cols].values
        
        # Check for any NaN or infinite values
        if np.any(np.isnan(model_data)) or np.any(np.isinf(model_data)):
            print("Warning: Found NaN or infinite values in data. Cleaning...")
            model_data = np.nan_to_num(model_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"Data shape: {model_data.shape}, Features: {len(feature_cols)}")
        
        # Use custom parameters if provided, otherwise use defaults
        if training_params and 'model_params' in training_params:
            model_params = training_params['model_params'].copy()
            model_params['input_dim'] = len(feature_cols)
        else:
            model_params = {
                'input_dim': len(feature_cols),
                'hidden_dim': 128,
                'num_actions': 3,
                'lookback_window': 60
            }
        
        if training_params and 'train_params' in training_params:
            train_params = training_params['train_params']
        else:
            train_params = {
                'learning_rate': 3e-4,
                'epochs': 100,
                'batch_size': 64,
                'ppo_epochs': 4,
                'clip_ratio': 0.2,
                'value_coef': 0.5,
                'entropy_coef': 0.01
            }
        
        config = {
            'model_params': model_params,
            'train_params': train_params
        }
        
        MODEL_NAME = args.model_name if args.model_name else "nq_ppo_trading_agent_cli"
        OUTPUT_DIR = f"./Models/PPO_{MODEL_NAME}"
        
        run_ppo_pipeline(
            model_name=MODEL_NAME, output_dir=OUTPUT_DIR,
            training_config=config, data=model_data, features=feature_cols
        )

if __name__ == '__main__':
    main()