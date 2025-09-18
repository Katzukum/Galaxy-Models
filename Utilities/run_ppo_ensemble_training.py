#!/usr/bin/env python3
"""
PPO Ensemble Training Utility

This script provides a command-line interface for training PPO-based ensemble models.
It can be used standalone or integrated with the main training pipeline.
"""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from NetworkConfigs.PPOEnsembleTrainer import run_ppo_ensemble_training


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        sys.exit(1)


def create_sample_config(output_path: str):
    """Create a sample configuration file for PPO ensemble training"""
    sample_config = {
        'model_name': 'ppo_ensemble_sample',
        'ensemble_type': 'ppo',
        'selected_models': [
            {
                'name': 'nn_model_1',
                'type': 'Neural Network (Regression)',
                'configPath': 'Models/NN_model1/config.yaml'
            },
            {
                'name': 'xgboost_model_1',
                'type': 'XGBoostClassifier',
                'configPath': 'Models/XGBoost_model1/config.yaml'
            }
        ],
        'ppo_params': {
            'learning_rate': 0.0003,
            'epochs': 100,
            'batch_size': 64,
            'sequence_length': 60,
            'gamma': 0.99,
            'clip_ratio': 0.2
        },
        'trading_params': {
            'initial_balance': 50000,
            'position_size': 0.1,
            'transaction_cost': 0.001
        },
        'features': ['close', 'volume', 'rsi', 'macd'],
        'csv_path': 'sample.csv'
    }
    
    config_path = os.path.join(output_path, 'ppo_ensemble_sample_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(sample_config, f, default_flow_style=False)
    
    print(f"Sample configuration created: {config_path}")
    return config_path


def validate_config(config: dict) -> bool:
    """Validate the configuration file"""
    required_fields = ['model_name', 'ensemble_type', 'selected_models']
    
    for field in required_fields:
        if field not in config:
            print(f"Error: Missing required field '{field}' in configuration")
            return False
    
    if config['ensemble_type'] != 'ppo':
        print(f"Error: ensemble_type must be 'ppo', got '{config['ensemble_type']}'")
        return False
    
    if not config['selected_models']:
        print("Error: selected_models cannot be empty")
        return False
    
    # Validate PPO parameters
    ppo_params = config.get('ppo_params', {})
    if 'learning_rate' in ppo_params and not (0 < ppo_params['learning_rate'] < 1):
        print("Error: learning_rate must be between 0 and 1")
        return False
    
    if 'epochs' in ppo_params and ppo_params['epochs'] <= 0:
        print("Error: epochs must be positive")
        return False
    
    if 'sequence_length' in ppo_params and ppo_params['sequence_length'] <= 0:
        print("Error: sequence_length must be positive")
        return False
    
    return True


def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(
        description='Train a PPO-based ensemble model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with configuration file
  python run_ppo_ensemble_training.py --config config.yaml --csv data.csv
  
  # Create sample configuration
  python run_ppo_ensemble_training.py --create-sample-config
  
  # Train with specific parameters
  python run_ppo_ensemble_training.py --model-name my_ppo_ensemble --csv data.csv --epochs 200
        """
    )
    
    parser.add_argument('--config', '-c', type=str, help='Path to configuration YAML file')
    parser.add_argument('--csv', type=str, help='Path to CSV data file')
    parser.add_argument('--model-name', type=str, help='Name for the trained model')
    parser.add_argument('--output-path', type=str, default='/models', help='Output directory for trained model')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, help='Learning rate for PPO training')
    parser.add_argument('--sequence-length', type=int, help='Sequence length for time-series data')
    parser.add_argument('--create-sample-config', action='store_true', help='Create a sample configuration file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Create sample config if requested
    if args.create_sample_config:
        create_sample_config('.')
        return
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        # Create minimal config from command line arguments
        config = {
            'model_name': args.model_name or 'ppo_ensemble_model',
            'ensemble_type': 'ppo',
            'selected_models': [],  # Will need to be specified
            'ppo_params': {},
            'trading_params': {},
            'features': [],
            'csv_path': args.csv or 'sample.csv'
        }
        
        # Override with command line arguments
        if args.epochs:
            config['ppo_params']['epochs'] = args.epochs
        if args.learning_rate:
            config['ppo_params']['learning_rate'] = args.learning_rate
        if args.sequence_length:
            config['ppo_params']['sequence_length'] = args.sequence_length
    
    # Override CSV path if provided
    if args.csv:
        config['csv_path'] = args.csv
    
    # Validate configuration
    if not validate_config(config):
        sys.exit(1)
    
    # Check if CSV file exists
    csv_path = config.get('csv_path', 'sample.csv')
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    # Check if selected models are specified
    if not config.get('selected_models'):
        print("Error: No selected models specified. Please provide a configuration file with selected_models.")
        print("Use --create-sample-config to create a sample configuration file.")
        sys.exit(1)
    
    # Print configuration summary
    print("PPO Ensemble Training Configuration:")
    print(f"  Model Name: {config['model_name']}")
    print(f"  CSV Data: {config['csv_path']}")
    print(f"  Selected Models: {len(config['selected_models'])}")
    print(f"  Output Path: {args.output_path}")
    
    if args.verbose:
        print(f"  PPO Parameters: {config.get('ppo_params', {})}")
        print(f"  Trading Parameters: {config.get('trading_params', {})}")
        print(f"  Features: {config.get('features', [])}")
    
    # Start training
    print("\nStarting PPO ensemble training...")
    try:
        result = run_ppo_ensemble_training(
            model_name=config['model_name'],
            config=config,
            output_path=args.output_path
        )
        
        if result['success']:
            print(f"\nTraining completed successfully!")
            print(f"Model saved to: {result.get('model_dir', 'Unknown')}")
            
            if 'results' in result and result['results']:
                print(f"Training results: {result['results']}")
        else:
            print(f"\nTraining failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during training: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()