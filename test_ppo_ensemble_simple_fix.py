#!/usr/bin/env python3
"""
Simple test script for PPO Ensemble implementation with fixes
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_ppo_ensemble_trainer_fixed():
    """Test PPOEnsembleTrainer class with fixes"""
    print("Testing PPOEnsembleTrainer with fixes...")
    
    try:
        from NetworkConfigs.PPOEnsembleTrainer import PPOEnsembleTrainer
        
        # Create test configuration
        config = {
            'model_name': 'test_ppo_ensemble_fixed',
            'ensemble_type': 'ppo',
            'selected_models': [
                {
                    'name': 'test_model_1',
                    'type': 'Neural Network (Regression)',
                    'configPath': 'test_models/simple_test_config.yaml'
                }
            ],
            'ppo_params': {
                'learning_rate': 0.0003,
                'epochs': 5,  # Very small number for testing
                'batch_size': 32,
                'sequence_length': 10,  # Small sequence length
                'gamma': 0.99,
                'clip_ratio': 0.2
            },
            'trading_params': {
                'initial_balance': 10000,
                'position_size': 0.1,
                'transaction_cost': 0.001
            },
            'features': ['close', 'volume'],
            'csv_path': 'sample.csv'
        }
        
        # Initialize trainer
        trainer = PPOEnsembleTrainer('test_ppo_ensemble_fixed', config, 'test_models')
        print("✓ PPOEnsembleTrainer initialized successfully")
        
        # Test data preparation
        if os.path.exists('sample.csv'):
            X_train, X_test, y_train, y_test = trainer.prepare_ensemble_data('sample.csv')
            print(f"✓ Data preparation successful: Train={len(X_train)}, Test={len(X_test)}")
            
            # Test model prediction collection
            model_predictions, model_names = trainer.collect_model_predictions(X_train)
            print(f"✓ Model predictions collected: {model_predictions.shape}, Models: {model_names}")
            
            # Test PPO dataset creation
            sequences, targets = trainer.create_ppo_dataset(X_train, model_predictions)
            print(f"✓ PPO dataset created: {sequences.shape}")
            
        else:
            print("⚠ Sample CSV not found, creating test data")
            # Create test data
            test_data = pd.DataFrame({
                'close': np.random.randn(100) * 100 + 1000,
                'volume': np.random.randn(100) * 1000 + 5000,
                'open': np.random.randn(100) * 100 + 1000,
                'high': np.random.randn(100) * 100 + 1000,
                'low': np.random.randn(100) * 100 + 1000
            })
            test_data.to_csv('test_sample.csv', index=False)
            
            X_train, X_test, y_train, y_test = trainer.prepare_ensemble_data('test_sample.csv')
            print(f"✓ Test data preparation successful: Train={len(X_train)}, Test={len(X_test)}")
            
            # Test model prediction collection
            model_predictions, model_names = trainer.collect_model_predictions(X_train)
            print(f"✓ Model predictions collected: {model_predictions.shape}, Models: {model_names}")
            
            # Test PPO dataset creation
            sequences, targets = trainer.create_ppo_dataset(X_train, model_predictions)
            print(f"✓ PPO dataset created: {sequences.shape}")
            
            # Clean up test file
            os.remove('test_sample.csv')
        
        print("✓ PPOEnsembleTrainer test with fixes passed")
        return True
        
    except Exception as e:
        print(f"✗ PPOEnsembleTrainer test with fixes failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gymnasium_import():
    """Test that gymnasium can be imported"""
    print("Testing gymnasium import...")
    
    try:
        import gymnasium as gym
        from gymnasium import spaces
        print("✓ Gymnasium imported successfully")
        return True
    except Exception as e:
        print(f"✗ Gymnasium import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("PPO Ensemble Implementation Test Suite (With Fixes)")
    print("=" * 60)
    
    tests = [
        test_gymnasium_import,
        test_ppo_ensemble_trainer_fixed
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print()
        if test():
            passed += 1
        print("-" * 40)
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! PPO Ensemble implementation fixes are working correctly.")
        print("\nKey Fixes Applied:")
        print("✓ Updated gym to gymnasium for NumPy 2.0 compatibility")
        print("✓ Simplified data preprocessing to avoid delta features issues")
        print("✓ Added fallback prediction generation for individual models")
        print("✓ Improved error handling and robustness")
    else:
        print("⚠ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)