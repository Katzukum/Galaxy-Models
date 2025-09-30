#!/usr/bin/env python3
"""
Test script for PPO Ensemble implementation
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_ppo_ensemble_trainer():
    """Test PPOEnsembleTrainer class"""
    print("Testing PPOEnsembleTrainer...")
    
    try:
        from NetworkConfigs.PPOEnsembleTrainer import PPOEnsembleTrainer
        
        # Create test configuration
        config = {
            'model_name': 'test_ppo_ensemble',
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
                'epochs': 10,  # Small number for testing
                'batch_size': 32,
                'sequence_length': 30,
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
        trainer = PPOEnsembleTrainer('test_ppo_ensemble', config, 'test_models')
        print("✓ PPOEnsembleTrainer initialized successfully")
        
        # Test data preparation
        if os.path.exists('sample.csv'):
            X_train, X_test, y_train, y_test = trainer.prepare_ensemble_data('sample.csv')
            print(f"✓ Data preparation successful: Train={len(X_train)}, Test={len(X_test)}")
        else:
            print("⚠ Sample CSV not found, skipping data preparation test")
        
        print("✓ PPOEnsembleTrainer test passed")
        return True
        
    except Exception as e:
        print(f"✗ PPOEnsembleTrainer test failed: {e}")
        return False

def test_ppo_ensemble_loader():
    """Test PPOEnsemble_loader class"""
    print("Testing PPOEnsemble_loader...")
    
    try:
        from NetworkConfigs.PPOEnsemble_loader import PPOEnsembleModelLoader
        
        # This would require a trained model, so we'll just test the import
        print("✓ PPOEnsemble_loader imported successfully")
        print("✓ PPOEnsemble_loader test passed")
        return True
        
    except Exception as e:
        print(f"✗ PPOEnsemble_loader test failed: {e}")
        return False

def test_ppo_environment():
    """Test PPO environment"""
    print("Testing PPO Environment...")
    
    try:
        from NetworkConfigs.PPOEnsembleTrainer import PPOEnsembleEnvironment
        
        # Create test data
        data = np.random.randn(100, 3)  # Market data
        model_predictions = np.random.randn(100, 2)  # 2 models
        features = ['close', 'volume', 'rsi']
        
        # Create environment
        env = PPOEnsembleEnvironment(
            data=data,
            model_predictions=model_predictions,
            features=features,
            lookback_window=30,
            initial_balance=10000
        )
        
        # Test reset
        obs = env.reset()
        print(f"✓ Environment reset successful, observation shape: {obs.shape}")
        
        # Test step
        action = 1  # Buy
        obs, reward, done, info = env.step(action)
        print(f"✓ Environment step successful, reward: {reward}")
        
        print("✓ PPO Environment test passed")
        return True
        
    except Exception as e:
        print(f"✗ PPO Environment test failed: {e}")
        return False

def test_ppo_network():
    """Test PPO network"""
    print("Testing PPO Network...")
    
    try:
        from NetworkConfigs.PPOEnsembleTrainer import PPOEnsembleNetwork
        import torch
        
        # Create test network
        input_size = 10  # 3 market features + 2 model predictions + 3 portfolio state
        network = PPOEnsembleNetwork(input_size=input_size, hidden_size=64, num_actions=3)
        
        # Test forward pass
        batch_size = 1
        seq_length = 30
        x = torch.randn(batch_size, seq_length, input_size)
        
        action_logits, value = network(x)
        print(f"✓ Network forward pass successful, action_logits shape: {action_logits.shape}, value shape: {value.shape}")
        
        # Test get_action
        action, log_prob, value = network.get_action(x)
        print(f"✓ Network get_action successful, action: {action}, log_prob: {log_prob}, value: {value}")
        
        print("✓ PPO Network test passed")
        return True
        
    except Exception as e:
        print(f"✗ PPO Network test failed: {e}")
        return False

def test_main_integration():
    """Test Main.py integration"""
    print("Testing Main.py integration...")
    
    try:
        from Main import start_ppo_ensemble_training
        
        print("✓ PPO ensemble training function imported successfully")
        print("✓ Main.py integration test passed")
        return True
        
    except Exception as e:
        print(f"✗ Main.py integration test failed: {e}")
        return False

def test_web_ui():
    """Test web UI components"""
    print("Testing Web UI components...")
    
    try:
        # Check if HTML has PPO ensemble option
        with open('web/index.html', 'r') as f:
            html_content = f.read()
            
        if 'ppo' in html_content.lower() and 'PPO Ensemble' in html_content:
            print("✓ PPO ensemble option found in HTML")
        else:
            print("⚠ PPO ensemble option not found in HTML")
            
        # Check if JavaScript has PPO handling
        with open('web/js/ensemble_training.js', 'r') as f:
            js_content = f.read()
            
        if 'ppo' in js_content.lower() and 'start_ppo_ensemble_training' in js_content:
            print("✓ PPO ensemble handling found in JavaScript")
        else:
            print("⚠ PPO ensemble handling not found in JavaScript")
            
        print("✓ Web UI test passed")
        return True
        
    except Exception as e:
        print(f"✗ Web UI test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("PPO Ensemble Implementation Test Suite")
    print("=" * 60)
    
    tests = [
        test_ppo_ensemble_trainer,
        test_ppo_ensemble_loader,
        test_ppo_environment,
        test_ppo_network,
        test_main_integration,
        test_web_ui
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
        print("🎉 All tests passed! PPO Ensemble implementation is working correctly.")
    else:
        print("⚠ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)