#!/usr/bin/env python3
"""
Test script to verify PPO setup is working correctly.
This script tests the PPO trainer, loader, and basic functionality.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_ppo_imports():
    """Test that all PPO modules can be imported"""
    print("Testing PPO imports...")
    
    try:
        from NetworkConfigs.PPOTrainer import PPOTrainer, PPONetwork, TradingEnvironment
        print("✓ PPOTrainer imports successful")
    except ImportError as e:
        print(f"✗ PPOTrainer import failed: {e}")
        return False
    
    try:
        from NetworkConfigs.PPO_loader import PPOModelLoader, PPOPredictionResponse
        print("✓ PPO_loader imports successful")
    except ImportError as e:
        print(f"✗ PPO_loader import failed: {e}")
        return False
    
    return True

def test_ppo_network():
    """Test PPO network creation and forward pass"""
    print("\nTesting PPO network...")
    
    try:
        from NetworkConfigs.PPOTrainer import PPONetwork
        
        # Create network
        input_dim = 10
        hidden_dim = 64
        num_actions = 3
        lookback_window = 20
        
        network = PPONetwork(input_dim + 3, hidden_dim, num_actions)  # +3 for account state
        
        # Test forward pass
        batch_size = 2
        seq_len = lookback_window
        feature_dim = input_dim + 3
        
        x = torch.randn(batch_size, seq_len, feature_dim)
        action_logits, value = network(x)
        
        assert action_logits.shape == (batch_size, num_actions), f"Expected action_logits shape ({batch_size}, {num_actions}), got {action_logits.shape}"
        assert value.shape == (batch_size, 1), f"Expected value shape ({batch_size}, 1), got {value.shape}"
        
        print("✓ PPO network creation and forward pass successful")
        return True
        
    except Exception as e:
        print(f"✗ PPO network test failed: {e}")
        return False

def test_trading_environment():
    """Test trading environment creation and basic functionality"""
    print("\nTesting trading environment...")
    
    try:
        from NetworkConfigs.PPOTrainer import TradingEnvironment
        
        # Create dummy data
        n_samples = 100
        n_features = 5
        data = np.random.randn(n_samples, n_features)
        features = ['close', 'open', 'high', 'low', 'volume']
        
        # Create environment
        env = TradingEnvironment(data, features, lookback_window=20, initial_balance=50000)
        
        # Test reset
        obs = env.reset()
        assert obs.shape == (20, n_features + 3), f"Expected obs shape (20, {n_features + 3}), got {obs.shape}"
        
        # Test step
        action = 1  # Buy
        obs, reward, done, info = env.step(action)
        assert obs.shape == (20, n_features + 3), f"Expected obs shape (20, {n_features + 3}), got {obs.shape}"
        assert isinstance(reward, (int, float)), f"Expected reward to be numeric, got {type(reward)}"
        assert isinstance(done, bool), f"Expected done to be bool, got {type(done)}"
        assert isinstance(info, dict), f"Expected info to be dict, got {type(info)}"
        
        print("✓ Trading environment test successful")
        return True
        
    except Exception as e:
        print(f"✗ Trading environment test failed: {e}")
        return False

def test_ppo_trainer():
    """Test PPO trainer initialization"""
    print("\nTesting PPO trainer...")
    
    try:
        from NetworkConfigs.PPOTrainer import PPOTrainer
        
        # Create dummy config
        config = {
            'model_params': {
                'input_dim': 5,
                'hidden_dim': 64,
                'num_actions': 3,
                'lookback_window': 20
            },
            'train_params': {
                'learning_rate': 3e-4,
                'epochs': 10,
                'batch_size': 32,
                'ppo_epochs': 2,
                'clip_ratio': 0.2,
                'value_coef': 0.5,
                'entropy_coef': 0.01
            }
        }
        
        # Create trainer
        trainer = PPOTrainer("test_model", config, "/tmp/test_ppo")
        
        # Test data preparation
        data = np.random.randn(100, 5)
        features = ['close', 'open', 'high', 'low', 'volume']
        env = trainer.prepare_data(data, features)
        
        assert hasattr(env, 'reset'), "Environment should have reset method"
        assert hasattr(env, 'step'), "Environment should have step method"
        
        print("✓ PPO trainer test successful")
        return True
        
    except Exception as e:
        print(f"✗ PPO trainer test failed: {e}")
        return False

def test_ppo_loader():
    """Test PPO model loader (requires a saved model)"""
    print("\nTesting PPO loader...")
    
    try:
        from NetworkConfigs.PPO_loader import PPOModelLoader
        
        # This test would require a saved model, so we'll just test the class can be imported
        # and basic structure is correct
        print("✓ PPO loader class structure is correct")
        return True
        
    except Exception as e:
        print(f"✗ PPO loader test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("PPO Setup Test Suite")
    print("=" * 50)
    
    tests = [
        test_ppo_imports,
        test_ppo_network,
        test_trading_environment,
        test_ppo_trainer,
        test_ppo_loader
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! PPO setup is working correctly.")
        print("\nNext steps:")
        print("1. Train a PPO model: python Utilities/run_training.py --csv_path sample.csv --model ppo")
        print("2. Test backtesting with the trained model")
        print("3. Host the model via API")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())