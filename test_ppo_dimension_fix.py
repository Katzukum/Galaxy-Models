#!/usr/bin/env python3
"""
Test script to verify the PPO dimension fix.
This script tests the dimension calculations to ensure they match.
"""

import numpy as np
import torch
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_ppo_dimensions():
    """Test PPO dimension calculations"""
    print("Testing PPO dimension calculations...")
    
    # Simulate the data from the error
    n_samples = 1000
    n_features = 34  # As shown in the error log
    data = np.random.randn(n_samples, n_features)
    features = [f'feature_{i}' for i in range(n_features)]
    
    print(f"Data shape: {data.shape}")
    print(f"Number of features: {len(features)}")
    
    # Test environment observation space
    from NetworkConfigs.PPOTrainer import TradingEnvironment
    
    lookback_window = 60
    env = TradingEnvironment(data, features, lookback_window=lookback_window)
    
    print(f"Environment observation space shape: {env.observation_space.shape}")
    print(f"Expected: ({lookback_window}, {n_features + 3})")
    print(f"Actual: {env.observation_space.shape}")
    
    # Test observation generation
    obs = env.reset()
    print(f"Generated observation shape: {obs.shape}")
    print(f"Expected: ({lookback_window}, {n_features + 3})")
    print(f"Actual: {obs.shape}")
    
    # Test network initialization
    from NetworkConfigs.PPOTrainer import PPONetwork
    
    data_input_dim = n_features  # Raw data features
    input_dim = data_input_dim + 3  # +3 for account state
    hidden_dim = 128
    num_actions = 3
    
    print(f"Network input_dim: {input_dim}")
    print(f"Expected: {n_features + 3}")
    
    network = PPONetwork(input_dim=input_dim, hidden_dim=hidden_dim, num_actions=num_actions)
    
    # Test forward pass
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)  # Add batch dimension
    print(f"Input tensor shape: {obs_tensor.shape}")
    print(f"Expected: (1, {lookback_window}, {n_features + 3})")
    
    try:
        action_logits, value = network(obs_tensor)
        print(f"✅ Forward pass successful!")
        print(f"Action logits shape: {action_logits.shape}")
        print(f"Value shape: {value.shape}")
        print(f"Expected action logits: (1, {num_actions})")
        print(f"Expected value: (1, 1)")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    return True

def test_ppo_trainer_initialization():
    """Test PPO trainer initialization"""
    print("\nTesting PPO trainer initialization...")
    
    from NetworkConfigs.PPOTrainer import PPOTrainer
    
    # Simulate config from the error
    config = {
        'model_params': {
            'input_dim': 34,  # Raw data features
            'hidden_dim': 128,
            'num_actions': 3,
            'lookback_window': 60
        },
        'train_params': {
            'learning_rate': 0.0003,
            'epochs': 100,
            'batch_size': 64,
            'ppo_epochs': 4,
            'clip_ratio': 0.2,
            'value_coef': 0.5,
            'entropy_coef': 0.01
        }
    }
    
    trainer = PPOTrainer("test_model", config, "/tmp/test_ppo")
    
    print(f"Trainer data_input_dim: {trainer.data_input_dim}")
    print(f"Trainer input_dim: {trainer.input_dim}")
    print(f"Expected data_input_dim: 34")
    print(f"Expected input_dim: 37")
    
    # Test data preparation
    n_samples = 1000
    n_features = 34
    data = np.random.randn(n_samples, n_features)
    features = [f'feature_{i}' for i in range(n_features)]
    
    try:
        env = trainer.prepare_data(data, features)
        print(f"✅ Data preparation successful!")
        print(f"Environment observation space: {env.observation_space.shape}")
        
        # Test observation
        obs = env.reset()
        print(f"Observation shape: {obs.shape}")
        
        # Test network forward pass
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        action_logits, value = trainer.network(obs_tensor)
        print(f"✅ Network forward pass successful!")
        print(f"Action logits shape: {action_logits.shape}")
        print(f"Value shape: {value.shape}")
        
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("PPO Dimension Fix Test Suite")
    print("=" * 60)
    
    tests = [
        test_ppo_dimensions,
        test_ppo_trainer_initialization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The dimension fix should work.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())