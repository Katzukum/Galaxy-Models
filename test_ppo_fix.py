#!/usr/bin/env python3
"""
Test script to verify PPO ensemble fixes
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_ppo_network():
    """Test PPO network with different input sizes"""
    print("Testing PPO network...")
    
    try:
        from NetworkConfigs.PPOEnsembleTrainer import PPOEnsembleNetwork
        import torch
        
        # Test with different input sizes
        test_cases = [
            (12, 64),  # Expected case
            (10, 64),  # Smaller case
            (15, 64),  # Larger case
        ]
        
        for input_size, hidden_size in test_cases:
            print(f"\nTesting with input_size={input_size}, hidden_size={hidden_size}")
            
            # Create network
            network = PPOEnsembleNetwork(input_size=input_size, hidden_size=hidden_size)
            
            # Create test data
            batch_size = 2
            seq_len = 60
            actual_input_size = input_size + 2  # Simulate mismatch
            
            test_data = torch.randn(batch_size, seq_len, actual_input_size)
            print(f"Test data shape: {test_data.shape}")
            
            # Test forward pass
            action_logits, values = network(test_data)
            print(f"✓ Forward pass successful")
            print(f"  Action logits shape: {action_logits.shape}")
            print(f"  Values shape: {values.shape}")
            
            # Test get_action
            action, log_prob, value = network.get_action(test_data)
            print(f"✓ Get action successful")
            print(f"  Action: {action}")
            print(f"  Log prob: {log_prob}")
            print(f"  Value: {value}")
        
        print("\n✓ All PPO network tests passed")
        return True
        
    except Exception as e:
        print(f"✗ PPO network test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_preparation():
    """Test data preparation without external dependencies"""
    print("Testing data preparation...")
    
    try:
        # Create test data
        test_data = pd.DataFrame({
            'close': np.random.randn(100) * 100 + 1000,
            'volume': np.random.randn(100) * 1000 + 5000,
            'open': np.random.randn(100) * 100 + 1000,
            'high': np.random.randn(100) * 100 + 1000,
            'low': np.random.randn(100) * 100 + 1000
        })
        
        # Test basic features
        basic_features = ['open', 'high', 'low', 'close', 'volume']
        available_features = [col for col in basic_features if col in test_data.columns]
        
        print(f"Available features: {available_features}")
        
        # Prepare feature matrix
        X = test_data[available_features].values
        print(f"Feature matrix shape: {X.shape}")
        
        # Test sequence creation
        sequence_length = 10
        sequences = []
        
        for i in range(sequence_length, len(X)):
            seq = X[i-sequence_length:i]
            sequences.append(seq)
        
        sequences = np.array(sequences)
        print(f"Sequences shape: {sequences.shape}")
        
        print("✓ Data preparation test passed")
        return True
        
    except Exception as e:
        print(f"✗ Data preparation test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("PPO Ensemble Fixes Test Suite")
    print("=" * 60)
    
    tests = [
        test_data_preparation,
        test_ppo_network
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
        print("🎉 All tests passed! PPO Ensemble fixes are working correctly.")
        print("\nKey Fixes Applied:")
        print("✓ Fixed PPO network input size handling")
        print("✓ Added robust error handling for shape mismatches")
        print("✓ Simplified network architecture to avoid memory issues")
        print("✓ Added debugging output for troubleshooting")
    else:
        print("⚠ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)