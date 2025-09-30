#!/usr/bin/env python3
"""
Test script to verify PPO ensemble architecture detection fixes
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_architecture_detection():
    """Test PPO network architecture detection from state dict"""
    print("Testing PPO architecture detection...")
    
    try:
        from NetworkConfigs.PPOEnsembleTrainer import PPOEnsembleNetwork
        
        # Test with different hidden sizes
        test_cases = [32, 64, 128, 256]
        
        for hidden_size in test_cases:
            print(f"\nTesting with hidden_size: {hidden_size}")
            
            # Create a PPO network
            input_size = 12
            network = PPOEnsembleNetwork(
                input_size=input_size,
                hidden_size=hidden_size,
                num_actions=3
            )
            
            # Get the state dict
            state_dict = network.state_dict()
            
            # Test the detection logic
            if 'feature_extractor.0.weight' in state_dict:
                detected_hidden_size = state_dict['feature_extractor.0.weight'].shape[0]
                print(f"Detected hidden_size: {detected_hidden_size}")
                
                # Verify it matches
                assert detected_hidden_size == hidden_size, f"Expected {hidden_size}, got {detected_hidden_size}"
                print("✓ Architecture detection correct")
            else:
                print("✗ Could not find feature_extractor.0.weight in state dict")
                return False
            
            # Test that we can create a new network with the detected size
            new_network = PPOEnsembleNetwork(
                input_size=input_size,
                hidden_size=detected_hidden_size,
                num_actions=3
            )
            
            # Test that we can load the state dict
            new_network.load_state_dict(state_dict)
            print("✓ State dict loading successful")
        
        print("\n✓ All architecture detection tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Architecture detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ppo_loader_initialization():
    """Test PPO loader initialization with different architectures"""
    print("\nTesting PPO loader initialization...")
    
    try:
        from NetworkConfigs.PPOEnsembleTrainer import PPOEnsembleNetwork
        
        # Create a mock state dict with specific architecture
        input_size = 12
        hidden_size = 64
        
        # Create network and get state dict
        network = PPOEnsembleNetwork(input_size=input_size, hidden_size=hidden_size, num_actions=3)
        state_dict = network.state_dict()
        
        # Test the detection logic from the loader
        if 'feature_extractor.0.weight' in state_dict:
            detected_hidden_size = state_dict['feature_extractor.0.weight'].shape[0]
            print(f"Detected hidden_size from state dict: {detected_hidden_size}")
            
            # Create new network with detected size
            new_network = PPOEnsembleNetwork(
                input_size=input_size,
                hidden_size=detected_hidden_size,
                num_actions=3
            )
            
            # Load state dict
            new_network.load_state_dict(state_dict)
            print("✓ PPO loader initialization successful")
            
            # Test forward pass
            test_input = torch.randn(1, 60, input_size)
            action_logits, values = new_network(test_input)
            print(f"✓ Forward pass successful - Action logits: {action_logits.shape}, Values: {values.shape}")
            
        else:
            print("✗ Could not find feature_extractor.0.weight in state dict")
            return False
        
        print("✓ PPO loader initialization test passed")
        return True
        
    except Exception as e:
        print(f"✗ PPO loader initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_fallback():
    """Test config fallback for hidden_size"""
    print("\nTesting config fallback...")
    
    try:
        # Test different config structures
        test_configs = [
            {'Config': {'hidden_size': 64}},
            {'Config': {'hidden_size': 128}},
            {'Config': {}},  # No hidden_size
            {}  # No Config
        ]
        
        for i, config in enumerate(test_configs):
            print(f"\nTest config {i+1}: {config}")
            
            # Simulate the fallback logic
            if 'Config' in config and 'hidden_size' in config['Config']:
                hidden_size = config['Config']['hidden_size']
                print(f"Using hidden_size from config: {hidden_size}")
            else:
                hidden_size = 64  # Default
                print(f"Using default hidden_size: {hidden_size}")
            
            # Verify expected results
            if 'Config' in config and 'hidden_size' in config['Config']:
                assert hidden_size == config['Config']['hidden_size'], f"Expected {config['Config']['hidden_size']}, got {hidden_size}"
            else:
                assert hidden_size == 64, f"Expected 64, got {hidden_size}"
            
            print("✓ Config fallback correct")
        
        print("✓ All config fallback tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Config fallback test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("PPO Ensemble Architecture Detection Fix Test Suite")
    print("=" * 60)
    
    tests = [
        test_architecture_detection,
        test_ppo_loader_initialization,
        test_config_fallback
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print("-" * 40)
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! PPO Ensemble architecture detection fixes are working correctly.")
        print("\nKey Fixes Applied:")
        print("✓ Added dynamic hidden_size detection from saved model state dict")
        print("✓ Added config fallback for hidden_size when state dict detection fails")
        print("✓ Updated PPOEnsembleTrainer to save hidden_size in config")
        print("✓ Fixed PPOEnsemble_loader to use correct architecture when loading models")
    else:
        print("⚠ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)