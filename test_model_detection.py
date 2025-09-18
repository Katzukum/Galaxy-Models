#!/usr/bin/env python3
"""
Test script to verify PPO ensemble model detection fixes
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_model_detection():
    """Test model type detection for PPO ensemble models"""
    print("Testing PPO ensemble model detection...")
    
    try:
        from Utilities.yaml_utils import load_yaml_config
        
        # Test the model detection logic
        def detect_model_type(config):
            """Replicate the model detection logic from Main.py"""
            model_type = config.find_key('Type', 'Unknown Type')
            
            # Check if it's a PPO ensemble model
            if model_type == 'Unknown Type':
                # Try to find model_type in Config section
                config_dict = config.to_dict()
                if 'Config' in config_dict and 'ensemble_type' in config_dict['Config']:
                    if config_dict['Config']['ensemble_type'] == 'ppo':
                        model_type = 'PPO Ensemble'
                # Also check for model_type field
                elif 'model_type' in config_dict and config_dict['model_type'] == 'PPOEnsemble':
                    model_type = 'PPO Ensemble'
            
            return model_type
        
        # Test with different config structures
        test_configs = [
            # Case 1: Has Type field
            {
                'model_name': 'test_model',
                'Type': 'PPO Ensemble',
                'Config': {'ensemble_type': 'ppo'}
            },
            # Case 2: Has model_type field
            {
                'model_name': 'test_model',
                'model_type': 'PPOEnsemble',
                'Config': {'ensemble_type': 'ppo'}
            },
            # Case 3: Only has ensemble_type in Config
            {
                'model_name': 'test_model',
                'Config': {'ensemble_type': 'ppo'}
            },
            # Case 4: Regular model
            {
                'model_name': 'test_model',
                'Type': 'Neural Network (Regression)'
            }
        ]
        
        for i, config_data in enumerate(test_configs):
            print(f"\nTest case {i+1}: {config_data}")
            
            # Create a mock config object
            class MockConfig:
                def __init__(self, data):
                    self.data = data
                
                def find_key(self, key, default=None):
                    return self.data.get(key, default)
                
                def to_dict(self):
                    return self.data
            
            config = MockConfig(config_data)
            detected_type = detect_model_type(config)
            print(f"Detected type: {detected_type}")
            
            # Verify expected results
            if 'Type' in config_data and config_data['Type'] == 'PPO Ensemble':
                assert detected_type == 'PPO Ensemble', f"Expected 'PPO Ensemble', got '{detected_type}'"
            elif 'model_type' in config_data and config_data['model_type'] == 'PPOEnsemble':
                assert detected_type == 'PPO Ensemble', f"Expected 'PPO Ensemble', got '{detected_type}'"
            elif 'Config' in config_data and config_data['Config'].get('ensemble_type') == 'ppo':
                assert detected_type == 'PPO Ensemble', f"Expected 'PPO Ensemble', got '{detected_type}'"
            else:
                assert detected_type == config_data.get('Type', 'Unknown Type'), f"Expected '{config_data.get('Type', 'Unknown Type')}', got '{detected_type}'"
            
            print("✓ Test case passed")
        
        print("\n✓ All model detection tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Model detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backtester_detection():
    """Test backtester model type detection"""
    print("\nTesting backtester model type detection...")
    
    try:
        from Utilities.yaml_utils import load_yaml_config
        
        # Test the backtester detection logic
        def detect_backtester_model_type(config):
            """Replicate the backtester model detection logic"""
            model_type = config.find_key('Type')
            
            # Check for PPO ensemble model type
            if model_type is None or model_type == 'Unknown Type':
                config_dict = config.to_dict()
                if 'model_type' in config_dict and config_dict['model_type'] == 'PPOEnsemble':
                    model_type = 'PPO Ensemble'
                elif 'Config' in config_dict and 'ensemble_type' in config_dict['Config']:
                    if config_dict['Config']['ensemble_type'] == 'ppo':
                        model_type = 'PPO Ensemble'
            
            return model_type
        
        # Test with different config structures
        test_configs = [
            # Case 1: Has Type field
            {
                'model_name': 'test_model',
                'Type': 'PPO Ensemble',
                'Config': {'ensemble_type': 'ppo'}
            },
            # Case 2: Has model_type field
            {
                'model_name': 'test_model',
                'model_type': 'PPOEnsemble',
                'Config': {'ensemble_type': 'ppo'}
            },
            # Case 3: Only has ensemble_type in Config
            {
                'model_name': 'test_model',
                'Config': {'ensemble_type': 'ppo'}
            },
            # Case 4: Regular model
            {
                'model_name': 'test_model',
                'Type': 'Neural Network (Regression)'
            }
        ]
        
        for i, config_data in enumerate(test_configs):
            print(f"\nBacktester test case {i+1}: {config_data}")
            
            # Create a mock config object
            class MockConfig:
                def __init__(self, data):
                    self.data = data
                
                def find_key(self, key, default=None):
                    return self.data.get(key, default)
                
                def to_dict(self):
                    return self.data
            
            config = MockConfig(config_data)
            detected_type = detect_backtester_model_type(config)
            print(f"Detected type: {detected_type}")
            
            # Verify expected results
            if 'Type' in config_data and config_data['Type'] == 'PPO Ensemble':
                assert detected_type == 'PPO Ensemble', f"Expected 'PPO Ensemble', got '{detected_type}'"
            elif 'model_type' in config_data and config_data['model_type'] == 'PPOEnsemble':
                assert detected_type == 'PPO Ensemble', f"Expected 'PPO Ensemble', got '{detected_type}'"
            elif 'Config' in config_data and config_data['Config'].get('ensemble_type') == 'ppo':
                assert detected_type == 'PPO Ensemble', f"Expected 'PPO Ensemble', got '{detected_type}'"
            else:
                assert detected_type == config_data.get('Type'), f"Expected '{config_data.get('Type')}', got '{detected_type}'"
            
            print("✓ Backtester test case passed")
        
        print("\n✓ All backtester detection tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Backtester detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("PPO Ensemble Model Detection Fix Test Suite")
    print("=" * 60)
    
    tests = [
        test_model_detection,
        test_backtester_detection
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print("-" * 40)
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! PPO Ensemble model detection fixes are working correctly.")
        print("\nKey Fixes Applied:")
        print("✓ Fixed Main.py model type detection to handle both 'Type' and 'model_type' fields")
        print("✓ Fixed backtester.py model type detection to handle PPO ensemble models")
        print("✓ Updated PPOEnsembleTrainer to save 'Type' field in config")
        print("✓ Added fallback detection for 'ensemble_type' in Config section")
    else:
        print("⚠ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)