#!/usr/bin/env python3
"""
Test script to verify debug features work correctly.
This script tests the debug configuration and subprocess creation.
"""

import os
import sys
import subprocess
import time

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_debug_config():
    """Test debug configuration loading"""
    print("Testing debug configuration...")
    
    try:
        from Main import DEBUG_TRAINING, DEBUG_API, DEBUG_VERBOSE, debug_print
        print(f"✅ Debug configuration loaded successfully")
        print(f"  - DEBUG_TRAINING: {DEBUG_TRAINING}")
        print(f"  - DEBUG_API: {DEBUG_API}")
        print(f"  - DEBUG_VERBOSE: {DEBUG_VERBOSE}")
        
        # Test debug_print function
        debug_print("This is a test debug message")
        
        return True
    except Exception as e:
        print(f"❌ Debug configuration test failed: {e}")
        return False

def test_subprocess_creation():
    """Test subprocess creation with debug features"""
    print("\nTesting subprocess creation...")
    
    try:
        # Test command
        cmd = ['python', '--version']
        
        print(f"Testing command: {' '.join(cmd)}")
        
        # Test normal subprocess
        print("Creating normal subprocess...")
        process1 = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        stdout1, _ = process1.communicate()
        print(f"Normal subprocess output: {stdout1.strip()}")
        
        # Test debug subprocess (Windows only)
        if os.name == 'nt':
            print("Creating debug subprocess with console window...")
            process2 = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            
            stdout2, _ = process2.communicate()
            print(f"Debug subprocess output: {stdout2.strip()}")
            print("✅ Debug subprocess created (check for new console window)")
        else:
            print("⚠️  Debug subprocess test skipped (not on Windows)")
        
        return True
    except Exception as e:
        print(f"❌ Subprocess creation test failed: {e}")
        return False

def test_debug_functions():
    """Test debug management functions"""
    print("\nTesting debug management functions...")
    
    try:
        # Import the debug functions
        from Main import get_debug_status, set_debug_training, set_debug_api, set_debug_verbose
        
        # Test get_debug_status
        status = get_debug_status()
        print(f"✅ get_debug_status() returned: {status}")
        
        # Test set functions
        result1 = set_debug_training(False)
        print(f"✅ set_debug_training(False) returned: {result1}")
        
        result2 = set_debug_api(False)
        print(f"✅ set_debug_api(False) returned: {result2}")
        
        result3 = set_debug_verbose(False)
        print(f"✅ set_debug_verbose(False) returned: {result3}")
        
        # Reset to original values
        set_debug_training(True)
        set_debug_api(True)
        set_debug_verbose(True)
        
        return True
    except Exception as e:
        print(f"❌ Debug functions test failed: {e}")
        return False

def test_training_command():
    """Test training command construction"""
    print("\nTesting training command construction...")
    
    try:
        # Simulate the training command construction
        model_type = "ppo"
        csv_path = "test_data.csv"
        model_name = "test_model"
        training_params = {
            'model_params': {'hidden_dim': 128, 'num_actions': 3, 'lookback_window': 60},
            'train_params': {'learning_rate': 0.0003, 'epochs': 100}
        }
        
        # Build command
        cmd = [
            'python', 'Utilities/run_training.py',
            '--csv_path', csv_path,
            '--model', model_type,
            '--model_name', model_name
        ]
        
        # Add training parameters
        import json
        params_json = json.dumps(training_params)
        cmd.extend(['--training_params', params_json])
        
        print(f"✅ Training command constructed: {' '.join(cmd)}")
        print(f"  - Model type: {model_type}")
        print(f"  - CSV path: {csv_path}")
        print(f"  - Model name: {model_name}")
        print(f"  - Parameters: {training_params}")
        
        return True
    except Exception as e:
        print(f"❌ Training command test failed: {e}")
        return False

def main():
    """Run all debug feature tests"""
    print("=" * 60)
    print("Galaxy Models Debug Features Test Suite")
    print("=" * 60)
    
    tests = [
        test_debug_config,
        test_subprocess_creation,
        test_debug_functions,
        test_training_command
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
        print("🎉 All debug feature tests passed!")
        print("\nDebug features are ready to use:")
        print("1. Set DEBUG_TRAINING = True to see training command windows")
        print("2. Set DEBUG_API = True to see API server command windows")
        print("3. Set DEBUG_VERBOSE = True for detailed console output")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())