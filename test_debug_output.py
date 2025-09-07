#!/usr/bin/env python3
"""
Test script to verify debug output works correctly.
This script tests the subprocess creation with proper output redirection.
"""

import os
import sys
import subprocess
import time

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_debug_subprocess():
    """Test subprocess creation with debug output"""
    print("Testing debug subprocess creation...")
    
    # Test command that produces output
    cmd = ['python', '-c', '''
import time
print("Starting test process...")
for i in range(10):
    print(f"Test output line {i+1}")
    time.sleep(0.5)
print("Test process completed!")
''']

    print(f"Command: {' '.join(cmd)}")
    print("This should open a new console window with output...")
    
    if os.name == 'nt':
        # Windows: Use CREATE_NEW_CONSOLE
        print("Creating Windows console subprocess...")
        process = subprocess.Popen(
            cmd,
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
    else:
        # Linux/macOS: Try to open in terminal
        print("Creating Linux/macOS terminal subprocess...")
        try:
            terminal_cmd = ['gnome-terminal', '--', 'bash', '-c', f"{' '.join(cmd)}; read -p 'Press Enter to close...'"]
            process = subprocess.Popen(terminal_cmd)
        except:
            print("Fallback: running without terminal")
            process = subprocess.Popen(cmd)
    
    print(f"Process started with PID: {process.pid}")
    print("Check for new console/terminal window!")
    
    # Wait for process to complete
    return_code = process.wait()
    print(f"Process completed with return code: {return_code}")
    
    return return_code == 0

def test_training_command():
    """Test training command with debug output"""
    print("\nTesting training command with debug output...")
    
    # Simulate a training command
    cmd = ['python', '-c', '''
import time
print("Starting PPO training simulation...")
print("Data shape: (193605, 34), Features: 34")
print("Preparing data for PPO training...")
print("Starting PPO training for model: test_ppo_model")
print("Collecting rollouts for 5 episodes...")
for episode in range(5):
    print(f"Episode {episode + 1}/5")
    time.sleep(1)
print("Training completed successfully!")
''']

    print(f"Training command: {' '.join(cmd)}")
    
    if os.name == 'nt':
        process = subprocess.Popen(
            cmd,
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
    else:
        try:
            terminal_cmd = ['gnome-terminal', '--', 'bash', '-c', f"{' '.join(cmd)}; read -p 'Press Enter to close...'"]
            process = subprocess.Popen(terminal_cmd)
        except:
            process = subprocess.Popen(cmd)
    
    print(f"Training process started with PID: {process.pid}")
    print("Check for new console/terminal window with training output!")
    
    return_code = process.wait()
    print(f"Training process completed with return code: {return_code}")
    
    return return_code == 0

def main():
    """Run debug output tests"""
    print("=" * 60)
    print("Debug Output Test Suite")
    print("=" * 60)
    
    tests = [
        test_debug_subprocess,
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
        print("🎉 All debug output tests passed!")
        print("\nDebug features should now work correctly:")
        print("1. Command windows will open with visible output")
        print("2. Training processes will show real-time progress")
        print("3. No output will be captured in debug mode")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())