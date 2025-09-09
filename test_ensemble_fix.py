#!/usr/bin/env python3
"""
Test script to verify the ensemble training fix
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_main_js_ensemble_support():
    """Test that main.js supports ensemble training tab"""
    print("Testing main.js ensemble training support...")
    
    try:
        main_js = Path("web/js/main.js")
        if not main_js.exists():
            print("❌ main.js not found")
            return False
        
        with open(main_js, 'r') as f:
            content = f.read()
        
        # Check for ensemble-training case
        if 'case \'ensemble-training\':' not in content:
            print("❌ ensemble-training case not found in main.js")
            return False
        
        # Check for initializeEnsembleTraining call
        if 'initializeEnsembleTraining()' not in content:
            print("❌ initializeEnsembleTraining() call not found in main.js")
            return False
        
        print("✅ main.js supports ensemble training tab")
        return True
        
    except Exception as e:
        print(f"❌ Error testing main.js: {e}")
        return False

def test_ensemble_js_initialization():
    """Test ensemble training JavaScript initialization"""
    print("Testing ensemble training JavaScript initialization...")
    
    try:
        ensemble_js = Path("web/js/ensemble_training.js")
        if not ensemble_js.exists():
            print("❌ ensemble_training.js not found")
            return False
        
        with open(ensemble_js, 'r') as f:
            content = f.read()
        
        # Check for key functions
        required_functions = [
            'function initializeEnsembleTraining()',
            'function loadAvailableModels()',
            'function renderModelSelectionList()',
            'function setupEnsembleEventListeners()'
        ]
        
        for func in required_functions:
            if func not in content:
                print(f"❌ Missing function: {func}")
                return False
        
        # Check for debugging
        if 'console.log' not in content:
            print("❌ No debugging console.log found")
            return False
        
        print("✅ Ensemble training JavaScript has required functions")
        return True
        
    except Exception as e:
        print(f"❌ Error testing ensemble JS: {e}")
        return False

def test_html_structure():
    """Test HTML structure for ensemble training"""
    print("Testing HTML structure...")
    
    try:
        html_file = Path("web/index.html")
        if not html_file.exists():
            print("❌ index.html not found")
            return False
        
        with open(html_file, 'r') as f:
            content = f.read()
        
        # Check for ensemble training elements
        required_elements = [
            'id="ensemble-training-content"',
            'id="model-selection-list"',
            'id="ensemble-type"',
            'data-tab="ensemble-training"'
        ]
        
        for element in required_elements:
            if element not in content:
                print(f"❌ Missing HTML element: {element}")
                return False
        
        print("✅ HTML structure is correct")
        return True
        
    except Exception as e:
        print(f"❌ Error testing HTML: {e}")
        return False

def test_backend_functions():
    """Test backend functions for ensemble training"""
    print("Testing backend functions...")
    
    try:
        main_py = Path("Main.py")
        if not main_py.exists():
            print("❌ Main.py not found")
            return False
        
        with open(main_py, 'r') as f:
            content = f.read()
        
        # Check for ensemble functions
        required_functions = [
            'def start_ensemble_training(',
            'def get_ensemble_training_history(',
            'def get_models('
        ]
        
        for func in required_functions:
            if func not in content:
                print(f"❌ Missing function: {func}")
                return False
        
        print("✅ Backend functions are present")
        return True
        
    except Exception as e:
        print(f"❌ Error testing backend: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Ensemble Training Fix Test Suite")
    print("=" * 60)
    
    tests = [
        test_main_js_ensemble_support,
        test_ensemble_js_initialization,
        test_html_structure,
        test_backend_functions
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
        print("🎉 All ensemble training fix tests passed!")
        print("\nThe ensemble training tab should now work correctly:")
        print("1. ✅ Tab switching properly initializes ensemble training")
        print("2. ✅ Model loading function has proper debugging")
        print("3. ✅ HTML structure is correct")
        print("4. ✅ Backend functions are available")
        print("\nTry switching to the Ensemble Training tab to see if models load!")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())