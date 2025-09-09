#!/usr/bin/env python3
"""
Test script to verify the ensemble training debug code
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_debug_code_added():
    """Test that debug code has been added to all files"""
    print("Testing debug code implementation...")
    
    # Test main.js debug code
    main_js = Path("web/js/main.js")
    if main_js.exists():
        with open(main_js, 'r', encoding='utf-8') as f:
            content = f.read()
        
        debug_checks = [
            'console.log(`[MAIN] Switching to tab:',
            'console.log(`[MAIN] loadTabContent called for:',
            'console.log(\'[MAIN] Loading ensemble-training content\')',
            'console.log(\'[MAIN] Checking for initializeEnsembleTraining function...\')'
        ]
        
        for check in debug_checks:
            if check not in content:
                print(f"❌ Missing debug code in main.js: {check}")
                return False
        
        print("✅ Debug code added to main.js")
    else:
        print("❌ main.js not found")
        return False
    
    # Test ensemble_training.js debug code
    ensemble_js = Path("web/js/ensemble_training.js")
    if ensemble_js.exists():
        with open(ensemble_js, 'r', encoding='utf-8') as f:
            content = f.read()
        
        debug_checks = [
            'console.log(\'[ENSEMBLE] DOM loaded, setting up ensemble training...\')',
            'console.log(\'[ENSEMBLE] loadAvailableModels called\')',
            'console.log(\'[ENSEMBLE] renderModelSelectionList called\')',
            'console.log(\'[ENSEMBLE] showModelSelectionError called with message:\', message)'
        ]
        
        for check in debug_checks:
            if check not in content:
                print(f"❌ Missing debug code in ensemble_training.js: {check}")
                return False
        
        print("✅ Debug code added to ensemble_training.js")
    else:
        print("❌ ensemble_training.js not found")
        return False
    
    # Test Main.py debug code
    main_py = Path("Main.py")
    if main_py.exists():
        with open(main_py, 'r', encoding='utf-8') as f:
            content = f.read()
        
        debug_checks = [
            'debug_print("get_models() called from frontend")',
            'debug_print(f"Models path: {models_path}")',
            'debug_print(f"Found {len(yaml_files)} YAML files: {yaml_files}")'
        ]
        
        for check in debug_checks:
            if check not in content:
                print(f"❌ Missing debug code in Main.py: {check}")
                return False
        
        print("✅ Debug code added to Main.py")
    else:
        print("❌ Main.py not found")
        return False
    
    return True

def test_html_structure():
    """Test that HTML structure is correct"""
    print("Testing HTML structure...")
    
    html_file = Path("web/index.html")
    if not html_file.exists():
        print("❌ index.html not found")
        return False
    
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for ensemble training elements
    required_elements = [
        'id="ensemble-training-content"',
        'id="model-selection-list"',
        'data-tab="ensemble-training"'
    ]
    
    for element in required_elements:
        if element not in content:
            print(f"❌ Missing HTML element: {element}")
            return False
    
    print("✅ HTML structure is correct")
    return True

def test_js_loading_order():
    """Test that JavaScript files are loaded in correct order"""
    print("Testing JavaScript loading order...")
    
    html_file = Path("web/index.html")
    if not html_file.exists():
        print("❌ index.html not found")
        return False
    
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check that ensemble_training.js is loaded after main.js
    main_js_pos = content.find('js/main.js')
    ensemble_js_pos = content.find('js/ensemble_training.js')
    
    if main_js_pos == -1:
        print("❌ main.js not found in HTML")
        return False
    
    if ensemble_js_pos == -1:
        print("❌ ensemble_training.js not found in HTML")
        return False
    
    if ensemble_js_pos < main_js_pos:
        print("❌ ensemble_training.js loaded before main.js")
        return False
    
    print("✅ JavaScript loading order is correct")
    return True

def main():
    """Run all debug tests"""
    print("=" * 60)
    print("Ensemble Training Debug Test Suite")
    print("=" * 60)
    
    tests = [
        test_debug_code_added,
        test_html_structure,
        test_js_loading_order
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
        print("🎉 All debug code tests passed!")
        print("\nDebug code has been added to:")
        print("1. ✅ main.js - Tab switching and function detection")
        print("2. ✅ ensemble_training.js - Model loading and rendering")
        print("3. ✅ Main.py - Backend model loading")
        print("4. ✅ HTML structure is correct")
        print("5. ✅ JavaScript loading order is correct")
        print("\nNow when you switch to the Ensemble Training tab, check the browser console for detailed debug output!")
        print("The debug messages will help identify exactly where the issue is occurring.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())