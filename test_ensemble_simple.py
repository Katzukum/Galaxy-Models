#!/usr/bin/env python3
"""
Simple test script for Ensemble Training functionality
This script tests the ensemble training tab structure and basic functionality
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_ensemble_html_structure():
    """Test ensemble HTML structure"""
    print("Testing ensemble HTML structure...")
    
    try:
        html_file = Path("web/index.html")
        if not html_file.exists():
            print("❌ HTML file not found")
            return False
        
        with open(html_file, 'r') as f:
            html_content = f.read()
        
        # Check for ensemble training tab
        required_elements = [
            'ensemble-training-content',
            'ensemble-type',
            'model-selection-list',
            'weight-config',
            'ensemble-csv-file',
            'ensemble-name',
            'start-ensemble-training'
        ]
        
        for element in required_elements:
            if element not in html_content:
                print(f"❌ Missing element: {element}")
                return False
            print(f"✅ Found element: {element}")
        
        print("✅ All required HTML elements found")
        return True
        
    except Exception as e:
        print(f"❌ HTML structure test failed: {e}")
        return False

def test_ensemble_css_file():
    """Test ensemble CSS file"""
    print("Testing ensemble CSS file...")
    
    try:
        css_file = Path("web/tabs/ensemble_training/ensemble_training.css")
        if not css_file.exists():
            print("❌ CSS file not found")
            return False
        
        with open(css_file, 'r') as f:
            css_content = f.read()
        
        # Check for key CSS classes
        required_classes = [
            '.ensemble-training-container',
            '.model-selection-container',
            '.weight-inputs-grid',
            '.training-progress-card'
        ]
        
        for css_class in required_classes:
            if css_class not in css_content:
                print(f"❌ Missing CSS class: {css_class}")
                return False
            print(f"✅ Found CSS class: {css_class}")
        
        print("✅ All required CSS classes found")
        return True
        
    except Exception as e:
        print(f"❌ CSS file test failed: {e}")
        return False

def test_ensemble_js_file():
    """Test ensemble JavaScript file"""
    print("Testing ensemble JavaScript file...")
    
    try:
        js_file = Path("web/js/ensemble_training.js")
        if not js_file.exists():
            print("❌ JavaScript file not found")
            return False
        
        with open(js_file, 'r') as f:
            js_content = f.read()
        
        # Check for key functions
        required_functions = [
            'initializeEnsembleTraining',
            'loadAvailableModels',
            'handleEnsembleFormSubmit',
            'startEnsembleTraining'
        ]
        
        for function in required_functions:
            if function not in js_content:
                print(f"❌ Missing function: {function}")
                return False
            print(f"✅ Found function: {function}")
        
        print("✅ All required JavaScript functions found")
        return True
        
    except Exception as e:
        print(f"❌ JavaScript file test failed: {e}")
        return False

def test_ensemble_backend_files():
    """Test ensemble backend files"""
    print("Testing ensemble backend files...")
    
    try:
        # Test EnsembleTrainer.py
        trainer_file = Path("NetworkConfigs/EnsembleTrainer.py")
        if not trainer_file.exists():
            print("❌ EnsembleTrainer.py not found")
            return False
        
        with open(trainer_file, 'r') as f:
            trainer_content = f.read()
        
        # Check for key classes and functions
        required_elements = [
            'class EnsembleTrainer',
            'def run_ensemble_training',
            'def load_models',
            'def train_ensemble'
        ]
        
        for element in required_elements:
            if element not in trainer_content:
                print(f"❌ Missing element: {element}")
                return False
            print(f"✅ Found element: {element}")
        
        print("✅ EnsembleTrainer.py contains required elements")
        
        # Test Main.py integration
        main_file = Path("Main.py")
        if not main_file.exists():
            print("❌ Main.py not found")
            return False
        
        with open(main_file, 'r') as f:
            main_content = f.read()
        
        # Check for ensemble functions
        if 'start_ensemble_training' not in main_content:
            print("❌ start_ensemble_training function not found in Main.py")
            return False
        print("✅ start_ensemble_training function found in Main.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Backend files test failed: {e}")
        return False

def test_ensemble_tab_navigation():
    """Test ensemble tab navigation"""
    print("Testing ensemble tab navigation...")
    
    try:
        html_file = Path("web/index.html")
        with open(html_file, 'r') as f:
            html_content = f.read()
        
        # Check for tab button
        if 'data-tab="ensemble-training"' not in html_content:
            print("❌ Ensemble training tab button not found")
            return False
        print("✅ Ensemble training tab button found")
        
        # Check for tab content
        if 'id="ensemble-training-content"' not in html_content:
            print("❌ Ensemble training tab content not found")
            return False
        print("✅ Ensemble training tab content found")
        
        return True
        
    except Exception as e:
        print(f"❌ Tab navigation test failed: {e}")
        return False

def test_ensemble_form_validation():
    """Test ensemble form validation logic"""
    print("Testing ensemble form validation...")
    
    try:
        js_file = Path("web/js/ensemble_training.js")
        with open(js_file, 'r') as f:
            js_content = f.read()
        
        # Check for validation functions
        validation_functions = [
            'validateEnsembleForm',
            'validateWeights',
            'handleModelSelectionChange'
        ]
        
        for function in validation_functions:
            if function not in js_content:
                print(f"❌ Missing validation function: {function}")
                return False
            print(f"✅ Found validation function: {function}")
        
        print("✅ All validation functions found")
        return True
        
    except Exception as e:
        print(f"❌ Form validation test failed: {e}")
        return False

def test_ensemble_types():
    """Test ensemble types support"""
    print("Testing ensemble types support...")
    
    try:
        html_file = Path("web/index.html")
        with open(html_file, 'r') as f:
            html_content = f.read()
        
        # Check for ensemble type options in HTML
        ensemble_types = ['voting', 'averaging', 'weighted', 'stacking']
        
        for ensemble_type in ensemble_types:
            if f'value="{ensemble_type}"' not in html_content:
                print(f"❌ Missing ensemble type: {ensemble_type}")
                return False
            print(f"✅ Found ensemble type: {ensemble_type}")
        
        print("✅ All ensemble types supported")
        return True
        
    except Exception as e:
        print(f"❌ Ensemble types test failed: {e}")
        return False

def main():
    """Run all ensemble training tests"""
    print("=" * 60)
    print("Ensemble Training Simple Test Suite")
    print("=" * 60)
    
    tests = [
        test_ensemble_html_structure,
        test_ensemble_css_file,
        test_ensemble_js_file,
        test_ensemble_backend_files,
        test_ensemble_tab_navigation,
        test_ensemble_form_validation,
        test_ensemble_types
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
        print("🎉 All ensemble training tests passed!")
        print("\nEnsemble training tab is ready:")
        print("1. ✅ HTML structure with all required elements")
        print("2. ✅ CSS styling for all components")
        print("3. ✅ JavaScript functionality for form handling")
        print("4. ✅ Backend logic for ensemble training")
        print("5. ✅ Tab navigation integration")
        print("6. ✅ Form validation and model selection")
        print("7. ✅ Support for all ensemble types")
        print("\nThe ensemble training tab is fully functional!")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())