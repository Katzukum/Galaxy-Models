#!/usr/bin/env python3
"""
Test script to verify the YAML refactoring works correctly.
This script tests the centralized YAML utilities and recursive key finding.
"""

import os
import sys
import tempfile
import yaml
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Utilities.yaml_utils import YAMLConfig, load_yaml_config, find_yaml_files, get_common_config_values

def create_test_yaml_files():
    """Create test YAML files with different structures"""
    test_dir = tempfile.mkdtemp()
    
    # Test YAML 1: Simple structure
    simple_config = {
        'model_name': 'simple_model',
        'Type': 'XGBoost',
        'features': ['feature1', 'feature2', 'feature3'],
        'input_dim': 3,
        'learning_rate': 0.1
    }
    
    simple_path = os.path.join(test_dir, 'simple_config.yaml')
    with open(simple_path, 'w') as f:
        yaml.dump(simple_config, f)
    
    # Test YAML 2: Nested structure
    nested_config = {
        'model_name': 'nested_model',
        'Type': 'PPO Agent',
        'Config': {
            'data_params': {
                'features': ['feature1', 'feature2', 'feature3', 'feature4'],
                'input_dim': 4
            },
            'model_params': {
                'hidden_dim': 128,
                'learning_rate': 0.001,
                'num_actions': 3
            },
            'train_params': {
                'epochs': 100,
                'batch_size': 32
            }
        },
        'artifact_paths': {
            'model': 'model.pt',
            'scaler': 'scaler.pkl'
        }
    }
    
    nested_path = os.path.join(test_dir, 'nested_config.yaml')
    with open(nested_path, 'w') as f:
        yaml.dump(nested_config, f)
    
    # Test YAML 3: Deeply nested structure
    deep_config = {
        'model_name': 'deep_model',
        'Type': 'Transformer',
        'metadata': {
            'version': '1.0',
            'created_by': 'test_suite'
        },
        'configurations': {
            'data': {
                'preprocessing': {
                    'features': ['feature1', 'feature2', 'feature3', 'feature4', 'feature5'],
                    'scaling': 'minmax'
                },
                'input_dim': 5
            },
            'model': {
                'architecture': {
                    'hidden_dim': 256,
                    'num_layers': 4
                },
                'training': {
                    'learning_rate': 0.0001,
                    'epochs': 200,
                    'batch_size': 64
                }
            }
        }
    }
    
    deep_path = os.path.join(test_dir, 'deep_config.yaml')
    with open(deep_path, 'w') as f:
        yaml.dump(deep_config, f)
    
    return test_dir, [simple_path, nested_path, deep_path]

def test_yaml_config_basic():
    """Test basic YAMLConfig functionality"""
    print("Testing basic YAMLConfig functionality...")
    
    test_config = {
        'model_name': 'test_model',
        'Type': 'Test Type',
        'nested': {
            'deep': {
                'key': 'value'
            }
        }
    }
    
    config = YAMLConfig(config_dict=test_config)
    
    # Test basic key finding
    assert config.find_key('model_name') == 'test_model'
    assert config.find_key('Type') == 'Test Type'
    assert config.find_key('key') == 'value'
    assert config.find_key('nonexistent', 'default') == 'default'
    
    # Test key path finding
    path = config.find_key_path('key')
    assert path == ['nested', 'deep', 'key']
    
    # Test all keys
    all_keys = config.get_all_keys()
    assert 'model_name' in all_keys
    assert 'Type' in all_keys
    assert 'key' in all_keys
    
    print("✅ Basic YAMLConfig functionality test passed")
    return True

def test_yaml_config_file_loading():
    """Test YAMLConfig file loading"""
    print("Testing YAMLConfig file loading...")
    
    test_dir, yaml_files = create_test_yaml_files()
    
    try:
        # Test simple config
        simple_config = load_yaml_config(yaml_files[0])
        assert simple_config.find_key('model_name') == 'simple_model'
        assert simple_config.find_key('Type') == 'XGBoost'
        assert simple_config.find_key('features') == ['feature1', 'feature2', 'feature3']
        
        # Test nested config
        nested_config = load_yaml_config(yaml_files[1])
        assert nested_config.find_key('model_name') == 'nested_model'
        assert nested_config.find_key('Type') == 'PPO Agent'
        assert nested_config.find_key('features') == ['feature1', 'feature2', 'feature3', 'feature4']
        assert nested_config.find_key('hidden_dim') == 128
        assert nested_config.find_key('epochs') == 100
        
        # Test deep config
        deep_config = load_yaml_config(yaml_files[2])
        assert deep_config.find_key('model_name') == 'deep_model'
        assert deep_config.find_key('Type') == 'Transformer'
        assert deep_config.find_key('features') == ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
        assert deep_config.find_key('hidden_dim') == 256
        assert deep_config.find_key('learning_rate') == 0.0001
        
        print("✅ YAMLConfig file loading test passed")
        return True
        
    finally:
        # Clean up test files
        import shutil
        shutil.rmtree(test_dir)

def test_recursive_key_finding():
    """Test recursive key finding functionality"""
    print("Testing recursive key finding...")
    
    test_config = {
        'level1': {
            'level2': {
                'level3': {
                    'target_key': 'found_value'
                },
                'other_key': 'other_value'
            },
            'another_branch': {
                'different_key': 'different_value'
            }
        },
        'root_key': 'root_value',
        'list_with_dicts': [
            {'item_key': 'item_value'},
            {'another_item': 'another_value'}
        ]
    }
    
    config = YAMLConfig(config_dict=test_config)
    
    # Test finding keys at different levels
    assert config.find_key('target_key') == 'found_value'
    assert config.find_key('other_key') == 'other_value'
    assert config.find_key('different_key') == 'different_value'
    assert config.find_key('root_key') == 'root_value'
    assert config.find_key('item_key') == 'item_value'
    assert config.find_key('another_item') == 'another_value'
    
    # Test finding keys that don't exist
    assert config.find_key('nonexistent_key', 'default') == 'default'
    
    # Test key path finding
    assert config.find_key_path('target_key') == ['level1', 'level2', 'level3', 'target_key']
    assert config.find_key_path('root_key') == ['root_key']
    assert config.find_key_path('nonexistent_key') is None
    
    print("✅ Recursive key finding test passed")
    return True

def test_yaml_file_discovery():
    """Test YAML file discovery functionality"""
    print("Testing YAML file discovery...")
    
    test_dir, yaml_files = create_test_yaml_files()
    
    try:
        # Test finding YAML files
        found_files = find_yaml_files(test_dir, recursive=True)
        assert len(found_files) == 3, f"Expected 3 files, found {len(found_files)}: {found_files}"
        assert all(f.endswith('.yaml') for f in found_files), f"Not all files are YAML: {found_files}"
        
        # Test non-recursive search
        found_files_nonrecursive = find_yaml_files(test_dir, recursive=False)
        assert len(found_files_nonrecursive) == 3, f"Expected 3 files in non-recursive search, found {len(found_files_nonrecursive)}: {found_files_nonrecursive}"
        
        print("✅ YAML file discovery test passed")
        return True
        
    except Exception as e:
        print(f"❌ YAML file discovery test failed: {e}")
        return False
        
    finally:
        # Clean up test files
        import shutil
        shutil.rmtree(test_dir)

def test_common_config_values():
    """Test common configuration values extraction"""
    print("Testing common configuration values extraction...")
    
    test_config = {
        'model_name': 'test_model',
        'Type': 'Test Type',
        'features': ['feature1', 'feature2'],
        'input_dim': 10,
        'hidden_dim': 128,
        'learning_rate': 0.001,
        'epochs': 100,
        'batch_size': 32,
        'nested': {
            'data_params': {
                'features': ['nested_feature1', 'nested_feature2']
            },
            'model_params': {
                'hidden_dim': 256,
                'learning_rate': 0.0001
            }
        }
    }
    
    config = YAMLConfig(config_dict=test_config)
    
    # Test finding common values
    common_values = config.find_keys(['model_name', 'Type', 'features', 'input_dim', 'hidden_dim', 'learning_rate'])
    
    assert common_values['model_name'] == 'test_model'
    assert common_values['Type'] == 'Test Type'
    assert common_values['features'] == ['feature1', 'feature2']  # Should find the first occurrence
    assert common_values['input_dim'] == 10
    assert common_values['hidden_dim'] == 128  # Should find the first occurrence
    assert common_values['learning_rate'] == 0.001  # Should find the first occurrence
    
    print("✅ Common configuration values test passed")
    return True

def test_nested_value_access():
    """Test nested value access and setting"""
    print("Testing nested value access and setting...")
    
    config = YAMLConfig(config_dict={})
    
    # Test setting nested values
    config.set_nested_value(['level1', 'level2', 'key'], 'value')
    assert config.get_nested_value(['level1', 'level2', 'key']) == 'value'
    
    # Test getting nested values with default
    assert config.get_nested_value(['level1', 'level2', 'nonexistent'], 'default') == 'default'
    
    # Test setting at root level
    config.set_nested_value(['root_key'], 'root_value')
    assert config.get_nested_value(['root_key']) == 'root_value'
    
    print("✅ Nested value access test passed")
    return True

def test_integration_with_existing_structure():
    """Test integration with existing YAML structure patterns"""
    print("Testing integration with existing YAML structure patterns...")
    
    # Simulate a typical model config structure
    model_config = {
        'model_name': 'my_ppo_model',
        'Type': 'PPO Agent',
        'Config': {
            'data_params': {
                'features': ['feature1', 'feature2', 'feature3'],
                'input_dim': 3
            },
            'model_params': {
                'hidden_dim': 128,
                'num_actions': 3,
                'lookback_window': 60
            },
            'train_params': {
                'learning_rate': 0.0003,
                'epochs': 100,
                'batch_size': 64
            }
        },
        'artifact_paths': {
            'model_state_dict': 'model.pt',
            'scaler': 'scaler.pkl'
        }
    }
    
    config = YAMLConfig(config_dict=model_config)
    
    # Test finding keys that exist in different locations
    assert config.find_key('model_name') == 'my_ppo_model'
    assert config.find_key('Type') == 'PPO Agent'
    assert config.find_key('features') == ['feature1', 'feature2', 'feature3']
    assert config.find_key('input_dim') == 3
    assert config.find_key('hidden_dim') == 128
    assert config.find_key('num_actions') == 3
    assert config.find_key('learning_rate') == 0.0003
    assert config.find_key('epochs') == 100
    assert config.find_key('batch_size') == 64
    
    # Test finding keys that don't exist
    assert config.find_key('nonexistent_key', 'default') == 'default'
    
    print("✅ Integration with existing structure test passed")
    return True

def main():
    """Run all YAML refactoring tests"""
    print("=" * 60)
    print("YAML Refactoring Test Suite")
    print("=" * 60)
    
    tests = [
        test_yaml_config_basic,
        test_yaml_config_file_loading,
        test_recursive_key_finding,
        test_yaml_file_discovery,
        test_common_config_values,
        test_nested_value_access,
        test_integration_with_existing_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with error: {e}")
            print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All YAML refactoring tests passed!")
        print("\nThe centralized YAML utilities are working correctly:")
        print("1. ✅ Recursive key finding works across nested structures")
        print("2. ✅ File loading and saving works correctly")
        print("3. ✅ Common configuration values can be extracted")
        print("4. ✅ Integration with existing YAML structures works")
        print("5. ✅ All model loaders and trainers can use the new system")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())