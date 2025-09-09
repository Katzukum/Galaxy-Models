# YAML Refactoring Guide

## 🎯 Overview

This guide documents the comprehensive refactoring of YAML handling throughout the Galaxy Models project. The refactoring introduces centralized YAML utilities with recursive key finding capabilities, making configuration management more robust and maintainable.

## 🔧 What Was Changed

### 1. Centralized YAML Utilities (`Utilities/yaml_utils.py`)

**New Features:**
- **Recursive Key Finding**: Find any key regardless of its location in nested structures
- **Unified Configuration Loading**: Consistent YAML loading across all components
- **Path-based Access**: Access values using specific paths in nested structures
- **Common Key Extraction**: Extract frequently used configuration values
- **File Discovery**: Find YAML files recursively in directories

### 2. Refactored Components

**Files Updated:**
- ✅ `Main.py` - Model discovery and details
- ✅ `NetworkConfigs/PPO_loader.py` - PPO model loading
- ✅ `Utilities/Api_Loader.py` - API server model detection
- ✅ `Utilities/backtester.py` - Backtesting configuration
- ✅ `Utilities/ppo_backtester.py` - PPO backtesting

**Files Pending:**
- ⏳ `NetworkConfigs/NN_loader.py` - Neural Network loader
- ⏳ `NetworkConfigs/Transformer_loader.py` - Transformer loader
- ⏳ `NetworkConfigs/XGBoost_loader.py` - XGBoost loader
- ⏳ All trainer files - Model training configuration

## 🚀 Key Benefits

### 1. **Recursive Key Finding**
```python
# Before: Manual nested access
if 'data_params' in config['Config'] and 'features' in config['Config']['data_params']:
    features = config['Config']['data_params']['features']
elif 'features' in config['Config']:
    features = config['Config']['features']

# After: Recursive key finding
features = config.find_key('features')
```

### 2. **Consistent Error Handling**
```python
# Before: Manual error handling
try:
    model_name = config['model_name']
except KeyError:
    model_name = 'Unknown Model'

# After: Built-in default values
model_name = config.find_key('model_name', 'Unknown Model')
```

### 3. **Unified Configuration Loading**
```python
# Before: Manual YAML loading
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# After: Centralized loading
config = load_yaml_config(config_path)
```

## 📊 Migration Examples

### Example 1: Model Loader Refactoring

**Before:**
```python
# Manual YAML loading and nested access
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

model_name = config['model_name']
if 'data_params' in config['Config'] and 'features' in config['Config']['data_params']:
    features = config['Config']['data_params']['features']
elif 'features' in config['Config']:
    features = config['Config']['features']
else:
    raise KeyError("Could not find features in configuration")
```

**After:**
```python
# Centralized YAML utilities with recursive key finding
config = load_yaml_config(config_path)

model_name = config.find_key('model_name')
features = config.find_key('features')
if not features:
    raise ValueError("No features found in config")
```

### Example 2: Model Discovery Refactoring

**Before:**
```python
# Manual YAML file discovery and loading
yaml_files = glob.glob(str(models_path / "**" / "*.yaml"), recursive=True)

for yaml_file in yaml_files:
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
        model_name = config.get('model_name', 'Unknown Model')
        model_type = config.get('Type', 'Unknown Type')
```

**After:**
```python
# Centralized file discovery and loading
yaml_files = find_yaml_files(str(models_path), recursive=True)

for yaml_file in yaml_files:
    config = load_yaml_config(yaml_file)
    model_name = config.find_key('model_name', 'Unknown Model')
    model_type = config.find_key('Type', 'Unknown Type')
```

### Example 3: Configuration Value Extraction

**Before:**
```python
# Manual extraction of common values
model_params = config['Config']['model_params']
data_input_dim = model_params.get('input_dim', len(features))
hidden_dim = model_params.get('hidden_dim', 128)
learning_rate = config['Config']['train_params'].get('learning_rate', 0.001)
```

**After:**
```python
# Recursive key finding for common values
data_input_dim = config.find_key('input_dim', len(features))
hidden_dim = config.find_key('hidden_dim', 128)
learning_rate = config.find_key('learning_rate', 0.001)
```

## 🛠️ New YAML Utilities API

### Core Classes

#### `YAMLConfig`
```python
# Initialize from file
config = YAMLConfig(config_path)

# Initialize from dictionary
config = YAMLConfig(config_dict=my_dict)

# Find keys recursively
value = config.find_key('key_name', default_value)

# Find multiple keys
values = config.find_keys(['key1', 'key2', 'key3'])

# Get all keys
all_keys = config.get_all_keys()

# Access nested values by path
value = config.get_nested_value(['level1', 'level2', 'key'])

# Set nested values
config.set_nested_value(['level1', 'level2', 'key'], 'value')
```

#### Convenience Functions
```python
# Load YAML configuration
config = load_yaml_config('path/to/config.yaml')

# Find YAML files
yaml_files = find_yaml_files('directory', recursive=True)

# Get common configuration values
common_values = get_common_config_values('path/to/config.yaml')
```

### Common Keys

The system recognizes these common configuration keys:
- `model_name` - Model name
- `Type` - Model type
- `features` - Feature list
- `input_dim` - Input dimension
- `hidden_dim` - Hidden dimension
- `learning_rate` - Learning rate
- `epochs` - Number of epochs
- `batch_size` - Batch size
- `data_params` - Data parameters
- `model_params` - Model parameters
- `train_params` - Training parameters

## 🧪 Testing

Run the comprehensive test suite:
```bash
python test_yaml_refactor.py
```

This tests:
- ✅ Basic YAMLConfig functionality
- ✅ File loading and saving
- ✅ Recursive key finding
- ✅ YAML file discovery
- ✅ Common configuration values
- ✅ Nested value access
- ✅ Integration with existing structures

## 📝 Migration Checklist

### For Each File to Refactor:

1. **Update Imports**
   ```python
   # Remove
   import yaml
   
   # Add
   from Utilities.yaml_utils import YAMLConfig, load_yaml_config
   ```

2. **Replace YAML Loading**
   ```python
   # Before
   with open(config_path, 'r') as f:
       config = yaml.safe_load(f)
   
   # After
   config = load_yaml_config(config_path)
   ```

3. **Replace Key Access**
   ```python
   # Before
   value = config.get('key', default)
   value = config['nested']['key']
   
   # After
   value = config.find_key('key', default)
   ```

4. **Replace Nested Access**
   ```python
   # Before
   if 'level1' in config and 'level2' in config['level1']:
       value = config['level1']['level2']['key']
   
   # After
   value = config.find_key('key', default)
   ```

5. **Update Error Handling**
   ```python
   # Before
   try:
       value = config['key']
   except KeyError:
       value = default
   
   # After
   value = config.find_key('key', default)
   ```

## 🔄 Backward Compatibility

The refactoring maintains backward compatibility:
- All existing YAML files continue to work
- No changes required to existing configuration files
- The recursive key finding works with any YAML structure
- Default values prevent breaking changes

## 🎯 Next Steps

1. **Complete Remaining Files**: Refactor all remaining model loaders and trainers
2. **Update Documentation**: Update all documentation to reflect new patterns
3. **Performance Testing**: Test performance with large configuration files
4. **User Training**: Train team members on new YAML utilities

## 📚 Additional Resources

- **Test Suite**: `test_yaml_refactor.py` - Comprehensive testing
- **API Documentation**: `Utilities/yaml_utils.py` - Full API reference
- **Examples**: See refactored files for usage examples
- **Migration Guide**: This document for step-by-step migration

The YAML refactoring provides a robust, maintainable foundation for configuration management across the entire Galaxy Models project! 🎉