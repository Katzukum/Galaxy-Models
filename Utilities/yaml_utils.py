#!/usr/bin/env python3
"""
Centralized YAML utilities for Galaxy Models.
Provides consistent YAML loading and recursive key finding functionality.
"""

import yaml
import os
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


class YAMLConfig:
    """
    Centralized YAML configuration handler with recursive key finding.
    """
    
    def __init__(self, config_path: str = None, config_dict: Dict = None):
        """
        Initialize YAMLConfig with either a file path or dictionary.
        
        Args:
            config_path: Path to YAML file
            config_dict: Pre-loaded configuration dictionary
        """
        self.config_path = config_path
        self.config_dict = config_dict or {}
        
        if config_path and not config_dict:
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str) -> 'YAMLConfig':
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML file
            
        Returns:
            Self for method chaining
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config_dict = yaml.safe_load(f)
            self.config_path = config_path
            return self
        except Exception as e:
            raise ValueError(f"Failed to load YAML from {config_path}: {e}")
    
    def save_to_file(self, output_path: str = None) -> 'YAMLConfig':
        """
        Save configuration to YAML file.
        
        Args:
            output_path: Path to save YAML file (uses config_path if None)
            
        Returns:
            Self for method chaining
        """
        save_path = output_path or self.config_path
        if not save_path:
            raise ValueError("No output path specified")
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config_dict, f, default_flow_style=False, sort_keys=False)
            return self
        except Exception as e:
            raise ValueError(f"Failed to save YAML to {save_path}: {e}")
    
    def find_key(self, key: str, default: Any = None) -> Any:
        """
        Recursively find a key anywhere in the nested configuration.
        
        Args:
            key: Key to search for
            default: Default value if key not found
            
        Returns:
            Value associated with key or default
        """
        return self._recursive_find(self.config_dict, key, default)
    
    def find_keys(self, keys: List[str], default: Any = None) -> Dict[str, Any]:
        """
        Find multiple keys and return as dictionary.
        
        Args:
            keys: List of keys to search for
            default: Default value for missing keys
            
        Returns:
            Dictionary with found keys and their values
        """
        result = {}
        for key in keys:
            result[key] = self.find_key(key, default)
        return result
    
    def find_key_path(self, key: str) -> Optional[List[str]]:
        """
        Find the path to a key in the nested structure.
        
        Args:
            key: Key to search for
            
        Returns:
            List representing the path to the key, or None if not found
        """
        path = self._recursive_find_path(self.config_dict, key, [])
        return path if path else None
    
    def get_nested_value(self, path: List[str], default: Any = None) -> Any:
        """
        Get value using a specific path in the nested structure.
        
        Args:
            path: List of keys representing the path
            default: Default value if path not found
            
        Returns:
            Value at the specified path or default
        """
        current = self.config_dict
        try:
            for key in path:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def set_nested_value(self, path: List[str], value: Any) -> 'YAMLConfig':
        """
        Set value using a specific path in the nested structure.
        
        Args:
            path: List of keys representing the path
            value: Value to set
            
        Returns:
            Self for method chaining
        """
        current = self.config_dict
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
        return self
    
    def _recursive_find(self, data: Any, key: str, default: Any = None) -> Any:
        """
        Recursively search for a key in nested data structures.
        
        Args:
            data: Data structure to search
            key: Key to find
            default: Default value if key not found
            
        Returns:
            Value associated with key or default
        """
        if isinstance(data, dict):
            # Check if key exists in current level
            if key in data:
                return data[key]
            
            # Recursively search in nested dictionaries
            for value in data.values():
                result = self._recursive_find(value, key, default)
                if result is not default:
                    return result
                    
        elif isinstance(data, list):
            # Search in list items
            for item in data:
                result = self._recursive_find(item, key, default)
                if result is not default:
                    return result
        
        return default
    
    def _recursive_find_path(self, data: Any, key: str, current_path: List[str]) -> Optional[List[str]]:
        """
        Recursively find the path to a key in nested data structures.
        
        Args:
            data: Data structure to search
            key: Key to find
            current_path: Current path being explored
            
        Returns:
            Path to the key or None if not found
        """
        if isinstance(data, dict):
            # Check if key exists in current level
            if key in data:
                return current_path + [key]
            
            # Recursively search in nested dictionaries
            for k, v in data.items():
                result = self._recursive_find_path(v, key, current_path + [k])
                if result:
                    return result
                    
        elif isinstance(data, list):
            # Search in list items
            for i, item in enumerate(data):
                result = self._recursive_find_path(item, key, current_path + [str(i)])
                if result:
                    return result
        
        return None
    
    def get_all_keys(self) -> List[str]:
        """
        Get all keys in the configuration (flattened).
        
        Returns:
            List of all keys found in the configuration
        """
        keys = set()
        self._collect_all_keys(self.config_dict, keys)
        return sorted(list(keys))
    
    def _collect_all_keys(self, data: Any, keys: set) -> None:
        """
        Collect all keys from nested data structures.
        
        Args:
            data: Data structure to traverse
            keys: Set to collect keys in
        """
        if isinstance(data, dict):
            for key, value in data.items():
                keys.add(key)
                self._collect_all_keys(value, keys)
        elif isinstance(data, list):
            for item in data:
                self._collect_all_keys(item, keys)
    
    def to_dict(self) -> Dict:
        """
        Get the configuration as a dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self.config_dict.copy()
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access using find_key."""
        return self.find_key(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-like setting (sets at root level)."""
        self.config_dict[key] = value
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists using find_key."""
        return self.find_key(key) is not None
    
    def __repr__(self) -> str:
        """String representation of the configuration."""
        return f"YAMLConfig(path={self.config_path}, keys={len(self.get_all_keys())})"


def load_yaml_config(config_path: str) -> YAMLConfig:
    """
    Convenience function to load a YAML configuration.
    
    Args:
        config_path: Path to YAML file
        
    Returns:
        YAMLConfig instance
    """
    return YAMLConfig(config_path)


def find_yaml_files(directory: str, recursive: bool = True) -> List[str]:
    """
    Find all YAML files in a directory.
    
    Args:
        directory: Directory to search
        recursive: Whether to search recursively
        
    Returns:
        List of YAML file paths
    """
    import glob
    
    pattern = "**/*.yaml" if recursive else "*.yaml"
    return glob.glob(os.path.join(directory, pattern), recursive=recursive)


def merge_yaml_configs(*config_paths: str) -> YAMLConfig:
    """
    Merge multiple YAML configurations.
    
    Args:
        *config_paths: Paths to YAML files to merge
        
    Returns:
        Merged YAMLConfig instance
    """
    merged_config = {}
    
    for config_path in config_paths:
        config = YAMLConfig(config_path)
        merged_config.update(config.to_dict())
    
    return YAMLConfig(config_dict=merged_config)


# Common configuration keys used across the project
COMMON_KEYS = {
    'model_name': 'model_name',
    'model_type': 'Type',
    'features': 'features',
    'input_dim': 'input_dim',
    'hidden_dim': 'hidden_dim',
    'learning_rate': 'learning_rate',
    'epochs': 'epochs',
    'batch_size': 'batch_size',
    'data_params': 'data_params',
    'model_params': 'model_params',
    'train_params': 'train_params'
}


def get_common_config_values(config_path: str) -> Dict[str, Any]:
    """
    Get common configuration values from a YAML file.
    
    Args:
        config_path: Path to YAML file
        
    Returns:
        Dictionary with common configuration values
    """
    config = YAMLConfig(config_path)
    return config.find_keys(list(COMMON_KEYS.keys()))


# Example usage and testing
if __name__ == "__main__":
    # Test the YAML utilities
    print("Testing YAML utilities...")
    
    # Create a test configuration
    test_config = {
        'model_name': 'test_model',
        'Type': 'PPO Agent',
        'Config': {
            'data_params': {
                'features': ['feature1', 'feature2'],
                'input_dim': 10
            },
            'model_params': {
                'hidden_dim': 128,
                'learning_rate': 0.001
            },
            'train_params': {
                'epochs': 100,
                'batch_size': 32
            }
        },
        'nested': {
            'deep': {
                'very_deep': {
                    'secret_key': 'found_it!'
                }
            }
        }
    }
    
    # Test YAMLConfig
    config = YAMLConfig(config_dict=test_config)
    
    print(f"Model name: {config.find_key('model_name')}")
    print(f"Model type: {config.find_key('Type')}")
    print(f"Features: {config.find_key('features')}")
    print(f"Hidden dim: {config.find_key('hidden_dim')}")
    print(f"Secret key: {config.find_key('secret_key')}")
    print(f"All keys: {config.get_all_keys()}")
    print(f"Path to secret_key: {config.find_key_path('secret_key')}")
    
    print("YAML utilities test completed!")