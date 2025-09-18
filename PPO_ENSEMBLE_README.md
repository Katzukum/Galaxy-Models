# PPO-Based Ensemble Model Implementation

## Overview

This implementation replaces the existing static ensemble training process with a PPO (Proximal Policy Optimization) reinforcement learning approach that uses individual model predictions as features to learn optimal trading strategies.

## Key Features

- **Reinforcement Learning**: Uses PPO to learn optimal trading actions based on actual market performance
- **Dynamic Weighting**: Learns to dynamically weight and combine model predictions
- **Trading-Focused**: Optimizes for trading performance rather than prediction accuracy
- **Full Integration**: Seamlessly integrated with existing model architecture and web interface

## Architecture

### Core Components

1. **PPOEnsembleTrainer** (`NetworkConfigs/PPOEnsembleTrainer.py`)
   - Main training class for PPO-based ensemble models
   - Handles data preparation, model prediction collection, and PPO training
   - Includes custom trading environment and PPO network architecture

2. **PPOEnsemble_loader** (`NetworkConfigs/PPOEnsemble_loader.py`)
   - Model loading and inference for trained PPO ensemble models
   - Provides prediction responses with action probabilities and individual model predictions

3. **Training Utility** (`Utilities/run_ppo_ensemble_training.py`)
   - Command-line interface for PPO ensemble training
   - Supports configuration files and parameter overrides

### Integration Points

- **Main.py**: Added `start_ppo_ensemble_training()` function
- **Backtester**: Updated to handle PPO ensemble models in backtesting
- **API Loader**: Updated to support PPO ensemble model serving
- **Web Interface**: Added PPO ensemble training options and configuration

## Usage

### Web Interface

1. Navigate to the Ensemble Training tab
2. Select "🤖 PPO Ensemble (Reinforcement Learning)" as the ensemble type
3. Configure PPO parameters:
   - Learning Rate (0.0001 - 0.01)
   - Epochs (10 - 500)
   - Sequence Length (30 - 120)
   - Gamma/Discount Factor (0.9 - 0.99)
   - Initial Balance
   - Position Size
4. Select individual models to combine
5. Start training

### Command Line

```bash
# Create sample configuration
python3 Utilities/run_ppo_ensemble_training.py --create-sample-config

# Train with configuration file
python3 Utilities/run_ppo_ensemble_training.py --config ppo_ensemble_sample_config.yaml --csv data.csv

# Train with specific parameters
python3 Utilities/run_ppo_ensemble_training.py --model-name my_ppo_ensemble --csv data.csv --epochs 200 --learning-rate 0.001
```

### Programmatic Usage

```python
from NetworkConfigs.PPOEnsembleTrainer import run_ppo_ensemble_training

config = {
    'model_name': 'my_ppo_ensemble',
    'ensemble_type': 'ppo',
    'selected_models': [
        {
            'name': 'model1',
            'type': 'Neural Network (Regression)',
            'configPath': 'Models/Model1/config.yaml'
        }
    ],
    'ppo_params': {
        'learning_rate': 0.0003,
        'epochs': 100,
        'sequence_length': 60,
        'gamma': 0.99
    },
    'trading_params': {
        'initial_balance': 50000,
        'position_size': 0.1,
        'transaction_cost': 0.001
    },
    'features': ['close', 'volume', 'rsi', 'macd'],
    'csv_path': 'data.csv'
}

result = run_ppo_ensemble_training('my_ppo_ensemble', config, 'Models')
```

## Configuration Format

```yaml
model_name: "ppo_ensemble_v1"
model_type: "PPOEnsemble"
config:
  ensemble_type: "ppo"
  selected_models:
    - name: "nn_model_1"
      type: "Neural Network (Regression)"
      configPath: "Models/NN_model1/config.yaml"
    - name: "xgboost_model_1"
      type: "XGBoostClassifier"
      configPath: "Models/XGBoost_model1/config.yaml"
  ppo_params:
    learning_rate: 0.0003
    epochs: 100
    batch_size: 64
    sequence_length: 60
    gamma: 0.99
    clip_ratio: 0.2
  trading_params:
    initial_balance: 50000
    position_size: 0.1
    transaction_cost: 0.001
  features: ["close", "volume", "rsi", "macd"]
```

## PPO Environment

### Actions
- **0**: Hold (no action)
- **1**: Buy (open long position)
- **2**: Sell (open short position)

### Observation Space
- Individual model predictions (N features)
- Market data features (price, volume, technical indicators)
- Portfolio state (balance, position, unrealized PnL)

### Reward Function
- Based on portfolio performance
- Includes transaction costs
- Penalizes excessive holding

## Network Architecture

### PPOEnsembleNetwork
- **Input**: Sequence of model predictions + market features + portfolio state
- **Shared Feature Extractor**: Processes all input types
- **LSTM Layer**: Handles time-series sequences
- **Actor Head**: Outputs action probabilities
- **Critic Head**: Estimates state values

## Training Process

1. **Data Preparation**: Load and preprocess historical data
2. **Model Prediction Collection**: Generate predictions from individual models
3. **PPO Dataset Creation**: Combine predictions with market data
4. **Environment Setup**: Initialize trading environment
5. **PPO Training**: Learn optimal trading policy through reinforcement learning
6. **Model Saving**: Save trained PPO model and configuration

## Integration with Existing System

### Backtesting
- PPO ensemble models are automatically detected
- Action-based predictions (0=Hold, 1=Buy, 2=Sell)
- Compatible with existing backtesting framework

### API Hosting
- PPO ensemble models can be served via API
- Returns action predictions with probabilities
- Includes individual model predictions for transparency

### Model Library
- PPO ensemble models appear in model library
- Full configuration and metadata support
- Compatible with existing model management

## Performance Considerations

### Training Time
- Depends on number of epochs and data size
- Typically 2-10x longer than static ensemble training
- Can be reduced by adjusting sequence length and batch size

### Memory Usage
- Higher memory usage due to sequence processing
- Scales with sequence length and number of models
- Consider reducing sequence length for large datasets

### Model Size
- PPO models are larger than static ensemble models
- Includes neural network weights and individual model references
- Typical size: 10-50MB depending on configuration

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Memory Issues**: Reduce sequence length or batch size
3. **Training Instability**: Adjust learning rate or clip ratio
4. **Poor Performance**: Check individual model quality and features

### Debug Mode

Enable debug mode in Main.py for detailed logging:
```python
DEBUG_VERBOSE = True
```

## Future Enhancements

1. **Advanced Reward Functions**: Implement risk-adjusted rewards
2. **Multi-Asset Support**: Extend to multiple trading instruments
3. **Online Learning**: Continuous model updates
4. **Ensemble Diversity**: Automatic model selection based on diversity
5. **Hyperparameter Optimization**: Automated parameter tuning

## Migration from Static Ensemble

1. **Backup Existing Models**: Save current ensemble models
2. **Test PPO Ensemble**: Start with small datasets and short training
3. **Compare Performance**: Use backtesting to compare approaches
4. **Gradual Migration**: Replace static ensembles gradually
5. **Monitor Results**: Track performance improvements

## Support

For issues or questions:
1. Check the test suite: `python3 test_ppo_ensemble_simple.py`
2. Review configuration format
3. Check debug logs
4. Verify individual model compatibility

## License

This implementation follows the same license as the main project.