# PPO Agent Integration for Galaxy Models

This document describes the PPO (Proximal Policy Optimization) agent integration into the Galaxy Models platform, following the existing architecture patterns for XGBoost and Transformer models.

## Overview

The PPO agent implementation provides:
- **Training**: Reinforcement learning agent that learns trading strategies through interaction with a custom trading environment
- **Backtesting**: Specialized backtesting for PPO agents with action-based trading logic
- **API Hosting**: REST API integration for real-time PPO agent predictions
- **Web Interface**: Integration with the existing web dashboard

## Architecture

### Components

1. **PPOTrainer** (`NetworkConfigs/PPOTrainer.py`)
   - Custom trading environment for PPO training
   - PPO Actor-Critic network implementation
   - Training pipeline with rollout collection and policy updates

2. **PPOModelLoader** (`NetworkConfigs/PPO_loader.py`)
   - Model loading and prediction for API serving
   - Handles feature scaling and account state management
   - Provides prediction responses with action confidence

3. **PPOBacktester** (`Utilities/ppo_backtester.py`)
   - Specialized backtesting for PPO agents
   - Action-based trading logic (Hold/Buy/Sell)
   - Performance metrics calculation

4. **Integration Updates**
   - `run_training.py`: Added PPO training pipeline
   - `Main.py`: Added PPO to valid model types
   - `Api_Loader.py`: Added PPO model loading support
   - `backtester.py`: Added PPO backtesting support

## Trading Environment

The PPO agent operates in a custom trading environment with:

### Actions
- **0 (Hold)**: No position change
- **1 (Buy)**: Open long position or close short position
- **2 (Sell)**: Open short position or close long position

### Observation Space
- Historical market features (scaled)
- Account state: normalized balance, position, unrealized PnL
- Sequence length: 60 timesteps (configurable)

### Rewards
- Based on equity changes
- Small penalty for holding to encourage trading
- Scaled for training stability

## Training Configuration

### Model Parameters
```yaml
model_params:
  input_dim: 27  # Number of input features
  hidden_dim: 128  # Hidden layer dimension
  num_actions: 3  # Hold/Buy/Sell
  lookback_window: 60  # Historical timesteps
```

### Training Parameters
```yaml
train_params:
  learning_rate: 3e-4  # Adam optimizer learning rate
  epochs: 100  # Training epochs
  batch_size: 64  # Training batch size
  ppo_epochs: 4  # PPO updates per iteration
  clip_ratio: 0.2  # PPO clipping parameter
  value_coef: 0.5  # Value function loss weight
  entropy_coef: 0.01  # Entropy bonus weight
```

## Usage

### Training a PPO Agent

```bash
# Basic training
python Utilities/run_training.py --csv_path sample.csv --model ppo

# Custom model name
python Utilities/run_training.py --csv_path sample.csv --model ppo --model_name my_ppo_agent

# With custom parameters
python Utilities/run_training.py --csv_path sample.csv --model ppo --training_params '{"train_params": {"epochs": 200}}'
```

### Backtesting

The PPO agent can be backtested through the web interface or programmatically:

```python
from Utilities.ppo_backtester import PPOBacktester

backtester = PPOBacktester("Models/PPO_my_ppo_agent/my_ppo_agent_config.yaml", "sample.csv")
backtester.load_artifacts()
results = backtester.run_with_results()
```

### API Hosting

PPO models can be hosted via the existing API infrastructure:

1. Start the API server through the web interface
2. Select a PPO model from the model library
3. The API will automatically detect and load the PPO model
4. Make predictions via POST requests to `/predict`

## Model Artifacts

A trained PPO model produces the following artifacts:

```
Models/PPO_model_name/
├── model_name_config.yaml      # Model configuration
├── model_name_scaler.pkl       # Feature scaler
└── model_name_model.pt         # PyTorch model state dict
```

## API Response Format

PPO predictions return:

```json
{
  "prediction": 1,
  "action_name": "Buy",
  "confidence": 0.85,
  "value_estimate": 0.12
}
```

## Integration with Existing Platform

The PPO agent seamlessly integrates with the existing Galaxy Models platform:

1. **Web Interface**: PPO appears as a model type in the training interface
2. **Model Library**: Trained PPO models appear alongside other model types
3. **Backtesting**: PPO models can be backtested with the same interface
4. **API Hosting**: PPO models can be hosted via the existing API infrastructure

## Dependencies

Additional dependencies for PPO support:
- `gym`: OpenAI Gym for environment interface
- `stable-baselines3`: PPO implementation (optional, we use custom implementation)

## Testing

Run the test suite to verify PPO setup:

```bash
python test_ppo_setup.py
```

This will test:
- Module imports
- Network creation and forward pass
- Trading environment functionality
- Trainer initialization
- Model loader structure

## Performance Considerations

- **Memory**: PPO training requires more memory due to rollout collection
- **Training Time**: RL training typically takes longer than supervised learning
- **Sequence Length**: Longer lookback windows increase memory usage
- **Batch Size**: Larger batches improve training stability but require more memory

## Future Enhancements

Potential improvements to the PPO implementation:

1. **Advanced Environments**: More sophisticated reward functions and market simulation
2. **Multi-Asset Support**: Extend to multiple trading instruments
3. **Risk Management**: Integrated position sizing and risk controls
4. **Online Learning**: Continuous learning from live market data
5. **Ensemble Methods**: Multiple PPO agents with different strategies

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Memory Issues**: Reduce batch size or sequence length
3. **Training Instability**: Adjust learning rate or PPO parameters
4. **Poor Performance**: Check reward function and environment design

### Debug Mode

Enable debug logging by setting environment variables:
```bash
export DEBUG=1
python Utilities/run_training.py --csv_path sample.csv --model ppo
```

## Contributing

When extending the PPO implementation:

1. Follow the existing code patterns and architecture
2. Update tests in `test_ppo_setup.py`
3. Update documentation
4. Ensure backward compatibility with existing models