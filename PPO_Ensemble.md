# Product Requirements Document (PRD)
## PPO-Based Ensemble Model Replacement

### 1. Executive Summary

**Objective**: Replace the existing ensemble training process with a PPO (Proximal Policy Optimization) reinforcement learning approach that uses individual model predictions as features to learn optimal trading strategies.

**Current State**: The existing ensemble system (`NetworkConfigs/EnsembleTrainer.py`) combines multiple model predictions using static methods (averaging, voting, weighted, stacking).

**Target State**: A PPO-based ensemble that learns to dynamically weight and combine model predictions through reinforcement learning, optimizing for trading performance rather than prediction accuracy.

### 2. Problem Statement

The current ensemble approach has limitations:
- Static combination methods don't adapt to market conditions
- No consideration of trading costs, risk, or portfolio management
- Models are combined based on prediction accuracy rather than actual trading performance
- No learning from past trading decisions and their outcomes

### 3. Solution Overview

Create a new `PPOEnsembleTrainer` class that:
1. Loads pre-trained individual models
2. Generates predictions for each model on historical data
3. Uses these predictions as features for PPO training
4. Learns optimal trading actions based on actual market performance
5. Maintains compatibility with existing model architecture

### 4. Technical Architecture

#### 4.1 Core Components

**New Files to Create:**
- `NetworkConfigs/PPOEnsembleTrainer.py` - Main training class
- `NetworkConfigs/PPOEnsemble_loader.py` - Model loading and inference
- `Utilities/run_ppo_ensemble_training.py` - Training pipeline utility

**Modified Files:**
- `Main.py` - Add PPO ensemble training option
- `web/index.html` - Add UI for PPO ensemble training
- `web/js/` - Add frontend logic for PPO ensemble

#### 4.2 Data Flow

```
CSV Data → Individual Model Predictions → PPO Training Data → PPO Model → Trading Actions
```

### 5. Detailed Requirements

#### 5.1 Data Preparation Process

**Step 1: Load and Prepare Historical Data**
- **File Reference**: `NetworkConfigs/PPOEnsembleTrainer.py:prepare_ensemble_data()`
- Load CSV data and normalize column names to lowercase
- Apply same data preprocessing as current ensemble (delta features)
- **Reference**: `NetworkConfigs/EnsembleTrainer.py:158-161` for target creation

**Step 2: Generate Individual Model Predictions**
- **File Reference**: `NetworkConfigs/PPOEnsembleTrainer.py:collect_model_predictions()`
- Load each selected model using existing loaders:
  - `NetworkConfigs/NN_loader.py` for Neural Networks
  - `NetworkConfigs/XGBoost_loader.py` for XGBoost models
  - `NetworkConfigs/Transformer_loader.py` for Transformer models
- Generate predictions for each row of historical data
- **Reference**: `NetworkConfigs/EnsembleTrainer.py:60-122` for model loading logic

**Step 3: Create PPO Training Dataset**
- **File Reference**: `NetworkConfigs/PPOEnsembleTrainer.py:create_ppo_dataset()`
- Combine individual model predictions into feature matrix
- Add market data features (price, volume, technical indicators)
- Create sequence data for time-series PPO training
- **Reference**: `NetworkConfigs/PPOTrainer.py:100-108` for sequence creation

#### 5.2 PPO Environment Design

**Step 4: Design PPO Trading Environment**
- **File Reference**: `NetworkConfigs/PPOEnsembleTrainer.py:PPOEnsembleEnvironment`
- **Actions**: 0=Hold, 1=Buy, 2=Sell (same as current PPO)
- **Observation Space**: 
  - Individual model predictions (N features)
  - Market data features (price, volume, technical indicators)
  - Portfolio state (balance, position, unrealized PnL)
- **Reward Function**: Based on portfolio performance
  - **Reference**: `NetworkConfigs/PPOTrainer.py:140-151` for reward calculation

**Step 5: PPO Network Architecture**
- **File Reference**: `NetworkConfigs/PPOEnsembleTrainer.py:PPOEnsembleNetwork`
- Input: Sequence of model predictions + market features
- Shared feature extractor for all inputs
- Actor head for action probabilities
- Critic head for value estimation
- **Reference**: `NetworkConfigs/PPOTrainer.py:171-222` for network structure

#### 5.3 Training Process

**Step 6: PPO Training Loop**
- **File Reference**: `NetworkConfigs/PPOEnsembleTrainer.py:train()`
- Use same PPO algorithm as current implementation
- **Reference**: `NetworkConfigs/PPOTrainer.py:382-450` for training loop
- Collect rollouts using ensemble environment
- Compute returns and advantages
- **Reference**: `NetworkConfigs/PPOTrainer.py:359-380` for advantage calculation

**Step 7: Model Saving and Configuration**
- **File Reference**: `NetworkConfigs/PPOEnsembleTrainer.py:save_model()`
- Save PPO model weights
- Save ensemble configuration (selected models, features)
- Save scalers for data preprocessing
- Create YAML configuration file
- **Reference**: `NetworkConfigs/EnsembleTrainer.py:580-640` for saving logic

#### 5.4 Integration with Existing System

**Step 8: Model Loading and Inference**
- **File Reference**: `NetworkConfigs/PPOEnsemble_loader.py:PPOEnsembleLoader`
- Load PPO model and individual model references
- Generate predictions from individual models
- Use PPO to make final trading decision
- **Reference**: `NetworkConfigs/Ensemble_loader.py:19-218` for loading structure

**Step 9: Web Interface Integration**
- **File Reference**: `Main.py:train_ppo_ensemble_model()`
- Add new training function to main application
- **Reference**: `Main.py:570-588` for existing training function structure
- Support same configuration format as current ensemble

### 6. Implementation Steps

#### Phase 1: Core PPO Ensemble Trainer
1. Create `PPOEnsembleTrainer` class with data preparation methods
2. Implement individual model prediction collection
3. Design PPO environment for ensemble trading
4. Create PPO network architecture

#### Phase 2: Training Pipeline
1. Implement PPO training loop
2. Add model saving and configuration management
3. Create training utility script
4. Add error handling and validation

#### Phase 3: Integration and Testing
1. Create model loader for inference
2. Integrate with main application
3. Add web interface support
4. Create test cases and validation

#### Phase 4: Documentation and Deployment
1. Update documentation
2. Add example configurations
3. Create migration guide from old ensemble
4. Performance testing and optimization

### 7. Configuration Format

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

### 8. Success Metrics

- **Training Performance**: PPO converges to stable policy
- **Trading Performance**: Positive Sharpe ratio, controlled drawdown
- **Integration**: Seamless integration with existing model architecture
- **Usability**: Same interface as current ensemble training
- **Scalability**: Support for 2-10 individual models

### 9. Risk Mitigation

- **Model Compatibility**: Ensure all individual model types are supported
- **Data Quality**: Validate prediction quality from individual models
- **Training Stability**: Implement early stopping and checkpointing
- **Performance**: Monitor training time and resource usage
- **Rollback Plan**: Maintain existing ensemble as fallback option

### 10. UI Changes Required

#### 10.1 Frontend JavaScript Updates
**File Reference**: `web/js/ensemble_training.js:777-829`
- **Step 1**: Add PPO ensemble option to ensemble type selection
- **Step 2**: Add PPO-specific configuration fields (learning rate, epochs, sequence length, gamma)
- **Step 3**: Update `startEnsembleTraining()` function to handle PPO ensemble type
- **Step 4**: Add PPO training progress monitoring with different status messages
- **Reference**: `web/js/ensemble_training.js:809-816` for existing training call structure

#### 10.2 HTML Interface Updates
**File Reference**: `web/index.html:432-638`
- **Step 1**: Add PPO ensemble option to ensemble type dropdown
- **Step 2**: Add PPO-specific configuration section with fields:
  - Learning rate slider (0.0001 - 0.01)
  - Epochs input (10 - 500)
  - Sequence length input (30 - 120)
  - Gamma value (0.9 - 0.99)
  - Initial balance input
- **Step 3**: Update progress display to show PPO-specific metrics (episode, reward, policy loss)
- **Reference**: `web/index.html:446-604` for existing form structure

#### 10.3 Training Tab Integration
**File Reference**: `web/tabs/training/training.html:18-26`
- **Step 1**: Add PPO Ensemble option to model type selection
- **Step 2**: Ensure PPO ensemble appears in training dropdown
- **Reference**: `web/tabs/training/training.html:18-24` for model type options

### 11. Backtesting Framework Changes

#### 11.1 Backtester Class Updates
**File Reference**: `Utilities/backtester.py:60-465`
- **Step 1**: Add PPO Ensemble model type detection
- **Step 2**: Update model loading logic to handle PPO ensemble
- **Step 3**: Modify prediction generation for PPO ensemble (action-based vs regression)
- **Step 4**: Update sequence length handling for PPO ensemble
- **Reference**: `Utilities/backtester.py:446-455` for existing ensemble handling
- **Reference**: `Utilities/backtester.py:648-670` for prediction generation logic

#### 11.2 PPO Ensemble Backtesting Logic
**File Reference**: `Utilities/backtester.py:404-465`
- **Step 1**: Implement PPO ensemble prediction method
- **Step 2**: Handle action-to-trading-signal conversion (0=Hold, 1=Buy, 2=Sell)
- **Step 3**: Add PPO-specific portfolio management logic
- **Step 4**: Update equity tracking for PPO ensemble trades
- **Reference**: `Utilities/backtester.py:650-670` for existing prediction handling

### 12. API Hosting Framework Changes

#### 12.1 API Loader Updates
**File Reference**: `Utilities/Api_Loader.py:60-103`
- **Step 1**: Add PPO Ensemble model type detection
- **Step 2**: Import PPOEnsemble_loader module
- **Step 3**: Add PPO ensemble loader creation logic
- **Reference**: `Utilities/Api_Loader.py:86-98` for existing model type handling

#### 12.2 API Prediction Endpoint
**File Reference**: `Utilities/Api_Loader.py:193-230`
- **Step 1**: Add PPO ensemble prediction handling in `/predict` endpoint
- **Step 2**: Update response model to include PPO ensemble response type
- **Step 3**: Handle PPO ensemble feature preprocessing
- **Reference**: `Utilities/Api_Loader.py:210-220` for existing prediction handling

#### 12.3 Main.py API Integration
**File Reference**: `Main.py:626-670`
- **Step 1**: Update `start_api_server()` to handle PPO ensemble models
- **Step 2**: Add PPO ensemble model type validation
- **Step 3**: Update API server startup command for PPO ensemble
- **Reference**: `Main.py:656-660` for existing API server startup

### 13. File Structure

```
NetworkConfigs/
├── PPOEnsembleTrainer.py          # Main training class
├── PPOEnsemble_loader.py          # Model loading and inference
├── EnsembleTrainer.py             # Existing (to be deprecated)
└── Ensemble_loader.py             # Existing (to be deprecated)

Utilities/
├── run_ppo_ensemble_training.py   # Training pipeline utility
├── run_training.py                # Existing (add PPO ensemble option)
├── backtester.py                  # Modified (add PPO ensemble support)
└── Api_Loader.py                  # Modified (add PPO ensemble API support)

web/
├── index.html                     # Modified (add PPO ensemble UI)
├── js/
│   ├── ensemble_training.js       # Modified (add PPO ensemble logic)
│   └── training.js                # Modified (add PPO ensemble option)
└── tabs/training/
    └── training.html              # Modified (add PPO ensemble option)

Main.py                            # Modified (add PPO ensemble training function)
```

### 14. Additional Implementation Details

#### 14.1 Main.py Training Function
**File Reference**: `Main.py:570-588`
- **Step 1**: Add `train_ppo_ensemble_model()` function
- **Step 2**: Implement PPO ensemble training endpoint
- **Step 3**: Add PPO ensemble model validation
- **Step 4**: Update model discovery to include PPO ensemble models
- **Reference**: `Main.py:158-167` for existing ensemble training function structure

#### 14.2 Model Type Detection Updates
**File Reference**: `Main.py:170-216`
- **Step 1**: Update `get_models()` to detect PPO ensemble models
- **Step 2**: Add PPO ensemble model type to model discovery
- **Step 3**: Update model details extraction for PPO ensemble
- **Reference**: `Main.py:196-207` for existing model info structure

#### 14.3 Backtesting Integration
**File Reference**: `Main.py:524-588`
- **Step 1**: Update `run_backtest()` to handle PPO ensemble models
- **Step 2**: Add PPO ensemble model type validation in backtesting
- **Step 3**: Update backtest parameter handling for PPO ensemble
- **Reference**: `Main.py:570-588` for existing backtest function structure

### 15. Dependencies

- **Existing**: All current dependencies from `requirements.txt`
- **Additional**: No new dependencies required
- **Compatibility**: Python 3.8+, PyTorch 1.9+, existing model loaders

### 16. Timeline

- **Phase 1**: 2-3 weeks (Core implementation)
- **Phase 2**: 1-2 weeks (Training pipeline)
- **Phase 3**: 2-3 weeks (UI, backtesting, and API integration)
- **Phase 4**: 1-2 weeks (Testing and documentation)

**Total Estimated Time**: 6-10 weeks

### 17. Testing Requirements

#### 17.1 Unit Testing
- PPO ensemble trainer functionality
- Model prediction collection
- PPO environment behavior
- Model loading and saving

#### 17.2 Integration Testing
- UI training workflow
- Backtesting with PPO ensemble
- API prediction endpoints
- End-to-end training pipeline

#### 17.3 Performance Testing
- Training time with multiple models
- Memory usage during training
- API response times
- Backtesting performance

### 18. Migration Strategy

#### 18.1 Backward Compatibility
- Maintain existing ensemble training as fallback
- Gradual migration of existing ensemble models
- Configuration migration tools

#### 18.2 Rollout Plan
- Phase 1: Deploy PPO ensemble alongside existing ensemble
- Phase 2: Migrate users to PPO ensemble
- Phase 3: Deprecate old ensemble system
- Phase 4: Remove old ensemble code

---

*This PRD replaces the existing ensemble training process with a PPO-based approach that learns optimal trading strategies from individual model predictions.*
