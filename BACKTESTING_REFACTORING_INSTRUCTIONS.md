# Backtesting Engine and API Refactoring Instructions

## Overview
This document provides detailed instructions for refactoring the backtesting engine and API hosting service to work with the new delta-based feature engineering system. The key challenge is that the loaders are now stateful and require previous data points to calculate deltas.

## 4.1: Refactoring the Backtesting Engine

### Key Changes Required

The backtesting engine must be modified to simulate the new stateful feature engineering by maintaining the state of the previous bar's feature dictionary.

### Implementation Steps

#### 1. State Management
The main backtesting loop must now maintain the state of the `previous_bar`'s feature dictionary:

```python
class BacktestingEngine:
    def __init__(self, model_loader, data, initial_balance=10000):
        self.model_loader = model_loader
        self.data = data
        self.initial_balance = initial_balance
        self.previous_bar_features = None  # Add this state variable
        
    def run_backtest(self):
        for i, current_bar in enumerate(self.data):
            # Skip the first bar as there's no previous bar to calculate deltas from
            if i == 0:
                self.previous_bar_features = self._extract_features(current_bar)
                continue
                
            # Calculate delta features
            current_bar_features = self._extract_features(current_bar)
            delta_features = self._calculate_delta_features(
                current_bar_features, 
                self.previous_bar_features
            )
            
            # Make prediction using delta features
            prediction = self.model_loader.predict(delta_features)
            
            # Execute trading logic based on prediction
            self._execute_trade(prediction, current_bar)
            
            # Update state for next iteration
            self.previous_bar_features = current_bar_features
```

#### 2. Delta Feature Calculation
Add a method to calculate delta features:

```python
def _calculate_delta_features(self, current_features, previous_features):
    """
    Calculate delta features by subtracting previous values from current values.
    Only applies to price-related features: close, open, high, low
    """
    delta_features = current_features.copy()
    
    price_columns = ['close', 'open', 'high', 'low']
    for col in price_columns:
        if col in delta_features and col in previous_features:
            delta_features[col] = current_features[col] - previous_features[col]
    
    return delta_features
```

#### 3. Feature Extraction
Ensure the feature extraction method returns a dictionary:

```python
def _extract_features(self, bar_data):
    """
    Extract features from a single bar of data.
    Returns a dictionary with feature names as keys and values as values.
    """
    features = {}
    
    # Extract OHLCV data
    features['close'] = bar_data['close']
    features['open'] = bar_data['open']
    features['high'] = bar_data['high']
    features['low'] = bar_data['low']
    features['volume'] = bar_data['volume']
    
    # Add any technical indicators
    # ... (RSI, MACD, etc.)
    
    return features
```

#### 4. Prediction Interpretation
Handle different model types appropriately:

```python
def _execute_trade(self, prediction, current_bar):
    """
    Execute trading logic based on model prediction.
    Handles both classification and regression models.
    """
    if isinstance(self.model_loader, (XGBoostModelLoader, PPOModelLoader)):
        # Classification models return action strings
        action = prediction  # 'Buy', 'Sell', 'Hold'
        self._execute_classification_trade(action, current_bar)
        
    elif isinstance(self.model_loader, NNModelLoader):
        # Regression models return price change in ticks
        price_change_ticks = prediction
        target_price = current_bar['close'] + (price_change_ticks * self.tick_size)
        self._execute_regression_trade(target_price, current_bar)
```

#### 5. Regression Model Handling
For NN and Transformer models, convert the predicted price change to a concrete price target:

```python
def _execute_regression_trade(self, target_price, current_bar):
    """
    Execute trade based on predicted price target.
    """
    current_price = current_bar['close']
    
    if target_price > current_price * 1.001:  # 0.1% threshold
        # Predicted upward movement - consider buying
        self._place_buy_order(current_price, target_price)
    elif target_price < current_price * 0.999:  # 0.1% threshold
        # Predicted downward movement - consider selling
        self._place_sell_order(current_price, target_price)
    else:
        # Predicted minimal movement - hold
        pass
```

### Files to Modify

1. **`Utilities/backtester.py`** - Main backtesting engine
2. **`Utilities/ppo_backtester.py`** - PPO-specific backtesting logic
3. **Any custom backtesting scripts** in the project

## 4.2: Refactoring the API Hosting Service

### Key Changes Required

The API hosting service must also maintain state for delta calculations and handle the stateful nature of the model loaders.

### Implementation Steps

#### 1. Stateful API Endpoints
Modify API endpoints to maintain state:

```python
class TradingAPI:
    def __init__(self):
        self.model_loaders = {}
        self.previous_features = {}  # Store previous features per model
        
    def predict_endpoint(self, model_name: str, features: Dict[str, float]):
        """
        API endpoint for making predictions.
        Maintains state for delta calculations.
        """
        if model_name not in self.model_loaders:
            raise ValueError(f"Model {model_name} not found")
            
        model_loader = self.model_loaders[model_name]
        
        # Store previous features for delta calculation
        if model_name not in self.previous_features:
            self.previous_features[model_name] = features
            raise ValueError("Not enough historical data. This is the first data point.")
        
        # Calculate deltas
        delta_features = self._calculate_delta_features(
            features, 
            self.previous_features[model_name]
        )
        
        # Make prediction
        prediction = model_loader.predict(delta_features)
        
        # Update state
        self.previous_features[model_name] = features
        
        return self._format_prediction_response(prediction, model_loader)
```

#### 2. Batch Prediction Handling
For batch predictions, maintain state across the batch:

```python
def batch_predict_endpoint(self, model_name: str, features_list: List[Dict[str, float]]):
    """
    API endpoint for batch predictions.
    Maintains state across the entire batch.
    """
    if model_name not in self.model_loaders:
        raise ValueError(f"Model {model_name} not found")
        
    model_loader = self.model_loaders[model_name]
    predictions = []
    
    for i, features in enumerate(features_list):
        if i == 0:
            # First prediction - need previous data
            if model_name not in self.previous_features:
                raise ValueError("No previous data available for first prediction in batch")
            delta_features = self._calculate_delta_features(
                features, 
                self.previous_features[model_name]
            )
        else:
            # Subsequent predictions - use previous in batch
            delta_features = self._calculate_delta_features(
                features, 
                features_list[i-1]
            )
        
        prediction = model_loader.predict(delta_features)
        predictions.append(prediction)
    
    return predictions
```

#### 3. Model Initialization
Ensure model loaders are properly initialized with state:

```python
def load_model(self, model_name: str, model_path: str):
    """
    Load a model and initialize its state.
    """
    if model_name.endswith('_xgboost'):
        from NetworkConfigs.XGBoost_loader import XGBoostModelLoader
        self.model_loaders[model_name] = XGBoostModelLoader(model_path)
    elif model_name.endswith('_nn'):
        from NetworkConfigs.NN_loader import NNModelLoader
        self.model_loaders[model_name] = NNModelLoader(model_path)
    elif model_name.endswith('_ppo'):
        from NetworkConfigs.PPO_loader import PPOModelLoader
        self.model_loaders[model_name] = PPOModelLoader(model_path)
    
    # Initialize state tracking
    self.previous_features[model_name] = None
```

#### 4. Error Handling
Add proper error handling for stateful operations:

```python
def _calculate_delta_features(self, current_features, previous_features):
    """
    Calculate delta features with error handling.
    """
    try:
        delta_features = current_features.copy()
        
        price_columns = ['close', 'open', 'high', 'low']
        for col in price_columns:
            if col in delta_features and col in previous_features:
                delta_features[col] = current_features[col] - previous_features[col]
        
        return delta_features
    except Exception as e:
        raise ValueError(f"Error calculating delta features: {e}")
```

### Files to Modify

1. **`Utilities/Api_Loader.py`** - Main API loader
2. **`web/js/api-hosting.js`** - Frontend API integration
3. **Any API hosting scripts** in the project

## 4.3: Testing the Refactored System

### Unit Tests
Create unit tests to verify delta calculations:

```python
def test_delta_calculation():
    """Test that delta features are calculated correctly."""
    current_features = {'close': 100.0, 'open': 99.0, 'high': 101.0, 'low': 98.0}
    previous_features = {'close': 99.0, 'open': 98.0, 'high': 100.0, 'low': 97.0}
    
    delta_features = calculate_delta_features(current_features, previous_features)
    
    assert delta_features['close'] == 1.0
    assert delta_features['open'] == 1.0
    assert delta_features['high'] == 1.0
    assert delta_features['low'] == 1.0
```

### Integration Tests
Test the full pipeline:

```python
def test_backtesting_with_deltas():
    """Test that backtesting works with delta features."""
    # Load test data
    data = load_test_data()
    
    # Initialize model loader
    model_loader = XGBoostModelLoader('path/to/model')
    
    # Run backtest
    backtester = BacktestingEngine(model_loader, data)
    results = backtester.run_backtest()
    
    # Verify results
    assert len(results) > 0
    assert 'total_return' in results
```

## 4.4: Migration Strategy

### Phase 1: Update Training
1. ✅ Create `Utilities/data_utils.py` with `prepare_delta_features`
2. ✅ Refactor all trainer files to use delta features
3. ✅ Retrain all models with delta features

### Phase 2: Update Loaders
1. ✅ Refactor all loader files to be stateful
2. ✅ Add delta calculation logic to predict methods
3. ✅ Test loaders individually

### Phase 3: Update Backtesting
1. 🔄 Refactor backtesting engine to maintain state
2. 🔄 Update feature extraction to return dictionaries
3. 🔄 Test backtesting with new delta features

### Phase 4: Update API
1. 🔄 Refactor API to maintain state per model
2. 🔄 Update batch prediction endpoints
3. 🔄 Test API with stateful predictions

### Phase 5: Integration Testing
1. 🔄 Test full pipeline end-to-end
2. 🔄 Compare results with previous system
3. 🔄 Performance testing

## 4.5: Important Considerations

### Data Consistency
- Ensure that the same feature extraction logic is used in training, backtesting, and API
- Verify that delta calculations are identical across all components

### State Management
- Consider using a database or cache for state persistence in production
- Implement state reset functionality for testing
- Add logging for state transitions

### Performance
- Delta calculations add minimal overhead
- State management may require additional memory
- Consider batching for high-frequency trading

### Error Handling
- First data point will always fail - handle gracefully
- Add validation for feature consistency
- Implement fallback strategies for missing data

## 4.6: Rollback Plan

If issues arise, the system can be rolled back by:
1. Reverting loader files to non-stateful versions
2. Using original feature extraction in backtesting
3. Removing delta calculations from API endpoints

The training data and models remain compatible, so no retraining is required for rollback.