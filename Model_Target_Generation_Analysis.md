# Model Target Generation Analysis

This document provides a detailed analysis of how each model type generates targets in the Galaxy Models training system.

## Overview

The training system includes 6 different model types, each with distinct target generation strategies:

1. **PPO Ensemble Trainer** - Reinforcement Learning with trading environment rewards
2. **Transformer Trainer** - Time-series forecasting with price change targets
3. **PPO Trainer** - Reinforcement Learning with equity-based rewards
4. **XGBoost Trainer** - Classification with magnitude-based trading signals
5. **Neural Network Trainer** - Regression with price change in ticks
6. **Ensemble Trainer** - Meta-learning combining multiple model predictions

---

## 1. PPO Ensemble Trainer (`PPOEnsembleTrainer.py`)

### Target Generation Method: **Reinforcement Learning Rewards**

The PPO Ensemble model doesn't generate traditional targets. Instead, it uses a **reward-based system** within a custom trading environment.

#### How Rewards are Generated:

**Environment Setup:**
- Creates a `PPOEnsembleEnvironment` that simulates trading
- Actions: 0=Hold, 1=Buy, 2=Sell
- Uses individual model predictions as features for decision making

**Reward Calculation Function:**
```python
def step(self, action):
    # 1. Calculate current portfolio equity
    current_equity = self.balance + self.unrealized_pnl
    
    # 2. Calculate reward based on equity change
    previous_equity = self.equity_history[-2]
    if previous_equity > 1e-6:
        equity_change = (current_equity - previous_equity) / previous_equity
        reward = equity_change * 100  # Scale reward
    else:
        reward = (current_equity - previous_equity)
    
    # 3. Add penalty for holding
    if action == 0:
        reward -= 0.001
    
    # 4. Bankruptcy penalty
    if current_equity < (self.initial_balance * 0.05):
        reward = -200  # Large penalty for going bankrupt
```

**Key Features:**
- **Equity-based rewards**: Rewards are proportional to portfolio value changes
- **Action penalties**: Small penalty for holding to encourage trading
- **Bankruptcy protection**: Large negative reward when equity drops below 5% of initial balance
- **Position management**: Tracks long/short positions and unrealized PnL

---

## 2. Transformer Trainer (`TransformerTrainer.py`)

### Target Generation Method: **Future Price Change Prediction**

The Transformer model generates targets by predicting future price changes using a **delta-based approach**.

#### Target Generation Process:

**1. Data Preprocessing:**
```python
def prepare_delta_data(data, feature_names, delta_cols):
    # Convert price data to deltas (differences)
    for col in delta_cols:  # ['close', 'open', 'high', 'low']
        if col in df.columns:
            df[col] = df[col].diff()  # Calculate price differences
    return df.iloc[1:].values  # Drop first row with NaN
```

**2. Sequence Creation:**
```python
def create_sequences(self, data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]  # Input sequence
        y = data[i + seq_length, 0]   # Target: next close price (first column)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)
```

**3. Scaling:**
- **Feature scaler**: StandardScaler on all input features
- **Target scaler**: StandardScaler on target values only
- Separate scaling prevents data leakage

**Key Features:**
- **Delta transformation**: Converts absolute prices to price changes
- **Sequence-based**: Uses 60 timesteps to predict next timestep
- **Dual scaling**: Separate scalers for features and targets
- **Time-series specific**: Designed for sequential data patterns

---

## 3. PPO Trainer (`PPOTrainer.py`)

### Target Generation Method: **Reinforcement Learning with Equity Rewards**

Similar to PPO Ensemble but operates on individual market data without ensemble predictions.

#### Reward Calculation:

**Environment Setup:**
- Uses `TradingEnvironment` class
- Actions: 0=Hold, 1=Buy, 2=Sell
- Tracks portfolio state (balance, position, unrealized PnL)

**Reward Function:**
```python
def step(self, action):
    # Calculate total equity
    total_equity = self.balance + self.unrealized_pnl
    
    # Reward based on equity change
    if len(self.equity_history) > 1:
        equity_change = (total_equity - self.equity_history[-2]) / self.initial_balance
        reward = equity_change * 100  # Scale reward
    
    # Penalty for holding
    if action == 0:
        reward -= 0.001
```

**Key Features:**
- **Direct equity tracking**: Monitors portfolio value changes
- **Position management**: Handles long/short positions
- **Action incentives**: Encourages trading through holding penalties
- **Scaled rewards**: Multiplies equity changes by 100 for better learning

---

## 4. XGBoost Trainer (`XGboostTrainer.py`)

### Target Generation Method: **Magnitude-Based Classification Labels**

Generates discrete trading signals based on future price movement magnitude.

#### Label Generation Process:

**1. Future Price Analysis:**
```python
def generate_labels(data, look_ahead_periods, min_tick_change, strong_tick_change, tick_size):
    # Find max high and min low across future periods
    future_highs = [data['high'].shift(-p) for p in look_ahead_periods]
    future_lows = [data['low'].shift(-p) for p in look_ahead_periods]
    
    max_future_high = pd.concat(future_highs, axis=1).max(axis=1)
    min_future_low = pd.concat(future_lows, axis=1).min(axis=1)
```

**2. Threshold-Based Classification:**
```python
# Calculate price thresholds
weak_price_threshold = min_tick_change * tick_size
strong_price_threshold = strong_tick_change * tick_size

# Define conditions
strong_buy_condition = (max_future_high >= data['close'] + strong_price_threshold)
weak_buy_condition = (max_future_high >= data['close'] + weak_price_threshold)
strong_sell_condition = (min_future_low <= data['close'] - strong_price_threshold)
weak_sell_condition = (min_future_low <= data['close'] - weak_price_threshold)
```

**3. Label Assignment:**
- **3-class system**: Strong Sell (0), Neutral (1), Strong Buy (2)
- **5-class system**: Strong Sell, Weak Sell, Hold, Weak Buy, Strong Buy
- Uses magnitude of future price movement, not direction

**Key Features:**
- **Magnitude-based**: Focuses on size of price moves, not just direction
- **Multi-period analysis**: Looks across multiple future periods
- **Tick-based thresholds**: Uses configurable tick sizes for thresholds
- **Flexible classification**: Supports both 3-class and 5-class systems

---

## 5. Neural Network Trainer (`NNTrainer.py`)

### Target Generation Method: **Price Change Regression in Ticks**

Generates continuous targets representing future price changes in tick units.

#### Target Generation Process:

**1. Future Price Calculation:**
```python
def prepare_regression_data(data, look_ahead_period, tick_size, columns_to_exclude):
    # Calculate future close price
    future_close = data['close'].shift(-look_ahead_period)
    
    # Create target as price change in ticks
    data['target'] = (future_close - data['close']) / tick_size
```

**2. Data Preparation:**
- Converts price changes to tick units (e.g., 0.25 for NQ futures)
- Drops NaN values created by the shift operation
- Aligns features with targets

**3. Delta Feature Processing:**
```python
# Convert features to deltas
feature_df_delta = prepare_delta_features(feature_df_raw)
X_sample = feature_df_delta.values
y_sample = y_sample_raw[1:]  # Align labels after delta processing
```

**Key Features:**
- **Tick-based targets**: Converts price changes to standardized tick units
- **Regression approach**: Predicts continuous values rather than discrete classes
- **Delta features**: Uses price changes as input features
- **Configurable lookahead**: Adjustable future prediction period

---

## 6. Ensemble Trainer (`EnsembleTrainer.py`)

### Target Generation Method: **Meta-Learning from Base Model Predictions**

The Ensemble trainer doesn't generate its own targets but combines predictions from multiple base models.

#### Target Generation Process:

**1. Base Model Prediction Collection:**
```python
def collect_predictions(self, X):
    predictions = {}
    for model_name, loader in self.model_loaders.items():
        # Collect predictions from each base model
        model_predictions = self._collect_sequential_predictions(loader, X, model_name)
        predictions[model_name] = np.array(model_predictions)
    return predictions
```

**2. Ensemble Combination Methods:**

**Averaging Ensemble:**
```python
def create_averaging_ensemble(self, predictions):
    pred_array = np.array(list(predictions.values()))
    ensemble_pred = np.mean(pred_array, axis=0)  # Simple average
```

**Weighted Ensemble:**
```python
def create_weighted_ensemble(self, predictions):
    ensemble_pred = None
    for model_name, pred in predictions.items():
        weight = self.weights.get(model_name, 1.0 / len(predictions))
        if ensemble_pred is None:
            ensemble_pred = weight * pred
        else:
            ensemble_pred += weight * pred
```

**Stacking Ensemble:**
```python
def create_stacking_ensemble(self, predictions):
    # Use base model predictions as features for meta-learner
    meta_features = np.column_stack(list(predictions.values()))
    meta_learner.fit(meta_features, self.y_train)
    ensemble_pred = meta_learner.predict(meta_features)
```

**Key Features:**
- **Meta-learning**: Uses base model predictions as features
- **Multiple combination methods**: Averaging, weighted, voting, stacking
- **Flexible architecture**: Can combine different model types
- **No direct target generation**: Inherits targets from base models

---

## Summary of Target Generation Approaches

| Model Type | Target Type | Generation Method | Key Characteristics |
|------------|-------------|-------------------|-------------------|
| **PPO Ensemble** | Rewards | Equity-based RL | Portfolio value changes, action penalties |
| **Transformer** | Regression | Future price deltas | Sequence-based, dual scaling |
| **PPO** | Rewards | Equity-based RL | Direct portfolio tracking |
| **XGBoost** | Classification | Magnitude thresholds | Multi-period analysis, tick-based |
| **Neural Network** | Regression | Price change in ticks | Delta features, continuous targets |
| **Ensemble** | Meta-learning | Base model predictions | Combination of multiple approaches |

## Common Patterns

1. **Delta Processing**: Most models use price changes rather than absolute prices
2. **Tick Standardization**: Many models convert prices to tick units for consistency
3. **Future Lookahead**: All models predict future values (1-5 periods ahead)
4. **Data Alignment**: Careful handling of NaN values from shift operations
5. **Scaling**: Most models use StandardScaler for feature normalization

This analysis shows that each model type is optimized for different aspects of trading prediction, from discrete classification to continuous regression to reinforcement learning approaches.
