# 🚀 Galaxy Models - Advanced Machine Learning Trading Platform

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Supported Model Types](#supported-model-types)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [API Documentation](#api-documentation)
- [Web Dashboard](#web-dashboard)
- [Configuration](#configuration)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🌟 Overview

Galaxy Models is a comprehensive machine learning platform designed specifically for financial trading applications. It provides a complete ecosystem for training, serving, and managing various types of ML models with a modern web-based interface and robust API infrastructure.

### Key Capabilities
- **Multi-Model Support**: Neural Networks, Transformers, XGBoost, PPO Agents, and Ensemble Models
- **Web Dashboard**: Modern, responsive interface for model management and training
- **REST API**: FastAPI-based serving infrastructure for production deployments
- **Backtesting Engine**: Comprehensive backtesting with detailed performance metrics
- **Real-time Training**: Live training progress monitoring and log streaming
- **Model Library**: Centralized model management and discovery

## ✨ Features

### 🎯 Core Functionality
- **Model Training**: Train multiple model types with customizable parameters
- **Ensemble Learning**: Combine multiple models for improved predictions
- **Backtesting**: Historical performance analysis with detailed metrics
- **API Hosting**: Deploy models as REST APIs for production use
- **Real-time Monitoring**: Live training progress and model performance tracking

### 🖥️ User Interface
- **Modern Design**: Glassmorphism UI with responsive mobile support
- **Tab-based Navigation**: Organized interface for different functionalities
- **Real-time Updates**: Live progress bars and status indicators
- **Interactive Charts**: Visual performance analysis and equity curves
- **File Management**: Drag-and-drop CSV upload and model file management

### 🔧 Technical Features
- **Modular Architecture**: Clean separation of concerns for maintainability
- **Error Handling**: Comprehensive error handling and user feedback
- **Debugging Tools**: Built-in debug modes for development and troubleshooting
- **Configuration Management**: YAML-based configuration system
- **Scalable Design**: Support for multiple concurrent training processes

## 🏗️ Architecture

### System Components

```
Galaxy Models
├── Main Application (Main.py)
│   ├── EEL Web Interface
│   ├── Training Management
│   ├── API Server Management
│   └── Model Discovery
├── Network Configs
│   ├── Model Loaders (NN, Transformer, XGBoost, PPO, Ensemble)
│   ├── Model Trainers
│   └── Model Architecture Definitions
├── Utilities
│   ├── API Loader (FastAPI)
│   ├── Backtester
│   ├── Data Utils
│   └── YAML Utils
└── Web Interface
    ├── HTML/CSS/JS Frontend
    ├── Tab-based UI Components
    └── Real-time Communication (EEL)
```

### Data Flow

1. **Training Pipeline**: CSV Data → Data Preprocessing → Model Training → Artifact Saving
2. **Prediction Pipeline**: Feature Input → Model Loading → Prediction → Response
3. **Backtesting Pipeline**: Historical Data → Model Loading → Simulation → Performance Metrics
4. **API Pipeline**: HTTP Request → Model Prediction → JSON Response

## 🤖 Supported Model Types

### 1. Neural Networks (Regression)
- **Purpose**: Price change prediction in ticks
- **Architecture**: Configurable multi-layer perceptron
- **Input**: OHLCV + technical indicators
- **Output**: Predicted price change in ticks
- **Use Case**: Short-term price movement prediction

**Key Parameters:**
- Hidden layers (1-5)
- Neurons per layer (16-512)
- Dropout rate (0-0.7)
- Optimizer (Adam, SGD, RMSprop)
- Loss function (MSE, L1, Smooth L1)

### 2. Time-Series Transformers
- **Purpose**: Future price forecasting
- **Architecture**: Transformer encoder with positional encoding
- **Input**: Sequence of historical data (60 timesteps)
- **Output**: Forecasted future price
- **Use Case**: Medium-term price forecasting

**Key Parameters:**
- Internal dimension (d_model): 32-512
- Attention heads: 1-16
- Encoder layers: 1-12
- Feedforward dimension: 64-1024
- Sequence length: 10-200
- Dropout rate: 0-0.5

### 3. XGBoost Classifiers
- **Purpose**: Trading action classification
- **Architecture**: Gradient boosting decision trees
- **Input**: OHLCV + technical indicators
- **Output**: Trading action (Strong Buy, Neutral, Strong Sell)
- **Use Case**: Categorical trading decisions

**Key Parameters:**
- Number of estimators: 50-1000
- Learning rate: 0.01-0.3
- Maximum depth: 2-10
- Look-ahead periods: Configurable
- Min/Strong tick thresholds: Customizable

### 4. PPO Agents (Reinforcement Learning)
- **Purpose**: Adaptive trading strategy learning
- **Architecture**: Actor-Critic neural network
- **Input**: Historical data + account state
- **Output**: Trading action (Hold, Buy, Sell)
- **Use Case**: Adaptive strategy optimization

**Key Parameters:**
- Hidden dimension: 64-512
- Lookback window: 20-200
- Learning rate: 0.0001-0.01
- PPO epochs: 1-10
- Clip ratio: 0.1-0.5
- Value/Entropy coefficients: Configurable

### 5. Ensemble Models
- **Purpose**: Combine multiple models for improved predictions
- **Types**: Voting, Averaging, Weighted, Stacking
- **Input**: Same as component models
- **Output**: Combined prediction
- **Use Case**: Robust prediction systems

**Ensemble Types:**
- **Voting**: Majority vote for classification
- **Averaging**: Mean prediction for regression
- **Weighted**: Custom weight combination
- **Stacking**: Meta-learning approach

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Installation Steps

1. **Clone the Repository**
```bash
git clone <repository-url>
cd galaxy-models
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify Installation**
```bash
python Main.py --help
```

### Dependencies
```
eel>=0.16.0          # Web interface framework
PyYAML>=6.0          # Configuration management
fastapi>=0.100.0     # API framework
uvicorn>=0.20.0      # ASGI server
requests>=2.28.0     # HTTP client
torch>=2.0.0         # PyTorch for neural networks
numpy>=1.24.0        # Numerical computing
scikit-learn>=1.3.0  # Machine learning utilities
pandas>=2.0.0        # Data manipulation
gym>=0.26.0          # Reinforcement learning environment
stable-baselines3>=2.0.0  # RL algorithms
xgboost>=1.7.0       # Gradient boosting
```

## 📖 Usage Guide

### Starting the Application

1. **Launch the Web Interface**
```bash
python Main.py
```

2. **Access the Dashboard**
- Open your browser to the automatically launched URL
- Default: `http://localhost:8000`

### Training Models

1. **Navigate to Training Tab**
2. **Select Model Type** (Neural Network, Transformer, XGBoost, PPO)
3. **Upload CSV Data** or select from available files
4. **Configure Parameters** using the parameter panels
5. **Start Training** and monitor progress in real-time

### Creating Ensemble Models

1. **Navigate to Ensemble Training Tab**
2. **Select Ensemble Type** (Voting, Averaging, Weighted, Stacking)
3. **Choose Component Models** from your model library
4. **Configure Weights** (for weighted ensembles)
5. **Set Advanced Options** (cross-validation, meta-learner)
6. **Start Ensemble Training**

### Running Backtests

1. **Navigate to Backtesting Tab**
2. **Select Model** from your trained models
3. **Choose Data File** for backtesting
4. **Configure Parameters**:
   - Initial capital
   - Take profit/Stop loss levels
   - Tick size and value
5. **Run Backtest** and analyze results

### API Hosting

1. **Navigate to API Hosting Tab**
2. **Select Model** to host
3. **Configure Server Settings**:
   - Host address
   - Port number
4. **Start API Server**
5. **Test Predictions** using the built-in tester

## 🔌 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### GET /info
Returns model metadata and configuration.

**Response:**
```json
{
  "model_name": "MyModel",
  "model_type": "Neural Network (Regression)",
  "features_required": ["open", "high", "low", "close", "volume"]
}
```

#### POST /predict
Makes predictions using the loaded model.

**Request:**
```json
{
  "features": {
    "open": 18000.25,
    "high": 18005.50,
    "low": 18000.00,
    "close": 18004.75,
    "volume": 1500
  }
}
```

**Response (Neural Network):**
```json
{
  "model_name": "MyModel",
  "model_type": "Neural Network (Regression)",
  "predicted_price_change_ticks": 2.5
}
```

**Response (Transformer):**
```json
{
  "model_name": "MyModel",
  "model_type": "Time-Series Transformer",
  "forecasted_value": 18007.25,
  "sequence_length": 60
}
```

**Response (XGBoost):**
```json
{
  "model_name": "MyModel",
  "model_type": "XGBoost Classifier",
  "predicted_action": "Strong Buy",
  "confidence": 0.85,
  "probabilities": {
    "Strong Sell": 0.05,
    "Neutral": 0.10,
    "Strong Buy": 0.85
  }
}
```

**Response (PPO Agent):**
```json
{
  "prediction": 1,
  "action_name": "Buy",
  "confidence": 0.78,
  "value_estimate": 0.65
}
```

**Response (Ensemble):**
```json
{
  "model_name": "MyEnsemble",
  "model_type": "Ensemble Model",
  "predicted_value": 2.3,
  "individual_predictions": {
    "Model1": 2.1,
    "Model2": 2.5,
    "Model3": 2.3
  },
  "ensemble_type": "averaging"
}
```

## 🖥️ Web Dashboard

### Tab Navigation

#### 📚 Model Library
- **Purpose**: Browse and manage trained models
- **Features**:
  - Model discovery and listing
  - Model details and configuration viewing
  - Model performance metrics
  - Model deletion and management

#### 🚀 Training
- **Purpose**: Train new machine learning models
- **Features**:
  - Model type selection
  - Parameter configuration
  - Real-time training progress
  - Training logs and monitoring
  - CSV file upload and management

#### 🎯 Ensemble Training
- **Purpose**: Create ensemble models from existing models
- **Features**:
  - Ensemble type selection
  - Model selection interface
  - Weight configuration
  - Advanced options (cross-validation, meta-learning)
  - Ensemble training progress

#### 📊 Backtesting
- **Purpose**: Test model performance on historical data
- **Features**:
  - Model and data selection
  - Backtest parameter configuration
  - Performance metrics calculation
  - Equity curve visualization
  - Trade analysis and statistics

#### 🌐 API Hosting
- **Purpose**: Deploy models as REST APIs
- **Features**:
  - Model selection for hosting
  - Server configuration
  - API testing interface
  - Server status monitoring
  - Log viewing and management

### User Interface Features

#### Responsive Design
- **Mobile Support**: Optimized for mobile devices
- **Adaptive Layout**: Adjusts to different screen sizes
- **Touch-Friendly**: Large buttons and touch targets

#### Real-time Updates
- **Live Progress**: Real-time training progress bars
- **Status Indicators**: Current operation status
- **Auto-refresh**: Automatic data updates
- **Notification System**: Success/error notifications

#### Interactive Elements
- **Drag & Drop**: File upload with drag-and-drop support
- **Parameter Panels**: Collapsible parameter configuration
- **Modal Dialogs**: Detailed information and confirmations
- **Charts & Graphs**: Interactive performance visualizations

## ⚙️ Configuration

### Model Configuration Files

Models are configured using YAML files with the following structure:

```yaml
model_name: "MyModel"
Type: "Neural Network (Regression)"
artifact_paths:
  scaler: "model_scaler.pkl"
  model: "model.pt"
Config:
  features: ["open", "high", "low", "close", "volume"]
  architecture:
    - type: "Linear"
      in_features: 5
      out_features: 64
    - type: "ReLU"
    - type: "Dropout"
      p: 0.3
    - type: "Linear"
      in_features: 64
      out_features: 1
```

### Training Parameters

#### Neural Network Parameters
```yaml
hidden_layers: 2
neurons_layer1: 64
neurons_layer2: 32
dropout: 0.3
optimizer: "Adam"
loss_function: "MSELoss"
learning_rate: 0.001
epochs: 100
```

#### Transformer Parameters
```yaml
d_model: 64
nhead: 4
num_encoder_layers: 3
dim_feedforward: 128
dropout: 0.1
sequence_length: 60
learning_rate: 0.0005
epochs: 25
batch_size: 32
```

#### XGBoost Parameters
```yaml
objective: "multi:softmax"
eval_metric: "mlogloss"
n_estimators: 150
learning_rate: 0.1
max_depth: 4
look_ahead: [3, 5]
min_tick: 20
strong_tick: 40
```

#### PPO Parameters
```yaml
hidden_dim: 128
num_actions: 3
lookback_window: 60
learning_rate: 0.0003
epochs: 100
batch_size: 64
ppo_epochs: 4
clip_ratio: 0.2
value_coef: 0.5
entropy_coef: 0.01
```

### Environment Variables

```bash
# Model directory (optional)
MODEL_DIR=/path/to/models

# Debug settings
DEBUG_TRAINING=true
DEBUG_API=true
DEBUG_VERBOSE=true
```

## 🛠️ Development

### Project Structure
```
galaxy-models/
├── Main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── NetworkConfigs/        # Model implementations
│   ├── NN_loader.py       # Neural network loader
│   ├── NN_trainer.py      # Neural network trainer
│   ├── Transformer_loader.py
│   ├── Transformer_trainer.py
│   ├── XGBoost_loader.py
│   ├── XGBoost_trainer.py
│   ├── PPO_loader.py
│   ├── PPOTrainer.py
│   ├── Ensemble_loader.py
│   └── EnsembleTrainer.py
├── Utilities/             # Utility modules
│   ├── Api_Loader.py      # FastAPI server
│   ├── backtester.py      # Backtesting engine
│   ├── data_utils.py      # Data processing utilities
│   ├── run_training.py    # Training script
│   └── yaml_utils.py      # YAML configuration utilities
├── web/                   # Web interface
│   ├── index.html         # Main HTML file
│   ├── css/               # Stylesheets
│   ├── js/                # JavaScript modules
│   └── tabs/              # Tab-specific content
└── Models/                # Trained models directory
```

### Adding New Model Types

1. **Create Model Loader** (`NetworkConfigs/NewModel_loader.py`)
2. **Create Model Trainer** (`NetworkConfigs/NewModel_trainer.py`)
3. **Update API Loader** to include new model type
4. **Add UI Components** for training and configuration
5. **Update Documentation** with new model parameters

### Debugging

#### Debug Modes
- **Training Debug**: Opens command windows for training processes
- **API Debug**: Opens command windows for API server processes
- **Verbose Debug**: Enables detailed logging output

#### Enabling Debug Modes
```python
# In Main.py
DEBUG_TRAINING = True   # Show training command windows
DEBUG_API = True        # Show API server command windows
DEBUG_VERBOSE = True    # Enable verbose logging
```

#### Common Debug Commands
```bash
# Check model directory
ls -la Models/

# Verify YAML configuration
python -c "import yaml; print(yaml.safe_load(open('model_config.yaml')))"

# Test API endpoint
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": {"open": 100, "high": 105, "low": 95, "close": 102, "volume": 1000}}'
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Model Loading Errors
**Problem**: Model fails to load
**Solutions**:
- Check model directory exists and contains required files
- Verify YAML configuration format
- Ensure all artifact files are present
- Check file permissions

#### 2. Training Failures
**Problem**: Training process fails or hangs
**Solutions**:
- Verify CSV data format and required columns
- Check available disk space
- Ensure sufficient memory for model training
- Enable debug mode to see detailed error messages

#### 3. API Server Issues
**Problem**: API server won't start or respond
**Solutions**:
- Check port availability
- Verify model directory path
- Check firewall settings
- Review server logs for error messages

#### 4. Web Interface Problems
**Problem**: Dashboard not loading or functioning
**Solutions**:
- Clear browser cache
- Check JavaScript console for errors
- Verify EEL connection
- Restart the application

### Performance Optimization

#### Memory Usage
- Use smaller batch sizes for large datasets
- Implement data streaming for very large files
- Monitor memory usage during training

#### Training Speed
- Use GPU acceleration when available
- Optimize data preprocessing
- Use appropriate model architectures for your data size

#### API Performance
- Implement model caching
- Use connection pooling
- Optimize prediction pipelines

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings for all functions and classes
- Include type hints where appropriate

### Testing
- Test new features thoroughly
- Ensure backward compatibility
- Update documentation for new features
- Test with different model types and configurations

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 📞 Support

For support, questions, or feature requests:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the documentation
- Contact the development team

---

**Galaxy Models** - Empowering traders with advanced machine learning capabilities 🚀
