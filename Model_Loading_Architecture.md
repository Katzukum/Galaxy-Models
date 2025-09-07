# Galaxy Models - Model Loading and Serving Architecture

## Overview
This document maps out how the Galaxy Models app loads and serves machine learning models through a REST API. The architecture involves multiple components working together to provide model hosting capabilities.

## Architecture Components

### 1. Main Application (`Main.py`)
The main EEL-based web application that provides the user interface and orchestrates model loading.

#### Key Global Variables
- `api_process` (line 394): Stores the subprocess handle for the running API server
- `api_server_config` (line 395): Stores the configuration used to start the API server
- `api_logs` (line 396): Array storing API server logs for display in the UI

#### Key Functions

##### `start_api_server(config)` (lines 399-450)
**Purpose**: Starts the API server subprocess with the specified model
**Parameters**:
- `config`: Dictionary containing model configuration
  - `model_path`: Path to the model's YAML config file
  - `host`: Host address for the API server
  - `port`: Port number for the API server

**Process Flow**:
1. Validates no API server is already running
2. Extracts model directory: `model_dir = os.path.dirname(config['model_path'])` (line 416)
3. Constructs uvicorn command: `['python', '-m', 'uvicorn', 'Utilities.Api_Loader:app', '--host', config['host'], '--port', str(config['port']), '--reload']` (lines 420-425)
4. Sets environment variable: `env['MODEL_DIR'] = model_dir` (line 434)
5. Starts subprocess with environment variables

##### `test_api_prediction(test_data)` (lines 544-567)
**Purpose**: Tests the running API server with sample data
**Parameters**:
- `test_data`: Dictionary containing `features` key with feature data

**Process Flow**:
1. Validates API server is running
2. Constructs API URL: `f"http://{api_server_config['host']}:{api_server_config['port']}/predict"` (line 553)
3. Makes HTTP POST request to `/predict` endpoint
4. Returns success/error response

### 2. API Loader (`Utilities/Api_Loader.py`)
The FastAPI application that loads and serves the actual machine learning models.

#### Key Global Variables
- `MODEL_DIR` (line 39): Path to the model directory, determined by `get_model_dir()`
- `model_loader` (line 89): Global instance of the model loader (NNModelLoader or TransformerModelLoader)

#### Key Functions

##### `get_model_dir()` (lines 22-37)
**Purpose**: Determines which model directory to load
**Priority Order**:
1. Command line argument `--model-dir`
2. Environment variable `MODEL_DIR`
3. Default fallback: `"../Models/NN_300Tick_NQ_SuperCCI"`

**Returns**: String path to model directory

##### `get_model_loader(model_dir)` (lines 47-74)
**Purpose**: Creates the appropriate model loader based on model type
**Parameters**:
- `model_dir`: Path to the model directory

**Process Flow**:
1. Finds YAML config file in the directory
2. Reads config to determine model type: `config.get('Type', 'nn').lower()` (line 62)
3. Returns appropriate loader:
   - `NNModelLoader` for 'nn', 'neural network', 'neural network (regression)'
   - `TransformerModelLoader` for 'transformer', 'time-series transformer'
   - Defaults to `NNModelLoader` for unknown types

**Returns**: Instance of model loader class

#### Model Loading Process (lines 85-100)
**Process Flow**:
1. Calls `get_model_dir()` to determine model directory
2. Validates directory exists
3. Calls `get_model_loader(MODEL_DIR)` to create loader
4. If successful: `model_loader` is set to the loader instance
5. If failed: `model_loader` is set to `None` and error is logged

#### API Endpoints

##### `GET /info` (lines 130-149)
**Purpose**: Returns model metadata
**Response**: `ModelInfoResponse` containing:
- `model_name`: Name of the loaded model
- `model_type`: Type of the model (from config)
- `features_required`: List of required feature names

**Error Handling**:
- Returns 503 if `model_loader` is `None`
- Returns 500 if error retrieving model information

##### `POST /predict` (lines 151-179)
**Purpose**: Makes predictions using the loaded model
**Request**: `PredictionRequest` containing:
- `features`: Dictionary of feature names to values

**Process Flow**:
1. Validates `model_loader` is not `None`
2. Removes 'date' and 'time' columns if present
3. Calls `model_loader.predict(request.features)`
4. Returns `PredictionResponse` with prediction result

**Error Handling**:
- Returns 503 if model not available
- Returns 400 for feature mismatch errors
- Returns 500 for other internal errors

### 3. Model Loaders

#### Neural Network Loader (`NetworkConfigs/NN_loader.py`)

##### `NNModelLoader.__init__(model_dir)` (lines 19-61)
**Purpose**: Loads a neural network model and its artifacts
**Process Flow**:
1. Validates model directory exists
2. Loads YAML config file
3. Extracts model metadata:
   - `self.model_name`: From `config['model_name']`
   - `self.features`: From `config['Config']['features']`
   - `self.architecture`: From `config['Config']['architecture']`
4. Loads scaler from pickle file
5. Builds PyTorch model from architecture config
6. Loads model state dict
7. Sets model to evaluation mode

##### `NNModelLoader.predict(feature_dict)` (lines 89-128)
**Purpose**: Makes predictions on single samples
**Process Flow**:
1. Validates feature names match expected features
2. Orders features according to model's expected order
3. Scales features using loaded scaler
4. Converts to PyTorch tensor
5. Runs inference with `torch.no_grad()`
6. Returns prediction as float

#### Transformer Loader (`NetworkConfigs/Transformer_loader.py`)

##### `TransformerModelLoader.__init__(model_dir)` (lines 94-141)
**Purpose**: Loads a transformer model and its artifacts
**Process Flow**:
1. Validates model directory exists
2. Loads YAML config file
3. Extracts model metadata:
   - `self.model_name`: From `config['model_name']`
   - `self.features`: From `config['Config']['data_params']['features']`
   - `self.sequence_length`: From `config['Config']['data_params']['sequence_length']`
4. Initializes history buffer: `self.history = deque(maxlen=self.sequence_length)`
5. Loads scaler from pickle file
6. Creates `TimeSeriesTransformer` model with config parameters
7. Loads model state dict
8. Sets model to evaluation mode

##### `TransformerModelLoader.predict(feature_dict)` (lines 151-183)
**Purpose**: Makes predictions using sequence history
**Process Flow**:
1. Validates feature names match expected features
2. Adds current features to history buffer
3. Checks if history buffer is full (has `sequence_length` entries)
4. Converts full sequence to numpy array
5. Scales entire sequence using scaler
6. Converts to PyTorch tensor with shape `[batch_size, seq_len, n_features]`
7. Runs inference with `torch.no_grad()`
8. Returns prediction as float

## Complete Data Flow

### 1. App Startup
```
Main.py → EEL Web Interface → User selects model → start_api_server(config)
```

### 2. API Server Startup
```
start_api_server() → subprocess.Popen() → uvicorn → Api_Loader.py → get_model_dir() → get_model_loader() → Model Loader.__init__()
```

### 3. Model Loading
```
get_model_loader() → Read YAML config → Determine model type → Create appropriate loader → Load artifacts → Load PyTorch model
```

### 4. API Request Processing
```
HTTP Request → FastAPI endpoint → model_loader.predict() → PyTorch inference → HTTP Response
```

### 5. Model Prediction Flow
```
Feature Dictionary → Validation → Scaling → Tensor Conversion → Model Forward Pass → Prediction Float
```

## Key Configuration Files

### Model YAML Structure
```yaml
model_name: "ModelName"
Type: "ModelType"  # "Neural Network (Regression)" or "Time-Series Transformer"
artifact_paths:
  scaler: "model_scaler.pkl"
  model_state_dict: "model_model.pt"  # or "model" for NN
Config:
  # For NN models:
  features: [list of feature names]
  architecture: [list of layer configs]
  
  # For Transformer models:
  data_params:
    features: [list of feature names]
    sequence_length: 60
  model_params:
    d_model: 64
    nhead: 4
    # ... other transformer parameters
```

## Error Handling Points

1. **Model Directory Not Found**: `EnvironmentError` in `get_model_dir()`
2. **Model Loading Failure**: `model_loader = None` in `Api_Loader.py`
3. **Feature Mismatch**: `ValueError` in model loader `predict()` methods
4. **API Server Not Running**: 503 responses from API endpoints
5. **Invalid JSON**: Syntax errors in test requests

## Environment Variables

- `MODEL_DIR`: Path to the model directory (set by `start_api_server()`)

## Dependencies

- **FastAPI**: Web framework for API endpoints
- **Uvicorn**: ASGI server for running FastAPI
- **PyTorch**: Deep learning framework for model inference
- **EEL**: Python-JavaScript bridge for web interface
- **YAML**: Configuration file parsing
- **Pickle**: Model artifact serialization
