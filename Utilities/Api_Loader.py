# File: main.py

import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Union
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# --- Core Component: Import the model loader ---
# Import all available model loaders and their response models
from NetworkConfigs.NN_loader import NNModelLoader, NNPredictionResponse
from NetworkConfigs.Transformer_loader import TransformerModelLoader, TransformerPredictionResponse
from NetworkConfigs.XGBoost_loader import XGBoostModelLoader, XGBoostPredictionResponse
from NetworkConfigs.PPO_loader import PPOModelLoader, PPOPredictionResponse



# --- Configuration ---
# Get model directory from command line argument or environment variable
import argparse

def get_model_dir():
    print("[DEBUG] get_model_dir() called")
    parser = argparse.ArgumentParser(description='ML Model API Server')
    parser.add_argument('--model-dir', type=str, help='Path to the model directory')
    args, unknown = parser.parse_known_args()
    print(f"[DEBUG] Command line args: {args}")
    print(f"[DEBUG] Unknown args: {unknown}")
    
    # Try command line argument first
    if args.model_dir:
        print(f"[DEBUG] Using command line model_dir: {args.model_dir}")
        return args.model_dir
    
    # Try environment variable
    model_dir = os.getenv('MODEL_DIR')
    print(f"[DEBUG] Environment MODEL_DIR: {model_dir}")
    if model_dir:
        print(f"[DEBUG] Using environment model_dir: {model_dir}")
        return model_dir
    
    # Default fallback
    default_dir = "Models/NN_300Tick_NQ_SuperCCI"
    print(f"[DEBUG] Using default model_dir: {default_dir}")
    return default_dir

print("[DEBUG] About to call get_model_dir() at module level")
print(f"[DEBUG] Environment MODEL_DIR at import: {os.getenv('MODEL_DIR')}")
MODEL_DIR = get_model_dir()
print(f"[DEBUG] MODEL_DIR set to: {MODEL_DIR}")
if not os.path.exists(MODEL_DIR):
    print(f"[DEBUG] Model directory does not exist: {MODEL_DIR}")
    raise EnvironmentError(
        f"Model directory '{MODEL_DIR}' not found. "
        "Please ensure the model has been trained and the path is correct."
    )
print(f"[DEBUG] Model directory exists: {MODEL_DIR}")

# --- Model Loader Selection ---
def get_model_loader(model_dir):
    """Determine and return the appropriate model loader based on the model type"""
    print(f"[DEBUG] get_model_loader() called with model_dir: {model_dir}")
    import yaml
    
    # Find the config file
    print(f"[DEBUG] Listing directory contents: {os.listdir(model_dir)}")
    config_files = [f for f in os.listdir(model_dir) if f.endswith('.yaml')]
    print(f"[DEBUG] Found YAML files: {config_files}")
    
    if not config_files:
        print(f"[DEBUG] No YAML config files found in {model_dir}")
        raise FileNotFoundError(f"No YAML config file found in {model_dir}")
    
    config_path = os.path.join(model_dir, config_files[0])
    print(f"[DEBUG] Using config file: {config_path}")
    
    # Read the config to determine model type
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"[DEBUG] Loaded config: {config}")
    
    model_type = config.get('Type', 'nn').lower()
    print(f"[DEBUG] Detected model type: {model_type}")
    
    # Select the appropriate loader
    if model_type in ['nn', 'neural network', 'neural network (regression)']:
        print(f"[DEBUG] Creating NNModelLoader")
        return NNModelLoader(model_dir=model_dir)
    # Add other model types as they become available
    elif model_type in ['transformer', 'time-series transformer']:
        print(f"[DEBUG] Creating TransformerModelLoader")
        return TransformerModelLoader(model_dir=model_dir)

    elif model_type in ['xgboost', 'xgboostclassifier']:
        return XGBoostModelLoader(model_dir=model_dir)
    elif model_type in ['ppo', 'ppo agent']:
        print(f"[DEBUG] Creating PPOModelLoader")
        return PPOModelLoader(model_dir=model_dir)
    else:
        # Default to NN loader
        print(f"[DEBUG] Unknown model type, defaulting to NNModelLoader")
        return NNModelLoader(model_dir=model_dir)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="ML Model Serving API",
    description="A generic API to serve predictions from various model types.",
    version="1.0.0"
)

# --- Global Model Loader Instance ---
# The model is loaded once when the application starts.
print(f"[DEBUG] Attempting to load model from: {MODEL_DIR}")
print(f"[DEBUG] Model directory exists: {os.path.exists(MODEL_DIR)}")

try:
    print(f"[DEBUG] Calling get_model_loader() with MODEL_DIR: {MODEL_DIR}")
    model_loader = get_model_loader(MODEL_DIR)
    print(f"[DEBUG] get_model_loader() returned: {model_loader}")
    print(f"[DEBUG] Successfully loaded model from: {MODEL_DIR}")
    print(f"[DEBUG] Model type: {type(model_loader).__name__}")
    print(f"[DEBUG] Model name: {model_loader.model_name}")
except Exception as e:
    print(f"[DEBUG] FATAL: Could not load the model. Error: {e}")
    print(f"[DEBUG] Error type: {type(e).__name__}")
    import traceback
    print("[DEBUG] Full traceback:")
    traceback.print_exc()
    # In a real scenario, you might want the app to fail startup if the model can't load.
    model_loader = None
    print(f"[DEBUG] Set model_loader to None due to error")


# --- API Data Models (using Pydantic) ---
class PredictionRequest(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        example={
            "Open": 18000.25, "High": 18005.50, "Low": 18000.00, "Close": 18004.75,
            "Volume": 1500, "Feature6": 0.5, "Feature7": -0.2, "Feature8": 1.1,
            "Feature9": 17990.0, "Feature10": 17985.5
        }
    )

# Individual response models are now imported from their respective loader files
# We'll use Union type for dynamic responses

class ModelInfoResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str
    model_type: str
    features_required: List[str]

# --- API Endpoints ---
@app.get("/", include_in_schema=False)
def root():
    return {"message": "ML Model Serving API is running. See /docs for details."}

@app.get("/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Returns metadata about the currently loaded model."""
    print("[DEBUG] /info endpoint called")
    print(f"[DEBUG] model_loader is: {model_loader}")
    print(f"[DEBUG] model_loader is None: {model_loader is None}")
    
    if not model_loader:
        print("[DEBUG] model_loader is None, returning 503 error")
        raise HTTPException(
            status_code=503, 
            detail="Model is not available or failed to load. Check the server logs for details."
        )
    
    try:
        print(f"[DEBUG] Getting model info from model_loader")
        print(f"[DEBUG] model_name: {model_loader.model_name}")
        print(f"[DEBUG] model_type: {model_loader.config.get('Type', 'Unknown')}")
        print(f"[DEBUG] features_required: {model_loader.features}")
        
        response = ModelInfoResponse(
            model_name=model_loader.model_name,
            model_type=model_loader.config.get('Type', 'Unknown'),
            features_required=model_loader.features
        )
        print(f"[DEBUG] Created response: {response}")
        return response
    except Exception as e:
        print(f"[DEBUG] Error in /info endpoint: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error retrieving model information: {str(e)}"
        )

@app.post("/predict", response_model=Union[NNPredictionResponse, TransformerPredictionResponse, XGBoostPredictionResponse, PPOPredictionResponse])
async def get_prediction(request: PredictionRequest):
    """
    Accepts a dictionary of features and returns a model prediction.
    The response format depends on the model type.
    """
    if not model_loader:
        raise HTTPException(status_code=503, detail="Model is not available or failed to load.")

    # Remove date/time columns if present
    if "date" in request.features:
        del request.features["date"]
    if "time" in request.features:
        del request.features["time"]

    try:
        # Use the appropriate response creation method based on model type
        if isinstance(model_loader, XGBoostModelLoader):
            return model_loader.create_prediction_response(request.features)
        elif isinstance(model_loader, TransformerModelLoader):
            prediction = model_loader.predict(request.features)
            return model_loader.create_prediction_response(prediction)
        elif isinstance(model_loader, NNModelLoader):
            prediction = model_loader.predict(request.features)
            return model_loader.create_prediction_response(prediction)
        elif isinstance(model_loader, PPOModelLoader):
            prediction_data = model_loader.predict(request.features)
            return model_loader.create_prediction_response(prediction_data)
        else:
            # Fallback for unknown model types
            raise HTTPException(status_code=500, detail="Unknown model type")
            
    except ValueError as e:
        # This catches feature mismatch errors from the loader
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Generic catch-all for other unexpected errors
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")