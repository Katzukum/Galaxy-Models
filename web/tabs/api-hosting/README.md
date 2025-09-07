# API Hosting Tab

The API Hosting tab allows you to host your trained machine learning models as REST APIs for real-time predictions.

## Features

- **Model Selection**: Choose from your trained models in the model library
- **API Configuration**: Configure host, port, title, and description
- **Real-time Status**: Monitor API server status and uptime
- **API Testing**: Test your API with sample data directly from the interface
- **Interactive Documentation**: Access FastAPI's automatic documentation
- **Logs Management**: View and export API server logs

## How to Use

1. **Select a Model**: Choose a trained model from the dropdown list
2. **Configure API Settings**: Set the host address, port, title, and description
3. **Start API Server**: Click "Start API Server" to launch the FastAPI server
4. **Test the API**: Use the built-in testing interface to verify predictions
5. **Access Documentation**: Click "View API Docs" to open the interactive documentation

## API Endpoints

When your API server is running, it provides the following endpoints:

- `GET /` - Root endpoint with basic information
- `GET /info` - Model information and required features
- `POST /predict` - Make predictions with feature data
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation (ReDoc)

## Example Usage

### Making a Prediction

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "features": {
         "Open": 18000.25,
         "High": 18005.50,
         "Low": 18000.00,
         "Close": 18004.75,
         "Volume": 1500,
         "Feature6": 0.5,
         "Feature7": -0.2,
         "Feature8": 1.1,
         "Feature9": 17990.0,
         "Feature10": 17985.5
       }
     }'
```

### Response

```json
{
  "model_name": "NN_300T_NQ_SuperCCI",
  "predicted_price_change_ticks": 2.5
}
```

## Model Support

The API Hosting tab supports all model types:

- **Neural Networks (NN)**: Regression models for price change prediction
- **Transformers**: Time-series forecasting models
- **XGBoost**: Classification models for trading actions

## Requirements

- FastAPI
- Uvicorn
- Requests
- Your trained model files

## Notes

- The API server runs as a separate process
- Model files are loaded dynamically based on your selection
- Temporary API loader files are created and cleaned up automatically
- The server can be stopped and restarted with different models
