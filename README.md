# Galaxy Models

## Overview

Galaxy Models is a robust platform for loading, serving, and interacting with machine learning models. The project is designed to provide a seamless way to host models through a REST API, and it features a modern web-based dashboard for user interaction. The system is modular, scalable, and built to support Agile workflows.

---

## Features

### **Model Loading and Serving**
- **Architecture**: Designed to load and serve machine learning models, including neural networks and transformers.
- **API Endpoints**:
  - `GET /info`: Retrieves metadata about the loaded model.
  - `POST /predict`: Sends prediction requests to the loaded model.
- **Error Handling**:
  - Graceful handling of model unavailability.
  - Comprehensive validation for input features.

### **Web Dashboard**
- **Theme System**:
  - CSS custom properties for easy theme changes.
  - Modern glassmorphism design with responsiveness for mobile devices.
- **Tab System**:
  - Modular design for functionality like model library display and training interface.
  - Easy maintenance by separating HTML, CSS, and JavaScript.

---

## Architecture

### Components
1. **Main Application (`Main.py`)**:
   - Orchestrates model loading.
   - Provides a user interface for interacting with models.
   - Handles API server subprocesses and logs.
2. **API Loader (`Utilities/Api_Loader.py`)**:
   - FastAPI application for serving models.
   - Manages global variables like `MODEL_DIR` and `model_loader`.
   - Handles key functions for model directory validation and loader initialization.
3. **Model Loaders**:
   - **Neural Network Loader (`NetworkConfigs/NN_loader.py`)**:
     - Initializes and serves neural network models.
     - Performs predictions using provided features.
   - **Transformer Loader (`NetworkConfigs/Transformer_loader.py`)**:
     - Initializes and serves transformer models.
     - Processes time-series data for predictions.

---

## Key Functionalities

### API Endpoints
- **GET /info**:
  - Returns model name, type, and required features.
  - Error codes for unavailability and server issues.
- **POST /predict**:
  - Predicts outcomes based on input features.
  - Validates input and handles errors like feature mismatches.

### Model Loading Process
1. Determine the model directory through command-line arguments, environment variables, or default paths.
2. Validate the directory and initialize the appropriate loader based on the model type.
3. Log errors for failed initialization.

---

## Development Benefits
- **Scalability**: Modular design supports team development and Agile workflows.
- **Ease of Maintenance**: Adding new features is as simple as creating new modules.
- **Separation of Concerns**: Logical separation of code into HTML, CSS, and JavaScript.

---

## Environment Setup
- **Dependencies**: All required libraries and packages are listed in the project's configuration files.
- **Environment Variables**:
  - `MODEL_DIR`: Path to the model directory.
  - Other variables for API server configuration.

---

## Contributions
Contributions are welcome! Please adhere to the project's coding standards and submit pull requests for review.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.
