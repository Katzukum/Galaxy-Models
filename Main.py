import eel
import os
import yaml
import glob
import subprocess
import threading
import time
import json
import requests
from pathlib import Path
from datetime import datetime
from Utilities.yaml_utils import YAMLConfig, load_yaml_config, find_yaml_files, get_common_config_values
from NetworkConfigs.EnsembleTrainer import run_ensemble_training

# Initialize EEL
eel.init('web')

# Debug configuration
DEBUG_TRAINING = True  # Set to True to open command windows for training subprocesses
DEBUG_API = True       # Set to True to open command windows for API subprocesses
DEBUG_VERBOSE = True   # Set to True for additional debug output

# Debug helper function
def debug_print(message):
    """Print debug messages if DEBUG_VERBOSE is enabled"""
    if DEBUG_VERBOSE:
        print(f"[DEBUG] {message}")

# Global variables for training management
training_processes = {}
training_logs = {}
training_status = {}

@eel.expose
def get_debug_status():
    """Get current debug configuration status"""
    return {
        'debug_training': DEBUG_TRAINING,
        'debug_api': DEBUG_API,
        'debug_verbose': DEBUG_VERBOSE
    }

@eel.expose
def set_debug_training(enabled):
    """Enable or disable debug mode for training processes"""
    global DEBUG_TRAINING
    DEBUG_TRAINING = enabled
    debug_print(f"Debug training mode: {'enabled' if enabled else 'disabled'}")
    return {'success': True, 'debug_training': DEBUG_TRAINING}

@eel.expose
def set_debug_api(enabled):
    """Enable or disable debug mode for API processes"""
    global DEBUG_API
    DEBUG_API = enabled
    debug_print(f"Debug API mode: {'enabled' if enabled else 'disabled'}")
    return {'success': True, 'debug_api': DEBUG_API}

@eel.expose
def set_debug_verbose(enabled):
    """Enable or disable verbose debug output"""
    global DEBUG_VERBOSE
    DEBUG_VERBOSE = enabled
    debug_print(f"Debug verbose mode: {'enabled' if enabled else 'disabled'}")
    return {'success': True, 'debug_verbose': DEBUG_VERBOSE}

@eel.expose
def start_ensemble_training(ensemble_type, ensemble_name, selected_models, csv_path, weights=None, advanced_options=None):
    """Start ensemble training process"""
    try:
        debug_print(f"Starting ensemble training: {ensemble_name}")
        debug_print(f"Ensemble type: {ensemble_type}")
        debug_print(f"Selected models: {len(selected_models)}")
        debug_print(f"CSV path: {csv_path}")
        
        # Generate training ID
        training_id = f"ensemble_{int(time.time())}"
        
        # Start training in background thread
        training_thread = threading.Thread(
            target=run_ensemble_training_thread,
            args=(training_id, ensemble_type, ensemble_name, selected_models, csv_path, weights, advanced_options)
        )
        training_thread.daemon = True
        training_thread.start()
        
        return training_id
        
    except Exception as e:
        debug_print(f"Error starting ensemble training: {e}")
        raise e

def run_ensemble_training_thread(training_id, ensemble_type, ensemble_name, selected_models, csv_path, weights, advanced_options):
    """Run ensemble training in background thread"""
    try:
        debug_print(f"Starting ensemble training thread: {training_id}")
        debug_print(f"CSV path received: {csv_path}")
        
        # Validate CSV file exists
        if not csv_path or not os.path.exists(csv_path):
            debug_print(f"CSV file not found: {csv_path}")
            return
        
        # Call the actual ensemble training function
        success = run_ensemble_training(
            ensemble_name=ensemble_name,
            ensemble_type=ensemble_type,
            selected_models=selected_models,
            csv_path=csv_path,
            weights=weights,
            advanced_options=advanced_options,
            output_dir="Models"
        )
        
        if success:
            debug_print(f"Ensemble training completed successfully: {training_id}")
        else:
            debug_print(f"Ensemble training failed: {training_id}")
            
    except Exception as e:
        debug_print(f"Ensemble training thread error: {e}")

@eel.expose
def get_ensemble_training_history():
    """Get ensemble training history"""
    try:
        # This would load from a database or file system
        # For now, return empty list
        return []
    except Exception as e:
        debug_print(f"Error getting ensemble training history: {e}")
        return []

@eel.expose
def get_models():
    """Scan the Models folder for YAML files and return model information"""
    debug_print("get_models() called from frontend")
    
    models = []
    models_path = Path("Models")
    
    debug_print(f"Models path: {models_path}")
    debug_print(f"Models path exists: {models_path.exists()}")
    
    if not models_path.exists():
        debug_print("Models directory does not exist, returning empty list")
        return models
    
    # Find all YAML files recursively in the Models folder
    debug_print("Finding YAML files...")
    yaml_files = find_yaml_files(str(models_path), recursive=True)
    debug_print(f"Found {len(yaml_files)} YAML files: {yaml_files}")
    
    for i, yaml_file in enumerate(yaml_files):
        debug_print(f"Processing YAML file {i+1}/{len(yaml_files)}: {yaml_file}")
        try:
            # Use centralized YAML utilities
            config = load_yaml_config(yaml_file)
            debug_print(f"Loaded config for {yaml_file}")
            
            # Extract model_name and Type using recursive key finding
            model_name = config.find_key('model_name', 'Unknown Model')
            model_type = config.find_key('Type', 'Unknown Type')
            debug_print(f"Extracted - Name: {model_name}, Type: {model_type}")
            
            model_info = {
                'name': model_name,
                'type': model_type,
                'config_path': yaml_file
            }
            models.append(model_info)
            debug_print(f"Added model: {model_info}")
            
        except Exception as e:
            debug_print(f"Error reading {yaml_file}: {e}")
            continue
    
    debug_print(f"Returning {len(models)} models: {models}")
    return models

@eel.expose
def get_model_details(config_path):
    """Get detailed information about a specific model"""
    try:
        # Use centralized YAML utilities
        config = load_yaml_config(config_path)
        return config.to_dict()
    except Exception as e:
        return {'error': str(e)}

@eel.expose
def start_training(model_type, csv_path, training_id=None, model_name=None, training_params=None):
    """Start a training process for the specified model type and CSV data"""
    try:
        if training_id is None:
            training_id = f"training_{int(time.time())}"
        
        # Validate model type
        valid_models = ['transformer', 'nn', 'xgboost', 'ppo']
        if model_type not in valid_models:
            return {'error': f'Invalid model type. Must be one of: {valid_models}'}
        
        # Validate CSV file exists
        if not os.path.exists(csv_path):
            return {'error': f'CSV file not found: {csv_path}'}
        
        # Generate model name if not provided
        if not model_name or model_name.strip() == '':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{model_type}_{timestamp}"
        
        # Clean model name (remove invalid characters)
        import re
        model_name = re.sub(r'[^\w\-_]', '_', model_name.strip())
        
        # Initialize training status
        training_status[training_id] = {
            'status': 'starting',
            'model_type': model_type,
            'model_name': model_name,
            'csv_path': csv_path,
            'start_time': datetime.now().isoformat(),
            'progress': 0,
            'message': 'Initializing training...',
            'training_params': training_params
        }
        training_logs[training_id] = []
        
        # Start training in a separate thread
        training_thread = threading.Thread(
            target=run_training_process,
            args=(training_id, model_type, csv_path, model_name, training_params)
        )
        training_thread.daemon = True
        training_thread.start()
        
        return {
            'success': True,
            'training_id': training_id,
            'model_name': model_name,
            'message': 'Training started successfully'
        }
        
    except Exception as e:
        return {'error': f'Failed to start training: {str(e)}'}

def run_training_process(training_id, model_type, csv_path, model_name, training_params=None):
    """Run the actual training process in a separate thread"""
    try:
        # Update status to running
        training_status[training_id]['status'] = 'running'
        training_status[training_id]['message'] = 'Training in progress...'
        
        # Build command for run_training.py
        # Change to the project root directory to ensure proper imports
        project_root = os.path.dirname(os.path.abspath(__file__))
        cmd = [
            'python', os.path.join(project_root, 'Utilities', 'run_training.py'),
            '--csv_path', csv_path,
            '--model', model_type,
            '--model_name', model_name
        ]
        
        # Add training parameters if provided
        if training_params:
            import json
            params_json = json.dumps(training_params)
            cmd.extend(['--training_params', params_json])
        
        # Start the training process
        debug_print(f"Starting {model_type} training for model: {model_name}")
        debug_print(f"Command: {' '.join(cmd)}")
        debug_print(f"Working directory: {project_root}")
        
        if DEBUG_TRAINING:
            # Debug mode: Open command window to show training output
            debug_print("Opening command window for training process")
            if os.name == 'nt':
                # Windows: Use cmd /k to keep window open after execution (even on error)
                # This allows viewing errors before the window closes
                debug_cmd = ['cmd', '/k'] + cmd
                process = subprocess.Popen(
                    debug_cmd,
                    cwd=project_root,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            else:
                # Linux/macOS: Use xterm or gnome-terminal if available
                try:
                    # Try to open in a new terminal window
                    terminal_cmd = ['gnome-terminal', '--', 'bash', '-c', f"cd {project_root} && {' '.join(cmd)}; read -p 'Press Enter to close...'"]
                    process = subprocess.Popen(terminal_cmd)
                except:
                    # Fallback: just run normally but don't capture output
                    process = subprocess.Popen(
                        cmd,
                        cwd=project_root
                    )
        else:
            # Normal mode: Capture output for web interface
            debug_print("Running training in background mode")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                cwd=project_root
            )
        
        training_processes[training_id] = process
        
        debug_print(f"Training process started with PID: {process.pid}")
        if DEBUG_TRAINING:
            debug_print("Command window opened - check for new console window")
            debug_print("Training output will be visible in the command window")
            debug_print("Note: Output is not captured in debug mode - check the command window")
        
        if DEBUG_TRAINING:
            # In debug mode, don't read output - just wait for process to complete
            debug_print("Waiting for training process to complete...")
        else:
            # Read output line by line only in normal mode
            for line in iter(process.stdout.readline, ''):
                if line:
                    training_logs[training_id].append({
                        'timestamp': datetime.now().isoformat(),
                        'message': line.strip()
                    })
                    
                    # Update progress based on keywords in output
                    if 'epoch' in line.lower() and 'loss' in line.lower():
                        # Extract epoch number for progress calculation
                        try:
                            epoch_part = line.split('epoch')[1].split('/')[0].strip()
                            if epoch_part.isdigit():
                                epoch = int(epoch_part)
                                # Assume 25 epochs for transformer, 100 for nn, 150 for xgboost
                                max_epochs = 25 if model_type == 'transformer' else (100 if model_type == 'nn' else 150)
                                progress = min(90, (epoch / max_epochs) * 90)  # Cap at 90% until completion
                                training_status[training_id]['progress'] = progress
                        except:
                            pass
        
        # Wait for process to complete
        return_code = process.wait()
        
        debug_print(f"Training process completed with return code: {return_code}")
        
        if return_code == 0:
            training_status[training_id]['status'] = 'completed'
            training_status[training_id]['progress'] = 100
            training_status[training_id]['message'] = 'Training completed successfully'
            training_status[training_id]['end_time'] = datetime.now().isoformat()
            debug_print("Training completed successfully")
        else:
            training_status[training_id]['status'] = 'failed'
            training_status[training_id]['message'] = f'Training failed with return code: {return_code}'
            training_status[training_id]['end_time'] = datetime.now().isoformat()
            debug_print(f"Training failed with return code: {return_code}")
            
    except Exception as e:
        debug_print(f"Training error occurred: {str(e)}")
        training_status[training_id]['status'] = 'error'
        training_status[training_id]['message'] = f'Training error: {str(e)}'
        training_status[training_id]['end_time'] = datetime.now().isoformat()
        training_logs[training_id].append({
            'timestamp': datetime.now().isoformat(),
            'message': f'ERROR: {str(e)}'
        })
    finally:
        # Clean up process reference
        if training_id in training_processes:
            del training_processes[training_id]

@eel.expose
def get_training_status(training_id):
    """Get the current status of a training process"""
    if training_id not in training_status:
        return {'error': 'Training ID not found'}
    
    return training_status[training_id]

@eel.expose
def get_training_logs(training_id, last_n=50):
    """Get the latest logs for a training process"""
    if training_id not in training_logs:
        return {'error': 'Training ID not found'}
    
    logs = training_logs[training_id]
    return logs[-last_n:] if len(logs) > last_n else logs

@eel.expose
def stop_training(training_id):
    """Stop a running training process"""
    try:
        if training_id not in training_processes:
            return {'error': 'Training process not found or already completed'}
        
        process = training_processes[training_id]
        process.terminate()
        
        training_status[training_id]['status'] = 'stopped'
        training_status[training_id]['message'] = 'Training stopped by user'
        training_status[training_id]['end_time'] = datetime.now().isoformat()
        
        del training_processes[training_id]
        
        return {'success': True, 'message': 'Training stopped successfully'}
        
    except Exception as e:
        return {'error': f'Failed to stop training: {str(e)}'}

@eel.expose
def get_available_csv_files():
    """Get list of available CSV files in the project directory"""
    try:
        csv_files = []
        project_root = Path(".")
        
        # Look for CSV files in the project root and common subdirectories
        search_paths = [project_root, project_root / "data", project_root / "datasets"]
        
        for search_path in search_paths:
            if search_path.exists():
                for csv_file in search_path.glob("*.csv"):
                    csv_files.append({
                        'name': csv_file.name,
                        'path': str(csv_file.absolute()),
                        'size': csv_file.stat().st_size,
                        'modified': datetime.fromtimestamp(csv_file.stat().st_mtime).isoformat()
                    })
        
        return csv_files
        
    except Exception as e:
        return {'error': f'Failed to get CSV files: {str(e)}'}

@eel.expose
def get_available_models():
    """Get list of available models for backtesting"""
    print("[DEBUG] get_available_models() called")
    try:
        models = []
        models_path = Path("Models")
        print(f"[DEBUG] Models path: {models_path}")
        print(f"[DEBUG] Models path exists: {models_path.exists()}")
        
        if not models_path.exists():
            print("[DEBUG] Models path does not exist, returning empty list")
            return models
        
        # Find all YAML files recursively in the Models folder
        yaml_files = find_yaml_files(str(models_path), recursive=True)
        print(f"[DEBUG] Found {len(yaml_files)} YAML files: {yaml_files}")
        
        for i, yaml_file in enumerate(yaml_files):
            print(f"[DEBUG] Processing YAML file {i+1}/{len(yaml_files)}: {yaml_file}")
            try:
                # Use centralized YAML utilities
                config = load_yaml_config(yaml_file)
                print(f"[DEBUG] Loaded config for {yaml_file}: {config.to_dict()}")
                
                # Extract model information using recursive key finding
                model_name = config.find_key('model_name', 'Unknown Model')
                model_type = config.find_key('Type', 'Unknown Type')
                config_path = str(Path(yaml_file).absolute())
                
                model_info = {
                    'name': model_name,
                    'type': model_type,
                    'config_path': config_path
                }
                print(f"[DEBUG] Created model info: {model_info}")
                models.append(model_info)
            except Exception as e:
                print(f"[DEBUG] Error reading {yaml_file}: {e}")
                continue
        
        print(f"[DEBUG] Returning {len(models)} models: {models}")
        return models
        
    except Exception as e:
        print(f"[DEBUG] Exception in get_available_models: {e}")
        return {'error': f'Failed to get models: {str(e)}'}

@eel.expose
def run_backtest(backtest_params):
    """Run a backtest using the specified model and data"""
    try:
        from Utilities.backtester import Backtester
        
        # Debug: Print received parameters
        print("Received backtest parameters:", backtest_params)
        
        # Extract parameters
        config_path = backtest_params.get('config_path')
        data_path = backtest_params.get('data_path')
        initial_capital = backtest_params.get('initial_capital', 50000)
        take_profit_pips = backtest_params.get('take_profit_pips', 50)
        stop_loss_pips = backtest_params.get('stop_loss_pips', 25)
        tick_size = backtest_params.get('tick_size', 0.25)
        tick_value = backtest_params.get('tick_value', 5)
        
        # Normalize file paths for Windows
        config_path = os.path.normpath(config_path)
        data_path = os.path.normpath(data_path)
        
        print(f"Config path: {config_path}")
        print(f"Data path: {data_path}")
        
        # Validate inputs
        if not config_path or not data_path:
            return {'success': False, 'error': 'Config path and data path are required'}
        
        if not os.path.exists(config_path):
            return {'success': False, 'error': f'Config file not found: {config_path}'}
        
        if not os.path.exists(data_path):
            return {'success': False, 'error': f'Data file not found: {data_path}'}
        
        print(f"Both files exist. Creating backtester...")
        
        # Create backtester instance
        backtester = Backtester(config_path=config_path, data_path=data_path)
        
        # Load model artifacts
        backtester.load_artifacts()
        
        # Run backtest and capture results
        results = backtester.run_with_results(
            initial_capital=initial_capital,
            take_profit_pips=take_profit_pips,
            stop_loss_pips=stop_loss_pips,
            tick_size=tick_size,
            tick_value=tick_value
        )
        
        return {
            'success': True,
            'data': results
        }
        
    except Exception as e:
        print(f"Backtest error: {str(e)}")
        return {'success': False, 'error': f'Backtest failed: {str(e)}'}

@eel.expose
def upload_csv_file(file_data):
    """Upload a CSV file and return the path where it was saved"""
    try:
        import base64
        import uuid
        from pathlib import Path
        
        # Create uploads directory if it doesn't exist
        uploads_dir = Path("uploads")
        uploads_dir.mkdir(exist_ok=True)
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file_data['name']}"
        file_path = uploads_dir / filename
        
        # Decode and save the file
        file_content = base64.b64decode(file_data['content'])
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        return {
            'success': True,
            'file_path': str(file_path.absolute()),
            'filename': filename
        }
        
    except Exception as e:
        return {'success': False, 'error': f'File upload failed: {str(e)}'}

# Global variables for API hosting
api_process = None
api_server_config = None
api_logs = []

@eel.expose
def start_api_server(config):
    """Start the API server with the specified configuration"""
    print("[DEBUG] start_api_server() called")
    print(f"[DEBUG] Received config: {config}")
    
    global api_process, api_server_config, api_logs
    
    try:
        print(f"[DEBUG] Current api_process: {api_process}")
        if api_process is not None:
            print("[DEBUG] API server already running, returning error")
            return {'success': False, 'error': 'API server is already running'}
        
        # Clear previous logs
        api_logs = []
        api_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Starting API server...")
        print("[DEBUG] Cleared previous logs and added startup message")
        
        # Store the configuration
        api_server_config = config
        api_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Config: {config}")
        print(f"[DEBUG] Stored config: {api_server_config}")
        
        # Get the model directory from the config path
        model_dir = os.path.dirname(config['model_path'])
        api_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Model directory: {model_dir}")
        print(f"[DEBUG] Extracted model directory: {model_dir}")
        print(f"[DEBUG] Model directory exists: {os.path.exists(model_dir)}")
        
        # Start the API server process using the existing Api_Loader.py
        cmd = [
            'python', '-m', 'uvicorn', 'Utilities.Api_Loader:app',
            '--host', config['host'],
            '--port', str(config['port'])
        ]
        
        # Set MODEL_DIR environment variable to ensure it's available to the worker process
        env = os.environ.copy()
        env['MODEL_DIR'] = model_dir
        print(f"[DEBUG] Constructed command: {cmd}")
        
        # Get the project root directory
        project_root = os.path.dirname(os.path.abspath(__file__))
        api_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Command: {' '.join(cmd)}")
        api_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Working directory: {project_root}")
        print(f"[DEBUG] Project root: {project_root}")
        print(f"[DEBUG] Project root exists: {os.path.exists(project_root)}")
        
        print(f"[DEBUG] Environment variables:")
        print(f"  - MODEL_DIR: {env['MODEL_DIR']}")
        print(f"  - Original MODEL_DIR: {os.environ.get('MODEL_DIR', 'Not set')}")
        
        debug_print("Starting API server subprocess...")
        if DEBUG_API:
            # Debug mode: Open command window to show API server output
            debug_print(f"Opening command window for API server: {' '.join(cmd)}")
            if os.name == 'nt':
                # Windows: Use cmd /k to keep window open after execution (even on error)
                # This allows viewing errors before the window closes
                debug_cmd = ['cmd', '/k'] + cmd
                api_process = subprocess.Popen(
                    debug_cmd,
                    cwd=project_root,
                    env=env,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            else:
                # Linux/macOS: Use xterm or gnome-terminal if available
                try:
                    # Try to open in a new terminal window
                    terminal_cmd = ['gnome-terminal', '--', 'bash', '-c', f"cd {project_root} && {' '.join(cmd)}; read -p 'Press Enter to close...'"]
                    api_process = subprocess.Popen(terminal_cmd)
                except:
                    # Fallback: just run normally but don't capture output
                    api_process = subprocess.Popen(
                        cmd,
                        cwd=project_root,
                        env=env
                    )
        else:
            # Normal mode: Capture output for web interface
            debug_print("Running API server in background mode")
            api_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                cwd=project_root,
                env=env
            )
        
        api_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Process started with PID: {api_process.pid}")
        
        debug_print(f"API server process started with PID: {api_process.pid}")
        if DEBUG_API:
            debug_print("Command window opened for API server - check for new console window")
            debug_print("API server output will be visible in the command window")
            debug_print("Note: Output is not captured in debug mode - check the command window")
        
        if DEBUG_API:
            # In debug mode, don't read output - just start the process
            debug_print("API server running in debug mode - output visible in command window")
        else:
            # Start a thread to read the output only in normal mode
            def read_output():
                try:
                    for line in iter(api_process.stdout.readline, ''):
                        if line:
                            api_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {line.strip()}")
                            # Keep only last 100 log entries
                            if len(api_logs) > 100:
                                api_logs.pop(0)
                except Exception as e:
                    api_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Error reading output: {str(e)}")
            
            output_thread = threading.Thread(target=read_output)
            output_thread.daemon = True
            output_thread.start()
        
        # Give the process a moment to start and check if it's still running
        import time
        time.sleep(3)
        
        # Check if the process is still running
        if api_process.poll() is not None:
            # Process has terminated, get the output
            stdout, stderr = api_process.communicate()
            error_msg = stdout if stdout else stderr if stderr else "Unknown error"
            api_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Process terminated: {error_msg}")
            return {'success': False, 'error': f'API server failed to start: {error_msg}'}
        
        api_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] API server started successfully")
        return {'success': True, 'message': 'API server started successfully'}
        
    except Exception as e:
        api_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Exception: {str(e)}")
        return {'success': False, 'error': f'Failed to start API server: {str(e)}'}

@eel.expose
def stop_api_server():
    """Stop the running API server"""
    global api_process, api_server_config, api_logs
    
    try:
        if api_process is None:
            return {'success': False, 'error': 'No API server is running'}
        
        api_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Stopping API server...")
        
        # Terminate the process
        api_process.terminate()
        api_process.wait(timeout=5)
        api_process = None
        api_server_config = None
        
        # No temporary files to clean up since we're using the existing Api_Loader.py
        
        api_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] API server stopped successfully")
        return {'success': True, 'message': 'API server stopped successfully'}
        
    except Exception as e:
        api_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Error stopping API server: {str(e)}")
        return {'success': False, 'error': f'Failed to stop API server: {str(e)}'}

@eel.expose
def get_api_server_status():
    """Get the current status of the API server"""
    global api_process, api_server_config
    
    if api_process is None:
        return {'running': False, 'message': 'No API server is running'}
    
    # Check if the process is still running
    if api_process.poll() is not None:
        # Process has terminated
        api_process = None
        api_server_config = None
        return {'running': False, 'message': 'API server process has terminated'}
    
    return {
        'running': True,
        'config': api_server_config,
        'message': 'API server is running'
    }

@eel.expose
def get_api_logs():
    """Get the API server logs"""
    global api_logs
    return api_logs

@eel.expose
def clear_api_logs():
    """Clear the API server logs"""
    global api_logs
    api_logs = []
    return {'success': True, 'message': 'API logs cleared'}

@eel.expose
def test_api_prediction(test_data):
    """Test the API with sample data"""
    try:
        if api_process is None or api_server_config is None:
            return {'success': False, 'error': 'API server is not running'}
        
        import json
        
        # Make a test request to the API
        api_url = f"http://{api_server_config['host']}:{api_server_config['port']}/predict"
        
        response = requests.post(
            api_url,
            json={'features': test_data['features']},
            timeout=10
        )
        
        if response.status_code == 200:
            return {'success': True, 'data': response.json()}
        else:
            return {'success': False, 'error': f'API request failed: {response.text}'}
            
    except Exception as e:
        return {'success': False, 'error': f'Test request failed: {str(e)}'}

# Removed create_api_loader_content and get_model_type_from_config functions
# since we're now using the existing Api_Loader.py directly


if __name__ == '__main__':
    # Start the EEL application
    eel.start('index.html', size=(1400, 900), mode='chrome')