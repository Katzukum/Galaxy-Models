# Debug Features for Galaxy Models

This document describes the debug features added to Galaxy Models to help with development and troubleshooting.

## 🐛 Debug Configuration

### Debug Variables in Main.py

```python
# Debug configuration
DEBUG_TRAINING = True  # Set to True to open command windows for training subprocesses
DEBUG_API = True       # Set to True to open command windows for API subprocesses
DEBUG_VERBOSE = True   # Set to True for additional debug output
```

### Debug Functions

- `debug_print(message)`: Print debug messages if DEBUG_VERBOSE is enabled
- `get_debug_status()`: Get current debug configuration status
- `set_debug_training(enabled)`: Enable/disable debug mode for training processes
- `set_debug_api(enabled)`: Enable/disable debug mode for API processes
- `set_debug_verbose(enabled)`: Enable/disable verbose debug output

## 🖥️ Command Window Features

### Training Processes

When `DEBUG_TRAINING = True`:
- Training subprocesses open in new command windows
- Full training output is visible in real-time
- Process IDs and command details are logged
- Useful for debugging training issues, especially with PPO

### API Server Processes

When `DEBUG_API = True`:
- API server subprocesses open in new command windows
- API server logs are visible in real-time
- Useful for debugging API hosting issues

## 🔧 How It Works

### Windows (NT)
- Uses `subprocess.CREATE_NEW_CONSOLE` flag
- Opens new console windows for subprocesses
- Each subprocess gets its own visible window

### Linux/Unix
- Uses `subprocess.Popen` with default settings
- May open new terminal windows depending on system configuration
- Output is still captured for web interface

## 📊 Debug Output Examples

### Training Debug Output
```
[DEBUG] Starting ppo training for model: my_ppo_agent
[DEBUG] Command: python Utilities/run_training.py --csv_path data.csv --model ppo
[DEBUG] Working directory: C:\Users\desk\Desktop\Javascript Projects\Galaxy Models
[DEBUG] Opening command window for training process
[DEBUG] Training process started with PID: 12345
[DEBUG] Command window opened - check for new console window
[DEBUG] Training output will be visible in the command window
```

### API Server Debug Output
```
[DEBUG] Starting API server subprocess...
[DEBUG] Opening command window for API server: python -m uvicorn Utilities.Api_Loader:app --host 127.0.0.1 --port 8000
[DEBUG] API server process started with PID: 67890
[DEBUG] Command window opened for API server - check for new console window
[DEBUG] API server output will be visible in the command window
```

## 🎯 Use Cases

### 1. Debugging PPO Training
- Set `DEBUG_TRAINING = True`
- Start PPO training from web interface
- Watch training progress in real-time in command window
- See detailed error messages and progress

### 2. Debugging API Issues
- Set `DEBUG_API = True`
- Start API server from web interface
- Monitor API server logs in real-time
- Debug model loading and prediction issues

### 3. Development and Testing
- Set `DEBUG_VERBOSE = True` for detailed console output
- Monitor all subprocess creation and management
- Track process IDs and working directories

## ⚙️ Configuration

### Quick Toggle
```python
# In Main.py, change these values:
DEBUG_TRAINING = False  # Disable training command windows
DEBUG_API = False       # Disable API command windows
DEBUG_VERBOSE = False   # Disable verbose output
```

### Runtime Toggle (via Web Interface)
```javascript
// Toggle debug modes from JavaScript
eel.set_debug_training(true)();   // Enable training debug
eel.set_debug_api(false)();       // Disable API debug
eel.set_debug_verbose(true)();    // Enable verbose output
```

## 🚀 Benefits

1. **Real-time Monitoring**: See training progress as it happens
2. **Error Debugging**: Catch and debug errors immediately
3. **Process Management**: Monitor subprocess creation and termination
4. **Development Aid**: Easier development and testing workflow
5. **User Experience**: Better visibility into long-running processes

## 🔍 Troubleshooting

### Command Windows Not Opening
- Check if `DEBUG_TRAINING` or `DEBUG_API` is set to `True`
- Verify Windows compatibility (uses `CREATE_NEW_CONSOLE`)
- Check console output for debug messages

### Too Much Debug Output
- Set `DEBUG_VERBOSE = False` to reduce console output
- Keep only the specific debug features you need

### Process Management Issues
- Check process IDs in debug output
- Use Task Manager to monitor subprocesses
- Check for zombie processes if training fails

## 📝 Notes

- Debug features are designed for development and troubleshooting
- Command windows may appear behind the main application window
- Debug output is in addition to the web interface logs
- All debug features can be toggled at runtime
- Debug configuration is stored in Main.py for easy access