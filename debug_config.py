# Debug Configuration for Galaxy Models
# This file contains debug settings that can be easily modified

# Debug flags - set to True to enable debug features
DEBUG_TRAINING = True   # Open command windows for training subprocesses
DEBUG_API = True        # Open command windows for API subprocesses  
DEBUG_VERBOSE = True    # Enable verbose debug output in console

# Debug output settings
DEBUG_SHOW_COMMANDS = True      # Show full command lines being executed
DEBUG_SHOW_PROCESS_IDS = True   # Show process IDs when starting subprocesses
DEBUG_SHOW_WORKING_DIRS = True  # Show working directories for subprocesses

# Platform-specific settings
WINDOWS_CREATE_CONSOLE = True   # Use CREATE_NEW_CONSOLE on Windows
LINUX_TERMINAL = "gnome-terminal"  # Terminal to use on Linux (if available)

def get_debug_config():
    """Return current debug configuration as a dictionary"""
    return {
        'debug_training': DEBUG_TRAINING,
        'debug_api': DEBUG_API,
        'debug_verbose': DEBUG_VERBOSE,
        'debug_show_commands': DEBUG_SHOW_COMMANDS,
        'debug_show_process_ids': DEBUG_SHOW_PROCESS_IDS,
        'debug_show_working_dirs': DEBUG_SHOW_WORKING_DIRS,
        'windows_create_console': WINDOWS_CREATE_CONSOLE,
        'linux_terminal': LINUX_TERMINAL
    }

def print_debug_config():
    """Print current debug configuration"""
    config = get_debug_config()
    print("=" * 50)
    print("Galaxy Models Debug Configuration")
    print("=" * 50)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 50)

if __name__ == "__main__":
    print_debug_config()