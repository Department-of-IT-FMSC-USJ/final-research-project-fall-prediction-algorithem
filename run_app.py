#!/usr/bin/env python3
"""
Launcher script for the Fall Prediction System Flask application.
This script allows you to run the refactored app from the root directory.
"""

import sys
import os
import subprocess

def main():
    """Launch the Flask application."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_dir = os.path.join(script_dir, "fall_prediction_app")
    
    # Check if the app directory exists
    if not os.path.exists(app_dir):
        print(f"Error: Application directory not found at {app_dir}")
        print("Please ensure the fall_prediction_app directory exists.")
        sys.exit(1)
    
    # Change to the app directory
    os.chdir(app_dir)
    
    # Get command line arguments (skip the script name)
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    
    # Build the command
    cmd = [sys.executable, "app.py"] + args
    
    try:
        print("Starting Fall Prediction System Flask Application...")
        print(f"Working directory: {os.getcwd()}")
        print(f"Command: {' '.join(cmd)}")
        print("-" * 50)
        
        # Run the application
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"Error running application: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 