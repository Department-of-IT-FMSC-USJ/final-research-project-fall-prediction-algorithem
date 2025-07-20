#!/usr/bin/env python3
"""Simple runner for the refactored fall detection application."""

import sys
import os

# Add the current directory to Python path so we can import services
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from refactored_webcam_feed import main

if __name__ == "__main__":
    main() 