#!/usr/bin/env python3
"""
Configuration utility to load paths from environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from the same directory as this script
_env_path = Path(__file__).parent / '.env'
load_dotenv(_env_path)

def get_data_base_root_path() -> str:
    """Get the data base root path from environment variable."""
    return os.getenv('DATA_BASE_ROOT_PATH', '/home/xliu316/scratchcxiao13/nanxi/Dataset')

def get_output_dir() -> str:
    """Get the output directory from environment variable."""
    return os.getenv('OUTPUT_DIR', './result')

def get_data_path(dataset_name: str) -> str:
    """Get the full path for a specific dataset."""
    base_path = get_data_base_root_path()
    return os.path.join(base_path, dataset_name)
