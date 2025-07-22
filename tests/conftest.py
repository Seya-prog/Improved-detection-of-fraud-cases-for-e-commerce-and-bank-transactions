"""
This file contains pytest fixtures and configurations.
"""
import os
import sys
import pytest

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Create directories if they don't exist
def ensure_directories_exist():
    """Ensure necessary directories exist"""
    dirs = [
        'data',
        'data/raw',
        'data/processed',
        'models'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    # Create .gitkeep files if they don't exist
    gitkeeps = [
        'data/raw/.gitkeep',
        'data/processed/.gitkeep',
        'models/.gitkeep'
    ]
    for gk in gitkeeps:
        if not os.path.exists(gk):
            with open(gk, 'w') as f:
                pass  # Create empty file

# Ensure directories exist at import time
ensure_directories_exist()

@pytest.fixture(scope="session", autouse=True)
def setup_directories():
    """Setup directories needed for tests"""
    ensure_directories_exist()

@pytest.fixture
def data_dir():
    """Return the path to the data directory"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

@pytest.fixture
def models_dir():
    """Return the path to the models directory"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))

@pytest.fixture
def src_dir():
    """Return the path to the src directory"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')) 