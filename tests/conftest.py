"""
This file contains pytest fixtures and configurations.
"""
import os
import sys
import pytest

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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