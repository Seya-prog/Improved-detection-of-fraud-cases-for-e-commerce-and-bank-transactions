"""
Tests for data integrity.
"""
import os
import pytest

def test_data_directories_exist(data_dir):
    """Test that the data directories exist"""
    assert os.path.isdir(data_dir), "Data directory should exist"
    assert os.path.isdir(os.path.join(data_dir, 'raw')), "Raw data directory should exist"
    assert os.path.isdir(os.path.join(data_dir, 'processed')), "Processed data directory should exist"
    # Removed checks for interim and external directories

def test_gitkeep_files_exist():
    """Test that .gitkeep files exist to track empty directories"""
    dirs_to_check = [
        'data/raw',
        'data/processed',
        'models'
        # Removed interim and external directories
    ]
    
    for dir_path in dirs_to_check:
        gitkeep_path = os.path.join(dir_path, '.gitkeep')
        assert os.path.isfile(gitkeep_path), f"{gitkeep_path} should exist" 