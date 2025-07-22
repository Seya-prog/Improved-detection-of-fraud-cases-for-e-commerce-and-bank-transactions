import os
import sys

# Add the project root directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_directory_structure():
    """Test that the required directories exist"""
    assert os.path.isdir("data"), "Data directory should exist"
    assert os.path.isdir("models"), "Models directory should exist"
    assert os.path.isdir("src"), "Source directory should exist"
    
def test_placeholder():
    """A placeholder test that always passes"""
    assert True, "This test should always pass"

if __name__ == "__main__":
    test_directory_structure()
    test_placeholder()
    print("All tests passed!") 