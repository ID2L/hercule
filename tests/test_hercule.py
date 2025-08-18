"""
Test module for hercule.
"""

import pytest
from hercule import __version__


def test_version():
    """Test that version is defined."""
    assert __version__ is not None
    assert isinstance(__version__, str)


def test_example():
    """Example test to demonstrate pytest usage."""
    assert 1 + 1 == 2


class TestExample:
    """Example test class to demonstrate pytest class-based tests."""
    
    def test_class_method(self):
        """Test method within a class."""
        assert True
        
    def test_with_fixture(self, example_fixture):
        """Test using a fixture."""
        assert example_fixture == "test_data"


@pytest.fixture
def example_fixture():
    """Example fixture for testing."""
    return "test_data"
