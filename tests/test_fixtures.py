"""Test to verify that global fixtures work correctly."""

from pathlib import Path


def test_temp_test_dir_fixture(temp_test_dir):
    """Test that the temp_test_dir fixture creates a temporary directory."""
    assert isinstance(temp_test_dir, Path)
    assert temp_test_dir.exists()
    assert temp_test_dir.is_dir()


def test_change_to_temp_dir_fixture(change_to_temp_dir):
    """Test that the change_to_temp_dir fixture changes working directory."""
    assert isinstance(change_to_temp_dir, Path)
    assert change_to_temp_dir.exists()
    assert Path.cwd() == change_to_temp_dir
