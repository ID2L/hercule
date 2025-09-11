"""Global test configuration and fixtures."""

import tempfile
from pathlib import Path

import click.testing
import pytest


@pytest.fixture(scope="function")
def temp_test_dir():
    """
    Create a temporary directory for each test function.

    This fixture ensures that each test runs in its own isolated directory,
    preventing interference between tests and keeping the workspace clean.

    Yields:
        Path: Path to the temporary directory
    """
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)

    yield temp_path

    # Cleanup: remove temporary directory
    import shutil

    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture(scope="function")
def change_to_temp_dir(temp_test_dir):
    """
    Change the current working directory to a temporary directory for the test.

    This fixture combines temp_test_dir with changing the working directory,
    ensuring tests run in isolation.

    Args:
        temp_test_dir: The temporary directory fixture

    Yields:
        Path: Path to the temporary directory (same as temp_test_dir)
    """
    import os

    original_cwd = Path.cwd()

    # Change to temporary directory
    os.chdir(temp_test_dir)

    yield temp_test_dir

    # Restore original working directory
    os.chdir(original_cwd)


@pytest.fixture(scope="function")
def runner():
    """
    Create a Click test runner for CLI testing.

    This fixture provides a Click test runner that can be used to invoke
    CLI commands in tests.

    Returns:
        click.testing.CliRunner: Click test runner instance
    """
    return click.testing.CliRunner()
