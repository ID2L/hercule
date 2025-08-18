## Pytest Testing Framework

This project uses pytest, a powerful and flexible testing framework for Python that makes it easy to write simple and scalable tests.

### Features

- **Simple test discovery**: Automatically finds and runs test files and functions
- **Fixtures**: Reusable test data and setup/teardown logic
- **Parametrized tests**: Run the same test with different inputs
- **Assertions**: Clear and informative assertion failures
- **Plugins ecosystem**: Extensive plugin system for additional functionality
- **Coverage reporting**: Integration with pytest-cov for code coverage

### Test Organization

Tests are organized in the `tests/` directory with the following structure:
- `tests/`: Main test directory
- `tests/test_*.py`: Test modules (must start with `test_`)
- Test functions must start with `test_`
- Test classes must start with `Test`

### Common Commands

```bash
# Run all tests
poetry run pytest

# Run tests with verbose output
poetry run pytest -v

# Run tests in a specific file
poetry run pytest tests/test_module.py

# Run a specific test function
poetry run pytest tests/test_module.py::test_function_name

# Run tests with coverage report
poetry run pytest --cov=src --cov-report=html

# Run tests and stop at first failure
poetry run pytest -x

# Run tests matching a pattern
poetry run pytest -k "test_pattern"

# Run tests with specific markers
poetry run pytest -m "unit"  # Run only unit tests
poetry run pytest -m "not slow"  # Skip slow tests
```

### Writing Tests

#### Basic Test Function
```python
def test_addition():
    """Test basic addition operation."""
    assert 1 + 1 == 2
    assert 2 + 3 == 5
```

#### Test Class
```python
class TestCalculator:
    """Test class for calculator operations."""
    
    def test_add(self):
        """Test addition method."""
        assert Calculator().add(2, 3) == 5
    
    def test_subtract(self):
        """Test subtraction method."""
        assert Calculator().subtract(5, 3) == 2
```

#### Using Fixtures
```python
import pytest

@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    return {"name": "test", "value": 42}

def test_with_fixture(sample_data):
    """Test using a fixture."""
    assert sample_data["name"] == "test"
    assert sample_data["value"] == 42
```

#### Parametrized Tests
```python
import pytest

@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_double(input, expected):
    """Test doubling function with multiple inputs."""
    assert double(input) == expected
```

### Test Markers

Use markers to categorize and control test execution:

```python
import pytest

@pytest.mark.unit
def test_unit_functionality():
    """Unit test marker."""
    pass

@pytest.mark.integration  
def test_integration_functionality():
    """Integration test marker."""
    pass

@pytest.mark.slow
def test_slow_operation():
    """Slow test marker."""
    pass
```

### Configuration

The project includes a `pytest.ini` file with default configuration:
- Test discovery paths
- Output formatting
- Default markers
- Coverage settings

### Best Practices

1. **Test naming**: Use descriptive names that explain what is being tested
2. **One assertion per test**: Keep tests focused and simple
3. **Arrange-Act-Assert**: Structure tests with clear setup, execution, and verification
4. **Use fixtures**: Share common setup logic across tests
5. **Test edge cases**: Include boundary conditions and error cases
6. **Keep tests fast**: Use markers to separate slow integration tests
7. **Mock dependencies**: Use `unittest.mock` or `pytest-mock` for external dependencies

### Example Test Structure

```python
"""
Test module for user authentication.
"""

import pytest
from unittest.mock import Mock, patch
from myproject.auth import UserAuth, AuthError


class TestUserAuth:
    """Test class for user authentication functionality."""
    
    @pytest.fixture
    def user_auth(self):
        """Create UserAuth instance for testing."""
        return UserAuth()
    
    @pytest.fixture
    def valid_credentials(self):
        """Provide valid test credentials."""
        return {"username": "testuser", "password": "testpass"}
    
    def test_valid_login(self, user_auth, valid_credentials):
        """Test successful login with valid credentials."""
        # Arrange
        expected_token = "abc123"
        
        # Act
        with patch.object(user_auth, '_generate_token', return_value=expected_token):
            token = user_auth.login(**valid_credentials)
        
        # Assert
        assert token == expected_token
    
    def test_invalid_login(self, user_auth):
        """Test login failure with invalid credentials."""
        # Arrange
        invalid_creds = {"username": "invalid", "password": "wrong"}
        
        # Act & Assert
        with pytest.raises(AuthError):
            user_auth.login(**invalid_creds)
    
    @pytest.mark.parametrize("username,password,expected_error", [
        ("", "password", "Username required"),
        ("user", "", "Password required"),
        (None, "password", "Username required"),
    ])
    def test_login_validation(self, user_auth, username, password, expected_error):
        """Test login input validation."""
        with pytest.raises(AuthError, match=expected_error):
            user_auth.login(username, password)
```

### Coverage Reports

Generate coverage reports to ensure your tests cover your code adequately:

```bash
# Generate HTML coverage report
poetry run pytest --cov=src --cov-report=html

# Generate terminal coverage report
poetry run pytest --cov=src --cov-report=term-missing

# Set minimum coverage threshold
poetry run pytest --cov=src --cov-fail-under=80
```

### Integration with CI/CD

Add pytest to your CI/CD pipeline:

```yaml
# GitHub Actions example
- name: Run tests
  run: |
    poetry install
    poetry run pytest --cov=src --cov-report=xml
    
- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

For more information, visit the [pytest documentation](https://docs.pytest.org/en/stable/). 