# Test Fixtures

This directory contains YAML configuration files used for testing the Hercule framework.

## Available Fixtures

### `simple_config.yaml`
A basic configuration with:
- Single environment: `CartPole-v1`
- Single model: `test_model` with basic hyperparameters
- 100 max iterations
- No evaluation configuration

### `complex_config.yaml`
A complex configuration with:
- Single environment: `FrozenLake-v1` with environment hyperparameters
- Single model: `complex_model` with complex hyperparameters (lists, booleans)
- 500 max iterations
- Evaluation configuration enabled
- Custom output directory

### `multi_config.yaml`
A multi-model, multi-environment configuration with:
- Two environments: `CartPole-v1` (string) and `LunarLander-v2` (with hyperparameters)
- Two models: `simple_model` and `advanced_model` with different hyperparameter types
- 1000 max iterations
- Evaluation configuration
- Custom output directory

## Usage in Tests

These fixtures are used across different test modules:

- **CLI tests**: Test basic CLI functionality with different configuration types
- **Configuration tests**: Test YAML loading with various configuration structures
- **Integration tests**: Test end-to-end functionality with realistic configurations

## Adding New Fixtures

When adding new fixtures:
1. Use descriptive names that indicate the configuration type
2. Include a variety of hyperparameter types (strings, numbers, booleans, lists)
3. Test both simple and complex configurations
4. Document the fixture's purpose and structure
