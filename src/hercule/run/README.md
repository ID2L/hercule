# Run Module - Training and Execution System

## Overview

The `run` module provides a comprehensive system for training reinforcement learning models and executing them after training. It includes separate components for training and execution, as well as a combined `RunManager` that handles both phases.

## Components

### 1. Training Components

#### `TrainingRunner`
Handles the training phase of RL models.

```python
from hercule.run import TrainingRunner

with TrainingRunner(config) as runner:
    result = runner.run_single_training(
        model=model,
        environment_name="Taxi-v3",
        model_name="simple_sarsa",
        hyperparameters=hyperparams
    )
```

#### `RunResult`
Container for training results.

```python
from hercule.run import RunResult

result = RunResult(
    environment_name="Taxi-v3",
    model_name="simple_sarsa",
    hyperparameters=hyperparams,
    metrics=training_metrics,
    success=True
)
```

### 2. Execution Components

#### `ModelExecutor`
Handles the execution phase of trained models.

```python
from hercule.run import ModelExecutor

with ModelExecutor(env_manager, log_level="INFO") as executor:
    result = executor.execute_model(
        model=trained_model,
        environment_name="Taxi-v3",
        model_name="simple_sarsa",
        num_episodes=10,
        max_steps_per_episode=100,
        render=False
    )
```

#### `ExecutionResult`
Container for execution results.

```python
from hercule.run import ExecutionResult

result = ExecutionResult(
    environment_name="Taxi-v3",
    model_name="simple_sarsa",
    episode_rewards=[-100.0, -150.0, -80.0],
    episode_lengths=[50, 75, 40],
    total_episodes=3,
    success=True
)

# Access computed metrics
print(f"Mean reward: {result.mean_reward}")
print(f"Std reward: {result.std_reward}")
print(f"Mean length: {result.mean_length}")
```

### 3. Combined Components

#### `RunManager`
Combines training and execution in a single interface.

```python
from hercule.run import RunManager

with RunManager(config, log_level="INFO") as manager:
    training_result, execution_result = manager.run_training_and_execution(
        model=model,
        environment_name="Taxi-v3",
        model_name="simple_sarsa",
        hyperparameters=hyperparams,
        execution_config={
            "num_episodes": 10,
            "max_steps_per_episode": 100,
            "render": False
        }
    )
```

## Usage Examples

### Basic Training and Execution

```python
from hercule.run import RunManager
from hercule.models import create_model
from hercule.config import HerculeConfig

# Create configuration
config = HerculeConfig(
    name="example_run",
    environments=["Taxi-v3"],
    models=[],
    max_iterations=1000,
    output_dir=Path("outputs"),
)

# Create model
model = create_model("simple_sarsa")

# Define hyperparameters
hyperparameters = {
    "learning_rate": 0.1,
    "discount_factor": 0.95,
    "epsilon": 0.1,
    "epsilon_decay": 0.005,
    "epsilon_min": 0.01,
    "seed": 42
}

# Run training and execution
with RunManager(config, log_level="INFO") as manager:
    training_result, execution_result = manager.run_training_and_execution(
        model=model,
        environment_name="Taxi-v3",
        model_name="simple_sarsa",
        hyperparameters=hyperparameters,
        execution_config={
            "num_episodes": 10,
            "max_steps_per_episode": 100,
            "render": False
        }
    )
    
    if training_result.success:
        print(f"Training completed: {training_result.metrics}")
    
    if execution_result and execution_result.success:
        print(f"Execution completed: {execution_result.mean_reward:.2f}")
```

### Execution Only (Pre-trained Model)

```python
from hercule.run import ModelExecutor
from hercule.environnements import EnvironmentManager

# Load pre-trained model
model = create_model("simple_sarsa")
model.load(Path("saved_model"))
model.configure(env, hyperparameters)

# Execute only
env_manager = EnvironmentManager(config)
with ModelExecutor(env_manager, log_level="DEBUG") as executor:
    result = executor.execute_model(
        model=model,
        environment_name="Taxi-v3",
        model_name="simple_sarsa",
        num_episodes=5,
        max_steps_per_episode=50,
        render=False
    )
    
    print(f"Mean reward: {result.mean_reward:.2f}")
    print(f"Individual rewards: {result.episode_rewards}")
```

### Multiple Execution Configurations

```python
from hercule.run import ModelExecutor

with ModelExecutor(env_manager, log_level="INFO") as executor:
    episode_configs = [
        {"num_episodes": 5, "max_steps_per_episode": 50, "render": False},
        {"num_episodes": 10, "max_steps_per_episode": 100, "render": False},
        {"num_episodes": 3, "max_steps_per_episode": 200, "render": True},
    ]
    
    results = executor.execute_multiple_episodes(
        model=model,
        environment_name="Taxi-v3",
        model_name="simple_sarsa",
        episode_configs=episode_configs
    )
    
    for i, result in enumerate(results):
        print(f"Config {i+1}: {result.mean_reward:.2f}")
```

### Saving Execution Results

```python
from hercule.run import save_execution_results

# Save results to JSON file
save_execution_results(results, Path("outputs"), "execution_results.json")
```

## Logging Levels

The execution system supports different logging levels:

- **DEBUG**: Detailed information including episode-by-episode progress
- **INFO**: Standard progress information (default)
- **WARNING**: Only warnings and errors
- **ERROR**: Only errors

```python
# Set log level when creating executor
executor = ModelExecutor(env_manager, log_level="DEBUG")
```

## Configuration File Example

```yaml
# config.yaml
name: "training_with_execution"
environments:
  - "Taxi-v3"
  - name: "FrozenLake-v1"
    hyperparameters:
      - key: "map_name"
        value: "4x4"
      - key: "is_slippery"
        value: false

models:
  - name: "simple_sarsa"
    hyperparameters:
      - key: "learning_rate"
        value: 0.1
      - key: "discount_factor"
        value: 0.95
      - key: "epsilon"
        value: 0.1
      - key: "epsilon_decay"
        value: 0.005
      - key: "epsilon_min"
        value: 0.01
      - key: "seed"
        value: 42

# Execution configuration
execution:
  num_episodes: 10
  max_steps_per_episode: 100
  render: false

max_iterations: 1000
output_dir: "outputs/example"
log_level: "INFO"
```

## Error Handling

The system includes comprehensive error handling:

```python
# Training errors
if not training_result.success:
    print(f"Training failed: {training_result.error_message}")

# Execution errors
if not execution_result.success:
    print(f"Execution failed: {execution_result.error_message}")
```

## Metrics and Results

### Training Metrics
- `mean_reward`: Average reward per episode
- `std_reward`: Standard deviation of rewards
- `min_reward` / `max_reward`: Min/max episode rewards
- `mean_episode_length`: Average episode length
- `episodes_completed`: Number of completed episodes
- `final_epsilon`: Final exploration rate
- `q_table_size`: Size of Q-table (for Q-learning methods)

### Execution Metrics
- `mean_reward`: Average reward across all episodes
- `std_reward`: Standard deviation of rewards
- `min_reward` / `max_reward`: Min/max episode rewards
- `mean_length`: Average episode length
- `std_length`: Standard deviation of episode lengths
- `episode_rewards`: List of individual episode rewards
- `episode_lengths`: List of individual episode lengths
- `total_episodes`: Total number of episodes executed

## Best Practices

1. **Use Context Managers**: Always use `with` statements to ensure proper resource cleanup
2. **Check Success**: Always check the `success` flag before accessing results
3. **Configure Logging**: Set appropriate log levels for your use case
4. **Save Results**: Use `save_execution_results()` to persist execution data
5. **Handle Errors**: Implement proper error handling for both training and execution phases

## Testing

Run the test scripts to verify functionality:

```bash
# Test execution functionality
python test_execution.py

# Run usage examples
python example_execution_usage.py
```
