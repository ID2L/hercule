# Simple SARSA Implementation

## Overview

This module provides a simple implementation of the SARSA (State-Action-Reward-State-Action) algorithm using a Q-table for discrete action and observation spaces.

## Algorithm Description

SARSA is an on-policy temporal difference learning algorithm that learns the Q-values for the policy being followed, rather than the optimal policy. The key difference from Q-learning is that SARSA uses the actual action taken in the next state for the update, making it on-policy.

### SARSA Update Rule

```
Q(s, a) ← Q(s, a) + α[r + γQ(s', a') - Q(s, a)]
```

Where:
- `Q(s, a)`: Current Q-value for state-action pair
- `α`: Learning rate
- `r`: Reward received
- `γ`: Discount factor
- `Q(s', a')`: Q-value for next state-action pair (using the policy)

## Features

- **Q-table based**: Uses a dictionary-based Q-table for efficient storage
- **Epsilon-greedy exploration**: Configurable exploration strategy
- **Discrete space validation**: Automatically validates that environment has discrete action and observation spaces
- **Hyperparameter configuration**: Supports all standard SARSA hyperparameters
- **Save/Load functionality**: Can save and load trained models
- **Training metrics**: Tracks comprehensive training metrics

## Hyperparameters

| Parameter | Type | Default | Description | Range |
|-----------|------|---------|-------------|-------|
| `learning_rate` | float | 0.1 | Learning rate (α) for Q-value updates | [0.0, 1.0] |
| `discount_factor` | float | 0.95 | Discount factor (γ) for future rewards | [0.0, 1.0] |
| `epsilon` | float | 1.0 | Initial exploration rate for epsilon-greedy | [0.0, 1.0] |
| `epsilon_decay` | float | 0.0 | Fraction of epsilon to decay per step (0.0 = no decay) | [0.0, 1.0] |
| `epsilon_min` | float | 0.0 | Minimum epsilon value | [0.0, 1.0] |
| `seed` | int | 42 | Random seed for reproducibility | Any integer |

## Usage

### Basic Usage

```python
from hercule.models import create_model
import gymnasium as gym

# Create environment
env = gym.make("Taxi-v3")

# Create SARSA model
model = create_model("simple_sarsa")

# Configure with hyperparameters (standard decay)
hyperparameters = {
    "learning_rate": 0.1,
    "discount_factor": 0.95,
    "epsilon": 0.1,
    "epsilon_decay": 0.005,  # Decay 0.5% of epsilon per step
    "epsilon_min": 0.01,
    "seed": 42
}

# Or use no decay (epsilon stays constant)
no_decay_hyperparameters = {
    "epsilon": 0.1,
    "epsilon_decay": 0.0,  # No decay
    "epsilon_min": 0.0
}

model.configure(env, hyperparameters)

# Train the model
metrics = model.train(env, hyperparameters, max_iterations=1000)

# Evaluate the model
eval_metrics = model.evaluate(num_episodes=10)
```

### Configuration File Usage

```yaml
# config.yaml
name: "sarsa_experiment"
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
        value: 0.005  # Decay 0.5% of epsilon per step
      - key: "epsilon_min"
        value: 0.01
      - key: "seed"
        value: 42
  
  # Example with no epsilon decay
  - name: "simple_sarsa"
    hyperparameters:
      - key: "epsilon"
        value: 0.1
      - key: "epsilon_decay"
        value: 0.0  # No decay (epsilon stays constant)
      - key: "epsilon_min"
        value: 0.0

max_iterations: 2000
```

## Environment Requirements

The SARSA model requires environments with:
- **Discrete action space**: Must be `gym.spaces.Discrete`
- **Discrete observation space**: Must be `gym.spaces.Discrete`

The model will automatically validate these requirements and raise a `ValueError` if they are not met.

## Training Process

1. **Initialization**: Q-table is initialized as an empty dictionary
2. **Episode Loop**: For each episode:
   - Reset environment and get initial observation
   - Select initial action using epsilon-greedy policy
   - For each step:
     - Take action and observe reward, next observation, done flag
     - Select next action using epsilon-greedy policy
     - Update Q-value using SARSA update rule
           - Decay epsilon: ε ← max(ε_min, ε × (1 - ε_decay)) (only if epsilon_decay > 0)
3. **Metrics Collection**: Tracks rewards, episode lengths, and Q-table size

## Hyperparameter Validation

All hyperparameters are validated to ensure they are within the correct ranges:
- **learning_rate**: Must be between 0.0 and 1.0
- **discount_factor**: Must be between 0.0 and 1.0
- **epsilon**: Must be between 0.0 and 1.0
- **epsilon_decay**: Must be between 0.0 and 1.0 (represents fraction to decay per step)
- **epsilon_min**: Must be between 0.0 and 1.0

If any hyperparameter is outside its valid range, a `ValueError` is raised with a descriptive message.

### Epsilon Decay Examples

- **epsilon_decay = 0.0**: No decay (epsilon stays constant)
- **epsilon_decay = 0.01**: Decay 1% of epsilon per step (slow decay)
- **epsilon_decay = 0.1**: Decay 10% of epsilon per step (fast decay)
- **epsilon_decay = 0.5**: Decay 50% of epsilon per step (very fast decay)

## Model Persistence

### Saving a Model

```python
from pathlib import Path

# Save trained model
save_path = Path("my_sarsa_model")
model.save(save_path)
```

### Loading a Model

```python
# Load model
new_model = create_model("simple_sarsa")
new_model.load(save_path)

# Reconfigure with environment (required after loading)
new_model.configure(env, hyperparameters)
```

## Training Metrics

The model tracks the following metrics during training:

- `mean_reward`: Average reward per episode
- `std_reward`: Standard deviation of rewards
- `min_reward`: Minimum episode reward
- `max_reward`: Maximum episode reward
- `mean_episode_length`: Average episode length
- `std_episode_length`: Standard deviation of episode lengths
- `episodes_completed`: Number of completed episodes
- `final_epsilon`: Final exploration rate
- `q_table_size`: Number of state-action pairs in Q-table

## Implementation Details

### Q-table Structure

The Q-table is implemented as a nested dictionary:
```python
{
    state_key: {
        action: q_value,
        ...
    },
    ...
}
```

Where `state_key` is a tuple representation of the discrete state.

### State Representation

For discrete observation spaces, states are converted to hashable tuples:
- Scalar observations: `(observation,)`
- Array observations: `tuple(observation.flatten())`

### Action Selection

Uses epsilon-greedy policy:
- **Exploit** (1-ε probability): Choose action with highest Q-value
- **Explore** (ε probability): Choose random action

## Limitations

- Only works with discrete action and observation spaces
- Q-table size grows with the number of states visited
- May not scale well to very large state spaces
- On-policy nature means it learns the policy being followed, not necessarily the optimal policy

## Comparison with Q-learning

| Aspect | SARSA | Q-learning |
|--------|-------|------------|
| Policy | On-policy | Off-policy |
| Update | Uses actual next action | Uses maximum Q-value |
| Learning | Learns policy being followed | Learns optimal policy |
| Safety | Generally safer in risky environments | May learn risky policies |
| Convergence | Slower convergence | Faster convergence |

## Testing

Run the test scripts to verify the implementation:

```bash
# Basic functionality test
python test_sarsa.py

# Model discovery and validation test
python test_sarsa_discovery.py

# Hyperparameter validation test
python test_sarsa_validation.py
```
