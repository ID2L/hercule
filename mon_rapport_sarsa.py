# -*- coding: utf-8 -*-
"""
Experiment Report
Generated automatically from experiment data
Experiment Path: outputs\simple_games\simple_games\FrozenLake-v1\is_sli_False__map_nam_4x4__max_epi_ste_200\simple_sarsa\dis_fac_0.8__eps_0.77__lea_rat_0.2
"""

import base64
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# =============================================================================
# Experiment Overview
# =============================================================================

print("# Experiment Report")
print(f"**Experiment Path:** `outputs\simple_games\simple_games\FrozenLake-v1\is_sli_False__map_nam_4x4__max_epi_ste_200\simple_sarsa\dis_fac_0.8__eps_0.77__lea_rat_0.2`")
print()

# Environment Information

print("## Environment Configuration")
print("```json")
print(json.dumps({
  "disable_env_checker": false,
  "id": "FrozenLake-v1",
  "kwargs": {
    "is_slippery": false,
    "map_name": "4x4"
  },
  "max_episode_steps": 200
}, indent=2))
print("```")
print()


# Model Information

print("## Model Configuration")
print("```json")
print(json.dumps({
  "q_table": [
    [
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      0.03200000000000001,
      0.0
    ],
    [
      0.0,
      0.0,
      0.36000000000000004,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0,
      0.0
    ]
  ]
}, indent=2))
print("```")
print()


# Training Information

print("## Training Information")
print(f"- **Learning Epochs:** 100")
print(f"- **Testing Epochs:** 100")
print()


# =============================================================================
# Data Loading and Preparation
# =============================================================================

# Load experiment data
learning_rewards = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
learning_steps = [8, 8, 3, 7, 10, 2, 12, 5, 2, 8, 9, 12, 11, 11, 4, 10, 3, 18, 2, 13, 8, 5, 15, 13, 16, 12, 2, 6, 7, 3, 2, 4, 12, 5, 7, 11, 12, 21, 8, 3, 3, 7, 14, 5, 18, 6, 8, 2, 6, 11, 45, 19, 8, 12, 2, 2, 3, 6, 12, 13, 7, 5, 4, 2, 4, 3, 16, 4, 7, 4, 5, 8, 14, 8, 3, 10, 9, 9, 3, 12, 4, 12, 5, 8, 11, 7, 4, 6, 3, 4, 5, 4, 27, 4, 3, 2, 6, 2, 16, 13]
testing_rewards = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
testing_steps = [12, 9, 2, 6, 3, 11, 3, 17, 7, 5, 15, 4, 11, 5, 6, 5, 3, 12, 19, 7, 2, 11, 9, 3, 6, 2, 3, 14, 9, 12, 8, 9, 13, 3, 8, 3, 21, 4, 3, 5, 19, 4, 17, 5, 4, 2, 6, 21, 15, 9, 2, 2, 3, 7, 9, 2, 3, 3, 4, 2, 7, 9, 8, 4, 2, 3, 4, 8, 9, 13, 6, 2, 8, 5, 7, 6, 10, 9, 7, 5, 2, 5, 9, 2, 11, 4, 5, 21, 5, 6, 8, 8, 17, 13, 3, 12, 2, 14, 2, 5]

print("## Data Summary")
print(f"- **Learning Episodes:** {len(learning_rewards)}")
print(f"- **Testing Episodes:** {len(testing_rewards)}")

print(f"- **Learning Reward Range:** [{min(learning_rewards):.3f}, {max(learning_rewards):.3f}]")
print(f"- **Learning Reward Mean:** {np.mean(learning_rewards):.3f} ± {np.std(learning_rewards):.3f}")


print(f"- **Testing Reward Range:** [{min(testing_rewards):.3f}, {max(testing_rewards):.3f}]")
print(f"- **Testing Reward Mean:** {np.mean(testing_rewards):.3f} ± {np.std(testing_rewards):.3f}")

print()

# =============================================================================
# Learning Progress Visualization
# =============================================================================


print("## Learning Progress")

# Create learning progress plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Learning rewards over time
ax1 = axes[0, 0]
ax1.plot(learning_rewards, alpha=0.7, label='Episode Reward', color='blue')
ax1.set_title('Learning Progress - Rewards')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward')
ax1.grid(True, alpha=0.3)

# Add moving average
window_size = min(50, len(learning_rewards) // 10)
if window_size > 1:
    moving_avg = pd.Series(learning_rewards).rolling(window=window_size).mean()
    ax1.plot(moving_avg, label=f'Moving Average (window={window_size})', linewidth=2, color='red')
    ax1.legend()

# Plot 2: Learning steps over time
ax2 = axes[0, 1]
ax2.plot(learning_steps, alpha=0.7, label='Episode Steps', color='green')
ax2.set_title('Learning Progress - Steps')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Steps')
ax2.grid(True, alpha=0.3)

if window_size > 1:
    moving_avg_steps = pd.Series(learning_steps).rolling(window=window_size).mean()
    ax2.plot(moving_avg_steps, label=f'Moving Average (window={window_size})', linewidth=2, color='red')
    ax2.legend()

# Plot 3: Reward distribution
ax3 = axes[1, 0]
ax3.hist(learning_rewards, bins=30, alpha=0.7, edgecolor='black', color='blue')
ax3.set_title('Reward Distribution (Learning)')
ax3.set_xlabel('Reward')
ax3.set_ylabel('Frequency')
ax3.grid(True, alpha=0.3)

# Plot 4: Steps distribution
ax4 = axes[1, 1]
ax4.hist(learning_steps, bins=30, alpha=0.7, edgecolor='black', color='green')
ax4.set_title('Steps Distribution (Learning)')
ax4.set_xlabel('Steps')
ax4.set_ylabel('Frequency')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Learning statistics
print("### Learning Statistics")
print(f"- **Mean Reward:** {np.mean(learning_rewards):.3f} ± {np.std(learning_rewards):.3f}")
print(f"- **Mean Steps:** {np.mean(learning_steps):.3f} ± {np.std(learning_steps):.3f}")
print(f"- **Success Rate:** {(np.array(learning_rewards) > 0).mean() * 100:.1f}%")
print()



# =============================================================================
# Final Model Evaluation
# =============================================================================


print("## Final Model Evaluation")

# Create evaluation plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Boxplot for testing rewards
ax1 = axes[0]
ax1.boxplot(testing_rewards, labels=['Testing Rewards'])
ax1.set_title('Final Model Evaluation - Rewards')
ax1.set_ylabel('Reward')
ax1.grid(True, alpha=0.3)

# Boxplot for testing steps
ax2 = axes[1]
ax2.boxplot(testing_steps, labels=['Testing Steps'])
ax2.set_title('Final Model Evaluation - Steps')
ax2.set_ylabel('Steps')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Evaluation statistics
print("### Evaluation Statistics")
print(f"- **Mean Reward:** {np.mean(testing_rewards):.3f} ± {np.std(testing_rewards):.3f}")
print(f"- **Mean Steps:** {np.mean(testing_steps):.3f} ± {np.std(testing_steps):.3f}")
print(f"- **Success Rate:** {(np.array(testing_rewards) > 0).mean() * 100:.1f}%")
print(f"- **Min Reward:** {np.min(testing_rewards):.3f}")
print(f"- **Max Reward:** {np.max(testing_rewards):.3f}")
print()



# =============================================================================
# Performance Analysis
# =============================================================================

print("## Performance Analysis")


# Compare learning vs testing performance
learning_mean = np.mean(learning_rewards)
testing_mean = np.mean(testing_rewards)
improvement = ((testing_mean - learning_mean) / abs(learning_mean)) * 100 if learning_mean != 0 else 0

print(f"- **Learning Performance:** {learning_mean:.3f} ± {np.std(learning_rewards):.3f}")
print(f"- **Testing Performance:** {testing_mean:.3f} ± {np.std(testing_rewards):.3f}")
print(f"- **Performance Change:** {improvement:+.1f}%")

# Learning curve analysis
if len(learning_rewards) > 100:
    early_performance = np.mean(learning_rewards[:len(learning_rewards)//3])
    late_performance = np.mean(learning_rewards[2*len(learning_rewards)//3:])
    learning_improvement = ((late_performance - early_performance) / abs(early_performance)) * 100 if early_performance != 0 else 0
    print(f"- **Learning Improvement:** {learning_improvement:+.1f}% (early vs late training)")


print()

# =============================================================================
# Conclusion
# =============================================================================

print("## Conclusion")

print(f"The experiment completed {len(learning_rewards)} learning episodes")

print(f"and {len(testing_rewards)} testing episodes.")


if np.mean(testing_rewards) > np.mean(learning_rewards):
    print("The model shows good generalization with testing performance exceeding learning performance.")
else:
    print("The model may be overfitting as testing performance is lower than learning performance.")



print("\n---")
print("*Report generated automatically by Hercule*")