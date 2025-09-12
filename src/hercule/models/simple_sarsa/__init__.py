"""Simple SARSA implementation with Q-table for discrete environments."""

from hercule.models.td_models import TDModel


class SimpleSarsaModel(TDModel):
    """
    Simple SARSA (State-Action-Reward-State-Action) implementation with Q-table.

    This model is designed for environments with discrete action and observation spaces.
    It uses a Q-table to store state-action values and updates them using the SARSA algorithm.

    SARSA is an on-policy temporal difference learning algorithm that learns the Q-values
    for the policy being followed, rather than the optimal policy.
    """

    # Class attribute for model name
    model_name: str = "simple_sarsa"

    def update(self, state: int, action: int, reward: float, next_state: int, next_action: int) -> None:
        """
        Update Q-values using SARSA (on-policy) temporal difference learning.

        SARSA uses the actual next action taken by the current policy for the update.

        Args:
            state: Current state
            action: Action taken in current state
            reward: Reward received
            next_state: Next state reached
            next_action: Next action taken by the current policy
        """
        self._q_table[state][action] += self._learning_rate * (
            reward + self._discount_factor * self._q_table[next_state, next_action] - self._q_table[state, action]
        )


__all__ = ["SimpleSarsaModel"]
