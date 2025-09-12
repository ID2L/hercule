from hercule.models.td_models import TDModel


class SimpleQLearningModel(TDModel):
    """
    Simple Q-Learning implementation with Q-table for discrete environments.

    This model is designed for environments with discrete action and observation spaces.
    It uses a Q-table to store state-action values and updates them using the Q-Learning algorithm.

    Q-Learning is an off-policy temporal difference learning algorithm that learns the Q-values
    for the optimal policy, regardless of the policy being followed.
    """

    model_name: str = "simple_q_learning"

    def update(self, state: int, action: int, reward: float, next_state: int, next_action: int) -> None:
        """
        Update Q-values using Q-Learning (off-policy) temporal difference learning.

        Q-Learning uses the best action for the next state (optimal policy) for the update,
        regardless of the actual next action taken.

        Args:
            state: Current state
            action: Action taken in current state
            reward: Reward received
            next_state: Next state reached
            next_action: Not used in Q-Learning (kept for interface compatibility)
        """
        # Find the best action for the next state (optimal policy)
        best_next_action = self.exploit(next_state)

        self._q_table[state][action] += self._learning_rate * (
            reward + self._discount_factor * self._q_table[next_state, best_next_action] - self._q_table[state, action]
        )
