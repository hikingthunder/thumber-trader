
import random
import math
from typing import Dict, Tuple

class StrategyEngine:
    """Base interface for pluggable trading strategy engines."""

    strategy_name = "base"

    async def run(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


class ExecutionRLAgent:
    """Lightweight RL execution policy for rest/chase/cancel decisions."""

    ACTION_REST = "REST"
    ACTION_CHASE = "CHASE"
    ACTION_CANCEL = "CANCEL"

    def __init__(
        self,
        learning_rate: float,
        discount: float,
        epsilon: float,
        min_epsilon: float,
        epsilon_decay: float,
    ):
        self.learning_rate = max(0.0001, float(learning_rate))
        self.discount = min(0.999, max(0.0, float(discount)))
        self.epsilon = min(1.0, max(0.0, float(epsilon)))
        self.min_epsilon = min(1.0, max(0.0, float(min_epsilon)))
        self.epsilon_decay = min(1.0, max(0.9, float(epsilon_decay)))
        self._q: Dict[Tuple[str, str], float] = {}

    @staticmethod
    def actions() -> Tuple[str, str, str]:
        return (ExecutionRLAgent.ACTION_REST, ExecutionRLAgent.ACTION_CHASE, ExecutionRLAgent.ACTION_CANCEL)

    def select_action(self, state: str) -> str:
        if random.random() < self.epsilon:
            return random.choice(self.actions())
        return max(self.actions(), key=lambda a: self._q.get((state, a), 0.0))

    def update(self, prev_state: str, action: str, reward: float, next_state: str) -> None:
        current_q = self._q.get((prev_state, action), 0.0)
        max_next = max(self._q.get((next_state, a), 0.0) for a in self.actions())
        target = reward + self.discount * max_next
        self._q[(prev_state, action)] = current_q + self.learning_rate * (target - current_q)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def fill_probability(self, state: str) -> float:
        raw = self._q.get((state, self.ACTION_REST), 0.0)
        return 1.0 / (1.0 + math.exp(-raw))
