"""
Base environment class for CodeFixRL.
All task environments inherit from BaseCodeEnv.
"""

from __future__ import annotations
import abc
from typing import Any, Dict, Optional, Tuple
from models.observation import ObservationModel
from models.action import ActionModel, RewardModel


class BaseCodeEnv(abc.ABC):
    """
    Abstract base class that defines the OpenEnv-compatible interface.

    The step() method returns a 4-tuple matching the OpenEnv specification:
        observation, reward, done, info = env.step(action)

    Subclasses must implement:
        _build_observation() -> ObservationModel
        _evaluate_action(action) -> Tuple[float, bool, Dict[str, Any]]
    """

    # Task metadata – override in subclasses
    TASK_NAME: str = "base"
    DIFFICULTY: str = "easy"
    MAX_STEPS: int = 5

    def __init__(self) -> None:
        self._step_count: int = 0
        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self._scenario_index: int = 0
        self._current_scenario: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Public OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> ObservationModel:
        """
        Reset the environment to the beginning of a new episode.
        Selects the next scenario in round-robin order.
        Returns the initial observation.
        """
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._done = False
        scenarios = self._get_scenarios()
        self._scenario_index = self._scenario_index % len(scenarios)
        self._current_scenario = scenarios[self._scenario_index]
        self._scenario_index += 1
        return self._build_observation()

    def step(self, action: ActionModel) -> Tuple[ObservationModel, float, bool, Dict[str, Any]]:
        """
        Execute one action step.

        Returns a 4-tuple per the OpenEnv specification:
            (observation, reward, done, info)

        Infinite-loop guard: applying -1.0 penalty when MAX_STEPS is exceeded.
        """
        if self._done:
            obs = self._build_observation()
            return obs, 0.0, True, {"error": "Episode already finished. Call reset()."}

        self._step_count += 1

        # Infinite-loop guard: penalise agent for exceeding max steps
        if self._step_count > self.MAX_STEPS:
            self._done = True
            obs = self._build_observation()
            penalty = -1.0
            self._cumulative_reward += penalty
            info = {
                "error": f"Max steps ({self.MAX_STEPS}) exceeded. Infinite-loop penalty applied.",
                "step": self._step_count,
                "max_steps": self.MAX_STEPS,
                "cumulative_reward": round(self._cumulative_reward, 4),
            }
            return obs, penalty, True, info

        reward, done, info = self._evaluate_action(action)

        # Clamp reward to [-1, 1]
        reward = max(-1.0, min(1.0, reward))
        self._cumulative_reward += reward

        # Episode ends when task is solved or max steps reached
        if done or self._step_count >= self.MAX_STEPS:
            self._done = True
            done = True

        info["step"] = self._step_count
        info["max_steps"] = self.MAX_STEPS
        info["cumulative_reward"] = round(self._cumulative_reward, 4)

        obs = self._build_observation()
        return obs, round(reward, 4), done, info

    def state(self) -> Dict[str, Any]:
        """
        Return the full current state of the environment.
        Useful for debugging and logging.
        """
        obs = self._build_observation() if self._current_scenario else None
        return {
            "task_name": self.TASK_NAME,
            "difficulty": self.DIFFICULTY,
            "step": self._step_count,
            "max_steps": self.MAX_STEPS,
            "cumulative_reward": self._cumulative_reward,
            "done": self._done,
            "observation": obs.model_dump() if obs else None,
        }

    def as_reward_model(
        self,
        obs: ObservationModel,
        reward: float,
        done: bool,
        info: Dict[str, Any],
    ) -> RewardModel:
        """
        Convenience helper: wrap a step() tuple into a RewardModel.
        Useful for code that prefers the structured model over the raw tuple.
        """
        return RewardModel(
            reward=reward,
            cumulative_reward=info.get("cumulative_reward", self._cumulative_reward),
            done=done,
            info=info,
        )

    # ------------------------------------------------------------------
    # Abstract methods for subclasses
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _get_scenarios(self) -> list[Dict[str, Any]]:
        """Return the list of all task scenarios."""

    @abc.abstractmethod
    def _build_observation(self) -> ObservationModel:
        """Build and return the current observation from the active scenario."""

    @abc.abstractmethod
    def _evaluate_action(
        self, action: ActionModel
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Evaluate the submitted action against the current scenario.
        Returns (reward, task_solved, info_dict).
        """
