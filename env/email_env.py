"""Email triage reinforcement learning environment.

This module implements an OpenEnv-style backend environment for classifying
emails into one of four labels:
    - spam
    - important
    - urgent
    - normal
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


VALID_ACTIONS = ("spam", "important", "urgent", "normal")


@dataclass(frozen=True)
class EmailSample:
    """Single email classification sample used by the environment."""

    text: str
    expected_label: str


class EmailTriageEnv:
    """OpenEnv-style environment for one-step email triage.

    API:
        - reset() -> str
        - state() -> Dict[str, Any]
        - step(action) -> (observation, reward, done, info)
    """

    def __init__(self, sample: EmailSample, partial_credit_map: Optional[Dict[Tuple[str, str], float]] = None) -> None:
        if sample.expected_label not in VALID_ACTIONS:
            raise ValueError(f"Invalid expected_label: {sample.expected_label}")

        self.sample = sample
        # Maps (predicted_label, expected_label) -> correctness score in [0.0, 1.0].
        # Reward is then derived from this score in step().
        self.partial_credit_map: Dict[Tuple[str, str], float] = partial_credit_map or {
            ("urgent", "important"): 0.5,
            ("important", "urgent"): 0.5,
            ("important", "spam"): 0.3,
            ("spam", "important"): 0.3,
            ("normal", "important"): 0.2,
            ("important", "normal"): 0.2,
            ("normal", "urgent"): 0.2,
            ("urgent", "normal"): 0.2,
        }
        self._done: bool = False
        self._last_action: Optional[str] = None
        self._last_score: float = 0.0
        self._last_reward: float = 0.0

    @property
    def action_space(self) -> Tuple[str, ...]:
        """Discrete action labels supported by this environment."""
        return VALID_ACTIONS

    @property
    def observation_space(self) -> str:
        """Simple textual description of the observation type."""
        return "email_text:str"

    def reset(self) -> str:
        """Reset episode state and return initial observation."""
        self._done = False
        self._last_action = None
        self._last_score = 0.0
        self._last_reward = 0.0
        return self.sample.text

    def state(self) -> Dict[str, Any]:
        """Return current environment state in a serializable format."""
        return {
            "email_text": self.sample.text,
            "done": self._done,
            "last_action": self._last_action,
            "last_score": self._last_score,
            "last_reward": self._last_reward,
            "action_space": list(self.action_space),
            "observation_space": self.observation_space,
        }

    def _score_prediction(self, predicted: str, expected: str) -> float:
        """Return correctness score in [0.0, 1.0]."""
        if predicted == expected:
            return 1.0
        return self.partial_credit_map.get((predicted, expected), 0.0)

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Apply action and return OpenEnv-style transition tuple.

        Returns:
            observation: str
                Observation returned after applying the action.
                For this one-step episode environment, it is the same email text.
            reward: float
                +1.0 for exact match, partial score value for partial matches
                (e.g. +0.5, +0.3, +0.2), and -1.0 for wrong classifications.
            done: bool
                True after one classification.
            info: Dict[str, Any]
                Extra metadata about grading.
        """
        if self._done:
            raise RuntimeError("Episode has ended. Call reset() before step().")

        normalized_action = action.strip().lower()
        if normalized_action not in self.action_space:
            raise ValueError(f"Invalid action '{action}'. Valid actions: {self.action_space}")

        expected = self.sample.expected_label
        score = self._score_prediction(normalized_action, expected)
        if score == 1.0:
            reward = 1.0
            grade = "correct"
        elif score > 0.0:
            reward = score
            grade = "partially_correct"
        else:
            reward = -1.0
            grade = "wrong"

        self._done = True
        self._last_action = normalized_action
        self._last_score = score
        self._last_reward = reward

        info = {
            "grade": grade,
            "score": score,
            "expected_label": expected,
            "predicted_label": normalized_action,
        }
        # Return the same observation for consistency; some validators dislike
        # returning `None` from `step()` even on terminal transitions.
        return self.sample.text, reward, True, info
