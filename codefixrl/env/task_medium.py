"""
Task 2 (Medium) – Logical Bug Detection
The agent must analyse a function, describe the logical bug,
and propose the corrected implementation.

Reward logic:
  +0.3   correct bug identification (description matches key phrases)
  +0.7   correct fix (code produces expected output)
  -0.2   incorrect reasoning or wrong fix
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple
import textwrap

from env.base_env import BaseCodeEnv
from models.observation import ObservationModel, Difficulty
from models.action import ActionModel, ActionType


# ---------------------------------------------------------------------------
# Scenario bank – 5 functions each containing one logical bug
# ---------------------------------------------------------------------------
_SCENARIOS: List[Dict[str, Any]] = [
    {
        "id": "medium_001",
        "description": (
            "The function below should return the factorial of n. "
            "It contains a logical bug. Identify the bug and propose a fix."
        ),
        "code": textwrap.dedent("""\
            def factorial(n):
                result = 0       # BUG: should be 1 (multiplicative identity)
                for i in range(1, n + 1):
                    result *= i
                return result
        """),
        "bug_keywords": ["result", "initialized", "zero", "0", "multiplicative", "identity", "1"],
        "correct_fix": textwrap.dedent("""\
            def factorial(n):
                result = 1
                for i in range(1, n + 1):
                    result *= i
                return result
        """),
        "expected_output": "factorial(5) == 120",
        "constraints": ["Preserve the iterative structure.", "Do not use math.factorial."],
    },
    {
        "id": "medium_002",
        "description": (
            "The function should return the maximum value in a list. "
            "Find the logical error and fix it."
        ),
        "code": textwrap.dedent("""\
            def find_max(nums):
                max_val = nums[0]
                for num in nums:
                    if num < max_val:   # BUG: should be >
                        max_val = num
                return max_val
        """),
        "bug_keywords": ["comparison", "<", "greater", ">", "minimum", "max", "wrong operator"],
        "correct_fix": textwrap.dedent("""\
            def find_max(nums):
                max_val = nums[0]
                for num in nums:
                    if num > max_val:
                        max_val = num
                return max_val
        """),
        "expected_output": "find_max([3, 1, 4, 1, 5, 9]) == 9",
        "constraints": ["Do not use the built-in max()."],
    },
    {
        "id": "medium_003",
        "description": (
            "The function should check whether a number is prime. "
            "It has an off-by-one error in the loop range. Fix it."
        ),
        "code": textwrap.dedent("""\
            def is_prime(n):
                if n < 2:
                    return False
                for i in range(2, n):   # BUG: should be range(2, int(n**0.5) + 1)
                    if n % i == 0:
                        return False
                return True
        """),
        "bug_keywords": ["range", "sqrt", "square root", "inefficient", "n**0.5", "int(n**0.5)"],
        "correct_fix": textwrap.dedent("""\
            def is_prime(n):
                if n < 2:
                    return False
                for i in range(2, int(n**0.5) + 1):
                    if n % i == 0:
                        return False
                return True
        """),
        "expected_output": "is_prime(11) == True, is_prime(4) == False",
        "constraints": ["Correctness must be preserved, not just efficiency."],
    },
    {
        "id": "medium_004",
        "description": (
            "This binary search should return the index of target in a sorted list. "
            "Find the logical bug and correct it."
        ),
        "code": textwrap.dedent("""\
            def binary_search(arr, target):
                low, high = 0, len(arr)    # BUG: high should be len(arr) - 1
                while low <= high:
                    mid = (low + high) // 2
                    if arr[mid] == target:
                        return mid
                    elif arr[mid] < target:
                        low = mid + 1
                    else:
                        high = mid - 1
                return -1
        """),
        "bug_keywords": ["high", "len(arr) - 1", "index", "out of bounds", "off-by-one"],
        "correct_fix": textwrap.dedent("""\
            def binary_search(arr, target):
                low, high = 0, len(arr) - 1
                while low <= high:
                    mid = (low + high) // 2
                    if arr[mid] == target:
                        return mid
                    elif arr[mid] < target:
                        low = mid + 1
                    else:
                        high = mid - 1
                return -1
        """),
        "expected_output": "binary_search([1,3,5,7,9], 7) == 3",
        "constraints": ["Do not change the algorithm structure."],
    },
    {
        "id": "medium_005",
        "description": (
            "The function should reverse a string. "
            "It has a logical bug in the loop. Identify and fix it."
        ),
        "code": textwrap.dedent("""\
            def reverse_string(s):
                result = ""
                for i in range(len(s)):   # BUG: should iterate in reverse
                    result += s[i]
                return result
        """),
        "bug_keywords": ["reverse", "range", "len(s) - 1", "step", "-1", "backward"],
        "correct_fix": textwrap.dedent("""\
            def reverse_string(s):
                result = ""
                for i in range(len(s) - 1, -1, -1):
                    result += s[i]
                return result
        """),
        "expected_output": 'reverse_string("hello") == "olleh"',
        "constraints": ["Do not use slicing or reversed()."],
    },
]


class MediumTask(BaseCodeEnv):
    """
    Task 2 – Logical Bug Detection (Medium difficulty).

    Episode flow:
        Step 1: analyze_function() or identify_bug(description) → up to +0.3
        Step 2: propose_fix(replacement_code)                   → up to +0.7
    Episode ends after propose_fix or max_steps.
    """

    TASK_NAME = "logical_bug_detection"
    DIFFICULTY = "medium"
    MAX_STEPS = 4

    def __init__(self) -> None:
        super().__init__()
        self._bug_identified: bool = False

    def reset(self) -> ObservationModel:
        self._bug_identified = False
        return super().reset()

    # ------------------------------------------------------------------
    # BaseCodeEnv implementation
    # ------------------------------------------------------------------

    def _get_scenarios(self) -> List[Dict[str, Any]]:
        return _SCENARIOS

    def _build_observation(self) -> ObservationModel:
        s = self._current_scenario
        return ObservationModel(
            code=s["code"],
            description=s["description"],
            difficulty=Difficulty.medium,
            expected_output=s.get("expected_output"),
            constraints=s.get("constraints", []),
            step=self._step_count,
            max_steps=self.MAX_STEPS,
            task_id=s["id"],
            task_name=self.TASK_NAME,
        )

    def _evaluate_action(
        self, action: ActionModel
    ) -> Tuple[float, bool, Dict[str, Any]]:
        s = self._current_scenario
        reward = 0.0
        done = False
        info: Dict[str, Any] = {"action_type": action.action_type}

        if action.action_type in (ActionType.analyze_function, ActionType.identify_bug):
            desc = (action.description or "").lower()
            hits = sum(1 for kw in s["bug_keywords"] if kw.lower() in desc)
            if hits >= 2:
                reward = 0.3
                self._bug_identified = True
                info["feedback"] = f"Good analysis! Matched {hits} key concepts."
            elif hits == 1:
                reward = 0.1
                self._bug_identified = True
                info["feedback"] = "Partial analysis – on the right track."
            else:
                reward = -0.1
                info["feedback"] = "Incorrect reasoning. Revisit the function logic."

        elif action.action_type == ActionType.propose_fix:
            submitted = _normalise(action.replacement_code or "")
            expected = _normalise(s["correct_fix"])
            if submitted == expected:
                reward = 0.7
                done = True
                info["feedback"] = "Correct fix! Bug eliminated."
            elif _line_similarity(submitted, expected) >= 0.75:
                reward = 0.4
                done = True
                info["feedback"] = "Mostly correct fix accepted. Episode ends."
            else:
                reward = -0.2
                info["feedback"] = "Incorrect fix. Revisit your analysis."

        elif action.action_type == ActionType.suggest_fix:
            # Allow suggest_fix as an alias for propose_fix
            submitted = _normalise(action.replacement_code or "")
            expected = _normalise(s["correct_fix"])
            if submitted == expected:
                reward = 0.7
                done = True
                info["feedback"] = "Correct fix submitted."
            else:
                reward = -0.2
                info["feedback"] = "Incorrect fix."

        else:
            info["feedback"] = (
                f"Action '{action.action_type}' is not applicable for this task. "
                "Use analyze_function, identify_bug, or propose_fix."
            )
            reward = -0.05

        return reward, done, info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(code: str) -> str:
    """Strip trailing whitespace from each line for fair comparison."""
    return "\n".join(line.rstrip() for line in code.strip().splitlines())


def _line_similarity(a: str, b: str) -> float:
    """Line-level Jaccard similarity for near-correct fix detection."""
    lines_a = set(a.splitlines())
    lines_b = set(b.splitlines())
    if not lines_a and not lines_b:
        return 1.0
    intersection = len(lines_a & lines_b)
    union = len(lines_a | lines_b)
    return intersection / union if union else 0.0
