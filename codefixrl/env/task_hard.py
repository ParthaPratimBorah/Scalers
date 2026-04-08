"""
Task 3 (Hard) – Code Optimization & Refactoring
The agent must detect an inefficiency, refactor the code,
and explain the improvement.

Reward logic:
  +0.2  correct inefficiency identified (keyword match)
  +0.5  refactored code is functionally correct and more efficient
  +0.3  quality explanation provided
  -0.2  incorrect refactor (changes behaviour)
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple
import textwrap

from env.base_env import BaseCodeEnv
from models.observation import ObservationModel, Difficulty
from models.action import ActionModel, ActionType


# ---------------------------------------------------------------------------
# Scenario bank – 5 inefficient code snippets
# ---------------------------------------------------------------------------
_SCENARIOS: List[Dict[str, Any]] = [
    {
        "id": "hard_001",
        "description": (
            "The function below finds duplicates in a list using an O(n²) approach. "
            "Refactor it to run in O(n) time using a set."
        ),
        "code": textwrap.dedent("""\
            def find_duplicates(items):
                duplicates = []
                for i in range(len(items)):
                    for j in range(i + 1, len(items)):
                        if items[i] == items[j] and items[i] not in duplicates:
                            duplicates.append(items[i])
                return duplicates
        """),
        "inefficiency_keywords": ["o(n²)", "nested loop", "quadratic", "n squared", "double loop", "inner loop"],
        "refactored_code": textwrap.dedent("""\
            def find_duplicates(items):
                seen = set()
                duplicates = set()
                for item in items:
                    if item in seen:
                        duplicates.add(item)
                    seen.add(item)
                return list(duplicates)
        """),
        "explanation_keywords": ["set", "o(n)", "linear", "hash", "lookup"],
        "expected_output": "find_duplicates([1,2,3,2,4,3]) == [2, 3] (order may vary)",
        "constraints": [
            "Must preserve functionality.",
            "Must use O(n) time complexity.",
            "Return type must be a list.",
        ],
    },
    {
        "id": "hard_002",
        "description": (
            "This function concatenates strings in a loop, creating O(n²) copies. "
            "Refactor it to use a list and join()."
        ),
        "code": textwrap.dedent("""\
            def build_csv(values):
                result = ""
                for v in values:
                    result += str(v) + ","
                return result.rstrip(",")
        """),
        "inefficiency_keywords": ["string concatenation", "+=", "immutable", "o(n²)", "copies", "join"],
        "refactored_code": textwrap.dedent("""\
            def build_csv(values):
                return ",".join(str(v) for v in values)
        """),
        "explanation_keywords": ["join", "list", "generator", "o(n)", "single allocation"],
        "expected_output": 'build_csv([1, 2, 3]) == "1,2,3"',
        "constraints": ["Must produce identical output.", "Must use str.join()."],
    },
    {
        "id": "hard_003",
        "description": (
            "The function computes Fibonacci numbers recursively without memoisation, "
            "causing exponential time complexity. Refactor using dynamic programming."
        ),
        "code": textwrap.dedent("""\
            def fib(n):
                if n <= 1:
                    return n
                return fib(n - 1) + fib(n - 2)
        """),
        "inefficiency_keywords": ["exponential", "o(2^n)", "recursion", "repeated", "recomputation", "memoization", "memoisation", "cache"],
        "refactored_code": textwrap.dedent("""\
            def fib(n):
                if n <= 1:
                    return n
                a, b = 0, 1
                for _ in range(2, n + 1):
                    a, b = b, a + b
                return b
        """),
        "explanation_keywords": ["iterative", "o(n)", "linear", "dp", "dynamic programming", "bottom-up"],
        "expected_output": "fib(10) == 55",
        "constraints": ["Must return correct Fibonacci numbers.", "Must run in O(n) time."],
    },
    {
        "id": "hard_004",
        "description": (
            "The function below has duplicate code blocks for even and odd number processing. "
            "Refactor to eliminate duplication."
        ),
        "code": textwrap.dedent("""\
            def process_numbers(nums):
                evens = []
                odds = []
                for n in nums:
                    if n % 2 == 0:
                        evens.append(n * 2)
                        evens.append(n * 2)   # duplicate append
                    else:
                        odds.append(n * 3)
                        odds.append(n * 3)    # duplicate append
                return evens, odds
        """),
        "inefficiency_keywords": ["duplicate", "repeated append", "redundant", "twice", "copy"],
        "refactored_code": textwrap.dedent("""\
            def process_numbers(nums):
                evens = [n * 2 for n in nums if n % 2 == 0]
                odds = [n * 3 for n in nums if n % 2 != 0]
                return evens * 2, odds * 2
        """),
        "explanation_keywords": ["list comprehension", "removed duplicate", "cleaner", "concise", "dry"],
        "expected_output": "process_numbers([1,2,3]) == ([4, 4], [3, 3, 9, 9])",
        "constraints": ["Output must match the original (including duplicates in result lists)."],
    },
    {
        "id": "hard_005",
        "description": (
            "The function checks membership in a list inside a loop, giving O(n²) complexity. "
            "Refactor using a set for O(n) performance."
        ),
        "code": textwrap.dedent("""\
            def common_elements(list_a, list_b):
                result = []
                for item in list_a:
                    if item in list_b and item not in result:
                        result.append(item)
                return result
        """),
        "inefficiency_keywords": ["in list_b", "o(n²)", "linear scan", "set", "membership", "lookup"],
        "refactored_code": textwrap.dedent("""\
            def common_elements(list_a, list_b):
                set_b = set(list_b)
                seen = set()
                result = []
                for item in list_a:
                    if item in set_b and item not in seen:
                        result.append(item)
                        seen.add(item)
                return result
        """),
        "explanation_keywords": ["set", "o(1) lookup", "o(n)", "hash", "convert"],
        "expected_output": "common_elements([1,2,3,4], [2,4,6]) == [2, 4]",
        "constraints": ["Preserve order of first occurrence.", "Must use a set for O(1) lookups."],
    },
]


class HardTask(BaseCodeEnv):
    """
    Task 3 – Code Optimization & Refactoring (Hard difficulty).

    Episode flow:
        Step 1: detect_inefficiency(description) → up to +0.2
        Step 2: refactor_code(replacement_code)  → up to +0.5
        Step 3: explain_improvement(explanation) → up to +0.3
    Episode ends after explain_improvement or max_steps.
    """

    TASK_NAME = "code_optimization_refactoring"
    DIFFICULTY = "hard"
    MAX_STEPS = 5

    def __init__(self) -> None:
        super().__init__()
        self._inefficiency_found: bool = False
        self._refactor_done: bool = False

    def reset(self) -> ObservationModel:
        self._inefficiency_found = False
        self._refactor_done = False
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
            difficulty=Difficulty.hard,
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

        if action.action_type == ActionType.detect_inefficiency:
            desc = (action.description or "").lower()
            hits = sum(1 for kw in s["inefficiency_keywords"] if kw.lower() in desc)
            if hits >= 2:
                reward = 0.2
                self._inefficiency_found = True
                info["feedback"] = f"Correct inefficiency detection ({hits} key concepts matched)."
            elif hits == 1:
                reward = 0.1
                self._inefficiency_found = True
                info["feedback"] = "Partially correct – keep digging."
            else:
                reward = -0.1
                info["feedback"] = "Did not identify the core inefficiency."

        elif action.action_type == ActionType.refactor_code:
            submitted = _normalise(action.replacement_code or "")
            expected = _normalise(s["refactored_code"])
            if submitted == expected:
                reward = 0.5
                self._refactor_done = True
                info["feedback"] = "Excellent refactor! Code is now efficient."
            elif _line_similarity(submitted, expected) >= 0.7:
                reward = 0.3
                self._refactor_done = True
                info["feedback"] = "Good refactor – minor differences from ideal."
            else:
                reward = -0.2
                info["feedback"] = "Refactor does not meet requirements."

        elif action.action_type == ActionType.explain_improvement:
            exp = (action.explanation or "").lower()
            exp_hits = sum(1 for kw in s["explanation_keywords"] if kw.lower() in exp)
            if exp_hits >= 2:
                reward = 0.3
                done = True
                info["feedback"] = f"Thorough explanation ({exp_hits} concepts covered). Episode complete!"
            elif exp_hits == 1:
                reward = 0.15
                done = True
                info["feedback"] = "Adequate explanation. Episode ends."
            else:
                reward = 0.0
                done = True
                info["feedback"] = "Explanation lacks depth. Episode ends anyway."

        else:
            info["feedback"] = (
                f"Action '{action.action_type}' is not valid for this task. "
                "Use detect_inefficiency, refactor_code, or explain_improvement."
            )
            reward = -0.05

        return reward, done, info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(code: str) -> str:
    """Strip trailing whitespace and normalise blank lines."""
    return "\n".join(line.rstrip() for line in code.strip().splitlines())


def _line_similarity(a: str, b: str) -> float:
    """Line-level Jaccard similarity for near-correct refactor detection."""
    lines_a = set(a.splitlines())
    lines_b = set(b.splitlines())
    if not lines_a and not lines_b:
        return 1.0
    intersection = len(lines_a & lines_b)
    union = len(lines_a | lines_b)
    return intersection / union if union else 0.0
