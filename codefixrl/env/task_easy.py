"""
Task 1 (Easy) – Syntax Error Detection
The agent must identify the line containing a syntax error
and suggest the corrected code.

Reward logic:
  +0.4  correct line number identified
  +0.6  correct fix provided
  -0.2  wrong fix / wrong line with correct fix
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple
import textwrap

from env.base_env import BaseCodeEnv
from models.observation import ObservationModel, Difficulty
from models.action import ActionModel, ActionType


# ---------------------------------------------------------------------------
# Scenario bank – 5 distinct Python snippets each with one syntax error
# ---------------------------------------------------------------------------
_SCENARIOS: List[Dict[str, Any]] = [
    {
        "id": "easy_001",
        "description": (
            "The function below contains a syntax error. "
            "Identify the faulty line number and suggest the corrected line."
        ),
        "code": textwrap.dedent("""\
            def greet(name):
                print("Hello, " + name)

            def farewell(name)
                print("Goodbye, " + name)
        """),
        "error_line": 4,                    # missing colon after def farewell(name)
        "correct_fix": "def farewell(name):",
        "expected_output": "No output; function must be syntactically valid.",
        "constraints": [
            "Fix must be a single line replacement.",
            "Do not change any other lines.",
        ],
    },
    {
        "id": "easy_002",
        "description": (
            "This code block has an indentation error. "
            "Identify the faulty line and provide the corrected line."
        ),
        "code": textwrap.dedent("""\
            def add(a, b):
            return a + b
        """),
        "error_line": 2,                    # missing indent before return
        "correct_fix": "    return a + b",
        "expected_output": "add(2, 3) == 5",
        "constraints": [
            "Use 4-space indentation.",
            "Do not modify the function signature.",
        ],
    },
    {
        "id": "easy_003",
        "description": (
            "The list comprehension below has a bracket mismatch. "
            "Find the line and fix it."
        ),
        "code": textwrap.dedent("""\
            numbers = [1, 2, 3, 4, 5]
            squares = [x**2 for x in numbers
            print(squares)
        """),
        "error_line": 2,                    # missing closing ]
        "correct_fix": "squares = [x**2 for x in numbers]",
        "expected_output": "[1, 4, 9, 16, 25]",
        "constraints": ["Fix must close the list comprehension on the same line."],
    },
    {
        "id": "easy_004",
        "description": (
            "There is a missing colon in an if-statement. "
            "Locate the line and suggest the fix."
        ),
        "code": textwrap.dedent("""\
            x = 10
            if x > 5
                print("x is greater than 5")
        """),
        "error_line": 2,                    # missing colon after if condition
        "correct_fix": "if x > 5:",
        "expected_output": "x is greater than 5",
        "constraints": ["Only fix the missing colon; do not alter logic."],
    },
    {
        "id": "easy_005",
        "description": (
            "A string is not properly closed. "
            "Find the line with the unterminated string literal."
        ),
        "code": textwrap.dedent("""\
            message = "Hello, world!
            print(message)
        """),
        "error_line": 1,                    # missing closing "
        "correct_fix": 'message = "Hello, world!"',
        "expected_output": "Hello, world!",
        "constraints": ["Use double quotes to close the string."],
    },
]


class EasyTask(BaseCodeEnv):
    """
    Task 1 – Syntax Error Detection (Easy difficulty).

    Episode flow:
        Step 1: Agent calls locate_error(line_number)      → up to +0.4
        Step 2: Agent calls suggest_fix(replacement_code)  → up to +0.6
    Episode ends after suggest_fix is called or max_steps is reached.
    """

    TASK_NAME = "syntax_error_detection"
    DIFFICULTY = "easy"
    MAX_STEPS = 3   # locate + fix + one optional retry

    def __init__(self) -> None:
        super().__init__()
        self._line_located: bool = False
        self._line_correct: bool = False

    def reset(self) -> ObservationModel:
        self._line_located = False
        self._line_correct = False
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
            difficulty=Difficulty.easy,
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

        if action.action_type == ActionType.locate_error:
            if action.line_number == s["error_line"]:
                reward = 0.4
                self._line_located = True
                self._line_correct = True
                info["feedback"] = f"Correct! The error is on line {s['error_line']}."
            else:
                reward = -0.1
                self._line_located = True
                self._line_correct = False
                info["feedback"] = (
                    f"Incorrect line. Hint: look between lines 1–{len(s['code'].splitlines())}."
                )

        elif action.action_type == ActionType.suggest_fix:
            fix = (action.replacement_code or "").strip()
            expected = s["correct_fix"].strip()
            if fix == expected:
                reward = 0.6
                done = True
                info["feedback"] = "Perfect fix! Syntax error resolved."
            elif _similarity(fix, expected) > 0.8:
                # Partial credit for near-correct fixes
                reward = 0.3
                done = True
                info["feedback"] = "Close fix accepted (minor differences). Episode ends."
            else:
                reward = -0.2
                info["feedback"] = "Incorrect fix. Try again."

        else:
            info["feedback"] = (
                f"Action '{action.action_type}' is not valid for this task. "
                "Use locate_error or suggest_fix."
            )
            reward = -0.05

        return reward, done, info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _similarity(a: str, b: str) -> float:
    """
    Simple character-level Jaccard similarity between two strings.
    Used for near-correct fix detection without heavy dependencies.
    """
    a_chars, b_chars = set(a.lower()), set(b.lower())
    if not a_chars and not b_chars:
        return 1.0
    intersection = len(a_chars & b_chars)
    union = len(a_chars | b_chars)
    return intersection / union if union else 0.0
