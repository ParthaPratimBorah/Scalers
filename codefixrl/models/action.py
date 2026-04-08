"""
Action and Reward schemas for the CodeFixRL environment.
Defines structured actions the agent can take, and the reward model
returned after each step.
"""

from __future__ import annotations
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class ActionType(str, Enum):
    """All valid action types across all tasks."""
    locate_error = "locate_error"         # Easy: point to the faulty line
    suggest_fix = "suggest_fix"           # Easy/Medium: provide corrected code
    analyze_function = "analyze_function" # Medium: describe function's intent
    identify_bug = "identify_bug"         # Medium: describe the logical bug
    propose_fix = "propose_fix"           # Medium: provide corrected function
    detect_inefficiency = "detect_inefficiency"  # Hard: describe the bottleneck
    refactor_code = "refactor_code"       # Hard: provide refactored code
    explain_improvement = "explain_improvement"  # Hard: justify the changes


class ActionModel(BaseModel):
    """
    Structured action submitted by the agent.
    Not all fields are required for every action_type;
    only relevant fields are validated by each task.
    """

    action_type: ActionType = Field(
        description="The type of action being taken."
    )
    line_number: Optional[int] = Field(
        default=None,
        ge=1,
        description="1-indexed line number (used with locate_error)."
    )
    replacement_code: Optional[str] = Field(
        default=None,
        description="Corrected or refactored code block."
    )
    description: Optional[str] = Field(
        default=None,
        description="Natural-language description (identify_bug, detect_inefficiency, analyze_function)."
    )
    explanation: Optional[str] = Field(
        default=None,
        description="Explanation of changes (explain_improvement)."
    )

    model_config = {"use_enum_values": True}


class RewardModel(BaseModel):
    """
    Reward signal returned after each environment step.
    """

    reward: float = Field(
        description="Immediate reward for the last action. Range: [-1.0, 1.0]."
    )
    cumulative_reward: float = Field(
        description="Total accumulated reward for the episode so far."
    )
    done: bool = Field(
        description="Whether the episode has ended."
    )
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional diagnostic information about the step."
    )
