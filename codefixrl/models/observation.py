"""
Observation schema for the CodeFixRL environment.
Defines the structured view the agent receives at each step.
"""

from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel, Field
from enum import Enum


class Difficulty(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


class ObservationModel(BaseModel):
    """
    Represents a single observation returned by the environment.
    Contains the code snippet, contextual metadata, and episode progress.
    """

    code: str = Field(
        description="The Python code snippet the agent must analyze."
    )
    description: str = Field(
        description="A natural-language description of the task objective."
    )
    difficulty: Difficulty = Field(
        description="Task difficulty level: easy | medium | hard."
    )
    expected_output: Optional[str] = Field(
        default=None,
        description="What the correct code should produce when executed."
    )
    constraints: List[str] = Field(
        default_factory=list,
        description="Any constraints the fix or refactor must satisfy."
    )
    step: int = Field(
        default=0,
        description="Current step number within the episode."
    )
    max_steps: int = Field(
        default=5,
        description="Maximum allowed steps per episode."
    )
    task_id: str = Field(
        description="Unique identifier for the current task scenario."
    )
    task_name: str = Field(
        description="Human-readable name of the active task."
    )
