"""
inference.py – CodeFixRL Baseline Agent
=========================================
Drives a full RL episode using an OpenAI-compatible LLM as the agent.

Environment variables required:
  API_BASE_URL  – Base URL of the OpenAI-compatible API endpoint
  MODEL_NAME    – Model identifier (e.g. gpt-4o-mini)

Optional:
  HF_TOKEN      – Hugging Face token (forwarded as bearer auth if set)
  TASK          – One of: easy | medium | hard   (default: easy)
  MAX_STEPS     – Override max steps per episode  (default: task default)

Output format (exact):
  [START] task=<task_name> env=codefixrl model=<model_name>
  [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END] success=<true|false> steps=<n> rewards=<r1,r2,...>
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Local imports – must run from the codefixrl/ root directory
# ---------------------------------------------------------------------------
from env.task_easy import EasyTask
from env.task_medium import MediumTask
from env.task_hard import HardTask
from models.action import ActionModel, ActionType
from models.observation import ObservationModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN: Optional[str] = os.environ.get("HF_TOKEN")  # required for HF Spaces; no default
TASK_NAME: str = os.environ.get("TASK", "easy").lower()

_TASK_MAP = {
    "easy": EasyTask,
    "medium": MediumTask,
    "hard": HardTask,
}

# ---------------------------------------------------------------------------
# System prompt – instructs the LLM to emit structured JSON actions
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """
You are a code-debugging AI agent operating inside the CodeFixRL environment.

At each step you receive an observation (JSON) containing a Python code snippet,
a task description, and the current step count.

You MUST respond with a single JSON object matching EXACTLY this schema:
{
  "action_type": "<one of: locate_error | suggest_fix | analyze_function | identify_bug | propose_fix | detect_inefficiency | refactor_code | explain_improvement>",
  "line_number": <integer | null>,
  "replacement_code": "<string | null>",
  "description": "<string | null>",
  "explanation": "<string | null>"
}

Rules:
- Do NOT include any text outside the JSON object.
- Pick action_type appropriate for the task difficulty in the observation.
- For easy tasks:  start with locate_error, then suggest_fix.
- For medium tasks: start with identify_bug, then propose_fix.
- For hard tasks:  start with detect_inefficiency, then refactor_code, then explain_improvement.
- Set unused fields to null.
""".strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_client() -> OpenAI:
    """
    Create an OpenAI-compatible client.
    HF_TOKEN is used as the API key when set (required for HuggingFace endpoints).
    Falls back to the OPENAI_API_KEY environment variable if HF_TOKEN is not set.
    """
    api_key = HF_TOKEN or os.environ.get("OPENAI_API_KEY", "no-key-set")
    return OpenAI(base_url=API_BASE_URL, api_key=api_key)


def _obs_to_prompt(obs: ObservationModel) -> str:
    """Serialize the observation as JSON for the user message."""
    return json.dumps(obs.model_dump(), indent=2)


def _call_llm(client: OpenAI, obs: ObservationModel) -> ActionModel:
    """
    Call the LLM and parse its response into an ActionModel.
    Raises ValueError if the response is not valid JSON or has bad fields.
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _obs_to_prompt(obs)},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    raw: str = response.choices[0].message.content or ""
    raw = raw.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

    data = json.loads(raw)
    return ActionModel(**data)


def _fmt_reward(r: float) -> str:
    return f"{r:.2f}"


def _fmt_bool(b: bool) -> str:
    return "true" if b else "false"


def _fmt_action(action: ActionModel) -> str:
    """Compact one-line representation of the action for the STEP line."""
    parts = [f"type={action.action_type}"]
    if action.line_number is not None:
        parts.append(f"line={action.line_number}")
    if action.replacement_code:
        snippet = action.replacement_code.replace("\n", "\\n")[:60]
        parts.append(f"fix={snippet!r}")
    if action.description:
        parts.append(f"desc={action.description[:60]!r}")
    if action.explanation:
        parts.append(f"expl={action.explanation[:60]!r}")
    return "{" + ", ".join(parts) + "}"


# ---------------------------------------------------------------------------
# Main episode loop
# ---------------------------------------------------------------------------

def run_episode() -> None:
    task_cls = _TASK_MAP.get(TASK_NAME)
    if task_cls is None:
        print(
            f"[START] task=unknown env=codefixrl model={MODEL_NAME}",
            flush=True,
        )
        print(
            f"[END] success=false steps=0 rewards=",
            flush=True,
        )
        sys.exit(1)

    env = task_cls()
    client = _build_client()

    # Print START line
    print(
        f"[START] task={env.TASK_NAME} env=codefixrl model={MODEL_NAME}",
        flush=True,
    )

    rewards: List[float] = []
    step_n = 0
    success = False
    obs: Optional[ObservationModel] = None

    try:
        obs = env.reset()

        while True:
            step_n += 1
            action_str = "null"
            error_str = "null"
            reward_val = 0.0
            done_val = False

            try:
                action = _call_llm(client, obs)
                action_str = _fmt_action(action)

                # OpenEnv spec: observation, reward, done, info = env.step(action)
                obs, reward_val, done_val, info = env.step(action)
                rewards.append(reward_val)
                cumulative = info.get("cumulative_reward", sum(rewards))

            except Exception as exc:
                error_str = str(exc).replace("\n", " ")[:120]
                rewards.append(0.0)
                done_val = True  # Abort episode on error
                cumulative = sum(rewards)

            print(
                f"[STEP] step={step_n} action={action_str} "
                f"reward={_fmt_reward(reward_val)} "
                f"done={_fmt_bool(done_val)} "
                f"error={error_str}",
                flush=True,
            )

            if done_val:
                success = cumulative > 0
                break

            # Safety valve – honour max_steps
            if step_n >= env.MAX_STEPS:
                break

    except Exception as fatal:
        error_detail = str(fatal).replace("\n", " ")[:120]
        print(
            f"[STEP] step={step_n} action=null reward=0.00 done=true error={error_detail}",
            flush=True,
        )

    rewards_str = ",".join(_fmt_reward(r) for r in rewards) if rewards else ""
    print(
        f"[END] success={_fmt_bool(success)} steps={step_n} rewards={rewards_str}",
        flush=True,
    )


if __name__ == "__main__":
    run_episode()
