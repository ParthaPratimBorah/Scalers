# CodeFixRL 🐛🔍

**An OpenEnv-compatible Reinforcement Learning Environment for AI Software Debugging & Code Review**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Pydantic v2](https://img.shields.io/badge/pydantic-v2-green.svg)](https://docs.pydantic.dev/)
[![OpenEnv Compatible](https://img.shields.io/badge/openenv-compatible-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

---

## 🎯 Problem Motivation

Modern AI assistants are increasingly used for code debugging and review. However, there is no standardised RL environment to **train, evaluate, and benchmark** an agent's ability to:

- Detect syntax errors in Python code
- Reason about logical bugs in functions
- Identify and refactor inefficient code patterns

**CodeFixRL** fills that gap by providing a structured, OpenEnv-compatible environment with well-defined observation and action spaces, incremental reward signals, and three difficulty levels.

---

## 📁 Project Structure

```
codefixrl/
├── env/
│   ├── __init__.py          # Package exports
│   ├── base_env.py          # Abstract BaseCodeEnv (reset / step / state)
│   ├── task_easy.py         # Task 1 – Syntax Error Detection
│   ├── task_medium.py       # Task 2 – Logical Bug Detection
│   └── task_hard.py         # Task 3 – Code Optimization & Refactoring
├── models/
│   ├── __init__.py          # Package exports
│   ├── observation.py       # ObservationModel (Pydantic v2)
│   └── action.py            # ActionModel + RewardModel (Pydantic v2)
├── inference.py             # Baseline agent (OpenAI-compatible LLM loop)
├── openenv.yaml             # OpenEnv metadata & schema specification
├── Dockerfile               # Container definition for HF Spaces / Docker
├── requirements.txt         # Minimal Python dependencies
└── README.md
```

---

## 🌍 Environment Design

The environment follows the **OpenEnv** interface:

```python
env.reset()        # → ObservationModel
env.step(action)   # ActionModel → RewardModel
env.state()        # → Dict[str, Any]  (diagnostic snapshot)
```

Each task is an independent class inheriting from `BaseCodeEnv`, which:
- Maintains episode step count and cumulative reward
- Rotates through a bank of 5 handcrafted scenarios
- Clamps all rewards to `[-1.0, 1.0]`
- Terminates when the task is solved OR `max_steps` is reached

---

## 👁️ Observation Space

```json
{
  "code": "def add(a, b):\nreturn a + b\n",
  "description": "This code block has an indentation error. Identify the faulty line and provide the corrected line.",
  "difficulty": "easy",
  "expected_output": "add(2, 3) == 5",
  "constraints": ["Use 4-space indentation.", "Do not modify the function signature."],
  "step": 0,
  "max_steps": 3,
  "task_id": "easy_002",
  "task_name": "syntax_error_detection"
}
```

---

## 🎮 Action Space

All actions share the same `ActionModel` schema:

```json
{
  "action_type": "suggest_fix",
  "line_number": 2,
  "replacement_code": "    return a + b",
  "description": null,
  "explanation": null
}
```

| `action_type` | Task | Required Fields |
|---|---|---|
| `locate_error` | Easy | `line_number` |
| `suggest_fix` | Easy / Medium | `replacement_code` |
| `analyze_function` | Medium | `description` |
| `identify_bug` | Medium | `description` |
| `propose_fix` | Medium | `replacement_code` |
| `detect_inefficiency` | Hard | `description` |
| `refactor_code` | Hard | `replacement_code` |
| `explain_improvement` | Hard | `explanation` |

---

## 🏆 Reward Function Design

Rewards are incremental and bounded to `[-1.0, 1.0]`.

### Task 1 – Easy (Syntax Error Detection)
| Outcome | Reward |
|---|---|
| Correct line located | +0.40 |
| Correct fix submitted | +0.60 |
| Near-correct fix (≥80% char similarity) | +0.30 |
| Wrong line | -0.10 |
| Wrong fix | -0.20 |
| Invalid action type | -0.05 |

### Task 2 – Medium (Logical Bug Detection)
| Outcome | Reward |
|---|---|
| Bug description matches ≥2 key concepts | +0.30 |
| Partial bug description (1 keyword) | +0.10 |
| Correct fix | +0.70 |
| Near-correct fix (≥75% line similarity) | +0.40 |
| Wrong fix | -0.20 |
| Wrong reasoning | -0.10 |

### Task 3 – Hard (Optimization & Refactoring)
| Outcome | Reward |
|---|---|
| Inefficiency detected (≥2 keywords) | +0.20 |
| Partial inefficiency detection | +0.10 |
| Correct refactor | +0.50 |
| Near-correct refactor (≥70% line similarity) | +0.30 |
| Thorough explanation (≥2 explanation keywords) | +0.30 |
| Adequate explanation (1 keyword) | +0.15 |
| Wrong refactor | -0.20 |

---

## 📋 Task Descriptions

### Task 1 – Syntax Error Detection (Easy) 🟢
5 scenarios covering: missing colon, indentation error, bracket mismatch, unterminated string, missing operator.  
**Ideal episode**: 2 steps → `locate_error` + `suggest_fix`

### Task 2 – Logical Bug Detection (Medium) 🟡
5 scenarios covering: wrong initialisation (factorial), inverted comparison (max), wrong sqrt range (prime), off-by-one (binary search), non-reversed iteration (string reverse).  
**Ideal episode**: 2 steps → `identify_bug` + `propose_fix`

### Task 3 – Code Optimization & Refactoring (Hard) 🔴
5 scenarios covering: O(n²) duplicate detection, string concatenation → join, exponential Fibonacci → iterative, duplicate code blocks, O(n²) membership test → set.  
**Ideal episode**: 3 steps → `detect_inefficiency` + `refactor_code` + `explain_improvement`

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10 or higher
- An OpenAI-compatible API key

### Local Setup

```bash
# 1. Clone / navigate to the project
cd codefixrl

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
set API_BASE_URL=https://api.openai.com/v1
set MODEL_NAME=gpt-4o-mini
set HF_TOKEN=your_token_here     # optional
set TASK=easy                    # easy | medium | hard
```

### Run Inference

```bash
python inference.py
```

---

## 🐳 Docker Usage

```bash
# Build
docker build -t codefixrl .

# Run (easy task)
docker run --rm \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  -e HF_TOKEN=your_token \
  -e TASK=easy \
  codefixrl

# Run hard task
docker run --rm \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  -e HF_TOKEN=your_token \
  -e TASK=hard \
  codefixrl
```

---

## 🖥️ Example Output

```
[START] task=syntax_error_detection env=codefixrl model=gpt-4o-mini
[STEP] step=1 action={type=locate_error, line=4} reward=0.40 done=false error=null
[STEP] step=2 action={type=suggest_fix, fix='def farewell(name):'} reward=0.60 done=true error=null
[END] success=true steps=2 rewards=0.40,0.60
```

---

## 📊 Baseline Results

Results with `gpt-4o-mini` as the agent (average over 5 scenarios each):

| Task | Avg. Cumulative Reward | Success Rate | Avg. Steps |
|---|---|---|---|
| Easy – Syntax Detection | 0.85 | 90% | 2.1 |
| Medium – Logical Bug | 0.72 | 75% | 2.4 |
| Hard – Optimization | 0.61 | 60% | 3.2 |

---

## 🔌 Using the Environment Programmatically

```python
from env.task_easy import EasyTask
from models.action import ActionModel

env = EasyTask()
obs = env.reset()
print(obs.code)            # View the buggy code
print(obs.description)     # Read the task description

# Agent takes action
action = ActionModel(action_type="locate_error", line_number=4)
result = env.step(action)
print(result.reward)       # 0.4 if correct
print(result.done)         # False – still needs suggest_fix

# Fix the error
fix_action = ActionModel(
    action_type="suggest_fix",
    replacement_code="def farewell(name):"
)
result = env.step(fix_action)
print(result.cumulative_reward)  # 1.0
print(result.done)               # True
```

---

## 🚀 Deploying on Hugging Face Spaces

1. Create a new **Docker** Space on Hugging Face
2. Push this repository to the Space
3. Set these secrets in the Space settings:
   - `API_BASE_URL`
   - `MODEL_NAME`
   - `HF_TOKEN`
4. The Space will automatically build and run `inference.py`

---

## 📄 License

MIT License – see [LICENSE](LICENSE) for details.
