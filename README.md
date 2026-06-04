# Dual-System Standalone

Standalone VLM -> subtask -> executor loop extracted from RLinf's Dual-System
prototype. No dependency on RLinf, Ray, Hydra, WorkerGroup, or simulator code.

## What It Does

One step looks like this:

1. Read an `Observation`: task text, images, optional robot state, metadata.
2. Build a memoryless or memory-enabled prompt.
3. Call a VLM adapter:
   - `openai_compatible`: commercial or self-hosted multimodal chat API.
   - `local_qwen`: local Qwen2.5-VL / Qwen3-VL via HuggingFace.
4. Parse the VLM output into a subtask and optional memory update.
5. Send the subtask to an executor (HTTP POST or custom Python client).
6. Save session state and JSONL trajectory logs.

## Architecture

### Module Structure

```
dualsystem/
├─ cli.py                         CLI entry point (parse args → assemble → call loop)
│   ├─► config.py                  YAML/JSON config loading
│   │     └─ DualSystemConfig, VLMConfig, ExecutorConfig, LoggingConfig
│   │
│   ├─► core/loop.py               ★ Core loop (VLM → parse → executor → result)
│   │     ├─► core/prompts.py        Prompt templates (memoryless / memory-enabled)
│   │     ├─► core/parsers.py        VLM output parser (JSON extraction + fallback)
│   │     └─► core/types.py          All shared data types
│   │
│   ├─► vlm/                       VLM adapters (pluggable)
│   │     ├─ base.py                 VLMClient protocol interface
│   │     ├─ openai_compatible.py    OpenAI-compatible API client
│   │     └─ local_qwen.py           Local Qwen2.5/3-VL via HuggingFace
│   │
│   ├─► executor/                   Executor adapters (pluggable)
│   │     ├─ base.py                 ExecutorClient protocol interface
│   │     └─ http_executor.py        HTTP POST executor client
│   │
│   ├─► io/
│   │     ├─ image.py                Image format normalization (path/base64/PIL/NumPy/Torch)
│   │     └─ trajectory_logger.py    JSONL trajectory logging
│   │
│   └─► state/
│       └─ session_store.py          File-backed session state persistence
│
└─ core/types.py                   ★ Global data types (shared by all modules)
    ├─ ImageInput                    Image payload (base64/url/path)
    ├─ Observation                   Input observation (task + images + state)
    ├─ SessionState                  Session state (memory + cache + subtask plan + counters)
    ├─ RunOptions                    Runtime options (memory/frequency/planning mode/params)
    ├─ PlannerInput / PlannerOutput  VLM interface types
    ├─ ExecutorInput / ExecutorOutput Executor interface types
    └─ StepResult                    Final step result (all fields aggregated)
```

### Data Flow

```
                          ┌──────────────────────────────┐
                          │     Caller (CLI or Python)    │
                          └──────┬──────────────┬────────┘
                                 │              │
                          Observation       SessionState
                       (task, images)    (memory, cached_subtask,
                                          optional subtasks)
                                 │              │
                                 ▼              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    DualSystemAgentLoop.step()                        │
│                                                                      │
│  ┌─────────────────────┐                                             │
│  │ Call VLM this step? │─── No ───► Reuse cached subtask             │
│  │ (cache miss or      │                                             │
│  │  step % freq == 0)  │                                             │
│  └────────┬────────────┘                                             │
│           │ Yes                                                      │
│           ▼                                                          │
│  ┌──────────────────┐                                                │
│  │ 1. Build prompt  │◄── core/prompts.py                            │
│  └────────┬─────────┘                                                │
│           ▼                                                          │
│  ┌──────────────────┐                                                │
│  │ 2. VLM generate  │──► vlm/openai_compatible.py                   │
│  │                  │    or vlm/local_qwen.py                        │
│  └────────┬─────────┘                                                │
│           ▼                                                          │
│  ┌──────────────────┐                                                │
│  │ 3. Parse output  │◄── core/parsers.py                            │
│  │  → subtask       │    (JSON parse → regex fallback → raw text)   │
│  │  → memory update │                                                │
│  └────────┬─────────┘                                                │
│           ▼                                                          │
│  ┌──────────────────┐                                                │
│  │ 4. Executor run  │──► executor/http_executor.py                  │
│  │                  │    POST subtask + images → actions             │
│  └────────┬─────────┘                                                │
│           ▼                                                          │
│  ┌──────────────────┐                                                │
│  │ 5. Assemble      │                                                │
│  │    StepResult    │                                                │
│  └────────┬─────────┘                                                │
└───────────┼──────────────────────────────────────────────────────────┘
            │
   ┌────────┴────────┐
   ▼                 ▼
StepResult      SessionState
  │                 │
  ▼                 ▼
TrajectoryLogger  SessionStore
→ trajectory.jsonl → sessions/*.json
```

## Install

```bash
pip install -e .
```

For local Qwen inference:

```bash
pip install -e ".[local-qwen]"
```

For development:

```bash
pip install -e ".[dev]"
```

## Configuration

Create `config.yaml` (JSON also works):

### Remote API VLM + HTTP Executor

```yaml
enable_memory: true
frequency: 1
planning_mode: direct
session_dir: ~/.dualsystem/sessions

vlm:
  provider: openai_compatible
  model: qwen-vl
  base_url: https://your-provider.example/v1
  api_key: ${OPENAI_API_KEY}
  sampling_params:
    temperature: 0.2
    max_new_tokens: 256

executor:
  provider: http
  endpoint: http://localhost:8000/execute
  timeout: 30

logging:
  enabled: true
  run_dir: ./runs
  copy_images: false
```

### Local Qwen VLM + HTTP Executor

```yaml
enable_memory: true
frequency: 1
planning_mode: direct

vlm:
  provider: local_qwen
  model_path: /path/to/Qwen2.5-VL-3B-Instruct
  model_family: auto       # "auto" infers from path; or "qwen2.5" / "qwen3"
  dtype: bf16
  device: cuda:0
  min_pixels: 200704       # 256 * 28 * 28
  max_pixels: 1003520      # 1280 * 28 * 28
  sampling_params:
    temperature: 1.0
    top_p: 0.95
    max_new_tokens: 512

executor:
  provider: http
  endpoint: http://localhost:8000/execute
  timeout: 30
```

Environment variables are expanded automatically (`${OPENAI_API_KEY}` etc.).
Fallback env vars: `DUALSYSTEM_VLM_API_KEY`, `DUALSYSTEM_VLM_BASE_URL`,
`DUALSYSTEM_EXECUTOR_URL`.

### Memory Mechanism

Memory is controlled by `enable_memory`.

When `enable_memory: false`, the VLM prompt asks for one plain next-subtask
sentence. The whole VLM text output is treated as the executor instruction, and
`ExecutorInput.memory` is `null`.

When `enable_memory: true`, the prompt includes the previous memory and asks the
VLM to return JSON:

```json
{"subtask": "<next subtask>", "memory": "<updated memory>"}
```

The memory string should be a compact summary of completed milestones, not a
full action log. For example, prefer:

```text
found the radio, approached the radio
```

over a detailed sequence of low-level motions. The loop updates
`SessionState.memory` only when the VLM is called and the parser recovers a
non-empty `memory` field. On skipped VLM turns, controlled by `frequency`, the
previous memory and cached subtask are reused.

`SessionState` is the persisted per-session state. It contains:

| Field | Meaning |
|-------|---------|
| `memory` | Compact milestone summary passed back into the memory prompt |
| `cached_subtask` | Last executor instruction, reused when the VLM is skipped |
| `cached_raw_output` | Last raw VLM output for the executor subtask decision |
| `cached_subtask_index` | Selected subtask index in `subtask_selection` mode |
| `cached_plan_raw_output` | Raw VLM output from the initial subtask-plan call |
| `subtasks` | Fixed subtask list generated by `subtask_selection` mode |
| `step_index` | Number of completed loop steps in this session |
| `trajectory_index` | Trajectory counter reserved for multi-trajectory use |

The CLI loads and saves this state through `SessionStore` under
`session_dir/{session_id}.json`. Use `--reset-session` to clear memory, cached
subtasks, and any fixed subtask plan before starting a new episode. In
`dualsystem.multi_run`, `--no-session` skips file persistence, but state is
still carried in memory during that Python process.

### Planning Routes

`planning_mode` selects how the VLM decides the executor subtask.

#### `direct`

This is the default route and preserves the original behavior:

1. Build a memoryless or memory-enabled prompt for the current image.
2. Ask the VLM for the immediate next subtask.
3. Parse the VLM output into one `subtask` and optional `memory`.
4. Send that single `subtask` to the executor.

`frequency` controls VLM cadence. With `frequency: 1`, every step calls the VLM.
With `frequency: 3`, only steps `0, 3, 6, ...` call the VLM; intermediate steps
reuse `SessionState.cached_subtask`.

#### `subtask_selection`

This route first creates a fixed task decomposition, then asks the VLM to choose
from that list at each later VLM turn.

On the first step of a session:

1. The VLM receives the current task and current image.
2. It generates an ordered plan:

   ```json
   {"subtasks": ["find the radio", "approach the radio", "press the power button"]}
   ```

3. The list is saved to `SessionState.subtasks`.
4. The loop immediately builds a second prompt containing that fixed numbered
   list and asks the VLM to choose one item for the current step.

On later steps, the VLM is not allowed to invent a new subtask. The selection
prompt includes the cached list and asks for:

```json
{"subtask_id": 2, "subtask": "approach the radio"}
```

With memory enabled, the required selection format is:

```json
{"subtask_id": 2, "subtask": "approach the radio", "memory": "found the radio"}
```

The parser normalizes the selection back to one of the cached list items. It can
recover from JSON, JSON-like fields, exact subtask text, or a plain numeric id
such as `2`. If the VLM selects something outside the fixed list, the parser
falls back to the first available subtask and marks `parse_ok: false`.

Use this route when you want a stable high-level task decomposition, while still
letting the VLM decide which planned milestone is appropriate for the current
image.

Configuration:

```yaml
enable_memory: true
frequency: 1
planning_mode: subtask_selection
```

The same settings are available as a ready-to-edit template in
`examples/config_subtask_selection.yaml`.

Run with the video-based entry point:

```bash
python -m dualsystem.multi_run \
  --config examples/config_subtask_selection.yaml \
  --task "turn on the radio" \
  --session-id task-radio-001 \
  --data-dir data/turn_on_radio_1 \
  --planning-mode subtask_selection
```

Python API:

```python
loop = DualSystemAgentLoop(
    vlm,
    executor,
    RunOptions(
        enable_memory=True,
        frequency=1,
        planning_mode="subtask_selection",
    ),
)
```

Operational notes:

- The first step in `subtask_selection` mode usually makes two VLM calls: one
  to generate the fixed `subtasks` list and one to select the current subtask.
- When the initial plan is created, `dualsystem.multi_run` prints
  `initial_subtasks=[...]`, and the same list is recorded in the step result.
- Later VLM turns only run the selection prompt unless the session is reset.
- `frequency` still controls the selection call cadence. Use `frequency: 1`
  when every sampled observation should trigger a fresh selection.
- Reuse the same `session_id` only when you want to keep the same memory and
  fixed subtask list. Use `--reset-session` or a new `session_id` to force a new
  decomposition for a new task or a new initial image.

## CLI Usage

Run one step:

```bash
python -m dualsystem.cli step \
  --config config.yaml \
  --session-id demo-001 \
  --task "turn on the radio" \
  --image main=./obs.jpg
```

Continue the same session (state is auto-loaded):

```bash
python -m dualsystem.cli step \
  --config config.yaml \
  --session-id demo-001 \
  --task "turn on the radio" \
  --image main=./obs2.jpg
```

Start a new episode (clear state):

```bash
python -m dualsystem.cli step \
  --config config.yaml \
  --session-id demo-001 \
  --reset-session \
  --task "pick up the cup" \
  --image main=./obs.jpg
```

Multiple images:

```bash
python -m dualsystem.cli step \
  --config config.yaml \
  --session-id demo-001 \
  --task "pick up the cup" \
  --image main=./main.jpg \
  --image wrist=./wrist.jpg
```

### CLI Reference

| Argument | Description |
|----------|-------------|
| `--config` | Path to YAML or JSON config file |
| `--session-id` | Session identifier for state persistence |
| `--task` | Task description text |
| `--image key=path` | Image input (repeatable): `main=photo.jpg`, `wrist=camera2.png` |
| `--state` | Robot state as JSON string |
| `--metadata` | Metadata as JSON string |
| `--reset-session` | Clear session state (new episode) |
| `--session-dir` | Override session storage directory |
| `--run-dir` | Override log output directory |
| `--no-log` | Disable JSONL trajectory logging |
| `--pretty` | Pretty-print JSON output |

## Realtime Robot Deployment

Use `dualsystem.real_robot_run` when camera observations are written as realtime
JPEG snapshots and the executor endpoint controls the VLA/robot. By default the
runner reads three images from `/tmp/img`:

- `cam_high`: `base_0_rgb.jpg`
- `cam_left_wrist`: `left_wrist_0_rgb.jpg`
- `cam_right_wrist`: `right_wrist_0_rgb.jpg`

```bash
python -m dualsystem.real_robot_run \
  --config examples/config_subtask_selection.yaml \
  --task "turn on the radio" \
  --session-id real-radio-001 \
  --image-dir /tmp/img \
  --control-hz 1 \
  --reset-session
```

After `pip install -e .`, the same entry point is available as
`dualsystem-real-robot`.

The realtime loop does this continuously:

1. Wait until all configured camera image files exist and become stable.
2. Convert the camera image files into VLM image inputs.
3. Build the executor payload with the selected instruction and camera metadata.
4. Run `DualSystemAgentLoop.step()`.
5. Send the selected subtask to the configured VLA executor.
6. Save session state and trajectory logs.

Useful options:

| Argument | Description |
|----------|-------------|
| `--image-dir DIR` | Directory containing realtime camera images; defaults to `/tmp/img`. |
| `--camera-file CAM=FILE` | Override or replace a camera filename mapping; repeatable. |
| `--control-hz N` | Observation/control loop frequency. This is separate from config `frequency`, which controls VLM decision cadence. |
| `--planning-mode subtask_selection` | Override config planning mode from the command line. |
| `--dry-run` | Run VLM planning and logging but do not call the real executor. |
| `--max-steps N` | Stop after N realtime iterations. |
| `--wait-timeout N` | Seconds to wait for camera files to exist and become stable. |
| `--stop-status STATUS` | Stop when the executor returns this status; defaults to `done`. |

When `subtask_selection` creates the first fixed plan, the script prints
`initial_subtasks=[...]`; the same list is recorded in the trajectory `result`.

## Python API

### Minimal Example

```python
from dualsystem import DualSystemAgentLoop, Observation, RunOptions, SessionState
from dualsystem.vlm import OpenAICompatibleVLMClient
from dualsystem.executor import HTTPExecutorClient

vlm = OpenAICompatibleVLMClient(
    model="qwen-vl",
    base_url="https://your-provider.example/v1",
    api_key="...",
)
executor = HTTPExecutorClient("http://localhost:8000/execute")
loop = DualSystemAgentLoop(vlm, executor, RunOptions(enable_memory=True))

result, state = loop.step(
    Observation(session_id="demo-001", task="turn on the radio"),
    SessionState(),
)
print(result.subtask)
print(result.executor_output.actions)
```

### Multi-Step Loop with Persistence

```python
from dualsystem import DualSystemAgentLoop, Observation, RunOptions
from dualsystem.vlm import OpenAICompatibleVLMClient
from dualsystem.executor import HTTPExecutorClient
from dualsystem.state import SessionStore
from dualsystem.io import image_from_path, TrajectoryLogger

vlm = OpenAICompatibleVLMClient(model="qwen-vl", base_url="...", api_key="...")
executor = HTTPExecutorClient("http://localhost:8000/execute")
logger = TrajectoryLogger(run_dir="./runs", copy_images=True)
store = SessionStore()
loop = DualSystemAgentLoop(vlm, executor, RunOptions(enable_memory=True))

session_id = "task-radio-001"
state = store.reset(session_id)

for step in range(100):
    image = image_from_path(f"/tmp/obs_step{step}.jpg")
    observation = Observation(
        session_id=session_id,
        task="turn on the radio",
        images={"main": image},
    )
    result, state = loop.step(observation, state)
    store.save(session_id, state)
    logger.log_step(result, observation, state)
    if result.executor_output.status == "done":
        break
```

### Local Qwen VLM

```python
from dualsystem.vlm.local_qwen import LocalQwenVLMClient

vlm = LocalQwenVLMClient(
    model_path="/path/to/Qwen2.5-VL-3B-Instruct",
    dtype="bf16",
    device="cuda:0",
    default_sampling_params={"temperature": 1.0, "max_new_tokens": 512},
)
# Model loads lazily on first generate() call
```

### Custom Executor

Implement the `ExecutorClient` protocol for local VLA inference or direct robot
control:

```python
from dualsystem.executor.base import ExecutorClient
from dualsystem.core.types import ExecutorInput, ExecutorOutput

class MyLocalExecutor:
    def __init__(self, vla_model):
        self.vla_model = vla_model

    def execute(self, executor_input: ExecutorInput) -> ExecutorOutput:
        actions = self.vla_model.predict(
            instruction=executor_input.subtask,
            images=executor_input.observation.images,
        )
        return ExecutorOutput(actions=actions, status="ok")

loop = DualSystemAgentLoop(vlm, MyLocalExecutor(my_vla), RunOptions())
```

### Custom VLM Client

```python
from dualsystem.vlm.base import VLMClient
from dualsystem.core.types import PlannerInput, PlannerOutput

class MyCustomVLM:
    def generate(self, planner_input: PlannerInput) -> PlannerOutput:
        text = call_my_model(planner_input.observation, planner_input.prompt)
        return PlannerOutput(raw_outputs=[text])
```

## HTTP Executor Contract

`HTTPExecutorClient` POSTs to your endpoint and expects:

Request:

```json
{
  "session_id": "demo-001",
  "task": "turn on the radio",
  "subtask": "approach the radio",
  "memory": "found the radio",
  "images": {"main": {"type": "base64", "data": "...", "mime_type": "image/jpeg"}},
  "state": {},
  "metadata": {"step_index": 0, "trajectory_index": 0}
}
```

In `subtask_selection` mode, `metadata` also includes the planning route, the
fixed subtask list, and the selected zero-based subtask index:

```json
{
  "planning_mode": "subtask_selection",
  "subtasks": ["find the radio", "approach the radio"],
  "subtask_index": 1
}
```

Response:

```json
{
  "status": "ok",
  "actions": [[0.1, 0.2]],
  "raw_response": {}
}
```

## Trajectory Logs

When logging is enabled, each step appends one JSON record to
`{run_dir}/{session_id}/trajectory.jsonl`:

```json
{
  "result": {
    "step_index": 0,
    "subtask": "find the radio",
    "memory": null,
    "vlm_raw_output": "find the radio",
    "executor_output": {"actions": [[0.1]], "status": "ok"},
    "skipped_vlm": false,
    "parse_ok": true,
    "planning_mode": "direct",
    "subtasks": [],
    "subtask_index": null,
    "created_subtask_plan": false
  },
  "observation": {
    "session_id": "demo-001",
    "task": "turn on the radio",
    "images": {"main": {"type": "base64", "data_length": 12345}}
  },
  "session_state": {
    "memory": "",
    "cached_subtask": "find the radio",
    "cached_subtask_index": null,
    "cached_plan_raw_output": null,
    "subtasks": [],
    "step_index": 1,
    "trajectory_index": 0
  }
}
```

Set `copy_images: true` to save observation images alongside the JSONL.

## Tests

```bash
python -m pytest -v
```

The test suite uses mock HTTP servers and does not require real VLM or robot
services.
