# Dual-System Standalone（独立双系统）

从 RLinf 的 Dual-System 原型中提取的独立 VLM → 子任务 → 执行器循环。
不依赖 RLinf、Ray、Hydra、WorkerGroup 或任何仿真器代码。

## 它做什么

每一步的流程：

1. 读取 `Observation`：任务文本、图片、可选的机器人状态和元数据。
2. 构建无记忆或带记忆的 Prompt。
3. 调用 VLM 适配器：
   - `openai_compatible`：兼容 OpenAI 格式的商业或自托管多模态 API。
   - `local_qwen`：通过 HuggingFace 本地加载 Qwen2.5-VL / Qwen3-VL。
4. 解析 VLM 输出，提取子任务和可选的记忆更新。
5. 将子任务发送给执行器（HTTP POST 或自定义 Python 客户端）。
6. 保存会话状态和 JSONL 轨迹日志。

## 架构

### 模块结构

```
dualsystem/
├─ cli.py                         命令行入口（解析参数 → 组装 → 调用 loop）
│   ├─► config.py                  YAML/JSON 配置加载
│   │     └─ DualSystemConfig, VLMConfig, ExecutorConfig, LoggingConfig
│   │
│   ├─► core/loop.py               ★ 核心循环（VLM → 解析 → 执行器 → 结果）
│   │     ├─► core/prompts.py        Prompt 模板（无记忆 / 带记忆）
│   │     ├─► core/parsers.py        VLM 输出解析（JSON 提取 + 兜底）
│   │     └─► core/types.py          所有共享数据类型
│   │
│   ├─► vlm/                       VLM 适配器（可插拔）
│   │     ├─ base.py                 VLMClient 协议接口
│   │     ├─ openai_compatible.py    OpenAI 兼容 API 客户端
│   │     └─ local_qwen.py           本地 Qwen2.5/3-VL（HuggingFace）
│   │
│   ├─► executor/                   执行器适配器（可插拔）
│   │     ├─ base.py                 ExecutorClient 协议接口
│   │     └─ http_executor.py        HTTP POST 执行器客户端
│   │
│   ├─► io/
│   │     ├─ image.py                图片格式统一（路径/base64/PIL/NumPy/Torch）
│   │     └─ trajectory_logger.py    JSONL 轨迹日志
│   │
│   └─► state/
│       └─ session_store.py          文件持久化的会话状态存储
│
└─ core/types.py                   ★ 全局数据类型（所有模块共享）
    ├─ ImageInput                    图片载荷（base64/url/path）
    ├─ Observation                   输入观测（task + images + state）
    ├─ SessionState                  会话状态（memory + cache + subtask plan + counters）
    ├─ RunOptions                    运行选项（memory/frequency/planning mode/params）
    ├─ PlannerInput / PlannerOutput  VLM 接口类型
    ├─ ExecutorInput / ExecutorOutput 执行器接口类型
    └─ StepResult                    最终步骤结果（所有字段汇总）
```

### 数据流

```
                          ┌──────────────────────────────┐
                          │       调用方（CLI 或 Python）   │
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
│  │ 本步需要调 VLM 吗？  │─── 否 ───► 复用缓存的 subtask               │
│  │ (缓存为空 或         │                                             │
│  │  step % freq == 0)  │                                             │
│  └────────┬────────────┘                                             │
│           │ 是                                                       │
│           ▼                                                          │
│  ┌──────────────────┐                                                │
│  │ 1. 构建 Prompt   │◄── core/prompts.py                            │
│  └────────┬─────────┘                                                │
│           ▼                                                          │
│  ┌──────────────────┐                                                │
│  │ 2. VLM 生成      │──► vlm/openai_compatible.py                   │
│  │                  │    或 vlm/local_qwen.py                        │
│  └────────┬─────────┘                                                │
│           ▼                                                          │
│  ┌──────────────────┐                                                │
│  │ 3. 解析输出      │◄── core/parsers.py                            │
│  │  → subtask       │    (JSON 解析 → 正则兜底 → 原始文本)            │
│  │  → memory 更新   │                                                │
│  └────────┬─────────┘                                                │
│           ▼                                                          │
│  ┌──────────────────┐                                                │
│  │ 4. 执行器执行    │──► executor/http_executor.py                  │
│  │                  │    POST subtask + images → actions             │
│  └────────┬─────────┘                                                │
│           ▼                                                          │
│  ┌──────────────────┐                                                │
│  │ 5. 组装结果      │                                                │
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

## 安装

```bash
pip install -e .
```

本地 Qwen 推理需要额外依赖：

```bash
pip install -e ".[local-qwen]"
```

开发模式：

```bash
pip install -e ".[dev]"
```

## 配置

创建 `config.yaml`（也支持 JSON 格式）：

### 远程 API VLM + HTTP 执行器

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

### 本地 Qwen VLM + HTTP 执行器

```yaml
enable_memory: true
frequency: 1
planning_mode: direct

vlm:
  provider: local_qwen
  model_path: /path/to/Qwen2.5-VL-3B-Instruct
  model_family: auto       # "auto" 从路径推断；也可写 "qwen2.5" 或 "qwen3"
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

环境变量会自动展开（如 `${OPENAI_API_KEY}`）。
备用环境变量：`DUALSYSTEM_VLM_API_KEY`、`DUALSYSTEM_VLM_BASE_URL`、`DUALSYSTEM_EXECUTOR_URL`。

### 记忆机制

记忆由 `enable_memory` 控制。

当 `enable_memory: false` 时，VLM Prompt 只要求输出一句“下一步子任务”。
VLM 返回的整段文本会被当作执行器指令，`ExecutorInput.memory` 为 `null`。

当 `enable_memory: true` 时，Prompt 会带上上一轮记忆，并要求 VLM 返回 JSON：

```json
{"subtask": "<next subtask>", "memory": "<updated memory>"}
```

`memory` 应该是已完成里程碑的紧凑摘要，而不是完整动作日志。例如优先写：

```text
found the radio, approached the radio
```

而不是记录每一个底层运动。Loop 只有在本步调用了 VLM，并且 parser 解析到非空
`memory` 字段时，才会更新 `SessionState.memory`。如果本步因为 `frequency`
跳过 VLM，则会复用上一轮 memory 和缓存的 subtask。

`SessionState` 是每个 session 持久化保存的状态，包含：

| 字段 | 含义 |
|------|------|
| `memory` | 传回 memory prompt 的紧凑里程碑摘要 |
| `cached_subtask` | 最近一次执行器指令；跳过 VLM 时复用 |
| `cached_raw_output` | 最近一次用于决定 executor subtask 的原始 VLM 输出 |
| `cached_subtask_index` | `subtask_selection` 模式下选中的子任务索引 |
| `cached_plan_raw_output` | 初始子任务列表规划调用的原始 VLM 输出 |
| `subtasks` | `subtask_selection` 模式生成并缓存的固定子任务列表 |
| `step_index` | 当前 session 已完成的 loop 步数 |
| `trajectory_index` | 预留的轨迹编号，用于多轨迹场景 |

CLI 会通过 `SessionStore` 将状态保存到
`session_dir/{session_id}.json`。使用 `--reset-session` 会清空 memory、缓存
subtask 以及固定子任务列表。在 `dualsystem.multi_run` 中，`--no-session` 会跳过
文件持久化，但在同一个 Python 进程内仍然会把 state 传给下一步。

### 规划路线

`planning_mode` 用来选择 VLM 如何决定执行器的 subtask。

#### `direct`

这是默认路线，也保留原始行为：

1. 基于当前图像构建无记忆或带记忆 Prompt。
2. 请求 VLM 直接输出当前 immediate next subtask。
3. 将 VLM 输出解析成一个 `subtask` 和可选的 `memory`。
4. 把这个单一 `subtask` 发给执行器。

`frequency` 控制调用 VLM 的频率。`frequency: 1` 表示每步都调用 VLM；
`frequency: 3` 表示只在 `0, 3, 6, ...` 等 step 调用 VLM，中间步骤复用
`SessionState.cached_subtask`。

#### `subtask_selection`

这条路线会先生成固定任务分解，然后每次让 VLM 从固定列表中选择当前应该执行的
子任务。

在 session 的第一步：

1. VLM 接收当前任务和当前图像。
2. VLM 生成有序计划：

   ```json
   {"subtasks": ["find the radio", "approach the radio", "press the power button"]}
   ```

3. 列表保存到 `SessionState.subtasks`。
4. Loop 立刻构建第二个 Prompt，把固定编号列表放进去，让 VLM 为当前 step 选择
   一个子任务。

后续步骤中，VLM 不允许创造新子任务。选择 Prompt 会包含缓存的固定列表，并要求
返回：

```json
{"subtask_id": 2, "subtask": "approach the radio"}
```

如果开启 memory，则选择阶段要求返回：

```json
{"subtask_id": 2, "subtask": "approach the radio", "memory": "found the radio"}
```

Parser 会把选择结果规整回缓存列表里的某一项。它可以从 JSON、类 JSON 字段、
完全一致的 subtask 文本，或纯数字 id（例如 `2`）中恢复选择。如果 VLM 选了
固定列表之外的内容，parser 会回退到第一个可用子任务，并将 `parse_ok` 标记为
`false`。

当你希望任务拥有稳定的高层分解，但又希望 VLM 根据当前图像动态判断执行哪一个
里程碑时，可以使用这条路线。

配置：

```yaml
enable_memory: true
frequency: 1
planning_mode: subtask_selection
```

同样的配置也已经放在 `examples/config_subtask_selection.yaml` 中，可直接复制修改。

使用视频入口运行：

```bash
python -m dualsystem.multi_run \
  --config examples/config_subtask_selection.yaml \
  --task "turn on the radio" \
  --session-id task-radio-001 \
  --data-dir data/turn_on_radio_1 \
  --planning-mode subtask_selection
```

Python API：

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

运行注意事项：

- `subtask_selection` 模式第一步通常会调用两次 VLM：一次生成固定 `subtasks`
  列表，一次选择当前 subtask。
- 当初始计划生成成功时，`dualsystem.multi_run` 会打印
  `initial_subtasks=[...]`，同一列表也会记录在 step result 中。
- 后续 VLM turn 只运行选择 Prompt，除非重置 session。
- `frequency` 仍然控制选择阶段的调用频率。如果希望每个采样 observation 都重新
  选择一次，请使用 `frequency: 1`。
- 只有当你希望保留同一份 memory 和固定子任务列表时，才复用同一个
  `session_id`。新任务或新初始图像建议使用 `--reset-session` 或新的
  `session_id`，以强制重新分解任务。

## 命令行使用

执行一步：

```bash
python -m dualsystem.cli step \
  --config config.yaml \
  --session-id demo-001 \
  --task "turn on the radio" \
  --image main=./obs.jpg
```

继续同一会话（状态自动加载）：

```bash
python -m dualsystem.cli step \
  --config config.yaml \
  --session-id demo-001 \
  --task "turn on the radio" \
  --image main=./obs2.jpg
```

开始新 episode（清除状态）：

```bash
python -m dualsystem.cli step \
  --config config.yaml \
  --session-id demo-001 \
  --reset-session \
  --task "pick up the cup" \
  --image main=./obs.jpg
```

多张图片：

```bash
python -m dualsystem.cli step \
  --config config.yaml \
  --session-id demo-001 \
  --task "pick up the cup" \
  --image main=./main.jpg \
  --image wrist=./wrist.jpg
```

### CLI 参数一览

| 参数 | 说明 |
|------|------|
| `--config` | 配置文件路径（YAML 或 JSON） |
| `--session-id` | 会话标识，用于状态持久化 |
| `--task` | 任务描述文本 |
| `--image key=path` | 图片输入（可多次使用）：`main=photo.jpg` |
| `--state` | 机器人状态，JSON 字符串 |
| `--metadata` | 元数据，JSON 字符串 |
| `--reset-session` | 清除会话状态（新 episode） |
| `--session-dir` | 覆盖状态存储目录 |
| `--run-dir` | 覆盖日志输出目录 |
| `--no-log` | 禁用 JSONL 轨迹日志 |
| `--pretty` | 格式化 JSON 输出 |

## 实时机器人部署

当相机观测以实时 JPEG 快照写入目录，并且 executor endpoint 负责控制
VLA/机器人时，可以使用 `dualsystem.real_robot_run`。默认从 `/tmp/img`
读取三张图：

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

执行 `pip install -e .` 后，也可以直接使用命令 `dualsystem-real-robot`。

实时循环会持续执行：

1. 等待所有配置的相机图片存在且文件大小稳定。
2. 将相机图片文件转成 VLM 图片输入。
3. 构造包含选中 instruction 和相机元数据的 executor payload。
4. 执行 `DualSystemAgentLoop.step()`。
5. 将选出的 subtask 发送给配置里的 VLA executor。
6. 保存 session state 和轨迹日志。

常用参数：

| 参数 | 说明 |
|------|------|
| `--image-dir DIR` | 实时相机图片目录；默认是 `/tmp/img`。 |
| `--camera-file CAM=FILE` | 覆盖或替换相机文件名映射；可重复。 |
| `--control-hz N` | 观测/控制循环频率。它和配置里的 `frequency` 不同，后者控制 VLM 决策频率。 |
| `--planning-mode subtask_selection` | 从命令行覆盖配置中的规划路线。 |
| `--dry-run` | 只运行 VLM 规划和日志，不调用真实 executor。 |
| `--max-steps N` | 运行 N 次实时迭代后停止。 |
| `--wait-timeout N` | 等待相机文件存在并稳定的秒数。 |
| `--stop-status STATUS` | executor 返回该状态时停止；默认是 `done`。 |

当 `subtask_selection` 第一次创建固定计划时，脚本会打印
`initial_subtasks=[...]`；同一列表也会记录到 trajectory 的 `result` 中。

## Python API

### 最简示例

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

### 多步循环 + 状态持久化

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

### 本地 Qwen VLM

```python
from dualsystem.vlm.local_qwen import LocalQwenVLMClient

vlm = LocalQwenVLMClient(
    model_path="/path/to/Qwen2.5-VL-3B-Instruct",
    dtype="bf16",
    device="cuda:0",
    default_sampling_params={"temperature": 1.0, "max_new_tokens": 512},
)
# 模型在第一次 generate() 时才加载（懒加载）
```

### 自定义执行器

实现 `ExecutorClient` 协议即可接入本地 VLA 或直接控制机器人：

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

### 自定义 VLM 客户端

```python
from dualsystem.vlm.base import VLMClient
from dualsystem.core.types import PlannerInput, PlannerOutput

class MyCustomVLM:
    def generate(self, planner_input: PlannerInput) -> PlannerOutput:
        text = call_my_model(planner_input.observation, planner_input.prompt)
        return PlannerOutput(raw_outputs=[text])
```

## HTTP 执行器接口

使用 `HTTPExecutorClient` 时，它会向指定 URL POST 如下 JSON：

请求格式：

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

在 `subtask_selection` 模式下，`metadata` 还会包含规划路线、固定子任务列表和
选中的零基子任务索引：

```json
{
  "planning_mode": "subtask_selection",
  "subtasks": ["find the radio", "approach the radio"],
  "subtask_index": 1
}
```

期望的响应格式：

```json
{
  "status": "ok",
  "actions": [[0.1, 0.2]],
  "raw_response": {}
}
```

## 轨迹日志

开启日志后（`logging.enabled: true`），每步会追加一条记录到
`{run_dir}/{session_id}/trajectory.jsonl`：

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

设置 `copy_images: true` 可将观测图片保存到 `{run_dir}/{session_id}/images/`。

## 测试

```bash
python -m pytest -v
```

测试套件使用 mock HTTP 服务器，不需要真实的 VLM 或机器人服务。
