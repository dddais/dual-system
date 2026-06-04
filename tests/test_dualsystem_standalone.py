# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from PIL import Image

from dualsystem.config import DualSystemConfig, load_config
from dualsystem.core import observation_from_raw_obs, to_jsonable
from dualsystem.core.loop import DualSystemAgentLoop
from dualsystem.core.parsers import (
    parse_subtask_plan,
    parse_subtask_selection,
    parse_vlm_output,
)
from dualsystem.core.prompts import (
    build_prompt,
    build_subtask_plan_prompt,
    build_subtask_selection_prompt,
)
from dualsystem.core.types import (
    ExecutorInput,
    ExecutorOutput,
    Observation,
    PlannerInput,
    PlannerOutput,
    RunOptions,
    SessionState,
)
from dualsystem.executor.http_executor import HTTPExecutorClient
from dualsystem.io.image import image_from_path
from dualsystem.real_robot_run import (
    DEFAULT_CAMERA_FILES,
    DryRunExecutor,
    build_observation_from_image_dir,
    parse_camera_file_specs,
)
from dualsystem.state.session_store import SessionStore
from dualsystem.vlm.openai_compatible import OpenAICompatibleVLMClient


class MockVLM:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.inputs = []

    def generate(self, planner_input):
        self.inputs.append(planner_input)
        output = self.outputs.pop(0)
        return PlannerOutput(raw_outputs=[output])


class MockExecutor:
    def __init__(self):
        self.inputs = []

    def execute(self, executor_input):
        self.inputs.append(executor_input)
        return ExecutorOutput(actions=[[1, 2, 3]], status="ok")


class FakeArray:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


def test_parser_plain_text():
    result = parse_vlm_output("approach the radio", enable_memory=False)
    assert result.subtask == "approach the radio"
    assert result.memory is None
    assert result.parse_ok


def test_parser_json_memory():
    result = parse_vlm_output(
        '{"subtask": "find the cup", "memory": "looked at table"}',
        enable_memory=True,
    )
    assert result.subtask == "find the cup"
    assert result.memory == "looked at table"
    assert result.parse_ok


def test_parser_markdown_json_memory():
    result = parse_vlm_output(
        '```json\n{"subtask": "open drawer", "memory": "drawer located"}\n```',
        enable_memory=True,
    )
    assert result.subtask == "open drawer"
    assert result.memory == "drawer located"


def test_parser_bad_json_fallback():
    result = parse_vlm_output("just keep going", enable_memory=True)
    assert result.subtask == "just keep going"
    assert result.memory is None
    assert not result.parse_ok


def test_parse_subtask_plan_json():
    result = parse_subtask_plan(
        '{"subtasks": ["find the radio", "approach the radio"]}'
    )
    assert result.subtasks == ["find the radio", "approach the radio"]
    assert result.parse_ok


def test_parse_subtask_plan_numbered_fallback():
    result = parse_subtask_plan("1. find the cup\n2. grasp the cup")
    assert result.subtasks == ["find the cup", "grasp the cup"]
    assert not result.parse_ok


def test_parse_subtask_selection_uses_fixed_plan():
    result = parse_subtask_selection(
        '{"subtask_id": 2, "subtask": "approach the radio", "memory": "radio visible"}',
        ["find the radio", "approach the radio"],
        enable_memory=True,
    )
    assert result.subtask == "approach the radio"
    assert result.subtask_index == 1
    assert result.memory == "radio visible"
    assert result.parse_ok


def test_parse_subtask_selection_falls_back_to_plan_item():
    result = parse_subtask_selection(
        '{"subtask": "invent a new action"}',
        ["find the radio", "approach the radio"],
    )
    assert result.subtask == "find the radio"
    assert result.subtask_index == 0
    assert not result.parse_ok


def test_parse_subtask_selection_plain_numeric_id():
    result = parse_subtask_selection(
        "2",
        ["find the radio", "approach the radio"],
    )
    assert result.subtask == "approach the radio"
    assert result.subtask_index == 1
    assert result.parse_ok


def test_prompt_contains_task_and_memory():
    prompt = build_prompt("turn on the radio", memory="found radio", enable_memory=True)
    assert "turn on the radio" in prompt
    assert "found radio" in prompt


def test_subtask_selection_prompts_contain_fixed_plan():
    plan_prompt = build_subtask_plan_prompt("turn on the radio")
    select_prompt = build_subtask_selection_prompt(
        "turn on the radio",
        ["find the radio", "approach the radio"],
        memory="found radio",
        enable_memory=True,
    )
    assert "turn on the radio" in plan_prompt
    assert "find the radio" in select_prompt
    assert "found radio" in select_prompt
    assert "subtask_id" in select_prompt


def test_session_store_load_save_reset(tmp_path):
    store = SessionStore(tmp_path)
    assert store.load("demo") == SessionState()

    state = SessionState(memory="done", cached_subtask="move", step_index=3)
    store.save("demo", state)
    loaded = store.load("demo")
    assert loaded.memory == "done"
    assert loaded.cached_subtask == "move"
    assert loaded.step_index == 3

    reset = store.reset("demo")
    assert reset == SessionState()
    assert store.load("demo") == SessionState()


def test_loop_memory_and_executor_input():
    vlm = MockVLM(['{"subtask": "approach radio", "memory": "radio visible"}'])
    executor = MockExecutor()
    loop = DualSystemAgentLoop(
        vlm_client=vlm,
        executor_client=executor,
        options=RunOptions(enable_memory=True),
    )
    obs = Observation(session_id="s1", task="turn on radio")

    result, state = loop.step(obs, SessionState())

    assert result.subtask == "approach radio"
    assert result.memory == "radio visible"
    assert result.executor_output.actions == [[1, 2, 3]]
    assert state.memory == "radio visible"
    assert state.step_index == 1
    assert executor.inputs[0].subtask == "approach radio"
    assert executor.inputs[0].memory == "radio visible"


def test_loop_frequency_cache_skips_vlm():
    vlm = MockVLM(["first subtask", "second subtask"])
    executor = MockExecutor()
    loop = DualSystemAgentLoop(
        vlm_client=vlm,
        executor_client=executor,
        options=RunOptions(frequency=2),
    )
    obs = Observation(session_id="s1", task="do thing")

    first, state = loop.step(obs, SessionState())
    second, state = loop.step(obs, state)
    third, state = loop.step(obs, state)

    assert first.subtask == "first subtask"
    assert not first.skipped_vlm
    assert second.subtask == "first subtask"
    assert second.skipped_vlm
    assert third.subtask == "second subtask"
    assert len(vlm.inputs) == 2


def test_loop_subtask_selection_plans_once_then_selects_from_plan():
    vlm = MockVLM(
        [
            '{"subtasks": ["find the radio", "approach the radio"]}',
            '{"subtask_id": 1, "subtask": "find the radio", "memory": "looking"}',
            '{"subtask_id": 2, "subtask": "approach the radio", "memory": "radio visible"}',
        ]
    )
    executor = MockExecutor()
    loop = DualSystemAgentLoop(
        vlm_client=vlm,
        executor_client=executor,
        options=RunOptions(enable_memory=True, planning_mode="subtask_selection"),
    )
    obs = Observation(session_id="s1", task="turn on radio")

    first, state = loop.step(obs, SessionState())
    second, state = loop.step(obs, state)

    assert first.subtasks == ["find the radio", "approach the radio"]
    assert first.subtask == "find the radio"
    assert first.subtask_index == 0
    assert first.created_subtask_plan
    assert state.subtasks == ["find the radio", "approach the radio"]
    assert second.subtask == "approach the radio"
    assert second.subtask_index == 1
    assert second.memory == "radio visible"
    assert not second.created_subtask_plan
    assert len(vlm.inputs) == 3
    assert "subtasks" in vlm.inputs[0].prompt
    assert "Available subtasks" in vlm.inputs[1].prompt
    assert executor.inputs[1].metadata["subtasks"] == state.subtasks


def test_real_robot_observation_helpers():
    raw_obs = {
        "frames": {"head": Image.new("RGB", (2, 2), color="white")},
        "state": {"joint_positions": FakeArray([1.0, 2.0])},
    }

    observation = observation_from_raw_obs(
        raw_obs=raw_obs,
        session_id="real",
        task="turn on radio",
        camera_names=["head"],
        metadata={"source": "test"},
    )

    assert observation.images["head"].type == "base64"
    assert observation.state == {"joint_positions": [1.0, 2.0]}
    assert to_jsonable(FakeArray([3])) == [3]

    executor = DryRunExecutor()
    output = executor.execute(
        ExecutorInput(observation=observation, subtask="find radio")
    )
    assert output.status == "dry_run"
    assert output.raw_response["subtask"] == "find radio"


def test_real_robot_image_dir_observation(tmp_path):
    for filename in DEFAULT_CAMERA_FILES.values():
        Image.new("RGB", (2, 2), color="white").save(tmp_path / filename)

    observation = build_observation_from_image_dir(
        image_dir=tmp_path,
        session_id="real",
        task="pick carrot",
        wait_timeout=1.0,
        poll_interval=0.01,
    )

    assert set(observation.images) == {
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    }
    assert observation.images["cam_high"].path == str(
        tmp_path / "base_0_rgb.jpg"
    )
    assert observation.metadata["camera_files"] == DEFAULT_CAMERA_FILES
    assert parse_camera_file_specs(["front=front.jpg"]) == {"front": "front.jpg"}


def test_http_executor_request_and_response(http_server):
    server, received = http_server(
        {"status": "ok", "actions": [[0.1]], "raw_response": {"trace": 1}}
    )
    obs = Observation(session_id="demo", task="pick cup")
    client = HTTPExecutorClient(f"http://127.0.0.1:{server.server_port}/execute")

    output = client.execute(
        ExecutorInput(
            observation=obs,
            subtask="reach cup",
            memory="cup visible",
            metadata={"source": "test"},
        )
    )

    assert output.actions == [[0.1]]
    assert received["path"] == "/execute"
    assert received["json"]["session_id"] == "demo"
    assert received["json"]["subtask"] == "reach cup"
    assert received["json"]["memory"] == "cup visible"


def test_openai_compatible_request(http_server, tmp_path):
    response = {
        "choices": [{"message": {"content": "approach the drawer"}}],
    }
    server, received = http_server(response)
    image_path = tmp_path / "obs.jpg"
    Image.new("RGB", (2, 2), color="white").save(image_path)
    image = image_from_path(image_path)
    obs = Observation(session_id="demo", task="open drawer", images={"main": image})
    client = OpenAICompatibleVLMClient(
        model="test-model",
        base_url=f"http://127.0.0.1:{server.server_port}/v1",
        api_key="test-key",
        default_sampling_params={"max_new_tokens": 7},
    )

    output = client.generate(
        PlannerInput(
            observation=obs,
            prompt="plan please",
            sampling_params={"temperature": 0.2},
        )
    )

    assert output.raw_outputs == ["approach the drawer"]
    assert received["path"] == "/v1/chat/completions"
    assert received["headers"]["Authorization"] == "Bearer test-key"
    assert received["json"]["model"] == "test-model"
    assert received["json"]["max_tokens"] == 7
    assert received["json"]["temperature"] == 0.2
    content = received["json"]["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[1] == {"type": "text", "text": "plan please"}


def test_cli_step_with_http_services(tmp_path, capsys):
    vlm_server, _ = _start_json_server(
        {"choices": [{"message": {"content": "move to the cup"}}]}
    )
    executor_server, _ = _start_json_server({"status": "ok", "actions": [[1]]})

    image_dir = tmp_path / "img"
    image_dir.mkdir()
    for filename in DEFAULT_CAMERA_FILES.values():
        Image.new("RGB", (2, 2), color="white").save(image_dir / filename)
    custom_camera_files = {
        "front": "base_0_rgb.jpg",
        "left": "left_wrist_0_rgb.jpg",
        "right": "right_wrist_0_rgb.jpg",
    }

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "session_dir": str(tmp_path / "sessions"),
                "real_robot": {
                    "task": "pick up the carrot",
                    "session_id": "real-demo",
                    "image_dir": str(image_dir),
                    "camera_files": custom_camera_files,
                    "control_hz": 1000,
                    "max_steps": 1,
                    "wait_timeout": 1,
                    "poll_interval": 0.01,
                },
                "vlm": {
                    "provider": "openai_compatible",
                    "model": "test-model",
                    "base_url": f"http://127.0.0.1:{vlm_server.server_port}/v1",
                },
                "executor": {
                    "provider": "http",
                    "endpoint": f"http://127.0.0.1:{executor_server.server_port}/execute",
                },
                "logging": {"enabled": True, "run_dir": str(tmp_path / "runs")},
            }
        )
    )
    image_path = tmp_path / "obs.jpg"
    Image.new("RGB", (2, 2), color="white").save(image_path)

    from dualsystem.cli import main

    exit_code = main(
        [
            "step",
            "--config",
            str(config_path),
            "--session-id",
            "demo",
            "--task",
            "pick up the cup",
            "--image",
            f"main={image_path}",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["subtask"] == "move to the cup"
    assert output["executor_output"]["actions"] == [[1]]
    assert (tmp_path / "sessions" / "demo.json").exists()
    assert (tmp_path / "runs" / "demo" / "trajectory.jsonl").exists()


def test_real_robot_run_from_image_dir(tmp_path, capsys):
    vlm_server, _ = _start_json_server(
        {"choices": [{"message": {"content": "move to the carrot"}}]}
    )
    executor_server, received = _start_json_server(
        {"status": "ok", "actions": [[1]]}
    )

    image_dir = tmp_path / "img"
    image_dir.mkdir()
    for filename in DEFAULT_CAMERA_FILES.values():
        Image.new("RGB", (2, 2), color="white").save(image_dir / filename)
    custom_camera_files = {
        "front": "base_0_rgb.jpg",
        "left": "left_wrist_0_rgb.jpg",
        "right": "right_wrist_0_rgb.jpg",
    }

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "session_dir": str(tmp_path / "sessions"),
                "real_robot": {
                    "task": "pick up the carrot",
                    "session_id": "real-demo",
                    "image_dir": str(image_dir),
                    "camera_files": custom_camera_files,
                    "control_hz": 1000,
                    "max_steps": 1,
                    "wait_timeout": 1,
                    "poll_interval": 0.01,
                },
                "vlm": {
                    "provider": "openai_compatible",
                    "model": "test-model",
                    "base_url": f"http://127.0.0.1:{vlm_server.server_port}/v1",
                },
                "executor": {
                    "provider": "http",
                    "endpoint": (
                        f"http://127.0.0.1:"
                        f"{executor_server.server_port}/execute"
                    ),
                },
                "logging": {"enabled": True, "run_dir": str(tmp_path / "runs")},
            }
        )
    )

    from dualsystem.real_robot_run import main

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--reset-session",
        ]
    )

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "instruction='move to the carrot'" in captured
    assert received["json"]["subtask"] == "move to the carrot"
    assert set(received["json"]["images"]) == set(custom_camera_files)
    assert received["json"]["metadata"]["image_dir"] == str(image_dir)
    assert (tmp_path / "sessions" / "real-demo.json").exists()
    assert (tmp_path / "runs" / "real-demo" / "trajectory.jsonl").exists()


def test_config_load_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
enable_memory: true
frequency: 3
planning_mode: subtask_selection
real_robot:
  image_dir: /tmp/img
  camera_files:
    cam_high: base_0_rgb.jpg
  control_hz: 2
  max_steps: 5
vlm:
  provider: openai_compatible
  model: qwen-vl
executor:
  provider: http
  endpoint: http://localhost:8000/execute
"""
    )
    config = load_config(config_path)
    assert isinstance(config, DualSystemConfig)
    assert config.enable_memory
    assert config.frequency == 3
    assert config.planning_mode == "subtask_selection"
    assert config.real_robot.image_dir == "/tmp/img"
    assert config.real_robot.camera_files == {"cam_high": "base_0_rgb.jpg"}
    assert config.real_robot.control_hz == 2.0
    assert config.real_robot.max_steps == 5
    assert config.vlm.model == "qwen-vl"


def test_config_expands_environment_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("DUALSYSTEM_TEST_KEY", "secret")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
vlm:
  provider: openai_compatible
  model: qwen-vl
  api_key: ${DUALSYSTEM_TEST_KEY}
"""
    )
    config = load_config(config_path)
    assert config.vlm.api_key == "secret"


def _start_json_server(response_json):
    received = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            received["path"] = self.path
            received["headers"] = dict(self.headers)
            received["json"] = json.loads(body.decode("utf-8"))
            payload = json.dumps(response_json).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, received


@pytest.fixture
def http_server():
    servers = []

    def factory(response_json):
        server, received = _start_json_server(response_json)
        servers.append(server)
        return server, received

    yield factory

    for server in servers:
        server.shutdown()
