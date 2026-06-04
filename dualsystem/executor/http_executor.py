"""HTTP JSON executor adapter."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from dualsystem.core.types import ExecutorInput, ExecutorOutput


class HTTPExecutorClient:
    """POST executor inputs to an external VLA or robot service."""

    def __init__(self, endpoint: str, timeout: float = 30.0) -> None:
        self.endpoint = endpoint
        self.timeout = timeout

    def execute(self, executor_input: ExecutorInput) -> ExecutorOutput:
        payload = executor_input_to_payload(executor_input)
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Executor request failed: HTTP {exc.code}: {body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Executor request failed: {exc}") from exc

        return ExecutorOutput.from_dict(response_data)


def executor_input_to_payload(executor_input: ExecutorInput) -> dict[str, Any]:
    """Convert an executor input to the stable HTTP JSON request body."""
    observation = executor_input.observation
    return {
        "session_id": observation.session_id,
        "task": observation.task,
        "subtask": executor_input.subtask,
        "memory": executor_input.memory,
        "images": {
            key: image.to_dict() for key, image in observation.images.items()
        },
        "state": observation.state or {},
        "metadata": executor_input.metadata or {},
    }
