"""File-backed session state store."""

from __future__ import annotations

import json
import re
from pathlib import Path

from dualsystem.core.types import SessionState


class SessionStore:
    """Persist session state as JSON files."""

    def __init__(self, base_dir: str | Path = "~/.dualsystem/sessions") -> None:
        self.base_dir = Path(base_dir).expanduser()

    def load(self, session_id: str) -> SessionState:
        path = self.path_for(session_id)
        if not path.exists():
            return SessionState()
        with open(path) as f:
            return SessionState.from_dict(json.load(f))

    def save(self, session_id: str, state: SessionState) -> Path:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(session_id)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(state.to_dict(), f, indent=2, sort_keys=True)
            f.write("\n")
        tmp_path.replace(path)
        return path

    def reset(self, session_id: str) -> SessionState:
        path = self.path_for(session_id)
        if path.exists():
            path.unlink()
        return SessionState()

    def path_for(self, session_id: str) -> Path:
        safe_id = _safe_session_id(session_id)
        return self.base_dir / f"{safe_id}.json"


def _safe_session_id(session_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", session_id.strip())
    if not cleaned:
        raise ValueError("session_id must not be empty")
    return cleaned
