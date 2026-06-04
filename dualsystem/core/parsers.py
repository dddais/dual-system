"""VLM output parsers for standalone Dual-System planning."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass
class ParseResult:
    """Parsed VLM output."""

    subtask: str
    memory: str | None = None
    subtask_index: int | None = None
    parse_ok: bool = True
    parse_error: str | None = None


@dataclass
class PlanParseResult:
    """Parsed initial subtask plan."""

    subtasks: list[str]
    parse_ok: bool = True
    parse_error: str | None = None


def parse_vlm_output(text: str, enable_memory: bool = False) -> ParseResult:
    """Parse a VLM output into subtask and optional memory."""
    if not enable_memory:
        subtask = text.strip()
        if subtask:
            return ParseResult(subtask=subtask, memory=None)
        return ParseResult(
            subtask="continue current action",
            memory=None,
            parse_ok=False,
            parse_error="empty VLM output",
        )

    cleaned = _strip_code_fences(text)
    json_result = _parse_json_object(cleaned)
    if json_result is not None:
        return json_result

    regex_result = _parse_json_like_fields(cleaned)
    if regex_result is not None:
        return regex_result

    fallback = cleaned or "continue current action"
    return ParseResult(
        subtask=fallback,
        memory=None,
        parse_ok=False,
        parse_error="failed to parse memory JSON",
    )


def parse_subtask_plan(text: str) -> PlanParseResult:
    """Parse a VLM output into an ordered subtask list."""
    cleaned = _strip_code_fences(text)
    data = _load_embedded_json(cleaned)
    if data is not None:
        candidates = data
        if isinstance(data, dict):
            candidates = (
                data.get("subtasks")
                or data.get("subtask_plan")
                or data.get("plan")
                or data.get("steps")
            )
        subtasks = _normalize_subtask_list(candidates)
        if subtasks:
            return PlanParseResult(subtasks=subtasks)

    numbered = _parse_numbered_or_bulleted_lines(cleaned)
    if numbered:
        return PlanParseResult(
            subtasks=numbered,
            parse_ok=False,
            parse_error="recovered subtask plan from numbered text",
        )

    fallback = cleaned or "continue current action"
    return PlanParseResult(
        subtasks=[fallback],
        parse_ok=False,
        parse_error="failed to parse subtask plan JSON",
    )


def parse_subtask_selection(
    text: str,
    subtasks: list[str],
    enable_memory: bool = False,
) -> ParseResult:
    """Parse a VLM output that chooses one subtask from a fixed plan."""
    available_subtasks = _normalize_subtask_list(subtasks)
    if not available_subtasks:
        return ParseResult(
            subtask="continue current action",
            memory=None,
            subtask_index=None,
            parse_ok=False,
            parse_error="no available subtasks to select from",
        )

    cleaned = _strip_code_fences(text)
    data = _load_embedded_json(cleaned)
    if isinstance(data, dict):
        return _selection_from_fields(
            data=data,
            subtasks=available_subtasks,
            enable_memory=enable_memory,
            parse_ok=True,
            parse_error=None,
        )

    regex_data = _parse_selection_json_like_fields(cleaned)
    if regex_data:
        return _selection_from_fields(
            data=regex_data,
            subtasks=available_subtasks,
            enable_memory=enable_memory,
            parse_ok=False,
            parse_error="recovered selection fields with regex after JSON parse failure",
        )

    selected_index = _resolve_selected_index(cleaned, cleaned, available_subtasks)
    if selected_index is not None:
        return ParseResult(
            subtask=available_subtasks[selected_index],
            memory=None,
            subtask_index=selected_index,
        )

    return ParseResult(
        subtask=available_subtasks[0],
        memory=None,
        subtask_index=0,
        parse_ok=False,
        parse_error="selection did not match available subtasks",
    )


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _parse_json_object(cleaned: str) -> ParseResult | None:
    data = _load_embedded_json(cleaned)
    if not isinstance(data, dict):
        return None
    subtask = str(data.get("subtask", "")).strip()
    if not subtask:
        return None
    memory_value = data.get("memory")
    memory = None if memory_value is None else str(memory_value).strip()
    return ParseResult(subtask=subtask, memory=memory)


def _parse_json_like_fields(cleaned: str) -> ParseResult | None:
    subtask_match = re.search(
        r'"subtask"\s*:\s*"([^"]*)"', cleaned, flags=re.DOTALL
    )
    if not subtask_match:
        return None
    memory_match = re.search(r'"memory"\s*:\s*"([^"]*)"', cleaned, flags=re.DOTALL)
    return ParseResult(
        subtask=subtask_match.group(1).strip(),
        memory=memory_match.group(1).strip() if memory_match else None,
        parse_ok=False,
        parse_error="recovered fields with regex after JSON parse failure",
    )


def _load_embedded_json(cleaned: str):
    json_match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.DOTALL)
    if not json_match:
        return None
    try:
        return json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return None


def _normalize_subtask_list(candidates) -> list[str]:
    if not isinstance(candidates, list):
        return []
    subtasks: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if isinstance(item, dict):
            item = (
                item.get("subtask")
                or item.get("description")
                or item.get("name")
                or item.get("step")
            )
        text = str(item).strip()
        if not text:
            continue
        normalized = _normalize_for_match(text)
        if normalized in seen:
            continue
        seen.add(normalized)
        subtasks.append(text)
    return subtasks


def _parse_numbered_or_bulleted_lines(cleaned: str) -> list[str]:
    subtasks: list[str] = []
    for line in cleaned.splitlines():
        match = re.match(r"^\s*(?:[-*]\s+|\d+[\).:-]\s+)(.+?)\s*$", line)
        if match:
            subtasks.append(match.group(1).strip())
    return _normalize_subtask_list(subtasks)


def _parse_selection_json_like_fields(cleaned: str) -> dict[str, str] | None:
    data: dict[str, str] = {}
    id_match = re.search(
        r'"(?:subtask_id|subtask_index|index|id)"\s*:\s*"?(\d+)"?',
        cleaned,
        flags=re.DOTALL,
    )
    if id_match:
        data["subtask_id"] = id_match.group(1)
    subtask_match = re.search(
        r'"subtask"\s*:\s*"([^"]*)"', cleaned, flags=re.DOTALL
    )
    if subtask_match:
        data["subtask"] = subtask_match.group(1)
    memory_match = re.search(r'"memory"\s*:\s*"([^"]*)"', cleaned, flags=re.DOTALL)
    if memory_match:
        data["memory"] = memory_match.group(1)
    return data or None


def _selection_from_fields(
    data: dict,
    subtasks: list[str],
    enable_memory: bool,
    parse_ok: bool,
    parse_error: str | None,
) -> ParseResult:
    raw_index = (
        data.get("subtask_id")
        if data.get("subtask_id") is not None
        else data.get("subtask_index")
    )
    if raw_index is None:
        raw_index = (
            data.get("index") if data.get("index") is not None else data.get("id")
        )
    raw_subtask = data.get("subtask")
    selected_index = _resolve_selected_index(raw_subtask, raw_index, subtasks)
    memory_value = data.get("memory") if enable_memory else None
    memory = None if memory_value is None else str(memory_value).strip()

    if selected_index is None:
        return ParseResult(
            subtask=subtasks[0],
            memory=memory,
            subtask_index=0,
            parse_ok=False,
            parse_error=parse_error or "selection did not match available subtasks",
        )

    selected_subtask = subtasks[selected_index]
    if (
        raw_subtask is not None
        and _find_subtask_by_text(str(raw_subtask), subtasks) is None
    ):
        parse_ok = False
        parse_error = parse_error or "selected subtask text was not in available subtasks"
    return ParseResult(
        subtask=selected_subtask,
        memory=memory,
        subtask_index=selected_index,
        parse_ok=parse_ok,
        parse_error=parse_error,
    )


def _resolve_selected_index(
    raw_subtask: object | None,
    raw_index: object | None,
    subtasks: list[str],
) -> int | None:
    index = _parse_index(raw_index, len(subtasks))
    if index is not None:
        return index
    if raw_subtask is None:
        return None
    return _find_subtask_by_text(str(raw_subtask), subtasks)


def _parse_index(raw_index: object | None, num_subtasks: int) -> int | None:
    if raw_index is None:
        return None
    try:
        index = int(str(raw_index).strip())
    except ValueError:
        return None
    if 1 <= index <= num_subtasks:
        return index - 1
    if 0 <= index < num_subtasks:
        return index
    return None


def _find_subtask_by_text(raw_subtask: str, subtasks: list[str]) -> int | None:
    cleaned = re.sub(r"^\s*subtask\s*:\s*", "", raw_subtask, flags=re.IGNORECASE)
    normalized = _normalize_for_match(cleaned)
    if not normalized:
        return None
    for index, subtask in enumerate(subtasks):
        if _normalize_for_match(subtask) == normalized:
            return index
    for index, subtask in enumerate(subtasks):
        candidate = _normalize_for_match(subtask)
        if candidate and (candidate in normalized or normalized in candidate):
            return index
    return None


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().casefold())
