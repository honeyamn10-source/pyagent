"""Parsing helpers that turn raw model text into structured tool calls.

Models do not always return a clean JSON object. The helpers in this module
tolerate surrounding prose, fenced code blocks and a stray trailing comma so
that an agent loop can reliably fetch the tool call out of a chat reply.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

_TRAILING_COMMA_RE = re.compile(r",\s*([\]}])")
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

_JSONDecodeError = json.JSONDecodeError if hasattr(json, "JSONDecodeError") else ValueError


@dataclass(frozen=True)
class ToolCall:
    """A structured request to execute a tool."""

    name: str
    arguments: Dict[str, Any]
    id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize into the OpenAI-style ``tool_calls`` payload shape.

        Returns:
            A dict with ``id`` and ``function`` keys whose ``arguments``
            value is a JSON-encoded string, as expected by the OpenAI
            chat-completions protocol.
        """
        return {
            "id": self.id or self.name,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


def _strip_code_fence(text: str) -> str:
    """Return the body of the first fenced code block, if any.

    Args:
        text: Raw model output.

    Returns:
        The code fence contents when a fence is detected, otherwise the
        original text unchanged.
    """
    match = _CODE_FENCE_RE.search(text)
    if match:
        return match.group(1)
    return text


def _balanced_objects(text: str) -> Iterator[str]:
    """Yield every maximal balanced ``{...}`` region in the text.

    Braces inside string literals are ignored so that message payloads that
    contain braces do not confuse the scan.

    Args:
        text: The text to scan.

    Yields:
        Each maximal balanced brace-delimited substring.
    """
    for start in range(len(text)):
        if text[start] != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for end in range(start, len(text)):
            char = text[end]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
            elif char == '"':
                in_string = not in_string
            elif not in_string and char == "{":
                depth += 1
            elif not in_string and char == "}":
                depth -= 1
                if depth == 0:
                    yield text[start : end + 1]
                    break


def _looks_like_tool_call(obj: Any) -> bool:
    """Return whether a parsed object resembles a tool-call payload.

    Args:
        obj: A JSON-decoded value.

    Returns:
        True when the value is a dict carrying a ``tool`` or ``tool_calls``
        key, meaning it is worth preferring over other candidates.
    """
    if not isinstance(obj, dict):
        return False
    return "tool" in obj or "tool_calls" in obj


def _parse_with_repair(candidate: str) -> Optional[Dict[str, Any]]:
    """Parse a candidate JSON text, attempting an unambiguous repair.

    Args:
        candidate: A brace-balanced substring.

    Returns:
        The decoded object, or None when it cannot be parsed even after
        removing a stray trailing comma.
    """
    try:
        value = json.loads(candidate)
    except _JSONDecodeError:
        repaired = repair_json(candidate)
        if repaired is None:
            return None
        try:
            value = json.loads(repaired)
        except _JSONDecodeError:
            return None
    if not isinstance(value, dict):
        return None
    return value


def repair_json(text: str) -> Optional[str]:
    """Remove a single unambiguous trailing comma from invalid JSON.

    The repair only fires when removing exactly the stray commas at the end
    of an object or array makes the whole text valid. Nothing more advanced
    is attempted, so genuinely malformed JSON is left untouched.

    Args:
        text: The JSON text to repair.

    Returns:
        The repaired text, or None when there is nothing unambiguous to fix.
    """
    if not text:
        return None
    repaired = _TRAILING_COMMA_RE.sub(r"\1", text)
    if repaired == text:
        return None
    try:
        json.loads(repaired)
    except _JSONDecodeError:
        return None
    return repaired


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract the most relevant JSON object from model output.

    A tool-call shaped object is preferred over other parseable objects so
    that prose that happens to contain JSON is not accidentally chosen.

    Args:
        text: Raw model output.

    Returns:
        A decoded JSON object, or None when no usable object is found.
    """
    if text is None:
        return None
    cleaned = _strip_code_fence(text).strip()
    candidates = list(_balanced_objects(cleaned))
    for candidate in candidates:
        parsed = _parse_with_repair(candidate)
        if parsed is not None and _looks_like_tool_call(parsed):
            return parsed
    for candidate in candidates:
        parsed = _parse_with_repair(candidate)
        if parsed is not None:
            return parsed
    return None


def parse_arguments(source: Any) -> Dict[str, Any]:
    """Coerce a tool-call ``arguments`` field into a JSON object.

    Handles an already-decoded mapping as well as a JSON-encoded string.
    When the string is not valid JSON it is treated as a single positional
    value stored under the key ``value``.

    Args:
        source: The arguments field from a tool call.

    Returns:
        A dict of arguments.
    """
    if source is None:
        return {}
    if isinstance(source, dict):
        return dict(source)
    if not isinstance(source, str):
        return {}
    cleaned = source.strip()
    if not cleaned:
        return {}
    try:
        value = json.loads(cleaned)
    except _JSONDecodeError:
        repaired = repair_json(cleaned)
        if repaired is not None:
            try:
                value = json.loads(repaired)
            except _JSONDecodeError:
                value = None
        else:
            value = None
    if isinstance(value, dict):
        return value
    if value is None:
        return {"value": cleaned}
    return {"value": value}


def _from_openai_shape(obj: Dict[str, Any]) -> Optional[ToolCall]:
    """Build a ToolCall from an OpenAI-style ``tool_calls`` payload.

    Args:
        obj: A decoded dict containing a ``tool_calls`` key.

    Returns:
        The first tool call, or None when the payload is malformed.
    """
    calls = obj.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        return None
    first = calls[0]
    if not isinstance(first, dict):
        return None
    function = first.get("function")
    if not isinstance(function, dict):
        function = {}
    name = function.get("name") or first.get("name")
    if isinstance(name, dict):
        name = name.get("name")
    if not isinstance(name, str) or not name:
        return None
    raw_arguments = function.get("arguments", first.get("arguments"))
    arguments = parse_arguments(raw_arguments)
    identifier = first.get("id")
    return ToolCall(
        name=name,
        arguments=arguments,
        id=str(identifier) if identifier else None,
    )


def _from_custom_shape(obj: Dict[str, Any]) -> Optional[ToolCall]:
    """Build a ToolCall from a compact ``{"tool": ..., "arguments": {...}}`` payload.

    Args:
        obj: A decoded dict containing a ``tool`` key.

    Returns:
        The parsed tool call, or None when the payload is malformed.
    """
    name = obj.get("tool")
    if isinstance(name, dict):
        name = name.get("name")
    if not isinstance(name, str) or not name:
        return None
    arguments = parse_arguments(obj.get("arguments"))
    identifier = obj.get("id")
    return ToolCall(
        name=name,
        arguments=arguments,
        id=str(identifier) if identifier else None,
    )


def extract_tool_call(text: str) -> Optional[ToolCall]:
    """Extract a single ToolCall from raw model output.

    Accepts both the compact ``{"tool": ..., "arguments": ...}`` shape and the
    OpenAI-style ``{"tool_calls": [...]}`` shape, with or without surrounding
    prose or code fences.

    Args:
        text: Raw model output.

    Returns:
        A ToolCall describing the requested tool, or None when the output
        does not contain a tool call.
    """
    obj = extract_json(text or "")
    if obj is None:
        return None
    if "tool_calls" in obj:
        return _from_openai_shape(obj)
    if "tool" in obj:
        return _from_custom_shape(obj)
    return None


__all__ = [
    "ToolCall",
    "extract_json",
    "extract_tool_call",
    "parse_arguments",
    "repair_json",
]