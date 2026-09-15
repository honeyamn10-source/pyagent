"""In-memory chat history for the agent loop."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterator, List, Optional, Union

_KNOWN_ROLES = ("system", "user", "assistant", "tool")
_MSG_OVERHEAD = 4

_URL_RE = re.compile(r"(?:https?|ftp)://", re.IGNORECASE)
_PATH_RE = re.compile(r"^(?:/{1,2}|[a-z]:[\\/])", re.IGNORECASE)
_EXT_RE = re.compile(r"\.(?:txt|json|md|py|csv|log|png|jpg|jpeg|pdf)$", re.IGNORECASE)


def _estimate_text_tokens(text: str) -> int:
    """Estimate the token count of a single piece of text.

    Runs at roughly 0.5 tokens per ASCII character. URLs, filesystem paths
    and file-like tokens are treated conservatively and count for more so
    that estimates lean towards over-estimation for expensive content.

    Args:
        text: The text to estimate.

    Returns:
        An integer estimate of the number of tokens.
    """
    total = 0
    for word in re.split(r"\s+", text.strip()):
        if not word:
            continue
        if _URL_RE.search(word) or _PATH_RE.match(word) or _EXT_RE.search(word):
            total += 4 + len(word) // 8
        else:
            total += (len(word) + 1) // 2
    return total


class Memory:
    """An ordered store of chat messages with a few practical helpers.

    Messages are plain dicts with at least a ``role`` key. The ``+`` operator
    appends a single message dict or merges another Memory into a brand new
    Memory, leaving the operands untouched.
    """

    def __init__(
        self,
        system: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Initialize the memory.

        Args:
            system: Optional system prompt placed at the front of history.
            messages: Optional initial list of messages to seed history with.
        """
        self._entries: List[Dict[str, Any]] = []
        for message in messages or ():
            self.add(message)
        if system is not None:
            self.set_system(system)

    @classmethod
    def from_json(cls, data: str) -> "Memory":
        """Build a Memory from the JSON produced by :meth:`to_json`.

        Args:
            data: Serialized message history.

        Returns:
            A new Memory instance.
        """
        try:
            messages = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid memory JSON: {exc}") from exc
        if not isinstance(messages, list):
            raise ValueError("memory JSON must be a list of messages")
        memory = cls()
        for message in messages:
            memory.add(message)
        return memory

    @property
    def messages(self) -> List[Dict[str, Any]]:
        """A shallow copy of the message history.

        Returns:
            The ordered list of message dicts.
        """
        return [dict(message) for message in self._entries]

    def _validate(self, message: Dict[str, Any]) -> None:
        """Validate a single message dict.

        Args:
            message: The message to validate.

        Raises:
            ValueError: When the message is not a well-formed dict.
        """
        if not isinstance(message, dict):
            raise ValueError("message must be a dict")
        role = message.get("role")
        if not isinstance(role, str) or role not in _KNOWN_ROLES:
            raise ValueError(f"message role must be one of {_KNOWN_ROLES}")
        if "content" not in message and role != "tool":
            raise ValueError("message must contain a 'content' key")

    def add(self, message: Dict[str, Any]) -> "Memory":
        """Append a message to the history.

        Args:
            message: A message dict such as ``{"role": "user", "content": "..."}``.

        Returns:
            This Memory so that calls can be chained.
        """
        self._validate(message)
        self._entries.append(dict(message))
        return self

    def append(self, message: Dict[str, Any]) -> "Memory":
        """Alias for :meth:`add`.

        Args:
            message: A message dict.

        Returns:
            This Memory so that calls can be chained.
        """
        return self.add(message)

    def set_system(self, content: str) -> "Memory":
        """Insert or replace the system message at the front of history.

        Args:
            content: The new system prompt text.

        Returns:
            This Memory so that calls can be chained.
        """
        if not isinstance(content, str):
            raise ValueError("system content must be a string")
        keep = [m for m in self._entries if m.get("role") != "system"]
        self._entries = [{"role": "system", "content": content}] + keep
        return self

    def trim(self, max_messages: int) -> "Memory":
        """Keep the system message and the most recent N non-system messages.

        Older non-system messages are dropped from the front of the history.

        Args:
            max_messages: Maximum number of non-system messages to keep.

        Returns:
            This Memory so that calls can be chained.

        Raises:
            ValueError: When ``max_messages`` is less than 1.
        """
        if max_messages < 1:
            raise ValueError("max_messages must be at least 1")
        system_messages = [m for m in self._entries if m.get("role") == "system"]
        rest = [m for m in self._entries if m.get("role") != "system"]
        if len(rest) > max_messages:
            rest = rest[-max_messages:]
        self._entries = system_messages + rest
        return self

    def clear(self) -> "Memory":
        """Remove all non-system messages, keeping the system prompt.

        Returns:
            This Memory so that calls can be chained.
        """
        self._entries = [m for m in self._entries if m.get("role") == "system"]
        return self

    def estimate_tokens(self) -> int:
        """Estimate the token cost of the whole history.

        Uses a cheap heuristic of roughly half a token per character plus a
        small fixed overhead per message. URLs, paths and file-like tokens
        are counted conservatively.

        Returns:
            An integer token estimate.
        """
        total = 0
        for message in self._entries:
            content = message.get("content", "")
            total += _MSG_OVERHEAD
            total += _estimate_text_tokens(str(content))
        return total

    def to_json(self) -> str:
        """Serialize the history to a JSON string.

        Returns:
            JSON text that can be passed to :meth:`from_json`.
        """
        return json.dumps(self._entries, ensure_ascii=False, default=str)

    def __add__(self, other: Union[Memory, Dict[str, Any]]) -> "Memory":
        """Combine this Memory with another Memory or a single message.

        Args:
            other: A Memory or a message dict to append.

        Returns:
            A new Memory containing the merged history.
        """
        new_memory = Memory()
        new_memory._entries = [dict(m) for m in self._entries]
        if isinstance(other, Memory):
            new_memory._entries.extend(dict(m) for m in other._entries)
        elif isinstance(other, dict):
            new_memory.add(other)
        else:
            raise TypeError("can only add a Memory or a message dict to a Memory")
        return new_memory

    def __iadd__(self, other: Union[Memory, Dict[str, Any]]) -> "Memory":
        """In-place append of another Memory or a single message.

        Args:
            other: A Memory or a message dict to append.

        Returns:
            This Memory so that calls can be chained.
        """
        if isinstance(other, Memory):
            self._entries.extend(dict(m) for m in other._entries)
        elif isinstance(other, dict):
            self.add(other)
        else:
            raise TypeError("can only add a Memory or a message dict to a Memory")
        return self

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over the message dicts.

        Yields:
            Each message in the history.
        """
        return iter([dict(m) for m in self._entries])

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return a message by position.

        Args:
            index: Integer position in the history.

        Returns:
            The message at that position.
        """
        return dict(self._entries[index])

    def __len__(self) -> int:
        """The number of stored messages.

        Returns:
            The message count.
        """
        return len(self._entries)

    def __bool__(self) -> bool:
        """Whether the history contains any messages.

        Returns:
            True when there is at least one message.
        """
        return bool(self._entries)

    def __repr__(self) -> str:
        """A compact developer-facing description.

        Returns:
            A string like ``Memory(3 messages)``.
        """
        return f"Memory({len(self._entries)} messages)"


__all__ = ["Memory"]