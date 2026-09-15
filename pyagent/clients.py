"""Language-model clients and the assistant message type.

:class:`OpenAICompatibleClient` speaks the OpenAI chat-completions protocol
using only the standard library's ``urllib`` module, so it works with OpenAI
and any compatible endpoint (OpenRouter, Ollama, vLLM, LM Studio, ...).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from pyagent.errors import APIError
from pyagent.parse import ToolCall, parse_arguments
from pyagent.tools import ToolRegistry, ensure_tool


@dataclass
class AssistantMessage:
    """A model response: free text plus an optional structured tool call."""

    text: str = ""
    tool_call: Optional[ToolCall] = None


def _tool_call_from_payload(raw: Dict[str, Any]) -> Optional[ToolCall]:
    """Build a ToolCall from one raw ``tool_calls[0]`` entry.

    Args:
        raw: A single tool-call entry as decoded from the API response.

    Returns:
        A ToolCall, or None when the entry is malformed.
    """
    function = raw.get("function")
    if not isinstance(function, dict):
        function = {}
    name = function.get("name") or raw.get("name")
    if isinstance(name, dict):
        name = name.get("name")
    if not isinstance(name, str) or not name:
        return None
    raw_arguments = function.get("arguments", raw.get("arguments"))
    identifier = raw.get("id")
    return ToolCall(
        name=name,
        arguments=parse_arguments(raw_arguments),
        id=str(identifier) if identifier else None,
    )


class BaseLLM(ABC):
    """Abstract interface every model client must implement."""

    @abstractmethod
    def chat(self, messages: List[Mapping[str, Any]]) -> AssistantMessage:
        """Send a conversation and return the assistant's reply.

        Args:
            messages: The full message history as role/content dicts.

        Returns:
            The model's reply as an :class:`AssistantMessage`.
        """


class OpenAICompatibleClient(BaseLLM):
    """A minimal OpenAI-compatible chat client built on ``urllib``.

    The client posts to ``<base_url>/chat/completions`` and parses the
    standard OpenAI response shape, including ``choices[0].message.tool_calls``.
    """

    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 30.0,
        retries: int = 0,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[Sequence[str]] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        retry_delay: float = 0.1,
    ) -> None:
        """Initialize the client.

        Args:
            api_key: Bearer token used for the ``Authorization`` header.
                May be omitted (None) for local endpoints that do not
                require authentication.
            base_url: Base URL of the OpenAI-compatible server.
            model: The model identifier to request.
            timeout: Socket timeout in seconds for each request.
            retries: Number of times to retry transient network failures.
            max_tokens: Optional ``max_tokens`` value sent with each request.
            temperature: Optional sampling temperature sent with each request.
            stop: Optional ``stop`` sequences sent with each request.
            extra_headers: Extra HTTP headers sent with each request.
            retry_delay: Base sleep in seconds between retry attempts.

        Raises:
            ValueError: When ``base_url`` is not a valid HTTP(S) URL.
        """
        if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        if retries < 0:
            raise ValueError("retries must be zero or greater")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.stop = list(stop) if stop else None
        self.extra_headers = dict(extra_headers) if extra_headers else None
        self.retry_delay = retry_delay

    def chat(self, messages: List[Mapping[str, Any]]) -> AssistantMessage:
        """Send the conversation and return the model's reply.

        Args:
            messages: The full message history as role/content dicts.

        Returns:
            An :class:`AssistantMessage` with the reply text and any tool
            call requested by the model.

        Raises:
            APIError: When the API is unreachable, returns an error or
                returns an unparseable response.
        """
        payload: Dict[str, Any] = {"model": self.model, "messages": list(messages)}
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.stop:
            payload["stop"] = self.stop
        data = self._post(payload)
        return self._parse_response(data)

    def format_tool_instructions(self, tools: Sequence[Any]) -> str:
        """Render the provided tools into an instruction block for a system prompt.

        Args:
            tools: Tools to document. Accepts Tool instances, callables or
                the global registry.

        Returns:
            A plain-text system prompt describing the tools and the exact
            JSON shapes the model should reply with.
        """
        lines = [
            "You are an automated agent with access to tools.",
            "",
            "When your task requires a tool, reply with a JSON object in one of these shapes:",
            '{"tool": "<name>", "arguments": { ... }}',
            '{"tool_calls": [{"function": {"name": "<name>", "arguments": "{...json...}"}}]}',
            "Fenced code blocks and surrounding prose are both accepted.",
            "After a tool result arrives, reason about it and either call another tool",
            "or give the final answer. When the task is complete, reply in plain text",
            "and do NOT call a tool.",
            "",
            "Available tools:",
        ]
        instances = []
        for item in tools or ():
            if isinstance(item, ToolRegistry):
                instances.extend(item)
            else:
                instances.append(ensure_tool(item))
        for instance in instances:
            signature = ", ".join(
                f"{param_name}: {spec['type']}"
                for param_name, spec in instance.parameters["properties"].items()
            )
            lines.append(f"- {instance.name}({signature}): {instance.description}")
        return "\n".join(lines)

    def _build_headers(self) -> Dict[str, str]:
        """Build the HTTP headers for a request.

        Returns:
            A dict of header names to values.
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for key, value in (self.extra_headers or {}).items():
            headers[key] = value
        return headers

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST the payload and return the decoded JSON response.

        Args:
            payload: The request body to send.

        Returns:
            The decoded JSON response body.

        Raises:
            APIError: When every attempt fails or the server returns an
                HTTP error status.
        """
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        headers = self._build_headers()
        last_error: Optional[BaseException] = None
        for attempt in range(self.retries + 1):
            try:
                raw = self._perform_request(url, body, headers)
                return json.loads(raw.decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:1000]
                raise APIError(
                    f"model API returned HTTP {exc.code}: {detail}",
                    status_code=exc.code,
                ) from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(self.retry_delay * (2 ** attempt))
        raise APIError(f"model API unreachable: {last_error}") from last_error

    def _perform_request(self, url: str, body: bytes, headers: Dict[str, str]) -> bytes:
        """Perform a single HTTP request and return the raw response body.

        Args:
            url: The URL to POST to.
            body: The encoded request body.
            headers: The request headers.

        Returns:
            The raw response body.

        Raises:
            urllib.error.URLError: On network or HTTP failures.
        """
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return response.read()

    def _parse_response(self, data: Dict[str, Any]) -> AssistantMessage:
        """Convert a decoded chat-completions response into an AssistantMessage.

        Args:
            data: The decoded JSON response body.

        Returns:
            An :class:`AssistantMessage` parsed from the response.

        Raises:
            APIError: When the response is missing choices, a message or any
                usable content.
        """
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise APIError("model response contained no choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise APIError("model response contained no message")
        content = message.get("content")
        text = content if isinstance(content, str) else ""
        tool_call: Optional[ToolCall] = None
        raw_calls = message.get("tool_calls")
        if isinstance(raw_calls, list) and raw_calls:
            tool_call = _tool_call_from_payload(raw_calls[0])
        if tool_call is None and not text.strip():
            raise APIError("model returned an empty response")
        return AssistantMessage(text=text, tool_call=tool_call)


__all__ = [
    "AssistantMessage",
    "BaseLLM",
    "OpenAICompatibleClient",
]