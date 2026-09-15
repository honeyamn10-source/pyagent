"""Tests for the urllib-based OpenAICompatibleClient (no network required)."""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from pyagent import APIError, AssistantMessage, BaseLLM, OpenAICompatibleClient, ToolCall

from conftest import add, double

CANNED_TEXT = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "Hello world"},
        }
    ],
}

CANNED_TOOL_CALL = {
    "id": "chatcmpl-tool",
    "choices": [
        {
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_9",
                        "type": "function",
                        "function": {
                            "name": "add",
                            "arguments": '{"a": 2, "b": 3}',
                        },
                    }
                ],
            },
        }
    ],
}


class StubTransport:
    """A fake transport that serves canned responses or raises canned errors."""

    def __init__(self, responses) -> None:
        """Initialize the queue of responses.

        Args:
            responses: An ordered list of bytes/dicts/byteables plus
                exception instances that should be raised.
        """
        self._responses = list(responses)
        self.recorded = []

    def __call__(self, url, body, headers):  # type: ignore[no-untyped-def]
        """Serve the next canned response and record the request.

        Args:
            url: The request URL.
            body: The encoded request body.
            headers: The request headers.

        Returns:
            The response bytes.

        Raises:
            The canned exception, when the next item is one.
        """
        self.recorded.append((url, body, dict(headers)))
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, bytes):
            return item
        return json.dumps(item).encode("utf-8")


def build_client(responses, **kwargs) -> tuple:  # noqa: ANN001
    """Build a client with a stubbed transport.

    Args:
        responses: The canned responses for the transport.
        **kwargs: Extra keyword arguments for OpenAICompatibleClient.

    Returns:
        A ``(client, transport)`` tuple.
    """
    options = {
        "base_url": "https://example.com/v1",
        "model": "tiny-model",
        "retry_delay": 0.0,
    }
    options.update(kwargs)
    options.setdefault("api_key", "sk-test")
    client = OpenAICompatibleClient(**options)
    transport = StubTransport(responses)
    client._perform_request = transport  # type: ignore[method-assign]
    return client, transport


class TestParsingResponses:
    def test_chat_returns_text_message(self) -> None:
        client, _ = build_client([CANNED_TEXT])
        reply = client.chat([{"role": "user", "content": "hi"}])
        assert reply == AssistantMessage(text="Hello world")
        assert reply.tool_call is None

    def test_chat_parses_tool_calls(self) -> None:
        client, _ = build_client([CANNED_TOOL_CALL])
        reply = client.chat([{"role": "user", "content": "add"}])
        assert isinstance(reply.tool_call, ToolCall)
        assert reply.tool_call.name == "add"
        assert reply.tool_call.arguments == {"a": 2, "b": 3}
        assert reply.tool_call.id == "call_9"

    def test_payload_includes_model_and_options(self) -> None:
        client, transport = build_client(
            [CANNED_TEXT],
            max_tokens=32,
            temperature=0.5,
            stop=["\n\n", "END"],
        )
        client.chat([{"role": "user", "content": "q"}])
        _, body, _ = transport.recorded[0]
        payload = json.loads(body)
        assert payload["model"] == "tiny-model"
        assert payload["max_tokens"] == 32
        assert payload["temperature"] == 0.5
        assert payload["stop"] == ["\n\n", "END"]
        assert payload["messages"] == [{"role": "user", "content": "q"}]

    def test_request_url_and_headers(self) -> None:
        client, transport = build_client([CANNED_TEXT])
        client.chat([{"role": "user", "content": "hi"}])
        url, _, headers = transport.recorded[0]
        assert url == "https://example.com/v1/chat/completions"
        assert headers["Authorization"] == "Bearer sk-test"
        assert headers["Content-Type"] == "application/json"

    def test_no_auth_header_without_api_key(self) -> None:
        client, transport = build_client([CANNED_TEXT], api_key=None)
        client.chat([{"role": "user", "content": "hi"}])
        _, _, headers = transport.recorded[0]
        assert "Authorization" not in headers

    def test_trailing_slash_on_base_url_tolerated(self) -> None:
        client, transport = build_client([CANNED_TEXT])
        client.base_url = "https://example.com/v1/"
        client.chat([{"role": "user", "content": "hi"}])
        url, _, _ = transport.recorded[0]
        assert url == "https://example.com/v1/chat/completions"


class TestRetries:
    def test_transient_network_error_is_retried(self) -> None:
        transient = urllib.error.URLError("connection reset")
        client, transport = build_client([transient, CANNED_TEXT], retries=2)
        reply = client.chat([{"role": "user", "content": "hi"}])
        assert reply.text == "Hello world"
        assert len(transport.recorded) == 2

    def test_retry_exhaustion_raises_api_error(self) -> None:
        transient = urllib.error.URLError("down")
        client, transport = build_client([transient] * 4, retries=3)
        with pytest.raises(APIError, match="unreachable"):
            client.chat([{"role": "user", "content": "hi"}])
        assert len(transport.recorded) == 4

    def test_http_error_raises_immediately_with_status(self) -> None:
        http_error = urllib.error.HTTPError(
            "https://example.com/v1/chat/completions",
            429,
            "Too Many Requests",
            {},
            io.BytesIO(b'{"error": "rate limited"}'),
        )
        client, transport = build_client([http_error], retries=3)
        with pytest.raises(APIError) as exc_info:
            client.chat([{"role": "user", "content": "hi"}])
        assert exc_info.value.status_code == 429
        assert len(transport.recorded) == 1


class TestErrorResponses:
    def test_no_choices_raises(self) -> None:
        client, _ = build_client([{"choices": []}])
        with pytest.raises(APIError, match="no choices"):
            client.chat([{"role": "user", "content": "x"}])

    def test_empty_message_raises(self) -> None:
        client, _ = build_client([{"choices": [{"message": {"content": None}}]}])
        with pytest.raises(APIError, match="empty response"):
            client.chat([{"role": "user", "content": "x"}])


class TestFormatToolInstructions:
    def test_lists_tool_names_and_types(self) -> None:
        client = OpenAICompatibleClient(api_key="sk-test")
        text = client.format_tool_instructions([add, double])
        assert "add" in text
        assert "double" in text
        assert "integer" in text
        assert "number" in text
        assert "Available tools:" in text

    def test_describes_json_shapes(self) -> None:
        client = OpenAICompatibleClient(api_key="sk-test")
        text = client.format_tool_instructions([add])
        assert '{"tool"' in text
        assert "tool_calls" in text

    def test_accepts_global_registry(self) -> None:
        from pyagent.tools import tool_registry

        registry = tool_registry()
        registry.add(add)
        client = OpenAICompatibleClient(api_key="sk-test")
        assert "add" in client.format_tool_instructions([registry])


class TestValidation:
    def test_invalid_base_url(self) -> None:
        with pytest.raises(ValueError):
            OpenAICompatibleClient(api_key="k", base_url="not a url")

    def test_negative_retries(self) -> None:
        with pytest.raises(ValueError):
            OpenAICompatibleClient(api_key="k", retries=-1)

    def test_base_llm_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            BaseLLM()  # type: ignore[abstract]

    def test_schemas_exported_from_client_tools(self) -> None:
        from pyagent.tools import tool

        @tool(name="squared")
        def a_number(n: int) -> int:
            """Square a number."""
            return n * n

        client = OpenAICompatibleClient(api_key="k")
        assert "squared" in client.format_tool_instructions([a_number])