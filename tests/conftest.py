"""Shared fixtures and helpers for the pyagent test suite."""

from __future__ import annotations

from typing import Any, Callable, List, Optional

import pytest

from pyagent import Agent, AssistantMessage, BaseLLM, ToolCall
from pyagent.errors import ToolError
from pyagent.tools import tool_registry


def text_message(text: str) -> AssistantMessage:
    """Build an AssistantMessage containing only text.

    Args:
        text: The reply text.

    Returns:
        An AssistantMessage with no tool call.
    """
    return AssistantMessage(text=text)


def tool_message(name: str, arguments=None, text: str = "") -> AssistantMessage:
    """Build an AssistantMessage that requests a tool call.

    Args:
        name: The tool name to call.
        arguments: The arguments to pass to the tool.
        text: Optional prose accompanying the call.

    Returns:
        An AssistantMessage carrying a ToolCall.
    """
    return AssistantMessage(
        text=text,
        tool_call=ToolCall(name=name, arguments=arguments or {}),
    )


class FakeLLM(BaseLLM):
    """A deterministic model client driven by an injected script.

    Each call pops the next scripted response. A response may also be a
    callable that receives the current message history and returns an
    AssistantMessage.
    """

    def __init__(self, responses: Optional[List[Any]] = None) -> None:
        """Initialize with the scripted responses.

        Args:
            responses: Ordered script of responses to serve.
        """
        self._responses: List[Any] = list(responses or [])
        self.calls: List[List[dict]] = []

    def queue(self, response: Any) -> None:
        """Append another response to the script.

        Args:
            response: An AssistantMessage or a callable receiving history.
        """
        self._responses.append(response)

    def chat(self, messages: List[dict]) -> AssistantMessage:
        """Serve the next scripted response.

        Args:
            messages: The current message history.

        Returns:
            The next scripted AssistantMessage.

        Raises:
            AssertionError: When the script has been exhausted.
        """
        history = [dict(message) for message in messages]
        self.calls.append(history)
        if not self._responses:
            raise AssertionError("FakeLLM ran out of scripted responses")
        response = self._responses.pop(0)
        if callable(response):
            return response(list(history))
        return response


def add(a: int, b: int) -> int:
    """Add two integers together."""
    return a + b


def double(x: float) -> float:
    """Double a number, returning a float result."""
    return 2.0 * x


def greet(name: str, *, enthusiastic: bool = True) -> str:
    """Greet someone by name.

    :param name: Who to greet.
    """
    if enthusiastic:
        return f"Hello, {name}!"
    return f"Hi, {name}."


def boom(reason: str) -> str:
    """A tool that always fails on purpose.

    :param reason: The reason to report in the failure.
    """
    raise ToolError(f"boom: {reason}")


@pytest.fixture(autouse=True)
def clean_registry() -> None:
    """Reset the global tool registry before and after every test."""
    registry = tool_registry()
    registry.clear()
    yield
    registry.clear()


@pytest.fixture
def make_agent():
    """Factory that builds an Agent backed by a fresh FakeLLM.

    Returns:
        A callable ``factory(**kwargs) -> (agent, llm)`` where extra keyword
        arguments are forwarded to the Agent constructor except ``responses``
        which scripts the FakeLLM.
    """

    def factory(**kwargs: Any):
        responses = kwargs.pop("responses", None)
        llm = kwargs.pop("llm", None)
        if llm is None:
            llm = FakeLLM(responses) if responses is not None else FakeLLM()
        agent = Agent(llm=llm, **kwargs)
        return agent, llm

    return factory