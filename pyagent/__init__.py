"""pyagent: a zero-dependency framework for building LLM agents.

Pure Python 3.9+ standard library only — no third-party imports anywhere in
the package. It implements a plain agent loop: user message -> model ->
tool calls -> execute -> repeat -> final answer.
"""

from pyagent.agent import Agent, format_tool_result, normalize_hooks, normalize_tools
from pyagent.clients import AssistantMessage, BaseLLM, OpenAICompatibleClient
from pyagent.errors import (
    APIError,
    MaxStepsExceeded,
    PyagentError,
    RetryExhausted,
    ToolError,
)
from pyagent.memory import Memory
from pyagent.middleware import Hooks, RetryPolicy, retry
from pyagent.parse import ToolCall, extract_json, extract_tool_call, parse_arguments
from pyagent.tools import Tool, ToolRegistry, ensure_tool, tool, tool_registry

__version__ = "0.1.0"

__all__ = [
    "APIError",
    "Agent",
    "AssistantMessage",
    "BaseLLM",
    "Hooks",
    "MaxStepsExceeded",
    "Memory",
    "OpenAICompatibleClient",
    "PyagentError",
    "RetryExhausted",
    "RetryPolicy",
    "Tool",
    "ToolCall",
    "ToolError",
    "ToolRegistry",
    "ensure_tool",
    "extract_json",
    "extract_tool_call",
    "format_tool_result",
    "normalize_hooks",
    "normalize_tools",
    "parse_arguments",
    "retry",
    "tool",
    "tool_registry",
]