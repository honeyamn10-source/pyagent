"""The Agent class that drives the agent loop."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

from pyagent.clients import BaseLLM
from pyagent.errors import MaxStepsExceeded, ToolError
from pyagent.memory import Memory
from pyagent.middleware import Hooks
from pyagent.tools import Tool, ToolRegistry, ensure_tool, tool_registry


def normalize_tools(tools: Optional[Any]) -> Dict[str, Tool]:
    """Coerce a tool collection into a name-to-Tool mapping.

    Accepts a :class:`ToolRegistry`, a mapping of names to tools, or an
    iterable of Tool instances and callables. Later entries overwrite
    earlier ones with the same name.

    Args:
        tools: The tool collection supplied by the user.

    Returns:
        A name-to-Tool dict.

    Raises:
        TypeError: When an item is neither a Tool nor callable.
    """
    result: Dict[str, Tool] = {}
    if isinstance(tools, ToolRegistry):
        source = tools.all().values()
    elif isinstance(tools, dict):
        source = list(tools.values())
    else:
        source = tools
    for item in source or ():
        instance = ensure_tool(item)
        result[instance.name] = instance
    return result


def normalize_hooks(hooks: Optional[Union[Hooks, Mapping[str, Any]]]) -> Hooks:
    """Coerce a hooks argument into a :class:`Hooks` instance.

    Args:
        hooks: A Hooks instance, a dict with ``on_step``/``on_message``
            callables, or None.

    Returns:
        A Hooks instance.

    Raises:
        TypeError: When the value is neither a Hooks instance nor a dict.
    """
    if hooks is None:
        return Hooks()
    if isinstance(hooks, Hooks):
        return hooks
    if isinstance(hooks, Mapping):
        return Hooks(
            on_step=hooks.get("on_step"),
            on_message=hooks.get("on_message"),
        )
    raise TypeError("hooks must be a Hooks instance, a dict, or None")


def format_tool_result(value: Any) -> str:
    """Convert a tool's return value into a string for the chat history.

    Args:
        value: The raw value returned by the tool.

    Returns:
        A string representation: the value as-is for strings, compact JSON
        for mappings and sequences, and ``"ok"`` for None.
    """
    if value is None:
        return "ok"
    if isinstance(value, str):
        return value
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


class Agent:
    """Drives the loop: user message -> model -> tool calls -> answer.

    The agent feeds the accumulated history to the model, hands any requested
    tool call to :meth:`execute_tool`, stores the result as a ``tool`` role
    message and repeats until the model replies with plain text.
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: Optional[Union[ToolRegistry, Mapping[str, Any], Iterable[Any]]] = None,
        system: Optional[str] = None,
        memory: Optional[Memory] = None,
        max_steps: int = 8,
        hooks: Optional[Union[Hooks, Mapping[str, Any]]] = None,
    ) -> None:
        """Initialize the agent.

        Args:
            llm: The model client, typically an :class:`OpenAICompatibleClient`.
            tools: Tools available to the agent. Defaults to the global
                :func:`pyagent.tool_registry`.
            system: Optional system prompt. When given it replaces any
                existing system message in the memory.
            memory: Optional shared memory. A fresh Memory is created when
                omitted so agent instances stay isolated.
            max_steps: Maximum number of tool-call rounds before
                :class:`MaxStepsExceeded` is raised.
            hooks: Optional Hooks instance or dict of callbacks.

        Raises:
            TypeError: When ``llm`` has no callable ``chat`` method.
            ValueError: When ``max_steps`` is less than 1.
        """
        if not callable(getattr(llm, "chat", None)):
            raise TypeError("llm must provide a chat(messages) method")
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.llm = llm
        self.tools: Dict[str, Tool] = (
            tool_registry().all() if tools is None else normalize_tools(tools)
        )
        self.system = system
        self.memory = memory if memory is not None else Memory()
        if system is not None:
            self.memory.set_system(system)
        self.max_steps = max_steps
        self.hooks = normalize_hooks(hooks)

    @property
    def available_tools(self) -> List[str]:
        """The names of the tools wired to this agent.

        Returns:
            A sorted list of tool names.
        """
        return sorted(self.tools)

    def tool_schemas(self) -> List[Dict[str, Any]]:
        """The OpenAI-style schema for every tool wired to the agent.

        Returns:
            A list of schema dicts suitable for a real API's ``tools``
            parameter.
        """
        return [tool.schema() for tool in self.tools.values()]

    def execute_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """Run a tool by name and return a string representation of the result.

        Failures do not crash the loop: a ``ToolError`` or any other
        exception raised while running the tool is converted into an error
        string prefixed with ``[tool error]`` so the model can see it in the
        following turn and recover.

        Args:
            name: The name of the tool to invoke.
            arguments: The arguments to pass to the tool.

        Returns:
            A string representation of the tool result or error.
        """
        instance = self.tools.get(name)
        if instance is None:
            available = ", ".join(sorted(self.tools)) or "(none)"
            return f"[tool error] unknown tool {name!r}; available: {available}"
        try:
            value = instance.run(arguments)
        except ToolError as exc:
            return f"[tool error] {exc}"
        except Exception as exc:
            return f"[tool error] {type(exc).__name__}: {exc}"
        return format_tool_result(value)

    def run(self, message: str) -> str:
        """Run the agent loop on a user message and return the final answer.

        Each iteration feeds the full history to the model. A reply carrying
        a tool call is executed and its result is appended as a ``tool`` role
        message before the next iteration. A plain-text reply ends the loop
        and is returned.

        Args:
            message: The user's message.

        Returns:
            The final assistant reply text.

        Raises:
            MaxStepsExceeded: When the model requests tools on every step up
                to ``max_steps`` without producing a plain-text answer.
        """
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")
        self.memory.add({"role": "user", "content": message})
        for step in range(1, self.max_steps + 1):
            reply = self.llm.chat(self.memory.messages)
            self.hooks.emit_message(reply)
            if reply.tool_call is None:
                answer = reply.text
                self.memory.add({"role": "assistant", "content": answer})
                return answer
            tool_call = reply.tool_call
            self.memory.add(
                {
                    "role": "assistant",
                    "content": reply.text,
                    "tool_calls": [tool_call.to_dict()],
                }
            )
            result = self.execute_tool(tool_call.name, tool_call.arguments)
            self.memory.add(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id or tool_call.name,
                    "content": result,
                }
            )
            self.hooks.emit_step(step, self.memory.messages, tool_call)
        raise MaxStepsExceeded(self.max_steps)

    def __repr__(self) -> str:
        """A compact developer-facing description.

        Returns:
            A string like ``Agent(max_steps=8, tools=['add'])``.
        """
        return f"Agent(max_steps={self.max_steps}, tools={self.available_tools})"


__all__ = ["Agent", "format_tool_result", "normalize_hooks", "normalize_tools"]