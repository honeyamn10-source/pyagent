"""Tests for the Agent loop."""

from __future__ import annotations

import json

import pytest

from pyagent import Agent, MaxStepsExceeded, Memory, Tool
from pyagent.middleware import Hooks
from pyagent.tools import tool_registry

from conftest import FakeLLM, add, boom, text_message, tool_message


def test_normal_tool_loop_ends_with_answer(make_agent) -> None:
    agent, llm = make_agent(
        tools=[add],
        responses=[
            tool_message("add", {"a": 2, "b": 3}),
            text_message("The sum is 5."),
        ],
    )
    answer = agent.run("Add 2 and 3")
    assert answer == "The sum is 5."
    assert len(llm.calls) == 2
    roles = [message["role"] for message in agent.memory.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    tool_message_entry = [
        message for message in agent.memory.messages if message["role"] == "tool"
    ][0]
    assert tool_message_entry["content"] == "5"
    assert tool_message_entry["tool_call_id"] == "add"
    assistant_entry = [
        message
        for message in agent.memory.messages
        if message["role"] == "assistant" and message.get("tool_calls")
    ][0]
    assert assistant_entry["tool_calls"][0]["function"]["name"] == "add"
    assert json.loads(assistant_entry["tool_calls"][0]["function"]["arguments"]) == {
        "a": 2,
        "b": 3,
    }


def test_prose_answer_stops_immediately(make_agent) -> None:
    agent, llm = make_agent(
        tools=[add],
        responses=[text_message("coffee is splendid")],
    )
    answer = agent.run("tell me about coffee")
    assert answer == "coffee is splendid"
    assert len(llm.calls) == 1


def test_max_steps_overflow_raises(make_agent) -> None:
    agent, llm = make_agent(
        tools=[add],
        max_steps=2,
        responses=[
            tool_message("add", {"a": 1, "b": 1}),
            tool_message("add", {"a": 2, "b": 2}),
            tool_message("add", {"a": 3, "b": 3}),
        ],
    )
    with pytest.raises(MaxStepsExceeded):
        agent.run("keep counting")
    assert len(llm.calls) == 2


def test_tool_error_propagates_to_model(make_agent) -> None:
    agent, llm = make_agent(
        tools=[boom],
        responses=[
            tool_message("boom", {"reason": "kaput"}),
            text_message("The tool failed because of kaput."),
        ],
    )
    answer = agent.run("please run the fragile tool")
    tool_entry = [
        message for message in agent.memory.messages if message["role"] == "tool"
    ][0]
    assert "[tool error]" in tool_entry["content"]
    assert "kaput" in tool_entry["content"]
    assert "kaput" in answer
    assert len(llm.calls) == 2


def test_unknown_tool_reported_as_error(make_agent) -> None:
    agent, llm = make_agent(
        tools=[add],
        responses=[
            tool_message("nope", {"a": 1}),
            text_message("That tool does not exist."),
        ],
    )
    answer = agent.run("call it")
    tool_entry = [
        message for message in agent.memory.messages if message["role"] == "tool"
    ][0]
    assert "[tool error] unknown tool" in tool_entry["content"]
    assert answer == "That tool does not exist."


def test_hooks_invoked_with_step_and_messages(make_agent) -> None:
    steps = []
    seen_messages = []
    hooks = {
        "on_step": lambda step, messages, tool_call: steps.append(
            (step, tool_call.name, len(messages))
        ),
        "on_message": seen_messages.append,
    }
    agent, llm = make_agent(
        tools=[add],
        hooks=hooks,
        responses=[
            tool_message("add", {"a": 1, "b": 1}),
            text_message("two"),
        ],
    )
    assert agent.run("go") == "two"
    assert steps == [(1, "add", 3)]
    assert len(seen_messages) == 2


def test_hooks_accept_hooks_instance(make_agent) -> None:
    events = []
    hooks = Hooks(on_step=lambda step, messages, tool_call: events.append(step))
    agent, llm = make_agent(
        tools=[add],
        hooks=hooks,
        responses=[
            tool_message("add", {"a": 5, "b": 6}),
            text_message("11"),
        ],
    )
    agent.run("sum")
    assert events == [1]


def test_system_message_prepended(make_agent) -> None:
    agent, llm = make_agent(system="Be terse.", responses=[text_message("ok")])
    agent.run("hi")
    assert agent.memory.messages[0] == {"role": "system", "content": "Be terse."}
    assert llm.calls[0][0]["role"] == "system"
    assert llm.calls[0][0]["content"] == "Be terse."


def test_system_overrides_existing_system() -> None:
    memory = Memory(system="old sys")
    agent = Agent(FakeLLM(responses=[]), memory=memory, system="new sys")
    assert agent.memory.messages[0]["content"] == "new sys"


def test_memory_continuity_across_runs(make_agent) -> None:
    agent, llm = make_agent(
        tools=[add],
        responses=[
            tool_message("add", {"a": 4, "b": 5}),
            text_message("nine"),
            text_message("done"),
        ],
    )
    assert agent.run("math") == "nine"
    assert agent.run("continue") == "done"
    assert len(llm.calls) == 3
    second_call = llm.calls[1]
    assert any(
        message["role"] == "user" and message["content"] == "math"
        for message in second_call
    )
    assert any(message["role"] == "tool" for message in second_call)


def test_messages_passed_to_llm_match_memory(make_agent) -> None:
    agent, llm = make_agent(responses=[text_message("yep")])
    agent.run("first")
    assert llm.calls[0][-1] == {"role": "user", "content": "first"}


def test_execute_tool_returns_json_for_structured_results(make_agent) -> None:
    def record() -> dict:
        """Return a structured record."""
        return {"sum": 1, "ok": True}

    agent, _ = make_agent(tools=[record])
    result = agent.execute_tool("record", {})
    assert json.loads(result) == {"sum": 1, "ok": True}


def test_execute_tool_unknown_name(make_agent) -> None:
    agent, _ = make_agent(tools=[add])
    result = agent.execute_tool("missing", {"a": 1})
    assert "[tool error] unknown tool 'missing'" in result


def test_execute_tool_generic_exception_is_caught(make_agent) -> None:
    def crash(x: int) -> int:
        """Crash on demand."""
        raise ZeroDivisionError("boom")

    agent, _ = make_agent(tools=[crash])
    result = agent.execute_tool("crash", {"x": 1})
    assert "[tool error] ZeroDivisionError: boom" == result


def test_default_tools_use_global_registry() -> None:
    tool_registry().add(add)
    agent = Agent(FakeLLM(responses=[]))
    assert agent.available_tools == ["add"]
    assert "add" in agent.tool_schemas()[0]["function"]["name"]


def test_agent_validates_constructor_args() -> None:
    with pytest.raises(TypeError):
        Agent(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Agent(FakeLLM(responses=[]), max_steps=0)
    with pytest.raises(TypeError):
        Agent(FakeLLM(responses=[]), tools=[42])


def test_run_rejects_empty_message(make_agent) -> None:
    agent, _ = make_agent(responses=[text_message("x")])
    with pytest.raises(ValueError):
        agent.run("   ")
    assert len(agent.memory) == 0


def test_agent_accepts_tool_mapping_and_registry() -> None:
    mapping = Agent(FakeLLM(responses=[]), tools={"add": Tool(add)})
    assert mapping.available_tools == ["add"]
    registry = tool_registry()
    registry.add(add)
    via_registry = Agent(FakeLLM(responses=[]), tools=registry)
    assert via_registry.available_tools == ["add"]