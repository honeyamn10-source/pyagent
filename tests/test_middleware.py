"""Tests for the retry policy and hooks."""

from __future__ import annotations

import pytest

from pyagent import AssistantMessage, RetryExhausted
from pyagent.errors import MaxStepsExceeded, ToolError
from pyagent.middleware import Hooks, RetryPolicy, retry


class TestRetryPolicy:
    def test_succeeds_after_transient_failures(self) -> None:
        policy = RetryPolicy(retries=2, delay=0.0)
        attempts = {"count": 0}

        def flaky() -> str:
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise ToolError("transient failure")
            return "recovered"

        wrapped = retry(policy)(flaky)
        assert wrapped() == "recovered"
        assert attempts["count"] == 3

    def test_raises_retry_exhausted_after_all_attempts(self) -> None:
        policy = RetryPolicy(retries=3, delay=0.0)
        attempts = {"count": 0}

        def always_fails() -> str:
            attempts["count"] += 1
            raise MaxStepsExceeded(8)

        wrapped = retry(policy)(always_fails)
        with pytest.raises(RetryExhausted) as exc_info:
            wrapped()
        assert exc_info.value.attempts == 4
        assert attempts["count"] == 4

    def test_non_pyagent_error_propagates_immediately(self) -> None:
        policy = RetryPolicy(retries=3, delay=0.0)

        def crashes() -> None:
            raise RuntimeError("not retryable")

        wrapped = retry(policy)(crashes)
        with pytest.raises(RuntimeError, match="not retryable"):
            wrapped()

    def test_success_on_first_attempt(self) -> None:
        policy = RetryPolicy(retries=5, delay=0.0)

        def works() -> str:
            return "ok"

        assert retry(policy)(works)() == "ok"

    def test_wrapper_preserves_function_name(self) -> None:
        policy = RetryPolicy(retries=1, delay=0.0)

        def target() -> str:
            """Docstring."""
            return "x"

        assert retry(policy)(target).__name__ == "target"

    def test_backoff_growth_is_computed(self) -> None:
        policy = RetryPolicy(retries=3, delay=1.0, backoff=2.0)
        assert policy.total_attempts == 4

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            RetryPolicy(retries=-1)
        with pytest.raises(ValueError):
            RetryPolicy(delay=-1.0)
        with pytest.raises(ValueError):
            RetryPolicy(backoff=0.5)


class TestHooks:
    def test_on_step_invoked(self) -> None:
        events = []
        hooks = Hooks(on_step=lambda step, messages, tool_call: events.append(step))
        hooks.emit_step(2, [], None)
        assert events == [2]

    def test_on_message_invoked(self) -> None:
        seen = []
        hooks = Hooks(on_message=seen.append)
        hooks.emit_message(AssistantMessage(text="hi"))
        assert [message.text for message in seen] == ["hi"]

    def test_none_callbacks_are_skipped(self) -> None:
        hooks = Hooks()
        hooks.emit_step(1, [], None)
        hooks.emit_message(AssistantMessage(text="hi"))

    def test_retry_policy_applied_to_agent_run(self, make_agent) -> None:
        from conftest import text_message, tool_message

        agent, _ = make_agent(
            tools=[],
            max_steps=1,
            responses=[
                tool_message("ghost", {}),
                tool_message("ghost", {}),
                tool_message("ghost", {}),
            ],
        )
        wrapped = retry(RetryPolicy(retries=2, delay=0.0))(agent.run)
        with pytest.raises(RetryExhausted):
            wrapped("go")