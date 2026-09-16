"""Retry policies and callback hooks for the agent loop."""

from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

from pyagent.errors import PyagentError, RetryExhausted


class RetryPolicy:
    """Describes how many times and how quickly a failing call is retried.

    Attempt times follow an exponential backoff: attempt ``n`` sleeps for
    ``delay * backoff ** (n - 1)`` seconds before running.
    """

    def __init__(
        self,
        retries: int = 2,
        delay: float = 0.0,
        backoff: float = 2.0,
    ) -> None:
        """Initialize the policy.

        Args:
            retries: Number of retries after the first attempt.
            delay: Base delay in seconds before the first retry.
            backoff: Multiplier applied to the delay on each retry.

        Raises:
            ValueError: When retries is negative, delay is negative or
                backoff is below 1.0.
        """
        if retries < 0:
            raise ValueError("retries must be zero or greater")
        if delay < 0:
            raise ValueError("delay must be zero or greater")
        if backoff < 1.0:
            raise ValueError("backoff must be at least 1.0")
        self.retries = retries
        self.delay = delay
        self.backoff = backoff

    @property
    def total_attempts(self) -> int:
        """The total number of attempts including the first.

        Returns:
            ``retries + 1``.
        """
        return self.retries + 1

    def sleep(self, attempt: int) -> None:
        """Sleep according to the backoff schedule for the given attempt.

        Args:
            attempt: The 1-based attempt number about to run.
        """
        seconds = self.delay * (self.backoff ** (attempt - 1))
        if seconds > 0:
            time.sleep(seconds)


class Hooks:
    """Callback hooks invoked by the agent during a run.

    Both callbacks are optional; a ``None`` callback is silently skipped.
    The same instance can be shared across many agent runs.
    """

    def __init__(
        self,
        on_step: Optional[Callable[[int, List[Dict[str, Any]], Any], None]] = None,
        on_message: Optional[Callable[[Any], None]] = None,
    ) -> None:
        """Initialize the hooks.

        Args:
            on_step: Called after each completed tool step with
                ``(step_number, messages, tool_call)``.
            on_message: Called with each assistant message the model produces.
        """
        self.on_step = on_step
        self.on_message = on_message

    def emit_step(
        self,
        step: int,
        messages: List[Dict[str, Any]],
        tool_call: Any,
    ) -> None:
        """Invoke the ``on_step`` callback, if one is registered.

        Args:
            step: The 1-based step number that just completed.
            messages: A snapshot of the message history after the step.
            tool_call: The ToolCall that was executed in this step.
        """
        if self.on_step is not None:
            self.on_step(step, messages, tool_call)

    def emit_message(self, reply: Any) -> None:
        """Invoke the ``on_message`` callback, if one is registered.

        Args:
            reply: The assistant message the model just produced.
        """
        if self.on_message is not None:
            self.on_message(reply)


def retry(policy: RetryPolicy) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate a callable so that it is retried under a RetryPolicy.

    Only failures that raise a :class:`pyagent.errors.PyagentError` (such
    as :class:`~pyagent.errors.MaxStepsExceeded`) trigger a retry; any other
    exception propagates immediately. When every attempt fails,
    :class:`~pyagent.errors.RetryExhausted` is raised.

    Example::

        from pyagent import Agent, RetryPolicy, retry

        run = retry(RetryPolicy(retries=2, delay=0.5))(agent.run)

    Args:
        policy: The retry policy to apply.

    Returns:
        A decorator that wraps a callable with retry logic.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(1, policy.total_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except PyagentError as exc:
                    if attempt > policy.retries:
                        raise RetryExhausted(attempt) from exc
                    policy.sleep(attempt)
            raise RetryExhausted(policy.total_attempts)

        return wrapper

    return decorator


__all__ = ["Hooks", "RetryPolicy", "retry"]