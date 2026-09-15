"""Exceptions raised by the pyagent framework."""

from __future__ import annotations

from typing import Optional


class PyagentError(Exception):
    """Base class for every exception raised by pyagent itself."""

    def __init__(self, message: str = "") -> None:
        """Initialize the error.

        Args:
            message: Human readable description of the failure.
        """
        super().__init__(message)
        self.message = message


class ToolError(PyagentError):
    """Raised when a tool cannot be executed or reports a failure.

    Errors of this type raised inside a tool are captured by the agent and
    forwarded to the model as a ``tool`` role message so that the model can
    recover instead of crashing the loop.
    """

    def __init__(self, message: str, tool_name: Optional[str] = None) -> None:
        """Initialize the error.

        Args:
            message: Description of what went wrong.
            tool_name: Optional name of the tool that raised the error.
        """
        super().__init__(message)
        self.message = message
        self.tool_name = tool_name


class MaxStepsExceeded(PyagentError):
    """Raised when the agent exceeds its configured step limit."""

    def __init__(self, max_steps: int) -> None:
        """Initialize the error.

        Args:
            max_steps: The step limit that was exceeded.
        """
        super().__init__(f"agent exceeded the maximum number of steps ({max_steps})")
        self.max_steps = max_steps


class RetryExhausted(PyagentError):
    """Raised when a retry policy runs out of retry attempts."""

    def __init__(self, attempts: int) -> None:
        """Initialize the error.

        Args:
            attempts: Total number of attempts made before giving up.
        """
        super().__init__(f"retry policy exhausted after {attempts} attempts")
        self.attempts = attempts


class APIError(PyagentError):
    """Raised when the model API is unreachable or returns an error."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
    ) -> None:
        """Initialize the error.

        Args:
            message: Description of the API failure.
            status_code: HTTP status code returned by the server, if any.
        """
        super().__init__(message)
        self.status_code = status_code


__all__ = [
    "APIError",
    "MaxStepsExceeded",
    "PyagentError",
    "RetryExhausted",
    "ToolError",
]