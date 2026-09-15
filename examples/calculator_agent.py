"""A runnable example agent with a safe calculator and a get_time tool.

Works with any OpenAI-compatible endpoint. Configuration is read from the
environment:

    OPENAI_API_KEY=sk-...      required (can be empty for local endpoints)
    OPENAI_BASE_URL=...        optional, defaults to https://api.openai.com/v1
    OPENAI_MODEL=...           optional, defaults to gpt-4o-mini

Run it with:

    python examples/calculator_agent.py
"""

from __future__ import annotations

import datetime
import os
import re
from typing import List, Union

from pyagent import Agent, OpenAICompatibleClient, tool
from pyagent.errors import ToolError

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def _tokenize(expression: str) -> List[Union[float, str]]:
    """Split an arithmetic expression into a token stream.

    Only numbers, the operators ``+ - * /`` and parentheses are accepted.
    Anything else — letters, commas, comparison operators — raises a
    :class:`ToolError`.

    Args:
        expression: The user-supplied expression.

    Returns:
        A flat list alternating floats and operator tokens, ending with
        the sentinel string ``"end"``.

    Raises:
        ToolError: When an invalid character appears in the expression.
    """
    tokens: List[Union[float, str]] = []
    index = 0
    length = len(expression)
    while index < length:
        char = expression[index]
        if char.isspace():
            index += 1
            continue
        match = _NUMBER_RE.match(expression, index)
        if match:
            tokens.append(float(match.group(0)))
            index = match.end()
            continue
        if char in "()+-*/":
            tokens.append(char)
            index += 1
            continue
        raise ToolError(f"invalid character {char!r} in expression")
    tokens.append("end")
    return tokens


class _ExpressionParser:
    """A small recursive-descent parser for basic arithmetic.

    Grammar::

        expr       := term (('+' | '-') term)*
        term       := factor (('*' | '/') factor)*
        factor     := ('+' | '-') factor | primary
        primary    := NUMBER | '(' expr ')'

    Python's ``eval`` is never used.
    """

    def __init__(self, tokens: List[Union[float, str]]) -> None:
        """Initialize the parser.

        Args:
            tokens: The token stream produced by :func:`_tokenize`.
        """
        self._tokens = tokens
        self._position = 0

    def _peek(self) -> Union[float, str, None]:
        """Return the current token without consuming it.

        Returns:
            The token at the cursor, or None past the end.
        """
        if self._position >= len(self._tokens):
            return None
        return self._tokens[self._position]

    def _advance(self) -> Union[float, str, None]:
        """Consume and return the current token.

        Returns:
            The consumed token, or None past the end.
        """
        token = self._peek()
        if token is not None:
            self._position += 1
        return token

    def parse(self) -> float:
        """Parse the full token stream into a numeric value.

        Returns:
            The value of the expression.

        Raises:
            ToolError: When the expression is malformed.
        """
        value = self._parse_expression()
        if self._peek() != "end":
            raise ToolError("unexpected trailing input in expression")
        return value

    def _parse_expression(self) -> float:
        """Parse an ``expr`` production.

        Returns:
            The running value.
        """
        value = self._parse_term()
        while self._peek() in ("+", "-"):
            operator = self._advance()
            operand = self._parse_term()
            value = value + operand if operator == "+" else value - operand
        return float(value)

    def _parse_term(self) -> float:
        """Parse a ``term`` production.

        Returns:
            The running value.

        Raises:
            ToolError: On division by zero.
        """
        value = self._parse_factor()
        while self._peek() in ("*", "/"):
            operator = self._advance()
            operand = self._parse_factor()
            if operator == "*":
                value = value * operand
            else:
                if operand == 0:
                    raise ToolError("division by zero")
                value = value / operand
        return float(value)

    def _parse_factor(self) -> float:
        """Parse a ``factor`` production, handling unary plus and minus.

        Returns:
            The value of the factor.
        """
        if self._peek() == "-":
            self._advance()
            return -self._parse_factor()
        if self._peek() == "+":
            self._advance()
            return self._parse_factor()
        return self._parse_primary()

    def _parse_primary(self) -> float:
        """Parse a ``primary`` production: a number or a parenthesized expr.

        Returns:
            The value of the primary.

        Raises:
            ToolError: When the expression is malformed.
        """
        token = self._advance()
        if isinstance(token, float):
            return token
        if token == "(":
            value = self._parse_expression()
            if self._advance() != ")":
                raise ToolError("missing closing parenthesis")
            return value
        raise ToolError("invalid expression")


@tool
def calculator(expr: str) -> Union[int, float]:
    """Evaluate a basic arithmetic expression safely.

    Supports numbers, the operators ``+ - * /``, parentheses and unary
    minus. No functions, variables or imports are allowed, and Python's
    ``eval`` is never used.

    :param expr: The arithmetic expression to evaluate.
    """
    value = _ExpressionParser(_tokenize(expr)).parse()
    if float(value).is_integer():
        return int(value)
    return float(value)


@tool
def get_time() -> str:
    """Return the current local date and time as an ISO 8601 string.

    The timestamp is timezone-aware and formatted to second precision,
    e.g. ``2026-09-15T18:13:42+05:30``.
    """
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def build_agent() -> Agent:
    """Construct the example agent from environment configuration.

    Returns:
        A configured :class:`Agent`.

    Raises:
        SystemExit: When neither OPENAI_API_KEY is set nor OPENAI_BASE_URL
            points at a local endpoint.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", OpenAICompatibleClient.DEFAULT_BASE_URL)
    model = os.environ.get("OPENAI_MODEL", OpenAICompatibleClient.DEFAULT_MODEL)
    local = "localhost" in base_url or "127.0.0.1" in base_url
    if not api_key and not local:
        raise SystemExit(
            "Set OPENAI_API_KEY to your API key (or OPENAI_BASE_URL to a "
            "local endpoint such as http://localhost:11434/v1)"
        )
    client = OpenAICompatibleClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=0.2,
    )
    tools = [calculator, get_time]
    agent = Agent(
        llm=client,
        tools=tools,
        system=client.format_tool_instructions(tools),
        max_steps=6,
    )
    return agent


def main() -> None:
    """Run a guided chat loop with the example agent."""
    agent = build_agent()
    print("pyagent example: calculator + get_time tools")
    print(f"model endpoint: {agent.llm.base_url} :: {agent.llm.model}")
    print('try: "what is (2 + 3) * 4?", "double that", "what time is it?"')
    print('type "exit" or press Ctrl-D to quit.\n')
    while True:
        try:
            user_input = input("You> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        command = user_input.strip()
        if not command:
            continue
        if command.lower() in ("exit", "quit"):
            break
        try:
            answer = agent.run(command)
        except Exception as exc:  # noqa: BLE001 - surface errors to the user
            print(f"Agent> error: {exc}\n")
            continue
        print(f"Agent> {answer}\n")


if __name__ == "__main__":
    main()