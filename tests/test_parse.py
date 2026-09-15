"""Tests for the tool-call parsing helpers."""

from __future__ import annotations

import pytest

from pyagent.parse import extract_json, extract_tool_call, parse_arguments, repair_json


class TestExtractJson:
    def test_plain_object(self) -> None:
        source = '{"tool": "add", "arguments": {"a": 1}}'
        assert extract_json(source) == {"tool": "add", "arguments": {"a": 1}}

    def test_object_wrapped_in_prose(self) -> None:
        source = (
            'First compute the sum.\n'
            '{"tool": "add", "arguments": {"a": 1, "b": 2}}\n'
            "Then reply with the answer."
        )
        parsed = extract_json(source)
        assert parsed == {"tool": "add", "arguments": {"a": 1, "b": 2}}

    def test_fenced_code_block(self) -> None:
        source = (
            "Here you go:\n```json\n"
            '{"tool": "add", "arguments": {"a": 1}}\n'
            "```\nthats all"
        )
        parsed = extract_json(source)
        assert parsed == {"tool": "add", "arguments": {"a": 1}}

    def test_fence_without_json_tag(self) -> None:
        source = '```\n{"tool": "get_time"}\n```'
        assert extract_json(source) == {"tool": "get_time"}

    def test_trailing_comma_repaired(self) -> None:
        source = '{"tool": "add", "arguments": {"a": 1, "b": 2,},}'
        assert extract_json(source) == {"tool": "add", "arguments": {"a": 1, "b": 2}}

    def test_no_json_returns_none(self) -> None:
        assert extract_json("No tools needed, this is plain prose.") is None

    def test_unbalanced_brace_in_prose_returns_none(self) -> None:
        assert extract_json("here is a { brace all alone") is None

    def test_none_input_returns_none(self) -> None:
        assert extract_json(None) is None  # type: ignore[arg-type]

    def test_multiple_candidates_prefers_tool_object(self) -> None:
        source = 'Result: {"total": 3}. Now call: {"tool": "add", "arguments": {"a": 1, "b": 2}}'
        assert extract_json(source)["tool"] == "add"

    def test_braces_inside_string_ignored(self) -> None:
        source = '{"tool": "echo", "arguments": {"text": "say {hi} now"}}'
        assert extract_json(source) == {"tool": "echo", "arguments": {"text": "say {hi} now"}}


class TestExtractToolCall:
    def test_custom_shape(self) -> None:
        call = extract_tool_call('{"tool": "add", "arguments": {"a": 1, "b": 2}}')
        assert call is not None
        assert call.name == "add"
        assert call.arguments == {"a": 1, "b": 2}

    def test_openai_shape_with_encoded_string_arguments(self) -> None:
        source = (
            '{"id": "x", "tool_calls": [{"id": "call_1", "type": "function", '
            '"function": {"name": "add", "arguments": "{\\"a\\": 2, \\"b\\": 3}"}}]}'
        )
        call = extract_tool_call(source)
        assert call is not None
        assert call.name == "add"
        assert call.arguments == {"a": 2, "b": 3}
        assert call.id == "call_1"

    def test_openai_shape_with_object_arguments(self) -> None:
        source = '{"tool_calls": [{"id": "c2", "function": {"name": "double", "arguments": {"x": 4}}}]}'
        call = extract_tool_call(source)
        assert call is not None
        assert call.name == "double"
        assert call.arguments == {"x": 4}

    def test_custom_shape_with_string_arguments(self) -> None:
        source = '{"tool": "double", "arguments": "{\\"x\\": 4}"}'
        call = extract_tool_call(source)
        assert call is not None
        assert call.arguments == {"x": 4}

    def test_non_json_string_arguments_become_value(self) -> None:
        call = extract_tool_call('{"tool": "say", "arguments": "hello there"}')
        assert call is not None
        assert call.arguments == {"value": "hello there"}

    def test_object_arguments_ignored_keys(self) -> None:
        call = extract_tool_call('{"tool": "add", "arguments": {"a": 5, "b": 6, "extra": 9}}')
        assert call is not None
        assert call.arguments == {"a": 5, "b": 6, "extra": 9}

    def test_prose_answer_has_no_tool_call(self) -> None:
        assert extract_tool_call("The answer is simply 42.") is None

    def test_empty_and_none_inputs(self) -> None:
        assert extract_tool_call("") is None
        assert extract_tool_call(None) is None  # type: ignore[arg-type]

    def test_malformed_tool_call_returns_none(self) -> None:
        assert extract_tool_call('{"tool_calls": []}') is None
        assert extract_tool_call('{"tool_calls": [{"function": {}}]}') is None


class TestParseArguments:
    def test_dict_passthrough(self) -> None:
        assert parse_arguments({"a": 1}) == {"a": 1}

    def test_string_json_parsed(self) -> None:
        assert parse_arguments('{"a": 1, "b": [true, null]}') == {"a": 1, "b": [True, None]}

    def test_none_returns_empty(self) -> None:
        assert parse_arguments(None) == {}

    def test_string_with_trailing_comma_repaired(self) -> None:
        assert parse_arguments('{"a": 1,}') == {"a": 1}

    def test_plain_string_becomes_value(self) -> None:
        assert parse_arguments("just words") == {"value": "just words"}

    def test_non_json_object_returned_empty(self) -> None:
        assert parse_arguments(42) == {}


class TestRepairJson:
    def test_removes_stray_trailing_commas(self) -> None:
        assert repair_json('{"a": 1,}') == '{"a": 1}'
        assert repair_json('[1, 2,]') == '[1, 2]'

    def test_valid_json_untouched(self) -> None:
        assert repair_json('{"a": 1}') is None

    def test_unfixable_json_returns_none(self) -> None:
        assert repair_json("{a: 1,}") is None

    def test_empty_input_returns_none(self) -> None:
        assert repair_json("") is None