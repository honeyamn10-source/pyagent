"""Tests for the Tool decorator, schema reflection and registries."""

from __future__ import annotations

from typing import Optional

import pytest

from pyagent import Tool, ToolError
from pyagent.tools import ToolRegistry, ensure_tool, tool, tool_registry

from conftest import add, double, greet


class TestSchemaReflection:
    def test_int_str_bool_reflection(self) -> None:
        def mix(a: int, b: str, c: bool) -> None:
            """Mix types."""

        instance = Tool(mix)
        properties = instance.parameters["properties"]
        assert properties["a"]["type"] == "integer"
        assert properties["b"]["type"] == "string"
        assert properties["c"]["type"] == "boolean"
        assert instance.parameters["required"] == ["a", "b", "c"]
        assert instance.parameters["type"] == "object"

    def test_float_maps_to_number(self) -> None:
        instance = Tool(double)
        assert instance.parameters["properties"]["x"]["type"] == "number"

    def test_defaults_not_required(self) -> None:
        def f(x: int, y: str = "fallback") -> None:
            """Have a default."""

        instance = Tool(f)
        assert instance.parameters["required"] == ["x"]
        assert instance.parameters["properties"]["y"]["default"] == "fallback"

    def test_keyword_only_params_are_reflected(self) -> None:
        instance = Tool(greet)
        properties = instance.parameters["properties"]
        assert properties["name"]["type"] == "string"
        assert properties["enthusiastic"]["type"] == "boolean"
        assert instance.parameters["required"] == ["name"]
        assert properties["enthusiastic"]["default"] is True

    def test_optional_type_unwraps_to_inner(self) -> None:
        def f(x: Optional[int]) -> None:
            """May or may not be provided."""

        instance = Tool(f)
        assert instance.parameters["properties"]["x"]["type"] == "integer"

    def test_unannotated_param_defaults_to_string(self) -> None:
        def f(value) -> None:  # noqa: ANN001
            """No annotation."""

        instance = Tool(f)
        assert instance.parameters["properties"]["value"]["type"] == "string"

    def test_varargs_are_skipped(self) -> None:
        def f(x: int, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            """Variadic signature."""

        instance = Tool(f)
        assert list(instance.parameters["properties"]) == ["x"]

    def test_docstring_params_and_description(self) -> None:
        def f(a: int, b: int) -> int:
            """Add two numbers together with care.

            :param a: the first addend
            :param b: the second addend
            """

        instance = Tool(f)
        assert "Add two numbers" in instance.description
        assert instance.parameters["properties"]["a"]["description"] == "the first addend"
        assert instance.parameters["properties"]["b"]["description"] == "the second addend"

    def test_description_falls_back_to_name(self) -> None:
        instance = Tool(add)
        assert instance.description == "Add two integers together."

    def test_openai_schema_shape(self) -> None:
        instance = Tool(add)
        schema = instance.schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "add"
        assert schema["function"]["parameters"]["type"] == "object"
        assert set(schema["function"]["parameters"]["properties"]) == {"a", "b"}


class TestToolRun:
    def test_run_coerces_types(self) -> None:
        def f(x: int, y: float, z: bool) -> str:
            """Coerce test."""
            return f"{x}|{y}|{z}"

        instance = Tool(f)
        assert instance.run({"x": "7", "y": 2, "z": "true"}) == "7|2.0|True"

    def test_missing_required_argument_raises(self) -> None:
        instance = Tool(add)
        with pytest.raises(ToolError, match="missing required"):
            instance.run({"a": 1})

    def test_extra_arguments_are_ignored(self) -> None:
        def f(a: int) -> int:
            """Single arg."""
            return a

        instance = Tool(f)
        assert instance.run({"a": 3, "unexpected": 99}) == 3

    def test_invalid_value_type_raises_toolerror(self) -> None:
        instance = Tool(add)
        with pytest.raises(ToolError, match="invalid value"):
            instance.run({"a": "oops", "b": 1})

    def test_non_dict_arguments_raise(self) -> None:
        instance = Tool(add)
        with pytest.raises(ToolError):
            instance.run("not a dict")  # type: ignore[arg-type]

    def test_direct_call_invokes_function(self) -> None:
        instance = Tool(add)
        assert instance(2, 3) == 5
        assert instance(a=2, b=3) == 5

    def test_bool_string_variants(self) -> None:
        def f(on: bool) -> bool:
            """Flip a bool."""
            return on

        instance = Tool(f)
        assert instance.run({"on": "false"}) is False
        assert instance.run({"on": 1}) is True
        assert instance.run({"on": "yes"}) is True

    def test_function_errors_propagate(self) -> None:
        def f(a: int) -> int:
            """Raise on purpose."""
            raise ZeroDivisionError("bad")

        instance = Tool(f)
        with pytest.raises(ZeroDivisionError):
            instance.run({"a": 1})


class TestDecoratorAndRegistry:
    def test_decorator_registers_globally(self) -> None:
        @tool
        def triple(x: int) -> int:
            """Triple a number."""
            return x * 3

        assert tool_registry().get("triple").run({"x": 4}) == 12
        assert "triple" in tool_registry().names()

    def test_decorator_with_overrides(self) -> None:
        @tool(name="exp2", description="The square function")
        def power_of_two(n: int) -> int:
            """Docstring not used."""
            return n * n

        registered = tool_registry().get("exp2")
        assert registered.name == "exp2"
        assert registered.description == "The square function"
        assert registered.run({"n": 5}) == 25

    def test_registry_duplicate_name_overwrites(self) -> None:
        def dup(x: int) -> int:
            """First version."""
            return 1

        registry = tool_registry()
        registry.add(dup)
        assert registry.get("dup").run({"x": 1}) == 1

        def dup(x: int) -> int:  # noqa: F811
            """Second version."""
            return 2

        registry.add(dup)
        assert registry.get("dup").run({"x": 1}) == 2
        assert len(registry) == 1

    def test_registry_contains_names_and_length(self) -> None:
        registry = ToolRegistry()
        registry.add(add)
        assert "add" in registry
        assert registry.get("missing") is None
        assert registry.names() == ["add"]
        assert len(registry) == 1

    def test_registry_constructed_with_tools(self) -> None:
        registry = ToolRegistry([add, double])
        assert set(registry.names()) == {"add", "double"}
        assert len(list(registry)) == 2

    def test_registry_remove_and_clear(self) -> None:
        registry = ToolRegistry([add, double])
        assert registry.remove("add") is True
        assert registry.remove("missing") is False
        registry.clear()
        assert len(registry) == 0

    def test_tool_registry_returns_global(self) -> None:
        assert isinstance(tool_registry(), ToolRegistry)

    def test_ensure_tool_coerces_callables(self) -> None:
        instance = ensure_tool(add)
        assert instance.name == "add"
        assert isinstance(ensure_tool(instance), Tool)
        with pytest.raises(TypeError):
            ensure_tool("not callable")

    def test_tool_requires_callable(self) -> None:
        with pytest.raises(TypeError):
            Tool("nope")  # type: ignore[arg-type]

    def test_tool_repr(self) -> None:
        assert "add" in repr(Tool(add))

    def test_greet_default_keyword_argument(self) -> None:
        instance = Tool(greet)
        assert instance.run({"name": "Bo"}) == "Hello, Bo!"
        assert instance.run({"name": "Bo", "enthusiastic": "false"}) == "Hi, Bo."