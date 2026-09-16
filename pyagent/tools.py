"""Tool primitives: the :func:`tool` decorator and tool registries.

The decorator reflects a function signature and docstring into a JSON schema
and registers the result in a module-level :class:`ToolRegistry` exposed
through :func:`tool_registry`.
"""

from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Pattern

from pyagent.errors import ToolError

_TYPE_MAP: Dict[str, str] = {
    "bool": "boolean",
    "boolean": "boolean",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "str": "string",
    "string": "string",
}

_PARAM_RE: Pattern[str] = re.compile(r":param\s+(?:([\w.]+)\s+)?(\w+)\s*:\s*(.*)")


@dataclass
class _DocstringInfo:
    """Parsed docstring: a free-form description and per-parameter notes."""

    description: str = ""
    params: Dict[str, str] = field(default_factory=dict)


def _parse_docstring(doc: Optional[str]) -> _DocstringInfo:
    """Extract the description and ``:param name:`` notes from a docstring.

    Args:
        doc: The raw docstring, or None.

    Returns:
        A :class:`_DocstringInfo` with the prose description and parameter
        descriptions parsed from ``:param x: text`` style lines.
    """
    if not doc:
        return _DocstringInfo()
    description_parts: List[str] = []
    params: Dict[str, str] = {}
    for raw_line in doc.strip().splitlines():
        line = raw_line.strip()
        match = _PARAM_RE.match(line)
        if match:
            params[match.group(2)] = match.group(3).strip()
        elif line.startswith(":param") or line.startswith(":return") or line.startswith(":raises"):
            continue
        else:
            description_parts.append(line)
    return _DocstringInfo(description="\n".join(description_parts).strip(), params=params)


def _json_type(annotation: Any) -> str:
    """Map a Python type annotation to a JSON schema type name.

    Handles real types, string annotations produced by ``from __future__
    import annotations`` and common typing generics such as
    ``Optional[int]`` by unwrapping to the innermost primitive.

    Args:
        annotation: The parameter annotation as exposed by ``inspect``.

    Returns:
        One of ``"string"``, ``"integer"``, ``"number"`` or ``"boolean"``.
    """
    if annotation is inspect.Parameter.empty:
        return "string"
    if isinstance(annotation, type):
        name = annotation.__name__.lower()
        return _TYPE_MAP.get(name, "string")
    name = str(annotation).lower()
    if name in _TYPE_MAP:
        return _TYPE_MAP[name]
    inner = re.search(r"\[([^,\[\]]+)\]", name)
    if inner and inner.group(1) in _TYPE_MAP:
        return _TYPE_MAP[inner.group(1)]
    return "string"


def _reflect_parameters(
    signature: inspect.Signature,
    param_docs: Dict[str, str],
) -> Dict[str, Any]:
    """Build a JSON schema object from an ``inspect.Signature``.

    Args:
        signature: The function signature being reflected.
        param_docs: Parameter descriptions parsed from the docstring.

    Returns:
        A JSON schema of the form ``{"type": "object", "properties": ...,
        "required": [...]}``.
    """
    properties: Dict[str, Any] = {}
    required: List[str] = []
    for param_name, parameter in signature.parameters.items():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        spec: Dict[str, Any] = {"type": _json_type(parameter.annotation)}
        if param_name in param_docs and param_docs[param_name]:
            spec["description"] = param_docs[param_name]
        if parameter.default is not inspect.Parameter.empty:
            spec["default"] = _json_default(parameter.default)
        else:
            required.append(param_name)
        properties[param_name] = spec
    return {"type": "object", "properties": properties, "required": required}


def _json_default(value: Any) -> Any:
    """Convert a parameter default into a JSON-serializable value.

    Args:
        value: The default value.

    Returns:
        The value when it is a JSON primitive, otherwise its string form.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _coerce(value: Any, json_type: str, name: str, tool_name: str) -> Any:
    """Coerce a raw argument value into the expected JSON schema type.

    Args:
        value: The value supplied by the model.
        json_type: The JSON schema type of the parameter.
        name: The parameter name, used in error messages.
        tool_name: The tool name, used in error messages.

    Returns:
        The coerced value.

    Raises:
        ToolError: When the value cannot be coerced.
    """
    try:
        if json_type == "integer":
            if isinstance(value, bool):
                raise ValueError(value)
            return int(value)
        if json_type == "number":
            return float(value)
        if json_type == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in ("true", "1", "yes", "y"):
                    return True
                if lowered in ("false", "0", "no", "n"):
                    return False
            if isinstance(value, (int, float)):
                return bool(value)
            raise ValueError(value)
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ToolError(
            f"invalid value {value!r} for argument {name!r} (expected {json_type})",
            tool_name=tool_name,
        ) from exc


class Tool:
    """A single callable tool exposed to the agent."""

    def __init__(
        self,
        func: Callable[..., Any],
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """Wrap a callable into a Tool with a reflected JSON schema.

        Args:
            func: The function backing the tool.
            name: Optional name override, defaults to the function name.
            description: Optional description override, defaults to the
                first paragraph of the function docstring.
        """
        if not callable(func):
            raise TypeError("a Tool requires a callable")
        self.func = func
        self.name = name or func.__name__
        doc = _parse_docstring(getattr(func, "__doc__", None))
        self.description = description or doc.description or self.name
        self._param_docs = doc.params
        self._schema_dict = _reflect_parameters(inspect.signature(func), doc.params)

    @property
    def parameters(self) -> Dict[str, Any]:
        """The JSON schema object for the tool arguments.

        Returns:
            A dict with ``type``, ``properties`` and ``required`` keys.
        """
        return {
            "type": "object",
            "properties": dict(self._schema_dict["properties"]),
            "required": list(self._schema_dict["required"]),
        }

    def schema(self) -> Dict[str, Any]:
        """The full OpenAI-style tool schema.

        Returns:
            A dict shaped like ``{"type": "function", "function": {...}}``.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def run(self, arguments: Optional[Dict[str, Any]] = None) -> Any:
        """Execute the tool with validated, coerced arguments.

        Extra keys in ``arguments`` are ignored; missing required keys or
        values that cannot be coerced raise :class:`ToolError`.

        Args:
            arguments: A mapping of argument names to values.

        Returns:
            Whatever the underlying function returns.

        Raises:
            ToolError: When arguments are invalid or a required argument is
                missing.
        """
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise ToolError("tool arguments must be a JSON object", tool_name=self.name)
        kwargs: Dict[str, Any] = {}
        for param_name, spec in self._schema_dict["properties"].items():
            if param_name not in arguments:
                continue
            kwargs[param_name] = _coerce(
                arguments[param_name],
                spec["type"],
                param_name,
                self.name,
            )
        missing = [param for param in self._schema_dict["required"] if param not in arguments]
        if missing:
            raise ToolError(
                f"missing required argument(s): {', '.join(missing)}",
                tool_name=self.name,
            )
        return self.func(**kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the underlying function directly, bypassing validation.

        Args:
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            Whatever the underlying function returns.
        """
        return self.func(*args, **kwargs)

    def __repr__(self) -> str:
        """A compact developer-facing description.

        Returns:
            A string like ``Tool(name='add')``.
        """
        return f"Tool(name={self.name!r})"


class ToolRegistry:
    """A named collection of Tools."""

    def __init__(self, tools: Optional[Iterable[Any]] = None) -> None:
        """Initialize an empty registry and optionally add tools.

        Args:
            tools: An iterable of Tools or callables to register.
        """
        self._tools: Dict[str, Tool] = {}
        for item in tools or ():
            self.add(item)

    def add(self, item: Any) -> Tool:
        """Register a Tool or callable under its name.

        Registering a second tool with the same name replaces the first.

        Args:
            item: A :class:`Tool` or any callable.

        Returns:
            The registered Tool instance.
        """
        instance = ensure_tool(item)
        self._tools[instance.name] = instance
        return instance

    def get(self, name: str) -> Optional[Tool]:
        """Look up a tool by name.

        Args:
            name: The tool name.

        Returns:
            The Tool, or None when no such tool is registered.
        """
        return self._tools.get(name)

    def remove(self, name: str) -> bool:
        """Remove a tool by name.

        Args:
            name: The tool name.

        Returns:
            True when a tool was removed, False when it did not exist.
        """
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def clear(self) -> None:
        """Remove every tool from the registry."""
        self._tools.clear()

    def all(self) -> Dict[str, Tool]:
        """A snapshot of all registered tools.

        Returns:
            A mapping of tool names to Tool instances.
        """
        return dict(self._tools)

    def names(self) -> List[str]:
        """The names of every registered tool.

        Returns:
            A list of tool names in registration order.
        """
        return list(self._tools)

    def schemas(self) -> List[Dict[str, Any]]:
        """The OpenAI-style schema for every registered tool.

        Returns:
            A list of schema dicts.
        """
        return [tool.schema() for tool in self._tools.values()]

    def __contains__(self, name: str) -> bool:
        """Whether a tool with the given name is registered.

        Args:
            name: The tool name.

        Returns:
            True when present.
        """
        return name in self._tools

    def __iter__(self) -> Iterable[Tool]:
        """Iterate over the registered Tool instances.

        Yields:
            Each Tool in the registry.
        """
        return iter(self._tools.values())

    def __len__(self) -> int:
        """The number of registered tools.

        Returns:
            The tool count.
        """
        return len(self._tools)

    def __repr__(self) -> str:
        """A compact developer-facing description.

        Returns:
            A string like ``ToolRegistry(3 tools)``.
        """
        return f"ToolRegistry({len(self._tools)} tools)"


_REGISTRY = ToolRegistry()


def ensure_tool(item: Any) -> Tool:
    """Coerce a callable or Tool into a Tool instance.

    Args:
        item: The value to coerce.

    Returns:
        A Tool wrapping the item.

    Raises:
        TypeError: When the item is neither a Tool nor callable.
    """
    if isinstance(item, Tool):
        return item
    if callable(item):
        return Tool(item)
    raise TypeError("expected a Tool or a callable, got " + type(item).__name__)


def tool(
    func: Optional[Callable[..., Any]] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Any:
    """Decorate a function to turn it into a registered Tool.

    The tool is registered in the global :func:`tool_registry` under its
    name and the decorator returns the :class:`Tool` instance.

    Args:
        func: The function to wrap, used when the decorator is applied
            directly as ``@tool``.
        name: Optional name override.
        description: Optional description override.

    Returns:
        The decorated function's Tool instance when this decorator was
        applied directly, otherwise a decorator that takes a function.
    """

    def _wrap(fn: Callable[..., Any]) -> Tool:
        instance = Tool(fn, name=name, description=description)
        _REGISTRY.add(instance)
        return instance

    if func is None:
        return _wrap
    return _wrap(func)


def tool_registry() -> ToolRegistry:
    """Return the global tool registry.

    Returns:
        The module-level :class:`ToolRegistry` that the :func:`tool`
        decorator populates.
    """
    return _REGISTRY


__all__ = [
    "Tool",
    "ToolRegistry",
    "ensure_tool",
    "tool",
    "tool_registry",
]