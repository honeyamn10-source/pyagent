# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Async agent loop.
- Parallel tool-call execution.
- Token streaming callbacks.

## [0.1.0] - 2026-09-15

### Added

- `Agent` with the core loop: user message, model call, tool execution,
  repeat, final plain-text answer.
- `@tool` decorator with signature/docstring reflection into JSON schemas
  for `bool`, `int`, `float` and `str` parameters.
- Global `ToolRegistry` exposed through `tool_registry()`.
- `BaseLLM` ABC and `OpenAICompatibleClient` built purely on `urllib`,
  compatible with OpenAI and any OpenAI-compatible endpoint.
- `format_tool_instructions()` helper that renders tool lists into a system
  prompt.
- Robust tool-call parsing (`parse.py`): prose, fenced code blocks, stray
  trailing commas, OpenAI-style `tool_calls` and string-encoded arguments.
- `Memory` with system injection, `+`/`+=` appending, trimming, JSON
  (de)serialization and a conservative token estimator.
- `RetryPolicy` + `retry()` helper and `Hooks` callbacks
  (`on_step`, `on_message`).
- Example agent (`examples/calculator_agent.py`) with a safe, eval-free
  arithmetic calculator and a `get_time` tool.
- Full test suite (25+ tests) runnable offline against a deterministic
  `FakeLLM` and a stubbed HTTP transport.

[Unreleased]: https://github.com/honeyamn10-source/pyagent/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/honeyamn10-source/pyagent/releases/tag/v0.1.0