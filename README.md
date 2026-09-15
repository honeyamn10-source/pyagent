# pyagent

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/github/license/honeyamn10-source/pyagent)
![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Stars](https://img.shields.io/github/stars/honeyamn10-source/pyagent)
![Forks](https://img.shields.io/github/forks/honeyamn10-source/pyagent)

**A zero-dependency Python framework for building LLM agents.**

`pyagent` is the smallest agent loop you will actually want to use. One user
message in, one final plain-text answer out — with a pure-Python tool loop in
between that is easy to read, easy to test, and easy to ship. It is built on
the **standard library only**: no `requests`, no `pydantic`, no `httpx`, no
anything. If you can run Python, you can run `pyagent`.

## What it does

```
user message → model → tool call → execute tool → repeat → final answer
```

The framework wires this loop for you:

1. **`@tool`** — decorate a function, get a JSON schema for free (reflected
   from the signature and docstring).
2. **`Agent`** — owns the loop, the memory, the step cap and the hooks.
3. **`OpenAICompatibleClient`** — talks to any OpenAI-compatible endpoint
   with nothing but `urllib`.
4. **`parse`** — pulls a tool call out of messy model output. Prose around
   the JSON, fenced code blocks, a stray trailing comma: all handled.
5. **`Memory`** — trimming, JSON persistence and a token estimator.

## Quick start

```python
from pyagent import Agent, OpenAICompatibleClient, tool

@tool
def add(a: int, b: int) -> int:
    """Add two integers together."""
    return a + b

client = OpenAICompatibleClient(api_key="sk-...", model="gpt-4o-mini")
agent = Agent(llm=client, tools=[add], system="You are a math agent.")
print(agent.run("What is 2 + 3?"))
```

Run a complete, working example (calculator + clock tools) against any
OpenAI-compatible endpoint:

```bash
OPENAI_API_KEY=sk-... python examples/calculator_agent.py
```

## Table of contents

- [Why zero-dependency](#why-zero-dependency)
- [How it works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

## Why zero-dependency

Most agent frameworks pull in hundreds of dependencies before you have
written a single line of your own logic. `pyagent` takes the opposite
position:

- **Total transparency.** `pyagent/` is a handful of small, readable modules.
  Read the whole agent loop in an afternoon.
- **No supply chain, no lockfile rot.** A dependency that pins the world ten
  years from now is a problem you never have.
- **Runs everywhere.** CI containers, offline boxes, edge functions — if the
  standard library is there, `pyagent` is there.
- **Swap anything.** Because the client is a 60-line ABC, swapping in your
  own transport or a different model API is a small, contained change.

The trade-off is deliberate: `pyagent` does one thing — the agent loop — and
does it with zero ceremony.

## How it works

```
        user                       tools
         │                           ▲
         ▼                           │
┌───────────────────┐   result   ┌───┴──────────────┐
│      Agent        │───────────▶│  ToolRegistry    │
│  run() loop       │            │  @tool functions │
└─────────┬─────────┘            └──────────────────┘
          │ messages
          ▼
┌──────────────────────────┐
│      BaseLLM (ABC)       │
│   OpenAICompatibleClient │   ← urllib, standard library only
└────────────┬─────────────┘
             │ raw output
             ▼
┌──────────────────────────┐
│  pyagent.parse           │   tolerates prose, fences, trailing commas
│  extract_tool_call()     │
└──────────────────────────┘
```

The agent keeps everything in a `Memory` (a plain list of
`{"role": ..., "content": ...}` dicts). Each iteration feeds the whole
history to the model. If the reply carries a tool call, `execute_tool` runs
it and the result comes back as a `tool` role message. The loop stops the
moment the model answers in plain text — or raises
`MaxStepsExceeded` when the step budget is used up.

## Installation

```bash
pip install pyagent        # once published to PyPI
# …or, right now, straight from this repo:
pip install -e .
```

Because `pyagent` is pure standard library, you can also vendor the
`pyagent/` folder into your project and `import pyagent` with zero setup.

## Usage

### Tools

```python
from pyagent import tool, tool_registry

@tool
def get_weather(city: str, units: str = "celsius") -> str:
    """Fetch the current weather for a city.

    :param city: The city to look up.
    :param units: "celsius" or "fahrenheit".
    """
    return f"It is 21° {units} in {city}."
```

Schema is reflected automatically — `int`/`float`/`str`/`bool` map to
`integer`/`number`/`string`/`boolean`, parameters with defaults are optional,
and `:param name:` docstring lines become field descriptions. List every
registered tool with `tool_registry()`.

### Agent

```python
from pyagent import Agent, OpenAICompatibleClient, Memory

client = OpenAICompatibleClient(
    api_key="sk-...", model="gpt-4o-mini",
    temperature=0.2, max_tokens=512,
)

agent = Agent(
    llm=client,
    tools=[get_weather],
    system="You are a helpful weather agent.",
    memory=Memory(system="Be terse."),
    max_steps=6,
    hooks={"on_step": lambda step, msgs, call: print("step", step, call.name)},
)

print(agent.run("Is it warm in Lisbon right now?"))
```

`hooks` accepts a `Hooks` instance, a dict, or nothing. `on_step` fires after
each executed tool call; `on_message` fires for every model reply — handy for
logs, dashboards, or streaming UIs.

### Retries

```python
from pyagent import RetryPolicy, retry

run = retry(RetryPolicy(retries=3, delay=0.5, backoff=2.0))(agent.run)
answer = run("What is the capital of Bolivia?")
```

### Memory

```python
memory = Memory(system="sys")            # inject a system message
memory += {"role": "user", "content": "hi"}
memory.trim(20)                          # keep system + last 20 messages
blob = memory.to_json()                  # persist
restored = Memory.from_json(blob)        # restore
memory.estimate_tokens()                 # cheap heuristic for cost planning
```

## Testing

```bash
python3 -m pytest tests -q
```

The suite is fully offline: a deterministic `FakeLLM` scripts the model's
responses and the HTTP client is tested against a stubbed transport, so no
network or API key is ever needed.

## Roadmap

- **Async support** — an `async` agent loop for high-concurrency servers.
- **Parallel tool calls** — batch `tool_calls` in a single step.
- **Streaming** — token-by-token `on_message` delivery.
- **More clients** — Anyscale, Together, and a native Anthropic ABC adapter,
  all still standard-library only.
- **Structured outputs** — optional schema validation of final answers.
- **1.0** — a stabilized public API and a published PyPI release.

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md)
first, and note that the zero-dependency guarantee is a hard constraint:
pull requests adding third-party imports will not be accepted. See the
[Contributor Covenant][coc] for our code of conduct.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Bittu Sharma.

[coc]: CODE_OF_CONDUCT.md