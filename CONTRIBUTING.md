# Contributing to pyagent

Thanks for your interest in `pyagent`. This project wants to stay small,
readable and dependency-free, and contributions that respect that goal are
very welcome.

## Ground rules

- **Zero dependency guarantee.** The package may only use the Python standard
  library. A pull request that adds a third-party `import` anywhere under
  `pyagent/` will be declined.
- **Tests must pass.** Every change lands with (or updates) tests in
  `tests/`. Run them before submitting:
  ```bash
  python3 -m pytest tests -q
  ```
- **Style.** The codebase is black-formatted, fully type-hinted and uses
  Google-style docstrings on every public function and class. No inline
  comments.
- **Be minimal.** When two designs both work, prefer the one with fewer lines
  and fewer public names. `pyagent` wins by being *small*.

## Getting started

```bash
git clone git@github.com:honeyamn10-source/pyagent.git
cd pyagent
python3 -m pip install -e .
python3 -m pytest tests -q
```

## How to contribute

1. Fork the repository and create a feature branch.
2. Make your change with tests.
3. Run the full suite and confirm it is green.
4. Open a pull request using the pull request template.
5. Mention the problem your change solves and, if applicable, the issue it
   closes.

## Areas that need help

Check the [Roadmap](README.md#roadmap) and open issues. Good first tasks are
usually: more tolerant parsing edge cases, additional `Memory` helpers, and
tests for edge conditions we have not covered yet.

Conventional commit messages are appreciated but not required.

## Code of conduct

By participating you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).