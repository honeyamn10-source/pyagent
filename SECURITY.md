# Security Policy

## Supported versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a vulnerability

Please do **not** open a public GitHub issue for security problems. Instead,
report them privately via GitHub's
[Security Advisories](https://github.com/honeyamn10-source/pyagent/security/advisories/new).

We aim to acknowledge reports within 48 hours and to ship a fix for
high-severity issues within a reasonable release cycle. Once a fix is
released, a security advisory is published with details and credits.

## What to include

- The version(s) affected.
- A minimal reproducer.
- The impact you observed.
- Any suggested remediation, if you have one.

## Notes

- The zero-dependency guarantee means the attack surface is small: there is
  no third-party code between your code and `pyagent`.
- The example calculator is intentionally dependency-free and eval-free; any
  tool that executes untrusted input should validate it rigorously on its own.