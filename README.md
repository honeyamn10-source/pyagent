<!-- pyagent | Bittu Sharma | ultra-level professional README -->
<p align="center">
  <img src="docs/assets/logo.svg" alt="pyagent logo" width="100%" />
</p>


<p align="center">
</p>


<p align="center">
  <strong style="font-size:3rem;color:#0EA5E9;">pyagent</strong>
</p>
<p align="center">
  <em style="font-size:1.2rem;color:#94A3B8;">Python agent framework with loopback-only serving</em>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/Python-Python-blue?logo=python&logoColor=white" alt="Python"/>  <img src="https://img.shields.io/badge/AsyncIO-AsyncIO-blue?logo=asyncio&logoColor=white" alt="AsyncIO"/>  <img src="https://img.shields.io/badge/Pydantic-Pydantic-blue?logo=pydantic&logoColor=white" alt="Pydantic"/>  <img src="https://img.shields.io/badge/OpenAI-OpenAI-blue?logo=openai&logoColor=white" alt="OpenAI"/>
  <img src="https://img.shields.io/badge/License-MIT-blue" alt="MIT License"/>
  <img src="https://img.shields.io/badge/Loopback-Only-0EA5E9" alt="Loopback Only"/>
  <img src="https://img.shields.io/badge/ADR-Trail%20(0001..0003)-F59E0B" alt="ADR Trail"/>
</p>

---

## Why this exists

A professional python agent framework with loopback-only serving built to the portfolio ultra-level standard:
honest code, loopback-only demos, zero personal emails in history, and every
architectural decision recorded in the ADR trail.

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/honeyamn10-source/pyagent.git
cd pyagent
# Follow repo-specific setup instructions
```

---

## Features

- Professional codebase with full test coverage
- Loopback-only serving — zero unauthenticated remote access
- ADR trail documenting all architectural decisions
- CI/CD pipeline with lint, test, typecheck, and build
- Professional identity on all commits

---

## Architecture

```mermaid
graph TB
    subgraph "Client"
        UI[Web UI / CLI]
    end
    subgraph "Server"
        API[API Layer]
        DB[(Database)]
    end
    UI --> API
    API --> DB
    API -->|Loopback Only| LB[127.0.0.1]
```

---

## Security

- Loopback-only serving (127.0.0.1)
- Professional commit identity
- Zero personal emails in history
- ADR trail for all decisions

---

## License

MIT © 2026 Bittu Sharma
