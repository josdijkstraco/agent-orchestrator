# agents

Agent definitions, one YAML per file. `harness.py:load_agent()` scans this directory and matches on the `name:` field (not the filename). Each agent declares a system `prompt`, a list of built-in `tools` (from `tools.py`), and optionally `skills`, `mcp` servers (from `.mcp.json`), a `model` override, and a structured `output` schema (which triggers the injected `submit_result` tool).

## Code in this directory

- `coordinator.yaml` — delegates a task across planner → implementer → reviewer; watches for `[AGENT_ERROR]`
- `planner.yaml` / `planner_no_ask.yaml` — produce a step-by-step plan (the `_no_ask` variant never calls `ask_user`)
- `implementer.yaml` — writes code and runs checks (read/write/bash/find); model `qwen/qwen3.6-plus`
- `reviewer.yaml` — returns structured `APPROVED`/`REJECTED` + feedback; model `z-ai/glm-5.1`
- `brainstormer.yaml` — turns ideas into specs using the `brainstorming` skill
- `requirements.yaml` — reviews software requirements for granularity/completeness/overlap gaps
- `judge.yaml` — compares competing plans (used by the multi-planner workflow)
- `linear.yaml` — Linear MCP assistant; `github.yaml` — opens PRs via the GitHub MCP
- `random.yaml` — picks `ONE`/`TWO` via `submit_result`; `hello.yaml` — emits "Hello world" (both for demo/test workflows)
