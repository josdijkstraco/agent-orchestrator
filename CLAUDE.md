# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A multi-agent coding harness that orchestrates LLM agents (Claude and others via OpenRouter) in configurable pipelines. Agents and workflows are defined as YAML — `harness.py` chains agents into a pipeline, passing each step's output to the next, with structured outputs, loop-back, and conditional steps. Python 3.12, managed with `uv`. See `README.md` for the full agent/workflow YAML schema.

## Commands

```bash
uv sync                                            # install deps

python main.py                                     # interactive single-agent REPL
python harness.py agent <name> ["<prompt>"]        # run one agent (one-shot or REPL)
python harness.py workflow <name> "<task>"         # run a workflow pipeline
python harness.py workflow <name> --dry-run        # preview resolved pipeline, no API calls
python harness.py trace list | show <id>           # inspect saved traces
python harness.py replay <id> --from-step <N>      # resume a pipeline from a step

pytest                                             # run the test suite
pytest tests/test_harness.py::test_name -v         # run a single test
```

Requires `OPENROUTER_API_KEY` in `.env`. Optional `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` enable Langfuse tracing.

## Architecture

- **Agent loop** (`agent_openrouter.py`): `agent_loop()` streams an OpenRouter chat completion, executes returned tool calls via `execute_tool()`, feeds results back, and retries on 429/529. Intercepts the injected `submit_result` tool to return a structured result. This is the active loop; `agent.py` is a standalone Anthropic-API reference loop not wired into the pipeline.
- **Pipeline executor + CLI** (`harness.py`): `load_workflow()`/`load_agent()` resolve YAML by `name:` field; `run_pipeline()` runs steps in sequence, resolving each step's `inputs:`, evaluating `when`/`stop_on`/`loop_on` conditions, extracting `outputs:` artifacts, and tracing every step.
- **Tools** (`tools.py`): built-in `read_file`, `write_file`, `bash`, `find_files`, `ask_user`, all sandboxed to the repo root. **MCP** (`mcp_client.py`): JSON-RPC over subprocess stdio; servers registered in `.mcp.json`, referenced per-agent.
- **Skills** (`skills_loader.py`): `skills/<name>/SKILL.md` plugins appended to an agent's prompt by name.
- **Tracing** (`trace.py`, `langfuse_client.py`): every pipeline run is saved to `traces/` as JSON with per-step message snapshots for replay.
- **Error flow**: tool exceptions are returned as `Error:`/`[AGENT_ERROR]` strings; agents (e.g. `coordinator`) are prompted to halt on `[AGENT_ERROR]`.

## Subdirectories

- `agents/` — agent definitions (YAML, one per file)
- `workflows/` — workflow pipeline definitions (YAML)
- `skills/` — SKILL.md plugins appended to agent prompts
- `tests/` — pytest suite
- `docs/` — design specs and implementation plans

Every source directory has a CLAUDE.md summarizing its files and subdirectories — when looking for functionality, consult the nearest CLAUDE.md before searching file contents.
