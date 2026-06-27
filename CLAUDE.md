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

The code is split into small, single-purpose modules. Most are pure and unit-tested in isolation; `harness.py` is a thin CLI that wires them together and re-exports their public names for back-compat.

**CLI & entry points**
- `harness.py` — argparse CLI: `workflow` / `agent` / `trace` / `replay`. Builds the parser, dispatches, and re-exports `load_workflow`, `load_agent`, `run_pipeline`, etc.
- `main.py` — interactive single-agent REPL (general coding agent with all skills + MCP servers).
- `repl.py` — shared REPL helpers (`select_model`, `run_cancellable`, `read_command`) used by both entry points.

**Loading & config**
- `workflow.py` — `load_workflow()` + `normalize_step()`: resolve a workflow YAML by `name:` and validate every step (unique ids, backward-only loops, known references, schema-consistent conditions) at load time.
- `agent_loader.py` — `load_agent()`: resolve an agent YAML by `name:`, mapping tool/skill/MCP names and appending the environment footer.

**Execution**
- `pipeline.py` — `run_pipeline()` via a small-method `_PipelineRun`: resolve input → run agent → interpret result (`stop_on`/STOP/`when`/`loop_on`). `dry_run_pipeline()` previews config without API calls.
- `agent_openrouter.py` — `agent_loop()`: drive one agent turn-by-turn over OpenRouter (stream → run tools → feed back), intercepting the injected `submit_result` tool for structured results. Retries on 429/529.
- `streaming.py` — `accumulate_stream()`: pure folding of SSE chunk deltas into text + tool calls + usage (no I/O, fully unit-tested).

**Pure helpers**
- `conditions.py` — control-flow predicates (`eval_condition`, `token_present`, `is_stop_signal`, `when_skips`).
- `step_io.py` — input assembly / output extraction / artifact parsing / `submit_result` schema.
- `display.py` — terminal headers and usage formatting.

**Infrastructure**
- `tools.py` — built-in `read_file`, `write_file`, `bash`, `find_files`, `ask_user`, all sandboxed to the repo root.
- `mcp_client.py` — MCP over JSON-RPC subprocess stdio; servers registered in `.mcp.json`, referenced per-agent.
- `skills_loader.py` — `skills/<name>/SKILL.md` plugins appended to an agent's prompt by name.
- `tracing.py` + `langfuse_client.py` — every pipeline run is saved to `traces/` as JSON with per-step message snapshots for replay; optional Langfuse spans.

**Conventions**
- Keep functions small and single-purpose; put pure logic in its own module with direct unit tests.
- A function's dependencies are patched in tests at the module that *calls* them (e.g. `run_pipeline`'s deps are patched as `pipeline.agent_loop` / `pipeline.load_agent`, since `run_pipeline` lives in `pipeline.py`).
- Tool/agent errors surface as `Error:` / `[AGENT_ERROR]` strings; agents (e.g. `coordinator`) are prompted to halt on `[AGENT_ERROR]`.

## Subdirectories

- `agents/` — agent definitions (YAML, one per file)
- `workflows/` — workflow pipeline definitions (YAML)
- `skills/` — SKILL.md plugins appended to agent prompts
- `tests/` — pytest suite
- `docs/` — design specs and implementation plans

Every source directory has a CLAUDE.md summarizing its files and subdirectories — when looking for functionality, consult the nearest CLAUDE.md before searching file contents.
