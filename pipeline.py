"""Workflow pipeline execution.

`run_pipeline()` runs a list of validated steps in sequence: it resolves each
step's input, runs the agent, then interprets the result for control flow —
`stop_on`/STOP to end early, `when` to skip, `loop_on`/`loop_to` to retry an
earlier step. The per-step logic lives on `_PipelineRun` as small methods so
each phase (input, execution, result handling, looping) can be read in isolation.
Every step is traced; the whole run is wrapped in an optional Langfuse span.
"""

import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from agent_loader import AgentConfig, load_agent
from agent_openrouter import MODEL as DEFAULT_MODEL, agent_loop
from conditions import eval_condition, is_stop_signal, token_present, when_skips
from display import print_agent_header, usage_totals
from langfuse_client import get_langfuse, lf_shutdown, null_ctx
from mcp_client import build_mcp_clients
from step_io import (
    INPUT_KEY,
    build_submit_result_tool,
    extract_output,
    parse_artifacts,
    resolve_input,
    substitute_prompt_vars,
)
from tracing import Trace, _preview
from workflow import DEFAULT_MAX_LOOPS, StepConfig


@contextmanager
def _mcp_session(mcp_names: list[str]) -> Iterator[list]:
    """Start MCP clients for the given server names and guarantee they're closed."""
    clients = build_mcp_clients(mcp_names) if mcp_names else []
    try:
        yield clients
    finally:
        for client in clients:
            client.close()


def _submit_schema(agent: AgentConfig, step: StepConfig) -> tuple[dict | None, dict | None]:
    """Merge the agent's canonical output schema with the step override.

    Returns (submit_result_tool, effective_schema); both None when no schema applies.
    """
    merged = {**(agent.get("output") or {}), **(step.get("output") or {})} or None
    return (build_submit_result_tool(merged) if merged else None), merged


class _PipelineRun:
    """Mutable state and per-step logic for a single `run_pipeline` invocation."""

    def __init__(self, steps: list[StepConfig], command: str, traces_dir: str | Path, lf, trace: Trace) -> None:
        self.steps = steps
        self.traces_dir = traces_dir
        self.lf = lf
        self.trace = trace
        self.step_index_map = {s["id"]: i for i, s in enumerate(steps)}
        self.agent_cache: dict[str, AgentConfig] = {}
        self.loop_counts: dict[str, int] = {}
        self.step_outputs: dict[str, str] = {INPUT_KEY: command}
        self.step_index = 0
        self.total_in = 0
        self.total_out = 0
        self.total_cost = 0.0

    # ── helpers ──────────────────────────────────────────────────────────────

    def print_totals(self) -> None:
        print(f"\n[total usage]  in={self.total_in:,}  out={self.total_out:,}  cost=${self.total_cost:.4f}")

    def _agent_for(self, step: StepConfig, step_id: str) -> AgentConfig:
        if step_id not in self.agent_cache:
            self.agent_cache[step_id] = load_agent(step["agent"])
        return self.agent_cache[step_id]

    def _effective_input(self, step: StepConfig) -> str:
        """The step's input: prior outputs resolved, with the step prompt prepended."""
        step_prompt = step.get("prompt")
        if step_prompt:
            step_prompt = substitute_prompt_vars(step_prompt, self.step_outputs)
        current_input = resolve_input(step.get("inputs") or [], self.step_outputs)
        return f"{step_prompt}\n\n{current_input}" if step_prompt else current_input

    # ── execution ────────────────────────────────────────────────────────────

    def _run_agent(self, step: StepConfig, step_id: str, step_label: str,
                   agent: AgentConfig, model: str, effective_input: str) -> tuple[dict, list, dict | None]:
        """Print the step header, run the agent, and return (usage, messages, effective_schema)."""
        print_agent_header(step["agent"], agent, leading_newline=True)
        print("\033[36m--- system prompt ---\033[0m")
        print(f"\033[2m{agent['prompt']}\033[0m")
        print("\033[36m--- user prompt ---\033[0m")
        print(f"\033[2m{effective_input}\033[0m")
        print("\033[36m--- response ---\033[0m")

        self.trace.log(step=step_label, event="step_start", model=model,
                       tools=agent["tool_names"], prompt_preview=_preview(effective_input),
                       system_prompt=agent["prompt"], user_prompt=effective_input)

        submit_schema, effective_schema = _submit_schema(agent, step)
        messages: list = [{"role": "system", "content": agent["prompt"]}]
        step_ctx = (self.lf.start_as_current_observation(
            name=step_label, as_type="agent",
            input={"prompt": effective_input, "model": model, "tools": agent["tool_names"]},
            metadata={"step_id": step_id},
        ) if self.lf else null_ctx())
        with step_ctx, _mcp_session(agent["mcp_names"]) as mcp_clients:
            usage = agent_loop(effective_input, messages, model=model, tools=agent["tools"],
                               mcp_clients=mcp_clients, trace=self.trace, step_label=step_label,
                               submit_result_schema=submit_schema)
        return usage, messages, effective_schema

    def _run_one(self, step: StepConfig) -> str:
        """Run a single step. Returns 'halt' to end the pipeline, else 'advance'/'loop'."""
        step_id = step.get("id") or step["agent"]
        step_label = f"{self.step_index}:{step_id}"

        when_expr = step.get("when")
        if when_expr and when_skips(when_expr, self.step_outputs):
            print(f"[skip] '{step_id}' when condition not met: {when_expr}", file=sys.stderr)
            self.trace.log(step=step_label, event="step_skip", when=when_expr)
            self.step_index += 1
            return "advance"

        effective_input = self._effective_input(step)
        try:
            agent = self._agent_for(step, step_id)
            model = agent["model"] or DEFAULT_MODEL
            step_start_time = time.time()
            usage, messages, effective_schema = self._run_agent(step, step_id, step_label, agent, model, effective_input)

            in_tok, out_tok, cost = usage_totals(usage)
            self.total_in += in_tok
            self.total_out += out_tok
            self.total_cost += cost
            print(f"[usage: {step_id}]  in={in_tok:,}  out={out_tok:,}  cost=${cost:.4f}")
            if usage.get("cancelled"):
                self.trace.status = "cancelled"
                print("\nPipeline cancelled.", file=sys.stderr)
                sys.exit(0)

            return self._handle_result(step, step_id, step_label, usage, messages,
                                       effective_schema, cost, step_start_time)
        except Exception:
            print(f"\n[error: {step_id}] step failed", file=sys.stderr)
            self.print_totals()
            raise

    # ── result handling ──────────────────────────────────────────────────────

    def _save_step_end(self, step_label: str, step_id: str, messages: list,
                       preview: str, cost: float, start_time: float) -> None:
        self.trace.log(step=step_label, event="step_end", output_preview=preview,
                       duration=time.time() - start_time, cost=cost)
        self.trace.save_snapshot(self.step_index, step_id, messages, traces_dir=self.traces_dir)

    def _handle_result(self, step: StepConfig, step_id: str, step_label: str, usage: dict,
                       messages: list, effective_schema: dict | None, cost: float, start_time: float) -> str:
        structured_result = usage.get("result")
        stop_on = step.get("stop_on")
        if stop_on and structured_result and eval_condition(stop_on, structured_result):
            print("Nothing to do.", file=sys.stderr)
            self._save_step_end(step_label, step_id, messages, "(stop_on triggered)", cost, start_time)
            return "halt"

        step_output = extract_output(structured_result, messages)
        if step_output is None:
            print(f"Warning: agent '{step_id}' produced no text output; passing previous input forward.", file=sys.stderr)
            self._save_step_end(step_label, step_id, messages, "(no output)", cost, start_time)
            self.step_index += 1
            return "advance"

        self.step_outputs[step_id] = step_output
        declared_outputs = step.get("outputs")
        if declared_outputs:
            for name, value in parse_artifacts(step_output, declared_outputs).items():
                self.step_outputs[name] = value

        self._save_step_end(step_label, step_id, messages, _preview(step_output), cost, start_time)

        if not stop_on and is_stop_signal(step_output):
            print("Nothing to do.", file=sys.stderr)
            return "halt"

        if self._loop_back(step, step_id, step_label, structured_result, effective_schema, step_output):
            return "loop"
        self.step_index += 1
        return "advance"

    def _loop_back(self, step: StepConfig, step_id: str, step_label: str,
                   structured_result: dict | None, effective_schema: dict | None, step_output: str) -> bool:
        """Jump back to loop_to when loop_on fires and the cap isn't hit. Returns True if it looped."""
        loop_on = step.get("loop_on")
        loop_to = step.get("loop_to")
        if not (loop_on and loop_to):
            return False
        if structured_result and effective_schema:
            triggered = eval_condition(loop_on, structured_result)
        else:
            triggered = token_present(loop_on, step_output)
        if not triggered:
            return False

        count = self.loop_counts.get(step_id, 0)
        max_loops = step.get("max_loops")
        if count < max_loops:
            self.loop_counts[step_id] = count + 1
            print(f"[loop] '{step_id}' triggered '{loop_on}' (loop {count + 1}/{max_loops}), jumping to '{loop_to}'", file=sys.stderr)
            self.trace.log(step=step_label, event="loop", loop_on=loop_on, loop_to=loop_to,
                           iteration=count + 1, max=max_loops)
            self.step_index = self.step_index_map[loop_to]
            return True
        print(f"[loop] '{step_id}' hit max_loops={max_loops}; continuing to next step.", file=sys.stderr)
        return False

    # ── driver ───────────────────────────────────────────────────────────────

    def execute(self) -> None:
        while self.step_index < len(self.steps):
            if self._run_one(self.steps[self.step_index]) == "halt":
                self.print_totals()
                self.trace.status = "completed"
                return
        self.trace.status = "completed"
        self.print_totals()


def run_pipeline(steps: list[StepConfig], command: str, traces_dir: str | Path = "traces",
                 workflow_name: str = "pipeline") -> None:
    """Run command through each agent in sequence, chaining responses and tracing the run."""
    lf = get_langfuse()
    trace = Trace(workflow=workflow_name, command=command)
    trace.log(event="pipeline_start", workflow=workflow_name, command=command)
    run = _PipelineRun(steps, command, traces_dir, lf, trace)
    start_time = time.time()

    pipeline_ctx = (lf.start_as_current_observation(
        name=workflow_name, as_type="span", input={"command": command},
        metadata={"workflow": workflow_name},
    ) if lf else null_ctx())
    try:
        with pipeline_ctx as pipeline_obs:
            try:
                run.execute()
            except Exception:
                if trace.status == "running":
                    trace.status = "failed"
                raise
            finally:
                duration = time.time() - start_time
                trace.log(event="pipeline_end", status=trace.status, total_input=run.total_in,
                          total_output=run.total_out, total_cost=run.total_cost, duration=duration)
                trace.save(traces_dir=traces_dir)
                if pipeline_obs is not None:
                    pipeline_obs.update(output={"status": trace.status, "total_cost": run.total_cost, "duration": duration})
    finally:
        lf_shutdown()


def dry_run_pipeline(workflow_name: str, steps: list[StepConfig]) -> None:
    """Print resolved pipeline config without making any API calls."""
    print(f"\n  Pipeline: {workflow_name} ({len(steps)} steps)\n")
    for i, step in enumerate(steps):
        agent = load_agent(step["agent"])
        step_id = step.get("id") or step["agent"]
        print(f"    {i + 1}. {step_id}  ({step['agent']})")
        print(f"       model: {agent['model'] or DEFAULT_MODEL}")
        print(f"       tools: {', '.join(agent['tool_names']) or 'none'}")
        if agent["skill_names"]:
            print(f"       skills: {', '.join(agent['skill_names'])}")
        print(f"       mcp: {', '.join(agent['mcp_names']) or 'none'}")
        inputs = step.get("inputs") or []
        print(f"       inputs: {', '.join(inputs) if inputs else '(none)'}")
        outputs = step.get("outputs")
        if outputs:
            print(f"       outputs: {', '.join(f'{k} ({v})' for k, v in outputs.items())}")
        if step.get("when"):
            print(f"       when: {step['when']}")
        loop_on, loop_to = step.get("loop_on"), step.get("loop_to")
        if loop_on and loop_to:
            print(f"       loop: {loop_on} → {loop_to} (max {step.get('max_loops') or DEFAULT_MAX_LOOPS})")
        print()
