#!/usr/bin/env python3
"""Agent pipeline executor — runs workflows through agent chains."""

import argparse
import datetime
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import NotRequired, TypedDict

import yaml
from prompt_toolkit import prompt as pt_prompt

from agent_openrouter import AVAILABLE_MODELS, MODEL as DEFAULT_MODEL, agent_loop
from mcp_client import build_mcp_clients, load_mcp_config
from repl_utils import COMMAND_COMPLETER, IS_TTY, status_text, watch_for_escape
from skills_loader import append_skills
from tools import ALL_TOOLS, Tool

_TOOL_MAP = {t.name: t for t in ALL_TOOLS}
_MCP_CONFIG = load_mcp_config()
_HERE = Path(__file__).parent


class AgentConfig(TypedDict):
    prompt: str
    tools: list[Tool]
    tool_names: list[str]
    skill_names: list[str]
    mcp_names: list[str]
    model: str | None
    output: dict | None


class StepConfig(TypedDict):
    agent: str
    id: str
    prompt: str | None
    inputs: list[str]
    outputs: NotRequired[dict[str, str] | None]
    when: NotRequired[str | None]
    loop_on: NotRequired[str | None]
    loop_to: NotRequired[str | None]
    max_loops: NotRequired[int | None]
    output: NotRequired[dict | None]
    stop_on: NotRequired[str | None]


class WorkflowConfig(TypedDict):
    steps: list[StepConfig]


def load_workflow(name: str, workflows_dir: Path = _HERE / "workflows") -> WorkflowConfig:
    """Scan workflows_dir for a YAML whose name: field matches name."""
    for path in sorted(workflows_dir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text())
        if data.get("name") == name:
            steps: list[StepConfig] = []
            seen_ids: set[str] = set()
            seen_artifact_names: set[str] = set()
            # Pre-collect all step ids and declared artifact names so inputs can forward-reference
            # a step that executes later (e.g., implementer consuming reviewer's feedback on loop-back).
            all_ids: set[str] = {"__input__"}
            all_artifact_names: set[str] = set()
            for step in data.get("steps", []):
                all_ids.add(step.get("id") or step["agent"])
                for artifact_name in (step.get("outputs") or {}):
                    all_artifact_names.add(artifact_name)
            for step in data.get("steps", []):
                step_agent = step["agent"]
                step_id = step.get("id") or step_agent
                if step_id in seen_ids:
                    raise ValueError(
                        f"Step id '{step_id}' is not unique. "
                        f"When the same agent appears multiple times, each step must have an explicit 'id:'."
                    )
                loop_on = step.get("loop_on") or None
                loop_to = step.get("loop_to") or None
                max_loops = step.get("max_loops")
                if "inputs" not in step:
                    raise ValueError(
                        f"Step '{step_id}' must declare 'inputs' explicitly. "
                        f"Use 'inputs: [__input__]' for the workflow prompt, "
                        f"'inputs: [<step_id>]' to consume a prior step, or 'inputs: []' for no input."
                    )
                raw_inputs = step["inputs"]
                inputs: list[str] = list(raw_inputs) if raw_inputs is not None else []
                raw_outputs = step.get("outputs")
                outputs: dict[str, str] | None = dict(raw_outputs) if raw_outputs else None
                when: str | None = step.get("when") or None
                if (loop_on is None) != (loop_to is None):
                    raise ValueError(f"Step '{step_id}' must have both loop_on and loop_to, or neither.")
                if loop_to is not None and loop_to not in seen_ids:
                    raise ValueError(f"Step '{step_id}' loop_to='{loop_to}' must refer to an earlier step.")
                if loop_on is not None and max_loops is None:
                    max_loops = 3
                if loop_on is not None and when is not None:
                    raise ValueError(f"Step '{step_id}' cannot have both loop_on and when.")
                for ref in inputs:
                    if ref not in all_ids and ref not in all_artifact_names:
                        raise ValueError(f"Step '{step_id}' inputs references unknown id '{ref}'.")
                if outputs is not None:
                    for artifact_name in outputs:
                        if artifact_name in seen_artifact_names or artifact_name in seen_ids:
                            raise ValueError(f"Step '{step_id}' output '{artifact_name}' conflicts with an existing id or artifact name.")
                        seen_artifact_names.add(artifact_name)
                if when is not None:
                    m = re.match(r'^(.+?)\s+in\s+(\w+)$', when)
                    if not m:
                        raise ValueError(f"Step '{step_id}' when='{when}' must be 'PATTERN in step_id'.")
                    ref_id = m.group(2)
                    if ref_id not in seen_ids and ref_id not in seen_artifact_names:
                        raise ValueError(f"Step '{step_id}' when references unknown id '{ref_id}'.")
                output: dict | None = step.get("output") or None
                stop_on: str | None = step.get("stop_on") or None
                if output is not None:
                    for field_name, field_def in output.items():
                        if "type" not in field_def:
                            raise ValueError(f"Step '{step_id}' output field '{field_name}' must have a 'type'.")
                        if "enum" in field_def and not isinstance(field_def["enum"], list):
                            raise ValueError(f"Step '{step_id}' output field '{field_name}' enum must be a list.")
                if stop_on is not None and output is None:
                    raise ValueError(f"Step '{step_id}' has stop_on but no output.")
                if stop_on is not None and loop_on is not None:
                    raise ValueError(f"Step '{step_id}' cannot have both stop_on and loop_on.")
                if loop_on is not None and output is not None:
                    parts = loop_on.split("==", 1)
                    if len(parts) != 2:
                        raise ValueError(f"Step '{step_id}' loop_on must be 'field == value' when output is set.")
                    field = parts[0].strip()
                    value = parts[1].strip()
                    if field not in output:
                        raise ValueError(f"Step '{step_id}' loop_on references unknown field '{field}' not in output.")
                    field_def = output[field]
                    if "enum" in field_def and value not in field_def["enum"]:
                        raise ValueError(f"Step '{step_id}' loop_on value '{value}' is not in enum {field_def['enum']}.")
                steps.append({
                    "agent": step_agent,
                    "id": step_id,
                    "prompt": step.get("prompt") or None,
                    "inputs": inputs,
                    "outputs": outputs,
                    "when": when,
                    "loop_on": loop_on,
                    "loop_to": loop_to,
                    "max_loops": max_loops,
                    "output": output,
                    "stop_on": stop_on,
                })
                seen_ids.add(step_id)
            return {"steps": steps}
    raise ValueError(f"No workflow named '{name}' found in {workflows_dir}/")


def _append_environment(prompt: str) -> str:
    """Append current working directory and date to a system prompt."""
    cwd = os.getcwd()
    today = datetime.date.today().isoformat()
    suffix = f"\n\n## Environment\n- Working directory: {cwd}\n- Current date: {today}"
    return (prompt or "") + suffix


def load_agent(name: str, agents_dir: Path = _HERE / "agents") -> AgentConfig:
    """Scan agents_dir for a YAML whose name: field matches name.

    Returns dict with keys: prompt, tools (list[Tool]), tool_names (list[str]), model (str|None), output (dict|None).
    Exits on unknown tool names.
    """
    for path in sorted(agents_dir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text())
        if data.get("name") == name:
            raw_tools = data.get("tools", [])
            tool_names = [t if isinstance(t, str) else t["name"] for t in raw_tools]
            tools = []
            for tool_name in tool_names:
                if tool_name not in _TOOL_MAP:
                    raise ValueError(f"Agent '{name}' references unknown tool '{tool_name}'")
                tools.append(_TOOL_MAP[tool_name])
            raw_skills = data.get("skills", [])
            skill_names = [s["name"] if isinstance(s, dict) else s for s in raw_skills]
            prompt = append_skills(data.get("prompt", ""), skill_names)
            prompt = _append_environment(prompt)
            raw_mcp = data.get("mcp", [])
            mcp_names = [m["name"] if isinstance(m, dict) else m for m in raw_mcp]
            for mcp_name in mcp_names:
                if mcp_name not in _MCP_CONFIG:
                    raise ValueError(f"Agent '{name}' references unknown MCP server '{mcp_name}'")
            output: dict | None = data.get("output") or None
            return {
                "prompt": prompt,
                "tools": tools,
                "model": data.get("model", None),
                "tool_names": tool_names,
                "skill_names": skill_names,
                "mcp_names": mcp_names,
                "output": output,
            }
    raise ValueError(f"No agent named '{name}' found in {agents_dir}/")


def substitute_prompt_vars(text: str, step_outputs: dict[str, str]) -> str:
    """Replace {name} placeholders in a step prompt.

    {step_id} expands to that step's output; {__input__} expands to the workflow's
    initial command. Unknown names are left unchanged so literal braces aren't
    consumed by accident.
    """
    def repl(m: re.Match) -> str:
        name = m.group(1).strip()
        if name in step_outputs:
            return step_outputs[name]
        return m.group(0)
    return re.sub(r"\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}", repl, text)


def parse_artifacts(raw_output: str, declared_outputs: dict[str, str]) -> dict[str, str]:
    """Extract declared artifacts from agent output.

    Looks for fenced code blocks tagged with the artifact name, e.g.:
        ```plan
        {"steps": [...]}
        ```
    Falls back to the full raw output if a block isn't found.
    """
    artifacts: dict[str, str] = {}
    for name in declared_outputs:
        pattern = rf'```{re.escape(name)}\s*\n(.*?)```'
        m = re.search(pattern, raw_output, re.DOTALL)
        artifacts[name] = m.group(1).strip() if m else raw_output
    return artifacts


def build_submit_result_tool(response_schema: dict) -> dict:
    """Generate an OpenAI-format tool definition from a response_schema dict."""
    properties = {}
    for field_name, field_def in response_schema.items():
        prop: dict = {"type": field_def["type"]}
        if "enum" in field_def:
            prop["enum"] = field_def["enum"]
        properties[field_name] = prop
    return {
        "type": "function",
        "function": {
            "name": "submit_result",
            "description": "Submit your final structured result for this step. You MUST call this tool when you have reached your conclusion.",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(response_schema.keys()),
            },
        },
    }


def eval_condition(expr: str, result: dict) -> bool:
    """Evaluate a 'field == value' expression against a structured result dict."""
    parts = expr.split("==", 1)
    if len(parts) != 2:
        return False
    field = parts[0].strip()
    value = parts[1].strip()
    return str(result.get(field, "")) == value


def run_pipeline(steps: list[StepConfig], command: str, traces_dir: str | Path = "traces", workflow_name: str = "pipeline") -> None:
    """Run command through each agent in sequence, chaining responses."""
    from trace import Trace, _preview
    from langfuse_client import get_langfuse, lf_shutdown, null_ctx

    lf = get_langfuse()
    step_index_map: dict[str, int] = {s["id"]: i for i, s in enumerate(steps)}
    agent_cache: dict[str, AgentConfig] = {}
    loop_counts: dict[str, int] = {}
    # Outputs keyed by step id; "__input__" holds the original command.
    step_outputs: dict[str, str] = {"__input__": command}
    step_index = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0

    trace = Trace(workflow=workflow_name, command=command)
    trace.log(event="pipeline_start", workflow=workflow_name, command=command)
    pipeline_start_time = time.time()

    pipeline_ctx = (lf.start_as_current_observation(
        name=workflow_name,
        as_type="span",
        input={"command": command},
        metadata={"workflow": workflow_name},
    ) if lf else null_ctx())
    pipeline_obs = pipeline_ctx.__enter__()

    try:
        while step_index < len(steps):
            step = steps[step_index]
            step_agent = step["agent"]
            step_id = step.get("id") or step_agent
            step_prompt = step.get("prompt")
            if step_prompt:
                step_prompt = substitute_prompt_vars(step_prompt, step_outputs)
            input_ids = step.get("inputs") or []
            step_label = f"{step_index}:{step_id}"

            # Evaluate when condition — skip step if false.
            when_expr = step.get("when")
            if when_expr:
                m = re.match(r'^(.+?)\s+in\s+(\w+)$', when_expr)
                if m:
                    pattern, ref_id = m.group(1).strip(), m.group(2)
                    if ref_id not in step_outputs or pattern not in step_outputs[ref_id]:
                        print(f"[skip] '{step_id}' when condition not met: {when_expr}", file=sys.stderr)
                        trace.log(step=step_label, event="step_skip", when=when_expr)
                        step_index += 1
                        continue

            # Determine what to feed this step.
            if len(input_ids) == 1 and input_ids[0] in step_outputs:
                current_input = step_outputs[input_ids[0]]
            else:
                parts = []
                for ref in input_ids:
                    if ref in step_outputs:
                        parts.append(f"## Input: {ref}\n{step_outputs[ref]}")
                current_input = "\n\n".join(parts)

            try:
                if step_id not in agent_cache:
                    agent_cache[step_id] = load_agent(step_agent)
                agent = agent_cache[step_id]
                model = agent["model"] or DEFAULT_MODEL
                messages: list = [{"role": "system", "content": agent["prompt"]}]
                tools_str = ", ".join(agent["tool_names"]) or "none"
                skills_str = ", ".join(agent["skill_names"]) or "none"
                mcp_names = agent["mcp_names"]
                mcp_str = ", ".join(mcp_names) or "none"
                mcp_clients = build_mcp_clients(mcp_names) if mcp_names else []
                print(f"\n\033[1m[agent: {step_agent}]\033[0m  tools: {tools_str}  |  skills: {skills_str}  |  mcp: {mcp_str}")
                effective_input = (step_prompt + "\n\n" + current_input) if step_prompt else current_input
                print("\033[36m--- system prompt ---\033[0m")
                print(f"\033[2m{agent['prompt']}\033[0m")
                print("\033[36m--- user prompt ---\033[0m")
                print(f"\033[2m{effective_input}\033[0m")
                print("\033[36m--- response ---\033[0m")

                trace.log(step=step_label, event="step_start", model=model,
                          tools=agent["tool_names"], prompt_preview=_preview(effective_input),
                          system_prompt=agent["prompt"], user_prompt=effective_input)
                step_start_time = time.time()

                # Merge agent's canonical output schema with step-level override.
                agent_output = agent.get("output") or {}
                step_output_schema = step.get("output") or {}
                effective_schema: dict | None = {**agent_output, **step_output_schema} or None
                submit_schema = build_submit_result_tool(effective_schema) if effective_schema else None
                step_ctx = (lf.start_as_current_observation(
                    name=step_label,
                    as_type="agent",
                    input={"prompt": effective_input, "model": model, "tools": agent["tool_names"]},
                    metadata={"step_id": step_id},
                ) if lf else null_ctx())
                step_obs = step_ctx.__enter__()
                try:
                    usage = agent_loop(effective_input, messages, model=model, tools=agent["tools"],
                                       mcp_clients=mcp_clients, trace=trace, step_label=step_label,
                                       submit_result_schema=submit_schema)
                finally:
                    for client in mcp_clients:
                        client.close()
                    step_ctx.__exit__(None, None, None)
                in_tok = usage.get("input_tokens", 0)
                out_tok = usage.get("output_tokens", 0)
                cost = usage.get("cost", 0.0)
                total_input_tokens += in_tok
                total_output_tokens += out_tok
                total_cost += cost
                print(f"[usage: {step_id}]  in={in_tok:,}  out={out_tok:,}  cost=${cost:.4f}")
                if usage.get("cancelled"):
                    trace.status = "cancelled"
                    print("\nPipeline cancelled.", file=sys.stderr)
                    sys.exit(0)
                structured_result = usage.get("result")
                stop_on = step.get("stop_on")
                if stop_on and structured_result and eval_condition(stop_on, structured_result):
                    print("Nothing to do.", file=sys.stderr)
                    trace.log(step=step_label, event="step_end", output_preview="(stop_on triggered)",
                              duration=time.time() - step_start_time, cost=cost)
                    trace.save_snapshot(step_index, step_id, messages, traces_dir=traces_dir)
                    print(f"\n[total usage]  in={total_input_tokens:,}  out={total_output_tokens:,}  cost=${total_cost:.4f}")
                    trace.status = "completed"
                    return
                step_output: str | None = None
                if structured_result:
                    step_output = "\n".join(f"{k}: {v}" for k, v in structured_result.items())
                else:
                    for msg in reversed(messages):
                        if msg["role"] == "assistant" and msg.get("content"):
                            step_output = msg["content"]
                            break
                if step_output is None:
                    print(f"Warning: agent '{step_id}' produced no text output; passing previous input forward.", file=sys.stderr)
                    trace.log(step=step_label, event="step_end", output_preview="(no output)",
                              duration=time.time() - step_start_time, cost=cost)
                    trace.save_snapshot(step_index, step_id, messages, traces_dir=traces_dir)
                    step_index += 1
                    continue
                step_outputs[step_id] = step_output
                declared_outputs = step.get("outputs")
                if declared_outputs:
                    artifacts = parse_artifacts(step_output, declared_outputs)
                    for artifact_name, artifact_value in artifacts.items():
                        step_outputs[artifact_name] = artifact_value

                trace.log(step=step_label, event="step_end", output_preview=_preview(step_output),
                          duration=time.time() - step_start_time, cost=cost)
                trace.save_snapshot(step_index, step_id, messages, traces_dir=traces_dir)

                if not stop_on and "STOP" in (step_output or ""):
                    print("Nothing to do.", file=sys.stderr)
                    print(f"\n[total usage]  in={total_input_tokens:,}  out={total_output_tokens:,}  cost=${total_cost:.4f}")
                    trace.status = "completed"
                    return
                loop_on = step.get("loop_on")
                loop_to = step.get("loop_to")
                max_loops = step.get("max_loops")
                if loop_on and loop_to:
                    loop_triggered = False
                    if structured_result and effective_schema:
                        loop_triggered = eval_condition(loop_on, structured_result)
                    else:
                        loop_triggered = loop_on in (step_output or "")
                    if loop_triggered:
                        count = loop_counts.get(step_id, 0)
                        if count < max_loops:
                            loop_counts[step_id] = count + 1
                            print(f"[loop] '{step_id}' triggered '{loop_on}' (loop {count + 1}/{max_loops}), jumping to '{loop_to}'", file=sys.stderr)
                            trace.log(step=step_label, event="loop", loop_on=loop_on, loop_to=loop_to,
                                      iteration=count + 1, max=max_loops)
                            step_index = step_index_map[loop_to]
                            continue
                        else:
                            print(f"[loop] '{step_id}' hit max_loops={max_loops}; continuing to next step.", file=sys.stderr)
                step_index += 1
            except Exception:
                print(f"\n[error: {step_id}] step failed", file=sys.stderr)
                print(f"\n[total usage]  in={total_input_tokens:,}  out={total_output_tokens:,}  cost=${total_cost:.4f}")
                raise

        trace.status = "completed"
        print(f"\n[total usage]  in={total_input_tokens:,}  out={total_output_tokens:,}  cost=${total_cost:.4f}")
    except Exception:
        if trace.status == "running":
            trace.status = "failed"
        raise
    finally:
        duration = time.time() - pipeline_start_time
        trace.log(event="pipeline_end", status=trace.status, total_input=total_input_tokens,
                  total_output=total_output_tokens, total_cost=total_cost, duration=duration)
        trace.save(traces_dir=traces_dir)
        if pipeline_obs is not None:
            pipeline_obs.update(output={"status": trace.status, "total_cost": total_cost, "duration": duration})
        pipeline_ctx.__exit__(None, None, None)
        lf_shutdown()


def _trace_list(traces_dir: str) -> None:
    """List recent traces."""
    import datetime
    from trace import Trace

    traces_path = Path(traces_dir)
    if not traces_path.is_dir():
        print("No traces found.")
        return
    trace_files = sorted(traces_path.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not trace_files:
        print("No traces found.")
        return
    print(f"{'ID':<10} {'Workflow':<20} {'Status':<12} {'Steps':>5} {'Cost':>10} {'Duration':>10} {'Started'}")
    for tf in trace_files:
        t = Trace.load(tf.stem, traces_dir=traces_dir)
        row = t.summary_row()
        started = datetime.datetime.fromtimestamp(row["started_at"]).strftime("%Y-%m-%d %H:%M")
        duration_s = f"{row['duration']:.0f}s"
        print(f"{row['id']:<10} {row['workflow']:<20} {row['status']:<12} {row['steps']:>5} ${row['cost']:>9.4f} {duration_s:>10} {started}")


def _trace_show(trace_id: str, traces_dir: str) -> None:
    """Show trace detail."""
    from trace import Trace

    t = Trace.load(trace_id, traces_dir=traces_dir)
    print(t.format_detail())


def _replay(trace_id: str, from_step: int, traces_dir: str, workflows_dir: str) -> None:
    """Replay a pipeline from a specific step."""
    from trace import Trace

    t = Trace.load(trace_id, traces_dir=traces_dir)
    wf = load_workflow(t.workflow, workflows_dir=Path(workflows_dir))
    all_steps = wf["steps"]
    if from_step >= len(all_steps):
        raise ValueError(f"--from-step {from_step} is out of range (workflow has {len(all_steps)} steps)")
    remaining_steps = all_steps[from_step:]
    run_pipeline(remaining_steps, t.command, traces_dir=traces_dir, workflow_name=t.workflow)


def _run_agent_oneshot(name: str, prompt: str, model_override: str | None = None) -> None:
    """Run a single agent with one prompt, print output, and exit."""
    agent = load_agent(name)
    model = model_override or agent["model"] or DEFAULT_MODEL
    mcp_names = agent["mcp_names"]
    mcp_clients = build_mcp_clients(mcp_names) if mcp_names else []
    tools_str = ", ".join(agent["tool_names"]) or "none"
    skills_str = ", ".join(agent["skill_names"]) or "none"
    mcp_str = ", ".join(mcp_names) or "none"
    print(f"\033[1m[agent: {name}]\033[0m  model: {model}  tools: {tools_str}  skills: {skills_str}  mcp: {mcp_str}")

    messages: list = [{"role": "system", "content": agent["prompt"]}]
    try:
        usage = agent_loop(prompt, messages, model=model, tools=agent["tools"], mcp_clients=mcp_clients)
    finally:
        for client in mcp_clients:
            client.close()

    in_tok = usage.get("input_tokens", 0)
    out_tok = usage.get("output_tokens", 0)
    cost = usage.get("cost", 0.0)
    print(f"\n[usage]  in={in_tok:,}  out={out_tok:,}  cost=${cost:.4f}")


def _select_model(current: str) -> str:
    """Prompt the user to pick a model."""
    print("Available models:")
    for i, m in enumerate(AVAILABLE_MODELS, 1):
        marker = " *" if m == current else ""
        print(f"  {i}. {m}{marker}")
    try:
        choice = input("Pick a number (or press Enter to keep current): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return current
    if not choice:
        return current
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(AVAILABLE_MODELS):
            selected = AVAILABLE_MODELS[idx]
            print(f"Model set to: {selected}")
            return selected
        print("Invalid selection, keeping current model.")
    except ValueError:
        print("Invalid input, keeping current model.")
    return current


def _run_agent_repl(name: str, model_override: str | None = None) -> None:
    """Run a single agent in interactive REPL mode."""
    agent = load_agent(name)
    model = model_override or agent["model"] or DEFAULT_MODEL
    mcp_names = agent["mcp_names"]
    mcp_clients = build_mcp_clients(mcp_names) if mcp_names else []
    tools_str = ", ".join(agent["tool_names"]) or "none"
    skills_str = ", ".join(agent["skill_names"]) or "none"
    mcp_str = ", ".join(mcp_names) or "none"
    print(f"\033[1m[agent: {name}]\033[0m  model: {model}  tools: {tools_str}  skills: {skills_str}  mcp: {mcp_str}")
    print("Interactive mode (type 'exit' to quit, '/model' to switch, '/clear' to reset)")

    messages: list = [{"role": "system", "content": agent["prompt"]}]
    current_model = model
    session_in = 0
    session_out = 0
    session_cost = 0.0
    turns = 0

    try:
        while True:
            try:
                if IS_TTY:
                    import warnings
                    def _toolbar() -> str:
                        return status_text(current_model, session_in, session_out, turns, session_cost)
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message=".*CPR.*")
                        user_input = pt_prompt(f"{name}> ", completer=COMMAND_COMPLETER, bottom_toolbar=_toolbar, refresh_interval=0.5)
                else:
                    user_input = input(f"{name}> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if user_input.strip().lower() in ("exit", "quit"):
                break
            if user_input.strip() == "/model":
                current_model = _select_model(current_model)
                continue
            if user_input.strip() == "/clear":
                messages.clear()
                messages.append({"role": "system", "content": agent["prompt"]})
                session_in = 0
                session_out = 0
                session_cost = 0.0
                turns = 0
                print("History cleared.")
                continue
            if not user_input.strip():
                continue

            cancel_event = threading.Event()
            done_event = threading.Event()
            result: dict = {}

            def _run() -> None:
                result["usage"] = agent_loop(
                    user_input, messages, model=current_model,
                    cancel_event=cancel_event, mcp_clients=mcp_clients,
                    tools=agent["tools"],
                )

            agent_thread = threading.Thread(target=_run, daemon=True)
            watcher_thread = threading.Thread(target=watch_for_escape, args=(cancel_event, done_event), daemon=True)
            watcher_thread.start()
            agent_thread.start()
            try:
                agent_thread.join()
            except KeyboardInterrupt:
                cancel_event.set()
                agent_thread.join()
            finally:
                done_event.set()

            usage = result.get("usage", {"input_tokens": 0, "output_tokens": 0, "cancelled": True})
            if usage.get("cancelled"):
                print("\nRequest interrupted.")
            else:
                session_in += usage["input_tokens"]
                session_out += usage["output_tokens"]
                session_cost += usage.get("cost", 0.0)
                turns = (len(messages) - 1) // 2
    finally:
        for client in mcp_clients:
            client.close()


def dry_run_pipeline(workflow_name: str, steps: list[StepConfig]) -> None:
    """Print resolved pipeline config without making any API calls."""
    print(f"\n  Pipeline: {workflow_name} ({len(steps)} steps)\n")
    for i, step in enumerate(steps):
        agent = load_agent(step["agent"])
        step_id = step.get("id") or step["agent"]
        print(f"    {i + 1}. {step_id}  ({step['agent']})")

        model = agent["model"] or DEFAULT_MODEL
        print(f"       model: {model}")

        tools = ", ".join(agent["tool_names"]) if agent["tool_names"] else "none"
        print(f"       tools: {tools}")

        if agent["skill_names"]:
            print(f"       skills: {', '.join(agent['skill_names'])}")

        if agent["mcp_names"]:
            print(f"       mcp: {', '.join(agent['mcp_names'])}")
        else:
            print(f"       mcp: none")

        inputs = step.get("inputs") or []
        if inputs:
            print(f"       inputs: {', '.join(inputs)}")
        else:
            print(f"       inputs: (none)")

        outputs = step.get("outputs")
        if outputs:
            print(f"       outputs: {', '.join(f'{k} ({v})' for k, v in outputs.items())}")

        when = step.get("when")
        if when:
            print(f"       when: {when}")

        loop_on = step.get("loop_on")
        loop_to = step.get("loop_to")
        if loop_on and loop_to:
            max_loops = step.get("max_loops") or 3
            print(f"       loop: {loop_on} \u2192 {loop_to} (max {max_loops})")

        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent pipelines")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # agent subcommand
    ag = subparsers.add_parser("agent", help="Run a single agent interactively or one-shot")
    ag.add_argument("name", help="agent name (matches name: field in agents/*.yaml)")
    ag.add_argument("prompt", nargs="?", help="one-shot prompt (omit for interactive REPL)")
    ag.add_argument("--model", help="override the agent's default model")

    # workflow subcommand
    wf = subparsers.add_parser("workflow", help="Run a workflow with an ad-hoc prompt")
    wf.add_argument("name", help="workflow name")
    wf.add_argument("prompt", nargs="?", help="initial prompt to send to the first agent")
    wf.add_argument("--dry-run", action="store_true", help="print resolved pipeline config without running")

    # trace subcommand
    tr = subparsers.add_parser("trace", help="Inspect traces")
    tr_sub = tr.add_subparsers(dest="trace_action", required=True)

    tr_list = tr_sub.add_parser("list", help="List recent traces")
    tr_list.add_argument("--traces-dir", default="traces", help="traces directory")

    tr_show = tr_sub.add_parser("show", help="Show trace detail")
    tr_show.add_argument("trace_id", help="trace ID")
    tr_show.add_argument("--traces-dir", default="traces", help="traces directory")

    # replay subcommand
    rp = subparsers.add_parser("replay", help="Replay a pipeline from a specific step")
    rp.add_argument("trace_id", help="trace ID to replay from")
    rp.add_argument("--from-step", type=int, required=True, help="step index to resume from")
    rp.add_argument("--traces-dir", default="traces", help="traces directory")
    rp.add_argument("--workflows-dir", default=str(_HERE / "workflows"), help="workflows directory")

    args = parser.parse_args()

    try:
        if args.subcommand == "agent":
            if args.prompt:
                _run_agent_oneshot(args.name, args.prompt, model_override=args.model)
            else:
                _run_agent_repl(args.name, model_override=args.model)
        elif args.subcommand == "workflow":
            wf = load_workflow(args.name)
            step_configs = wf["steps"]
            if args.dry_run:
                dry_run_pipeline(args.name, step_configs)
            else:
                run_pipeline(step_configs, args.prompt or "", workflow_name=args.name)
        elif args.subcommand == "trace":
            if args.trace_action == "list":
                _trace_list(args.traces_dir)
            elif args.trace_action == "show":
                _trace_show(args.trace_id, args.traces_dir)
        elif args.subcommand == "replay":
            _replay(args.trace_id, args.from_step, args.traces_dir, args.workflows_dir)
        else:
            parser.error(f"Unknown subcommand: {args.subcommand}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
