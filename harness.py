#!/usr/bin/env python3
"""Multi-agent harness CLI.

Thin entry point that wires the focused modules together behind an argparse
interface: `workflow` runs a pipeline, `agent` runs one agent (one-shot or REPL),
`trace` inspects saved runs, and `replay` resumes a run from a step. The real
logic lives in workflow.py, agent_loader.py, pipeline.py, and trace.py — this
module also re-exports their public names so `from harness import …` keeps working.
"""

import argparse
import datetime
import sys
from pathlib import Path

# Re-exported so existing imports / patch targets (`from harness import X`) keep working.
from agent_loader import load_agent
from agent_openrouter import MODEL as DEFAULT_MODEL, agent_loop
from conditions import eval_condition
from display import print_agent_header, usage_totals
from pipeline import _mcp_session, dry_run_pipeline, run_pipeline
from repl import read_command, run_cancellable, select_model
from step_io import build_submit_result_tool, parse_artifacts, substitute_prompt_vars
from workflow import load_workflow

_HERE = Path(__file__).parent


# ── trace inspection ─────────────────────────────────────────────────────────


def _trace_list(traces_dir: str) -> None:
    """Print a summary table of recent traces."""
    from tracing import Trace

    traces_path = Path(traces_dir)
    trace_files = sorted(traces_path.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True) if traces_path.is_dir() else []
    if not trace_files:
        print("No traces found.")
        return
    print(f"{'ID':<10} {'Workflow':<20} {'Status':<12} {'Steps':>5} {'Cost':>10} {'Duration':>10} {'Started'}")
    for tf in trace_files:
        row = Trace.load(tf.stem, traces_dir=traces_dir).summary_row()
        started = datetime.datetime.fromtimestamp(row["started_at"]).strftime("%Y-%m-%d %H:%M")
        duration_s = f"{row['duration']:.0f}s"
        print(f"{row['id']:<10} {row['workflow']:<20} {row['status']:<12} {row['steps']:>5} ${row['cost']:>9.4f} {duration_s:>10} {started}")


def _trace_show(trace_id: str, traces_dir: str) -> None:
    """Print the step-by-step detail of one trace."""
    from tracing import Trace

    print(Trace.load(trace_id, traces_dir=traces_dir).format_detail())


def _replay(trace_id: str, from_step: int, traces_dir: str, workflows_dir: str) -> None:
    """Re-run a previous pipeline from a specific step."""
    from tracing import Trace

    t = Trace.load(trace_id, traces_dir=traces_dir)
    all_steps = load_workflow(t.workflow, workflows_dir=Path(workflows_dir))["steps"]
    if from_step >= len(all_steps):
        raise ValueError(f"--from-step {from_step} is out of range (workflow has {len(all_steps)} steps)")
    run_pipeline(all_steps[from_step:], t.command, traces_dir=traces_dir, workflow_name=t.workflow)


# ── single-agent runners ─────────────────────────────────────────────────────


def _run_agent_oneshot(name: str, prompt: str, model_override: str | None = None) -> None:
    """Run a single agent with one prompt, print output, and exit."""
    agent = load_agent(name)
    model = model_override or agent["model"] or DEFAULT_MODEL
    print_agent_header(name, agent, model=model)

    messages: list = [{"role": "system", "content": agent["prompt"]}]
    with _mcp_session(agent["mcp_names"]) as mcp_clients:
        usage = agent_loop(prompt, messages, model=model, tools=agent["tools"], mcp_clients=mcp_clients)

    in_tok, out_tok, cost = usage_totals(usage)
    print(f"\n[usage]  in={in_tok:,}  out={out_tok:,}  cost=${cost:.4f}")


def _run_agent_repl(name: str, model_override: str | None = None) -> None:
    """Run a single agent in an interactive REPL."""
    agent = load_agent(name)
    current_model = model_override or agent["model"] or DEFAULT_MODEL
    print_agent_header(name, agent, model=current_model)
    print("Interactive mode (type 'exit' to quit, '/model' to switch, '/clear' to reset)")

    messages: list = [{"role": "system", "content": agent["prompt"]}]
    session_in = session_out = turns = 0
    session_cost = 0.0

    with _mcp_session(agent["mcp_names"]) as mcp_clients:
        while True:
            try:
                user_input = _read_user_input(name, current_model, session_in, session_out, turns, session_cost)
            except (EOFError, KeyboardInterrupt):
                print()
                break
            command = user_input.strip()
            if command.lower() in ("exit", "quit"):
                break
            if command == "/model":
                current_model = select_model(current_model)
                continue
            if command == "/clear":
                messages[:] = [{"role": "system", "content": agent["prompt"]}]
                session_in = session_out = turns = 0
                session_cost = 0.0
                print("History cleared.")
                continue
            if not command:
                continue

            usage = run_cancellable(lambda ce: agent_loop(
                user_input, messages, model=current_model, cancel_event=ce,
                mcp_clients=mcp_clients, tools=agent["tools"]))
            if usage.get("cancelled"):
                print("\nRequest interrupted.")
                continue
            in_tok, out_tok, cost = usage_totals(usage)
            session_in += in_tok
            session_out += out_tok
            session_cost += cost
            turns = (len(messages) - 1) // 2


# ── argument parsing ─────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run agent pipelines")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    ag = subparsers.add_parser("agent", help="Run a single agent interactively or one-shot")
    ag.add_argument("name", help="agent name (matches name: field in agents/*.yaml)")
    ag.add_argument("prompt", nargs="?", help="one-shot prompt (omit for interactive REPL)")
    ag.add_argument("--model", help="override the agent's default model")

    wf = subparsers.add_parser("workflow", help="Run a workflow with an ad-hoc prompt")
    wf.add_argument("name", help="workflow name")
    wf.add_argument("prompt", nargs="?", help="initial prompt to send to the first agent")
    wf.add_argument("--dry-run", action="store_true", help="print resolved pipeline config without running")

    tr = subparsers.add_parser("trace", help="Inspect traces")
    tr_sub = tr.add_subparsers(dest="trace_action", required=True)
    tr_list = tr_sub.add_parser("list", help="List recent traces")
    tr_list.add_argument("--traces-dir", default="traces", help="traces directory")
    tr_show = tr_sub.add_parser("show", help="Show trace detail")
    tr_show.add_argument("trace_id", help="trace ID")
    tr_show.add_argument("--traces-dir", default="traces", help="traces directory")

    rp = subparsers.add_parser("replay", help="Replay a pipeline from a specific step")
    rp.add_argument("trace_id", help="trace ID to replay from")
    rp.add_argument("--from-step", type=int, required=True, help="step index to resume from")
    rp.add_argument("--traces-dir", default="traces", help="traces directory")
    rp.add_argument("--workflows-dir", default=str(_HERE / "workflows"), help="workflows directory")
    return parser


def _dispatch(args: argparse.Namespace) -> None:
    if args.subcommand == "agent":
        if args.prompt:
            _run_agent_oneshot(args.name, args.prompt, model_override=args.model)
        else:
            _run_agent_repl(args.name, model_override=args.model)
    elif args.subcommand == "workflow":
        steps = load_workflow(args.name)["steps"]
        if args.dry_run:
            dry_run_pipeline(args.name, steps)
        else:
            run_pipeline(steps, args.prompt or "", workflow_name=args.name)
    elif args.subcommand == "trace":
        if args.trace_action == "list":
            _trace_list(args.traces_dir)
        elif args.trace_action == "show":
            _trace_show(args.trace_id, args.traces_dir)
    elif args.subcommand == "replay":
        _replay(args.trace_id, args.from_step, args.traces_dir, args.workflows_dir)


def main() -> None:
    args = _build_parser().parse_args()
    try:
        _dispatch(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
