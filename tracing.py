"""Structured trace logging for workflow pipeline runs."""

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

PREVIEW_MAX = 500
PARAMS_MAX = 50


def _preview(text: str) -> str:
    """Truncate text to PREVIEW_MAX characters."""
    if len(text) <= PREVIEW_MAX:
        return text
    return text[:PREVIEW_MAX] + "..."


def _truncate_params(params: dict) -> str:
    """Render a params dict as key=value pairs, capped at PARAMS_MAX chars."""
    s = ", ".join(f"{k}={v!r}" for k, v in params.items())
    if len(s) <= PARAMS_MAX:
        return s
    return s[: PARAMS_MAX - 1] + "…"


def _color_enabled() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(text: str, code: str) -> str:
    if not _color_enabled():
        return text
    return f"\033[{code}m{text}\033[0m"


@dataclass(frozen=True)
class TraceEvent:
    timestamp: float
    step: str | None
    event: str
    data: dict

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "step": self.step,
            "event": self.event,
            "data": self.data,
        }


class Trace:
    def __init__(self, workflow: str, command: str) -> None:
        self.id = uuid4().hex[:8]
        self.workflow = workflow
        self.command = command
        self.started_at = time.time()
        self.events: list[TraceEvent] = []
        self.status = "running"

    def log(self, step: str | None = None, event: str = "", **data: object) -> None:
        self.events.append(TraceEvent(
            timestamp=time.time(),
            step=step,
            event=event,
            data=data,
        ))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "workflow": self.workflow,
            "command": self.command,
            "started_at": self.started_at,
            "status": self.status,
            "events": [e.to_dict() for e in self.events],
        }

    def save(self, traces_dir: str | Path = "traces") -> Path:
        """Save trace to a JSON file. Returns the path to the saved file."""
        traces_dir = Path(traces_dir)
        traces_dir.mkdir(parents=True, exist_ok=True)
        path = traces_dir / f"{self.id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, trace_id: str, traces_dir: str | Path = "traces") -> "Trace":
        """Load a trace from its JSON file."""
        path = Path(traces_dir) / f"{trace_id}.json"
        data = json.loads(path.read_text())
        trace = cls.__new__(cls)
        trace.id = data["id"]
        trace.workflow = data["workflow"]
        trace.command = data["command"]
        trace.started_at = data["started_at"]
        trace.status = data["status"]
        trace.events = [
            TraceEvent(
                timestamp=e["timestamp"],
                step=e["step"],
                event=e["event"],
                data=e["data"],
            )
            for e in data["events"]
        ]
        return trace

    def save_snapshot(self, step_index: int, step_name: str, messages: list, traces_dir: str | Path = "traces") -> Path:
        """Save the messages list after a step completes."""
        traces_dir = Path(traces_dir)
        snapshot_dir = traces_dir / f"{self.id}_messages"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        path = snapshot_dir / f"step_{step_index}_{step_name}.json"
        path.write_text(json.dumps(messages, indent=2))
        return path

    @staticmethod
    def load_snapshot(trace_id: str, step_index: int, step_name: str, traces_dir: str | Path = "traces") -> list:
        """Load a conversation snapshot for a specific step."""
        path = Path(traces_dir) / f"{trace_id}_messages" / f"step_{step_index}_{step_name}.json"
        return json.loads(path.read_text())

    def summary_row(self) -> dict:
        """Return a dict summarizing this trace for table display."""
        step_count = sum(1 for e in self.events if e.event == "step_start")
        pipeline_end = next((e for e in self.events if e.event == "pipeline_end"), None)
        cost = pipeline_end.data.get("total_cost", 0.0) if pipeline_end else 0.0
        duration = pipeline_end.data.get("duration", 0.0) if pipeline_end else 0.0
        return {
            "id": self.id,
            "workflow": self.workflow,
            "status": self.status,
            "steps": step_count,
            "cost": cost,
            "duration": duration,
            "started_at": self.started_at,
        }

    def format_detail(self) -> str:
        """Format trace as human-readable step-by-step detail."""
        header = _c(f"Trace {self.id} — {self.workflow} ({self.status})", "1")
        lines = [header, f'Command: "{self.command}"', ""]

        def ts(e: TraceEvent) -> str:
            return _c(f"[+{e.timestamp - self.started_at:5.1f}s]", "2")

        arrow_call = _c("->", "32")
        arrow_result = _c("<-", "34")

        for e in self.events:
            if e.event == "step_start":
                model = e.data.get("model", "?")
                lines.append(f"{ts(e)} {_c(f'Step {e.step}', '1;36')}  (model: {model})")
                system_prompt = e.data.get("system_prompt")
                user_prompt = e.data.get("user_prompt")
                if system_prompt:
                    lines.append(f"{ts(e)}   {_c('--- system prompt ---', '36')}")
                    lines.append(_c(system_prompt, "2"))
                if user_prompt:
                    lines.append(f"{ts(e)}   {_c('--- user prompt ---', '36')}")
                    lines.append(_c(user_prompt, "2"))
            elif e.event == "tool_call":
                tool = e.data.get("tool", "?")
                params = e.data.get("params", {})
                lines.append(f"{ts(e)}   {arrow_call} {tool}({_truncate_params(params)})")
            elif e.event == "tool_result":
                tool = e.data.get("tool", "?")
                preview = e.data.get("result_preview", "")
                error = e.data.get("error")
                if error:
                    lines.append(f"{ts(e)}   {arrow_result} {tool}: {_c(f'ERROR: {error}', '31')}")
                else:
                    lines.append(f"{ts(e)}   {arrow_result} {tool}: {preview[:100]}")
            elif e.event == "step_end":
                preview = e.data.get("output_preview", "")
                duration = e.data.get("duration", 0)
                cost = e.data.get("cost", 0)
                lines.append(f'{ts(e)}   {arrow_call} output: "{preview[:200]}"')
                lines.append(f"{ts(e)}   ({duration:.1f}s, ${cost:.4f})")
                lines.append("")
            elif e.event == "pipeline_end":
                total_cost = e.data.get("total_cost", 0)
                duration = e.data.get("duration", 0)
                total_in = e.data.get("total_input", 0)
                total_out = e.data.get("total_output", 0)
                total = f"Total: {duration:.1f}s, ${total_cost:.4f}, {total_in:,} in / {total_out:,} out"
                lines.append(f"{ts(e)} {_c(total, '1')}")
        return "\n".join(lines)
