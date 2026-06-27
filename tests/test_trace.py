import json
import time
from unittest.mock import patch

from tracing import TraceEvent, Trace


def test_trace_event_creation():
    ts = time.time()
    event = TraceEvent(timestamp=ts, step="0:planner", event="tool_call", data={"tool": "read_file"})
    assert event.timestamp == ts
    assert event.step == "0:planner"
    assert event.event == "tool_call"
    assert event.data == {"tool": "read_file"}


def test_trace_event_to_dict():
    ts = 1000.0
    event = TraceEvent(timestamp=ts, step=None, event="pipeline_start", data={"workflow": "example"})
    d = event.to_dict()
    assert d == {"timestamp": 1000.0, "step": None, "event": "pipeline_start", "data": {"workflow": "example"}}


def test_trace_log_appends_event():
    trace = Trace(workflow="example", command="Fix bug")
    trace.log(step="0:planner", event="tool_call", tool="read_file", params={"path": "foo.py"})
    assert len(trace.events) == 1
    e = trace.events[0]
    assert e.step == "0:planner"
    assert e.event == "tool_call"
    assert e.data == {"tool": "read_file", "params": {"path": "foo.py"}}
    assert isinstance(e.timestamp, float)


def test_trace_to_dict():
    trace = Trace(workflow="example", command="Fix bug")
    trace.status = "completed"
    d = trace.to_dict()
    assert d["workflow"] == "example"
    assert d["command"] == "Fix bug"
    assert d["status"] == "completed"
    assert isinstance(d["id"], str)
    assert len(d["id"]) == 8
    assert d["events"] == []


def test_trace_save_creates_json_file(tmp_path):
    trace = Trace(workflow="example", command="Fix bug")
    trace.log(step="0:planner", event="step_start", model="qwen")
    trace.status = "completed"
    trace.save(traces_dir=tmp_path)

    trace_file = tmp_path / f"{trace.id}.json"
    assert trace_file.exists()
    data = json.loads(trace_file.read_text())
    assert data["id"] == trace.id
    assert data["workflow"] == "example"
    assert data["status"] == "completed"
    assert len(data["events"]) == 1


def test_trace_load_roundtrips(tmp_path):
    trace = Trace(workflow="example", command="Fix bug")
    trace.log(step="0:planner", event="step_start", model="qwen")
    trace.log(step="0:planner", event="tool_call", tool="read_file", params={"path": "x.py"})
    trace.status = "completed"
    trace.save(traces_dir=tmp_path)

    loaded = Trace.load(trace.id, traces_dir=tmp_path)
    assert loaded.id == trace.id
    assert loaded.workflow == "example"
    assert loaded.command == "Fix bug"
    assert loaded.status == "completed"
    assert len(loaded.events) == 2
    assert loaded.events[0].event == "step_start"
    assert loaded.events[1].data["tool"] == "read_file"


def test_trace_save_snapshot(tmp_path):
    trace = Trace(workflow="example", command="Fix bug")
    messages = [
        {"role": "system", "content": "You are a planner."},
        {"role": "user", "content": "Fix bug"},
        {"role": "assistant", "content": "Here is the plan."},
    ]
    trace.save_snapshot(step_index=0, step_name="planner", messages=messages, traces_dir=tmp_path)

    snapshot_dir = tmp_path / f"{trace.id}_messages"
    assert snapshot_dir.is_dir()
    snapshot_file = snapshot_dir / "step_0_planner.json"
    assert snapshot_file.exists()
    loaded = json.loads(snapshot_file.read_text())
    assert len(loaded) == 3
    assert loaded[2]["content"] == "Here is the plan."


def test_trace_load_snapshot(tmp_path):
    trace = Trace(workflow="example", command="Fix bug")
    messages = [{"role": "user", "content": "hello"}]
    trace.save_snapshot(step_index=2, step_name="reviewer", messages=messages, traces_dir=tmp_path)

    loaded = Trace.load_snapshot(trace.id, step_index=2, step_name="reviewer", traces_dir=tmp_path)
    assert loaded == [{"role": "user", "content": "hello"}]


def test_trace_summary_row():
    trace = Trace(workflow="pick-and-fix", command="Fix the auth bug")
    trace.started_at = 1000.0
    trace.log(step=None, event="pipeline_start", workflow="pick-and-fix", command="Fix the auth bug")
    trace.log(step="0:planner", event="step_start")
    trace.log(step="0:planner", event="step_end", output_preview="plan done")
    trace.log(step="1:implementer", event="step_start")
    trace.log(step="1:implementer", event="step_end", output_preview="impl done")
    trace.log(step=None, event="pipeline_end", status="completed", total_cost=0.0342, duration=45.0, total_input=1000, total_output=500)
    trace.status = "completed"

    row = trace.summary_row()
    assert row["id"] == trace.id
    assert row["workflow"] == "pick-and-fix"
    assert row["status"] == "completed"
    assert row["steps"] == 2
    assert row["cost"] == 0.0342
    assert row["duration"] == 45.0


def test_trace_format_detail():
    trace = Trace(workflow="example", command="Fix bug")
    trace.log(step=None, event="pipeline_start", workflow="example", command="Fix bug")
    trace.log(step="0:planner", event="step_start", model="qwen", tools=["read_file"])
    trace.log(step="0:planner", event="tool_call", tool="read_file", params={"path": "foo.py"})
    trace.log(step="0:planner", event="tool_result", tool="read_file", result_preview="contents...")
    trace.log(step="0:planner", event="step_end", output_preview="Here is the plan.", duration=3.4, cost=0.005)
    trace.log(step=None, event="pipeline_end", status="completed", total_cost=0.005, duration=3.4, total_input=500, total_output=200)
    trace.status = "completed"

    output = trace.format_detail()
    assert "example" in output
    assert "Fix bug" in output
    assert "planner" in output
    assert "read_file" in output
    assert "Here is the plan." in output


def test_agent_loop_logs_trace_events():
    """agent_loop logs api_call, tool_call, and tool_result events to trace."""
    from agent_openrouter import agent_loop
    from tools import read_file as rf

    trace = Trace(workflow="test", command="test")

    call_count = [0]

    def fake_call_api_streaming(messages, tools, model, cancel_event=None):
        call_count[0] += 1
        if call_count[0] == 1:
            yield {
                "choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "read_file", "arguments": '{"path": "foo.py"}'}}]}, "finish_reason": None}],
            }
            yield {
                "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.001},
            }
        else:
            yield {
                "choices": [{"delta": {"content": "Done."}, "finish_reason": None}],
            }
            yield {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 150, "completion_tokens": 10, "cost": 0.001},
            }

    messages = [{"role": "system", "content": "You are helpful."}]

    with patch("agent_openrouter.call_api_streaming", side_effect=fake_call_api_streaming):
        agent_loop("test prompt", messages, tools=[rf], trace=trace, step_label="0:planner")

    event_types = [e.event for e in trace.events]
    assert "api_call" in event_types
    assert "tool_call" in event_types
    assert "tool_result" in event_types
    assert event_types.count("api_call") == 2
    for e in trace.events:
        assert e.step == "0:planner"
