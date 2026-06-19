# tests/test_harness.py
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_load_workflow_finds_by_name(tmp_path):
    """Scans directory and returns step dicts for a matching workflow name."""
    wf = tmp_path / "mywf.yaml"
    wf.write_text(
        "name: mywf\nsteps:\n"
        "  - agent: agent1\n    inputs: [__input__]\n"
        "  - agent: agent2\n    inputs: [agent1]\n"
    )
    from harness import load_workflow
    result = load_workflow("mywf", workflows_dir=tmp_path)
    assert result["steps"] == [
        {"agent": "agent1", "id": "agent1", "inputs": ["__input__"], "prompt": None, "outputs": None, "when": None, "loop_on": None, "loop_to": None, "max_loops": None, "output": None, "stop_on": None},
        {"agent": "agent2", "id": "agent2", "inputs": ["agent1"], "prompt": None, "outputs": None, "when": None, "loop_on": None, "loop_to": None, "max_loops": None, "output": None, "stop_on": None},
    ]


def test_load_workflow_missing_inputs_raises(tmp_path):
    """Raises ValueError when a step omits the inputs field."""
    wf = tmp_path / "mywf.yaml"
    wf.write_text("name: mywf\nsteps:\n  - agent: agent1\n")
    from harness import load_workflow
    with pytest.raises(ValueError, match="must declare 'inputs' explicitly"):
        load_workflow("mywf", workflows_dir=tmp_path)


def test_load_workflow_step_with_prompt(tmp_path):
    """Step dict includes prompt when present in YAML."""
    wf = tmp_path / "mywf.yaml"
    wf.write_text(
        "name: mywf\nsteps:\n  - agent: agent1\n    inputs: [__input__]\n    prompt: 'Focus on tests.'\n"
    )
    from harness import load_workflow
    steps = load_workflow("mywf", workflows_dir=tmp_path)["steps"]
    assert steps == [{"agent": "agent1", "id": "agent1", "inputs": ["__input__"], "prompt": "Focus on tests.", "outputs": None, "when": None, "loop_on": None, "loop_to": None, "max_loops": None, "output": None, "stop_on": None}]


def test_load_workflow_step_without_prompt_is_none(tmp_path):
    """Step dict has prompt=None when prompt field is absent."""
    wf = tmp_path / "mywf.yaml"
    wf.write_text("name: mywf\nsteps:\n  - agent: agent1\n    inputs: [__input__]\n")
    from harness import load_workflow
    steps = load_workflow("mywf", workflows_dir=tmp_path)["steps"]
    assert steps[0]["prompt"] is None


def test_substitute_prompt_vars():
    """{step_id} expands to that step's output; {__input__} expands to the workflow command; unknown names left alone."""
    from harness import substitute_prompt_vars
    outputs = {"__input__": "Fix login", "planner_1": "Plan A"}
    assert substitute_prompt_vars("{__input__}", outputs) == "Fix login"
    assert substitute_prompt_vars("See {planner_1}", outputs) == "See Plan A"
    assert substitute_prompt_vars("{unknown}", outputs) == "{unknown}"


def test_load_workflow_not_found_raises(tmp_path):
    """Raises SystemExit when no workflow matches the name."""
    from harness import load_workflow
    with pytest.raises(ValueError):
        load_workflow("missing", workflows_dir=tmp_path)


def test_load_agent_finds_by_name(tmp_path):
    """Scans directory and returns AgentConfig for a matching agent name."""
    ag = tmp_path / "myagent.yaml"
    ag.write_text(
        "name: myagent\nprompt: Do stuff.\ntools:\n  - read_file\n"
    )
    from harness import load_agent
    config = load_agent("myagent", agents_dir=tmp_path)
    assert config["prompt"].startswith("Do stuff.")
    assert "Working directory:" in config["prompt"]
    assert "Current date:" in config["prompt"]
    assert config["tool_names"] == ["read_file"]
    assert config["model"] is None  # not set in yaml


def test_load_agent_with_model(tmp_path):
    """Model field is read from agent YAML when present."""
    ag = tmp_path / "myagent.yaml"
    ag.write_text(
        "name: myagent\nmodel: google/gemini-2.5-flash-preview\nprompt: Hi.\ntools:\n  - bash\n"
    )
    from harness import load_agent
    config = load_agent("myagent", agents_dir=tmp_path)
    assert config["model"] == "google/gemini-2.5-flash-preview"


def test_load_agent_not_found_raises(tmp_path):
    """Raises SystemExit when no agent matches the name."""
    from harness import load_agent
    with pytest.raises(ValueError):
        load_agent("missing", agents_dir=tmp_path)


def test_load_agent_unknown_tool_raises(tmp_path):
    """Raises SystemExit when agent YAML references an unknown tool name."""
    ag = tmp_path / "myagent.yaml"
    ag.write_text(
        "name: myagent\nprompt: Hi.\ntools:\n  - nonexistent_tool\n"
    )
    from harness import load_agent
    with pytest.raises(ValueError):
        load_agent("myagent", agents_dir=tmp_path)


def test_main_workflow_subcommand(monkeypatch):
    """main() with workflow subcommand loads workflow and runs pipeline."""
    from unittest.mock import patch
    from harness import main

    # Mock load_workflow and run_pipeline
    with patch("harness.load_workflow") as mock_load_wf, \
         patch("harness.run_pipeline") as mock_run_pipeline:
        mock_steps = [{"agent": "agent1", "id": "agent1", "prompt": None}, {"agent": "agent2", "id": "agent2", "prompt": None}]
        mock_load_wf.return_value = {"steps": mock_steps}

        # Set sys.argv for argparse
        test_argv = ["harness.py", "workflow", "example", "Fix the bug"]
        monkeypatch.setattr(sys, "argv", test_argv)

        main()

        # Assert load_workflow was called with correct name
        mock_load_wf.assert_called_once_with("example")
        # Assert run_pipeline was called with steps and prompt
        mock_run_pipeline.assert_called_once_with(
            mock_steps,
            "Fix the bug",
            workflow_name="example",
        )


def test_main_workflow_no_prompt_errors(monkeypatch):
    """main() errors when no prompt is given."""
    from unittest.mock import patch
    from harness import main

    with patch("harness.load_workflow") as mock_load_wf:
        mock_load_wf.return_value = {"steps": [{"agent": "agent1", "id": "agent1", "prompt": None}]}

        test_argv = ["harness.py", "workflow", "example"]
        monkeypatch.setattr(sys, "argv", test_argv)

        with pytest.raises(SystemExit):
            main()


def _agent_config():
    return {
        "prompt": "System prompt",
        "tools": [],
        "tool_names": [],
        "skill_names": [],
        "mcp_names": [],
        "model": None,
        "output": None,
    }


def test_run_pipeline_appends_step_prompt(monkeypatch):
    """Step prompt is prepended to current_input with double newline separator."""
    from unittest.mock import patch
    from harness import run_pipeline

    captured_inputs = []

    def fake_agent_loop(user_message, messages, **kwargs):
        captured_inputs.append(user_message)
        messages.append({"role": "assistant", "content": "output"})
        return {}

    with patch("harness.load_agent", return_value=_agent_config()), \
         patch("harness.agent_loop", side_effect=fake_agent_loop), \
         patch("harness.build_mcp_clients", return_value=[]):
        run_pipeline([{"agent": "agent1", "id": "agent1", "prompt": "Extra guidance", "inputs": ["__input__"]}], "Initial command")

    assert captured_inputs[0] == "Extra guidance\n\nInitial command"


def test_run_pipeline_stops_on_stop_signal(monkeypatch, capsys):
    """Pipeline exits early when an agent ends its output with a standalone STOP line."""
    from unittest.mock import patch
    from harness import run_pipeline

    call_count = 0

    def fake_agent_loop(user_message, messages, **kwargs):
        nonlocal call_count
        call_count += 1
        messages.append({"role": "assistant", "content": "Nothing needed here.\nSTOP"})
        return {}

    with patch("harness.load_agent", return_value=_agent_config()), \
         patch("harness.agent_loop", side_effect=fake_agent_loop), \
         patch("harness.build_mcp_clients", return_value=[]):
        run_pipeline(
            [{"agent": "agent1", "id": "agent1", "prompt": None, "inputs": ["__input__"]},
             {"agent": "agent2", "id": "agent2", "prompt": None, "inputs": ["agent1"]}],
            "Do the thing"
        )

    assert call_count == 1
    captured = capsys.readouterr()
    assert "Nothing to do." in captured.err


def test_run_pipeline_does_not_stop_on_incidental_stop(monkeypatch, capsys):
    """Prose mentioning STOP (or NONSTOP) must not halt the pipeline — only a final STOP line does."""
    from unittest.mock import patch
    from harness import run_pipeline

    outputs = ["I did not STOP the process; this is a NONSTOP job.", "All done."]
    call_count = [0]

    def fake_agent_loop(user_message, messages, **kwargs):
        messages.append({"role": "assistant", "content": outputs[call_count[0]]})
        call_count[0] += 1
        return {}

    with patch("harness.load_agent", return_value=_agent_config()), \
         patch("harness.agent_loop", side_effect=fake_agent_loop), \
         patch("harness.build_mcp_clients", return_value=[]):
        run_pipeline(
            [{"agent": "agent1", "id": "agent1", "prompt": None, "inputs": ["__input__"]},
             {"agent": "agent2", "id": "agent2", "prompt": None, "inputs": ["agent1"]}],
            "Do the thing"
        )

    assert call_count[0] == 2  # both steps ran; incidental "STOP"/"NONSTOP" did not halt
    captured = capsys.readouterr()
    assert "Nothing to do." not in captured.err


def test_run_pipeline_loop_on_does_not_fire_on_embedded_token():
    """loop_on 'APPROVED' must not fire inside 'UNAPPROVED' (whole-word match, not substring)."""
    from unittest.mock import patch
    from harness import run_pipeline

    call_count = [0]

    def fake_agent_loop(user_message, messages, **kwargs):
        call_count[0] += 1
        messages.append({"role": "assistant", "content": "Still UNAPPROVED."})
        return {}

    steps = [
        {"agent": "implementer", "id": "implementer", "prompt": None, "inputs": ["__input__", "reviewer"]},
        {"agent": "reviewer", "id": "reviewer", "prompt": None, "inputs": ["implementer"],
         "loop_on": "APPROVED", "loop_to": "implementer", "max_loops": 3},
    ]

    with patch("harness.load_agent", return_value=_agent_config()), \
         patch("harness.agent_loop", side_effect=fake_agent_loop), \
         patch("harness.build_mcp_clients", return_value=[]):
        run_pipeline(steps, "Fix the bug")

    assert call_count[0] == 2  # no loop-back: 'APPROVED' is only a substring of 'UNAPPROVED'


def test_run_pipeline_no_step_prompt_passes_input_unchanged(monkeypatch):
    """Input is passed unchanged to agent_loop when step has no prompt."""
    from unittest.mock import patch
    from harness import run_pipeline

    captured_inputs = []

    def fake_agent_loop(user_message, messages, **kwargs):
        captured_inputs.append(user_message)
        messages.append({"role": "assistant", "content": "output"})
        return {}

    with patch("harness.load_agent", return_value=_agent_config()), \
         patch("harness.agent_loop", side_effect=fake_agent_loop), \
         patch("harness.build_mcp_clients", return_value=[]):
        run_pipeline([{"agent": "agent1", "id": "agent1", "prompt": None, "inputs": ["__input__"]}], "Initial command")

    assert captured_inputs[0] == "Initial command"


# --- Loop-back: load_workflow tests ---

def test_load_workflow_step_with_loop_fields(tmp_path):
    """Loop fields are parsed from YAML and returned in step dict."""
    wf = tmp_path / "mywf.yaml"
    wf.write_text(
        "name: mywf\nsteps:\n"
        "  - agent: implementer\n    inputs: [__input__]\n"
        "  - agent: reviewer\n    inputs: [implementer]\n"
        "    loop_on: UNAPPROVED\n"
        "    loop_to: implementer\n"
        "    max_loops: 2\n"
    )
    from harness import load_workflow
    steps = load_workflow("mywf", workflows_dir=tmp_path)["steps"]
    assert steps[1]["loop_on"] == "UNAPPROVED"
    assert steps[1]["loop_to"] == "implementer"
    assert steps[1]["max_loops"] == 2


def test_load_workflow_step_loop_default_max_loops(tmp_path):
    """max_loops defaults to 3 when loop_on/loop_to are set but max_loops is absent."""
    wf = tmp_path / "mywf.yaml"
    wf.write_text(
        "name: mywf\nsteps:\n"
        "  - agent: implementer\n    inputs: [__input__]\n"
        "  - agent: reviewer\n    inputs: [implementer]\n"
        "    loop_on: UNAPPROVED\n"
        "    loop_to: implementer\n"
    )
    from harness import load_workflow
    steps = load_workflow("mywf", workflows_dir=tmp_path)["steps"]
    assert steps[1]["max_loops"] == 3


def test_load_workflow_step_no_loop_fields_are_none(tmp_path):
    """Steps without loop fields have all loop fields as None."""
    wf = tmp_path / "mywf.yaml"
    wf.write_text("name: mywf\nsteps:\n  - agent: agent1\n    inputs: [__input__]\n")
    from harness import load_workflow
    steps = load_workflow("mywf", workflows_dir=tmp_path)["steps"]
    assert steps[0]["loop_on"] is None
    assert steps[0]["loop_to"] is None
    assert steps[0]["max_loops"] is None


def test_load_workflow_loop_on_without_loop_to_raises(tmp_path):
    """loop_on without loop_to triggers SystemExit."""
    wf = tmp_path / "mywf.yaml"
    wf.write_text(
        "name: mywf\nsteps:\n"
        "  - agent: agent1\n    inputs: [__input__]\n"
        "  - agent: agent2\n    inputs: [agent1]\n"
        "    loop_on: UNAPPROVED\n"
    )
    from harness import load_workflow
    with pytest.raises(ValueError):
        load_workflow("mywf", workflows_dir=tmp_path)


def test_load_workflow_loop_to_forward_reference_raises(tmp_path):
    """loop_to referencing a later step triggers SystemExit."""
    wf = tmp_path / "mywf.yaml"
    wf.write_text(
        "name: mywf\nsteps:\n"
        "  - agent: agent1\n    inputs: [__input__]\n"
        "    loop_on: RETRY\n"
        "    loop_to: agent2\n"
        "  - agent: agent2\n    inputs: [agent1]\n"
    )
    from harness import load_workflow
    with pytest.raises(ValueError):
        load_workflow("mywf", workflows_dir=tmp_path)


# --- Loop-back: run_pipeline tests ---


def test_run_pipeline_loops_back_on_keyword():
    """Reviewer outputs UNAPPROVED once then clean; pipeline runs implementer twice."""
    from unittest.mock import patch
    from harness import run_pipeline

    call_log = []
    outputs = {
        "implementer": ["impl output v1", "impl output v2"],
        "reviewer": ["This needs work. UNAPPROVED", "Looks good."],
    }
    counters = {"implementer": 0, "reviewer": 0}

    def fake_load_agent(name, **kwargs):
        return _agent_config()

    def fake_agent_loop(user_message, messages, **kwargs):
        call_log.append(user_message)
        call_index = len(call_log) - 1
        sequence = ["implementer", "reviewer", "implementer", "reviewer"]
        agent = sequence[call_index]
        output = outputs[agent][counters[agent]]
        counters[agent] += 1
        messages.append({"role": "assistant", "content": output})
        return {}

    steps = [
        {"agent": "implementer", "id": "implementer", "prompt": None, "inputs": ["__input__", "reviewer"]},
        {"agent": "reviewer", "id": "reviewer", "prompt": None, "inputs": ["implementer"], "loop_on": "UNAPPROVED", "loop_to": "implementer", "max_loops": 3},
    ]

    with patch("harness.load_agent", side_effect=fake_load_agent), \
         patch("harness.agent_loop", side_effect=fake_agent_loop), \
         patch("harness.build_mcp_clients", return_value=[]):
        run_pipeline(steps, "Fix the bug")

    assert len(call_log) == 4
    # Second implementer call receives the reviewer's UNAPPROVED feedback
    assert "UNAPPROVED" in call_log[2]


def test_run_pipeline_loop_respects_max_loops(capsys):
    """Reviewer always outputs UNAPPROVED; pipeline stops looping after max_loops and continues."""
    from unittest.mock import patch
    from harness import run_pipeline

    call_count = [0]

    def fake_load_agent(name, **kwargs):
        return _agent_config()

    def fake_agent_loop(user_message, messages, **kwargs):
        call_count[0] += 1
        messages.append({"role": "assistant", "content": "UNAPPROVED always"})
        return {}

    steps = [
        {"agent": "implementer", "id": "implementer", "prompt": None, "inputs": ["__input__", "reviewer"]},
        {"agent": "reviewer", "id": "reviewer", "prompt": None, "inputs": ["implementer"], "loop_on": "UNAPPROVED", "loop_to": "implementer", "max_loops": 2},
    ]

    with patch("harness.load_agent", side_effect=fake_load_agent), \
         patch("harness.agent_loop", side_effect=fake_agent_loop), \
         patch("harness.build_mcp_clients", return_value=[]):
        run_pipeline(steps, "Fix the bug")

    # implementer(1) → reviewer(1,loop1) → implementer(2) → reviewer(2,loop2) → implementer(3) → reviewer(3,max exceeded) = 6
    assert call_count[0] == 6
    captured = capsys.readouterr()
    assert "max_loops" in captured.err


def test_run_pipeline_no_loop_when_keyword_absent():
    """Pipeline runs linearly when loop_on keyword is not in output."""
    from unittest.mock import patch
    from harness import run_pipeline

    call_count = [0]

    def fake_load_agent(name, **kwargs):
        return _agent_config()

    def fake_agent_loop(user_message, messages, **kwargs):
        call_count[0] += 1
        messages.append({"role": "assistant", "content": "Looks great. APPROVED"})
        return {}

    steps = [
        {"agent": "implementer", "id": "implementer", "prompt": None, "inputs": ["__input__", "reviewer"]},
        {"agent": "reviewer", "id": "reviewer", "prompt": None, "inputs": ["implementer"], "loop_on": "UNAPPROVED", "loop_to": "implementer", "max_loops": 3},
    ]

    with patch("harness.load_agent", side_effect=fake_load_agent), \
         patch("harness.agent_loop", side_effect=fake_agent_loop), \
         patch("harness.build_mcp_clients", return_value=[]):
        run_pipeline(steps, "Fix the bug")

    assert call_count[0] == 2


def test_run_pipeline_step_prompt_not_duplicated_on_loop():
    """Step prompt is prepended once per execution, not accumulated across loop iterations."""
    from unittest.mock import patch
    from harness import run_pipeline

    captured_inputs = []
    call_count = [0]

    def fake_load_agent(name, **kwargs):
        return _agent_config()

    def fake_agent_loop(user_message, messages, **kwargs):
        captured_inputs.append(user_message)
        call_count[0] += 1
        # implementer runs twice (calls 0 and 2), reviewer runs twice (calls 1 and 3)
        if call_count[0] == 1:
            messages.append({"role": "assistant", "content": "impl done"})
        elif call_count[0] == 2:
            messages.append({"role": "assistant", "content": "UNAPPROVED please fix"})
        elif call_count[0] == 3:
            messages.append({"role": "assistant", "content": "impl done v2"})
        else:
            messages.append({"role": "assistant", "content": "Looks good."})
        return {}

    steps = [
        {"agent": "implementer", "id": "implementer", "prompt": "Do the work.", "inputs": ["__input__", "reviewer"]},
        {"agent": "reviewer", "id": "reviewer", "prompt": "Review it.", "inputs": ["implementer"], "loop_on": "UNAPPROVED", "loop_to": "implementer", "max_loops": 3},
    ]

    with patch("harness.load_agent", side_effect=fake_load_agent), \
         patch("harness.agent_loop", side_effect=fake_agent_loop), \
         patch("harness.build_mcp_clients", return_value=[]):
        run_pipeline(steps, "Fix the bug")

    # implementer called twice: prompt should be "Do the work.\n\n<input>" each time
    assert captured_inputs[0].startswith("Do the work.\n\n")
    assert captured_inputs[2].startswith("Do the work.\n\n")
    # reviewer called twice: prompt should be "Review it.\n\n<input>" each time
    assert captured_inputs[1].startswith("Review it.\n\n")
    assert captured_inputs[3].startswith("Review it.\n\n")
    # The implementer's prompt must NOT appear in the reviewer's input
    assert "Do the work." not in captured_inputs[1]
    assert "Do the work." not in captured_inputs[3]


def test_run_pipeline_creates_trace(tmp_path):
    """run_pipeline creates a trace JSON file when traces_dir is provided."""
    import json
    from unittest.mock import patch
    from harness import run_pipeline

    def fake_load_agent(name, **kwargs):
        return _agent_config()

    def fake_agent_loop(user_message, messages, **kwargs):
        messages.append({"role": "assistant", "content": "output from agent"})
        return {"input_tokens": 100, "output_tokens": 50, "cost": 0.001}

    steps = [
        {"agent": "planner", "id": "planner", "prompt": None, "inputs": ["__input__"], "loop_on": None, "loop_to": None, "max_loops": None},
    ]

    with patch("harness.load_agent", side_effect=fake_load_agent), \
         patch("harness.agent_loop", side_effect=fake_agent_loop), \
         patch("harness.build_mcp_clients", return_value=[]):
        run_pipeline(steps, "Fix the bug", traces_dir=tmp_path)

    trace_files = list(tmp_path.glob("*.json"))
    assert len(trace_files) == 1

    data = json.loads(trace_files[0].read_text())
    assert data["workflow"] == "pipeline"
    assert data["command"] == "Fix the bug"
    assert data["status"] == "completed"

    event_types = [e["event"] for e in data["events"]]
    assert "pipeline_start" in event_types
    assert "step_start" in event_types
    assert "step_end" in event_types
    assert "pipeline_end" in event_types


def test_run_pipeline_saves_snapshots(tmp_path):
    """run_pipeline saves conversation snapshots per step."""
    from unittest.mock import patch
    from harness import run_pipeline

    def fake_load_agent(name, **kwargs):
        return _agent_config()

    def fake_agent_loop(user_message, messages, **kwargs):
        messages.append({"role": "assistant", "content": "step output"})
        return {"input_tokens": 50, "output_tokens": 25, "cost": 0.0005}

    steps = [
        {"agent": "planner", "id": "planner", "prompt": None, "inputs": ["__input__"], "loop_on": None, "loop_to": None, "max_loops": None},
        {"agent": "implementer", "id": "implementer", "prompt": None, "inputs": ["planner"], "loop_on": None, "loop_to": None, "max_loops": None},
    ]

    with patch("harness.load_agent", side_effect=fake_load_agent), \
         patch("harness.agent_loop", side_effect=fake_agent_loop), \
         patch("harness.build_mcp_clients", return_value=[]):
        run_pipeline(steps, "Fix the bug", traces_dir=tmp_path)

    trace_file = list(tmp_path.glob("*.json"))[0]
    trace_id = trace_file.stem

    snapshot_dir = tmp_path / f"{trace_id}_messages"
    assert snapshot_dir.is_dir()
    assert (snapshot_dir / "step_0_planner.json").exists()
    assert (snapshot_dir / "step_1_implementer.json").exists()


def test_run_pipeline_trace_on_failure(tmp_path):
    """Trace is saved with status 'failed' when an agent raises."""
    import json
    from unittest.mock import patch
    from harness import run_pipeline

    def fake_load_agent(name, **kwargs):
        return _agent_config()

    def fake_agent_loop(user_message, messages, **kwargs):
        raise RuntimeError("API exploded")

    steps = [
        {"agent": "planner", "id": "planner", "prompt": None, "inputs": ["__input__"], "loop_on": None, "loop_to": None, "max_loops": None},
    ]

    with patch("harness.load_agent", side_effect=fake_load_agent), \
         patch("harness.agent_loop", side_effect=fake_agent_loop), \
         patch("harness.build_mcp_clients", return_value=[]):
        with pytest.raises(RuntimeError):
            run_pipeline(steps, "Fix the bug", traces_dir=tmp_path)

    trace_files = list(tmp_path.glob("*.json"))
    assert len(trace_files) == 1
    data = json.loads(trace_files[0].read_text())
    assert data["status"] == "failed"


def test_trace_list_command(tmp_path, monkeypatch, capsys):
    """trace list subcommand prints a summary table of traces."""
    import json
    from harness import main

    trace_data = {
        "id": "abc12345",
        "workflow": "example",
        "command": "Fix bug",
        "started_at": 1000.0,
        "status": "completed",
        "events": [
            {"timestamp": 1000.0, "step": "0:planner", "event": "step_start", "data": {}},
            {"timestamp": 1001.0, "step": None, "event": "pipeline_end",
             "data": {"total_cost": 0.0342, "duration": 45.0, "total_input": 1000, "total_output": 500, "status": "completed"}},
        ],
    }
    (tmp_path / "abc12345.json").write_text(json.dumps(trace_data))

    monkeypatch.setattr(sys, "argv", ["harness.py", "trace", "list", "--traces-dir", str(tmp_path)])
    main()

    captured = capsys.readouterr()
    assert "abc12345" in captured.out
    assert "example" in captured.out
    assert "completed" in captured.out


def test_trace_show_command(tmp_path, monkeypatch, capsys):
    """trace show subcommand prints detail view of a trace."""
    import json
    from harness import main

    trace_data = {
        "id": "abc12345",
        "workflow": "example",
        "command": "Fix bug",
        "started_at": 1000.0,
        "status": "completed",
        "events": [
            {"timestamp": 1000.0, "step": None, "event": "pipeline_start", "data": {"workflow": "example", "command": "Fix bug"}},
            {"timestamp": 1000.1, "step": "0:planner", "event": "step_start", "data": {"model": "qwen", "tools": ["read_file"]}},
            {"timestamp": 1000.5, "step": "0:planner", "event": "step_end", "data": {"output_preview": "Here is the plan.", "duration": 3.4, "cost": 0.005}},
            {"timestamp": 1001.0, "step": None, "event": "pipeline_end",
             "data": {"total_cost": 0.005, "duration": 3.4, "total_input": 500, "total_output": 200, "status": "completed"}},
        ],
    }
    (tmp_path / "abc12345.json").write_text(json.dumps(trace_data))

    monkeypatch.setattr(sys, "argv", ["harness.py", "trace", "show", "abc12345", "--traces-dir", str(tmp_path)])
    main()

    captured = capsys.readouterr()
    assert "abc12345" in captured.out
    assert "planner" in captured.out
    assert "Here is the plan." in captured.out


# --- Artifacts system tests ---


def test_load_workflow_step_with_outputs(tmp_path):
    """outputs field is parsed from YAML and returned in step dict."""
    wf = tmp_path / "mywf.yaml"
    wf.write_text(
        "name: mywf\nsteps:\n"
        "  - agent: planner\n"
        "    id: plan\n"
        "    inputs: [__input__]\n"
        "    outputs:\n"
        "      plan: json\n"
        "      summary: text\n"
    )
    from harness import load_workflow
    steps = load_workflow("mywf", workflows_dir=tmp_path)["steps"]
    assert steps[0]["outputs"] == {"plan": "json", "summary": "text"}


def test_load_workflow_step_without_outputs_is_none(tmp_path):
    """Steps without outputs have outputs=None."""
    wf = tmp_path / "mywf.yaml"
    wf.write_text("name: mywf\nsteps:\n  - agent: agent1\n    inputs: [__input__]\n")
    from harness import load_workflow
    steps = load_workflow("mywf", workflows_dir=tmp_path)["steps"]
    assert steps[0]["outputs"] is None


def test_parse_artifacts_extracts_fenced_blocks():
    """parse_artifacts extracts content from fenced code blocks tagged with artifact names."""
    from harness import parse_artifacts
    raw = 'Some text\n```plan\n{"steps": [1, 2]}\n```\nMore text\n```summary\nDone.\n```'
    result = parse_artifacts(raw, {"plan": "json", "summary": "text"})
    assert result["plan"] == '{"steps": [1, 2]}'
    assert result["summary"] == "Done."


def test_parse_artifacts_fallback_to_raw():
    """parse_artifacts falls back to full raw output when no fenced block found."""
    from harness import parse_artifacts
    raw = "Just plain text output with no fenced blocks."
    result = parse_artifacts(raw, {"plan": "json"})
    assert result["plan"] == raw


def test_run_pipeline_artifacts_injected_with_labels():
    """Downstream step receives labeled inputs when multiple sources are selected."""
    from unittest.mock import patch
    from harness import run_pipeline

    captured_inputs = []

    def fake_agent_loop(user_message, messages, **kwargs):
        captured_inputs.append(user_message)
        messages.append({"role": "assistant", "content": f"output_{len(captured_inputs)}"})
        return {}

    steps = [
        {"agent": "planner", "id": "plan", "prompt": None, "inputs": ["__input__"],
         "outputs": None, "when": None,
         "loop_on": None, "loop_to": None, "max_loops": None},
        {"agent": "researcher", "id": "research", "prompt": None, "inputs": ["__input__"],
         "outputs": None, "when": None,
         "loop_on": None, "loop_to": None, "max_loops": None},
        {"agent": "implementer", "id": "implementer", "prompt": None, "inputs": ["plan", "research"],
         "outputs": None, "when": None,
         "loop_on": None, "loop_to": None, "max_loops": None},
    ]

    with patch("harness.load_agent", return_value=_agent_config()), \
         patch("harness.agent_loop", side_effect=fake_agent_loop), \
         patch("harness.build_mcp_clients", return_value=[]):
        run_pipeline(steps, "Do the thing")

    # Third step has two inputs -> each gets a labeled header
    assert captured_inputs[2] == "## Input: plan\noutput_1\n\n## Input: research\noutput_2"


def test_run_pipeline_single_input_is_unlabeled():
    """A step with exactly one input receives its raw content, no header."""
    from unittest.mock import patch
    from harness import run_pipeline

    captured_inputs = []

    def fake_agent_loop(user_message, messages, **kwargs):
        captured_inputs.append(user_message)
        messages.append({"role": "assistant", "content": "output"})
        return {}

    steps = [
        {"agent": "planner", "id": "plan", "prompt": None, "inputs": ["__input__"],
         "outputs": None, "when": None,
         "loop_on": None, "loop_to": None, "max_loops": None},
        {"agent": "implementer", "id": "implementer", "prompt": None, "inputs": ["plan"],
         "outputs": None, "when": None,
         "loop_on": None, "loop_to": None, "max_loops": None},
    ]

    with patch("harness.load_agent", return_value=_agent_config()), \
         patch("harness.agent_loop", side_effect=fake_agent_loop), \
         patch("harness.build_mcp_clients", return_value=[]):
        run_pipeline(steps, "Do the thing")

    assert captured_inputs[1] == "output"


def test_run_pipeline_artifacts_stored_from_fenced_blocks():
    """Artifacts declared in outputs are parsed and stored in step_outputs."""
    from unittest.mock import patch
    from harness import run_pipeline

    captured_inputs = []

    def fake_agent_loop(user_message, messages, **kwargs):
        captured_inputs.append(user_message)
        if len(captured_inputs) == 1:
            messages.append({"role": "assistant", "content": 'Here is the plan:\n```plan\n{"steps": ["a", "b"]}\n```\nDone.'})
        else:
            messages.append({"role": "assistant", "content": "implemented"})
        return {}

    steps = [
        {"agent": "planner", "id": "planner_step", "prompt": None, "inputs": ["__input__"],
         "outputs": {"plan": "json"}, "when": None,
         "loop_on": None, "loop_to": None, "max_loops": None},
        {"agent": "implementer", "id": "implementer", "prompt": None, "inputs": ["plan"],
         "outputs": None, "when": None,
         "loop_on": None, "loop_to": None, "max_loops": None},
    ]

    with patch("harness.load_agent", return_value=_agent_config()), \
         patch("harness.agent_loop", side_effect=fake_agent_loop), \
         patch("harness.build_mcp_clients", return_value=[]):
        run_pipeline(steps, "Do the thing")

    # Single input -> raw content with no header
    assert captured_inputs[1] == '{"steps": ["a", "b"]}'


# --- Conditional branching (when) tests ---


def test_load_workflow_step_with_when(tmp_path):
    """when field is parsed from YAML and returned in step dict."""
    wf = tmp_path / "mywf.yaml"
    wf.write_text(
        "name: mywf\nsteps:\n"
        "  - agent: reviewer\n"
        "    id: review\n"
        "    inputs: [__input__]\n"
        "  - agent: implementer\n"
        "    inputs: [review]\n"
        "    when: 'REVISION_NEEDED in review'\n"
    )
    from harness import load_workflow
    steps = load_workflow("mywf", workflows_dir=tmp_path)["steps"]
    assert steps[1]["when"] == "REVISION_NEEDED in review"


def test_run_pipeline_when_true_runs_step():
    """Step executes when its when condition matches."""
    from unittest.mock import patch
    from harness import run_pipeline

    call_count = [0]

    def fake_agent_loop(user_message, messages, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            messages.append({"role": "assistant", "content": "REVISION_NEEDED: fix tests"})
        else:
            messages.append({"role": "assistant", "content": "fixed"})
        return {}

    steps = [
        {"agent": "reviewer", "id": "review", "prompt": None, "inputs": ["__input__"],
         "outputs": None, "when": None,
         "loop_on": None, "loop_to": None, "max_loops": None},
        {"agent": "implementer", "id": "implementer", "prompt": None, "inputs": ["review"],
         "outputs": None, "when": "REVISION_NEEDED in review",
         "loop_on": None, "loop_to": None, "max_loops": None},
    ]

    with patch("harness.load_agent", return_value=_agent_config()), \
         patch("harness.agent_loop", side_effect=fake_agent_loop), \
         patch("harness.build_mcp_clients", return_value=[]):
        run_pipeline(steps, "Check it")

    assert call_count[0] == 2  # both steps ran


def test_run_pipeline_when_false_skips_step():
    """Step is skipped when its when condition does not match."""
    from unittest.mock import patch
    from harness import run_pipeline

    call_count = [0]

    def fake_agent_loop(user_message, messages, **kwargs):
        call_count[0] += 1
        messages.append({"role": "assistant", "content": "APPROVED: looks great"})
        return {}

    steps = [
        {"agent": "reviewer", "id": "review", "prompt": None, "inputs": ["__input__"],
         "outputs": None, "when": None,
         "loop_on": None, "loop_to": None, "max_loops": None},
        {"agent": "implementer", "id": "implementer", "prompt": None, "inputs": ["review"],
         "outputs": None, "when": "REVISION_NEEDED in review",
         "loop_on": None, "loop_to": None, "max_loops": None},
    ]

    with patch("harness.load_agent", return_value=_agent_config()), \
         patch("harness.agent_loop", side_effect=fake_agent_loop), \
         patch("harness.build_mcp_clients", return_value=[]):
        run_pipeline(steps, "Check it")

    assert call_count[0] == 1  # only reviewer ran, implementer skipped


def test_load_workflow_when_references_unknown_id_raises(tmp_path):
    """when referencing an unknown step id triggers ValueError."""
    wf = tmp_path / "mywf.yaml"
    wf.write_text(
        "name: mywf\nsteps:\n"
        "  - agent: agent1\n    inputs: [__input__]\n"
        "  - agent: agent2\n    inputs: [agent1]\n"
        "    when: 'FOO in nonexistent'\n"
    )
    from harness import load_workflow
    with pytest.raises(ValueError):
        load_workflow("mywf", workflows_dir=tmp_path)


def test_load_workflow_when_and_loop_on_raises(tmp_path):
    """Step cannot have both loop_on and when."""
    wf = tmp_path / "mywf.yaml"
    wf.write_text(
        "name: mywf\nsteps:\n"
        "  - agent: implementer\n"
        "    id: impl\n"
        "    inputs: [__input__]\n"
        "  - agent: reviewer\n"
        "    inputs: [impl]\n"
        "    when: 'FOO in impl'\n"
        "    loop_on: UNAPPROVED\n"
        "    loop_to: implementer\n"
    )
    from harness import load_workflow
    with pytest.raises(ValueError):
        load_workflow("mywf", workflows_dir=tmp_path)


def test_load_workflow_outputs_conflict_with_existing_id_raises(tmp_path):
    """Output artifact name that conflicts with an existing step id raises ValueError."""
    wf = tmp_path / "mywf.yaml"
    wf.write_text(
        "name: mywf\nsteps:\n"
        "  - agent: planner\n"
        "    id: plan\n"
        "    inputs: [__input__]\n"
        "  - agent: implementer\n"
        "    id: impl\n"
        "    inputs: [plan]\n"
        "    outputs:\n"
        "      plan: json\n"  # conflicts with planner's id
    )
    from harness import load_workflow
    with pytest.raises(ValueError):
        load_workflow("mywf", workflows_dir=tmp_path)


def test_load_workflow_inputs_can_reference_artifact_names(tmp_path):
    """inputs field can reference artifact names from outputs, not just step ids."""
    wf = tmp_path / "mywf.yaml"
    wf.write_text(
        "name: mywf\nsteps:\n"
        "  - agent: planner\n"
        "    id: plan_step\n"
        "    inputs: [__input__]\n"
        "    outputs:\n"
        "      plan: json\n"
        "  - agent: implementer\n"
        "    inputs: [plan]\n"
    )
    from harness import load_workflow
    steps = load_workflow("mywf", workflows_dir=tmp_path)["steps"]
    assert steps[1]["inputs"] == ["plan"]


def test_replay_command(tmp_path, monkeypatch):
    """replay subcommand loads trace and re-runs from given step."""
    import json
    from unittest.mock import patch as mock_patch
    from harness import main

    trace_data = {
        "id": "abc12345",
        "workflow": "example",
        "command": "Fix bug",
        "started_at": 1000.0,
        "status": "completed",
        "events": [],
    }
    (tmp_path / "abc12345.json").write_text(json.dumps(trace_data))

    wf_dir = tmp_path / "workflows"
    wf_dir.mkdir()
    (wf_dir / "example.yaml").write_text(
        "name: example\nsteps:\n"
        "  - agent: planner\n    inputs: [__input__]\n"
        "  - agent: implementer\n    inputs: [planner]\n"
    )

    captured_args = {}

    def fake_run_pipeline(steps, command, **kwargs):
        captured_args["steps"] = steps
        captured_args["command"] = command
        captured_args.update(kwargs)

    monkeypatch.setattr(sys, "argv", [
        "harness.py", "replay", "abc12345", "--from-step", "1",
        "--traces-dir", str(tmp_path), "--workflows-dir", str(wf_dir),
    ])

    with mock_patch("harness.run_pipeline", side_effect=fake_run_pipeline):
        main()

    assert len(captured_args["steps"]) == 1
    assert captured_args["steps"][0]["agent"] == "implementer"


# --- Structured outputs: build_submit_result_tool tests ---


def test_build_submit_result_tool_generates_openai_format():
    """build_submit_result_tool generates an OpenAI-compatible tool dict from response_schema."""
    from harness import build_submit_result_tool

    schema = {
        "decision": {"type": "string", "enum": ["APPROVED", "REJECTED"]},
        "feedback": {"type": "string"},
    }
    tool = build_submit_result_tool(schema)

    assert tool["type"] == "function"
    assert tool["function"]["name"] == "submit_result"
    params = tool["function"]["parameters"]
    assert params["type"] == "object"
    assert params["properties"]["decision"] == {"type": "string", "enum": ["APPROVED", "REJECTED"]}
    assert params["properties"]["feedback"] == {"type": "string"}
    assert set(params["required"]) == {"decision", "feedback"}


def test_build_submit_result_tool_number_and_boolean():
    """build_submit_result_tool handles number and boolean types."""
    from harness import build_submit_result_tool

    schema = {
        "confidence": {"type": "number"},
        "approved": {"type": "boolean"},
    }
    tool = build_submit_result_tool(schema)

    params = tool["function"]["parameters"]
    assert params["properties"]["confidence"] == {"type": "number"}
    assert params["properties"]["approved"] == {"type": "boolean"}


# --- Structured outputs: eval_condition tests ---


def test_eval_condition_match():
    """eval_condition returns True when field matches value."""
    from harness import eval_condition
    assert eval_condition("decision == REJECTED", {"decision": "REJECTED", "feedback": "bad"}) is True


def test_eval_condition_no_match():
    """eval_condition returns False when field does not match value."""
    from harness import eval_condition
    assert eval_condition("decision == REJECTED", {"decision": "APPROVED", "feedback": "good"}) is False


def test_eval_condition_missing_field():
    """eval_condition returns False when field is not in result."""
    from harness import eval_condition
    assert eval_condition("decision == REJECTED", {"feedback": "good"}) is False


def test_eval_condition_whitespace_handling():
    """eval_condition handles extra whitespace around field and value."""
    from harness import eval_condition
    assert eval_condition("  decision  ==  REJECTED  ", {"decision": "REJECTED"}) is True


# --- Structured outputs: load_workflow validation tests ---


def test_load_workflow_step_with_output(tmp_path):
    """output is parsed from YAML and returned in step dict."""
    wf = tmp_path / "mywf.yaml"
    wf.write_text(
        "name: mywf\nsteps:\n"
        "  - agent: reviewer\n"
        "    inputs: [__input__]\n"
        "    output:\n"
        "      decision:\n"
        "        type: string\n"
        "        enum: [APPROVED, REJECTED]\n"
        "      feedback:\n"
        "        type: string\n"
    )
    from harness import load_workflow
    steps = load_workflow("mywf", workflows_dir=tmp_path)["steps"]
    assert steps[0]["output"] == {
        "decision": {"type": "string", "enum": ["APPROVED", "REJECTED"]},
        "feedback": {"type": "string"},
    }


def test_load_workflow_step_without_output_is_none(tmp_path):
    """Steps without output have output=None."""
    wf = tmp_path / "mywf.yaml"
    wf.write_text("name: mywf\nsteps:\n  - agent: agent1\n    inputs: [__input__]\n")
    from harness import load_workflow
    steps = load_workflow("mywf", workflows_dir=tmp_path)["steps"]
    assert steps[0]["output"] is None
    assert steps[0]["stop_on"] is None


def test_load_workflow_stop_on_without_output_raises(tmp_path):
    """stop_on without output raises ValueError."""
    wf = tmp_path / "mywf.yaml"
    wf.write_text(
        "name: mywf\nsteps:\n"
        "  - agent: agent1\n"
        "    inputs: [__input__]\n"
        "    stop_on: status == STOP\n"
    )
    from harness import load_workflow
    with pytest.raises(ValueError, match="output"):
        load_workflow("mywf", workflows_dir=tmp_path)


def test_load_workflow_stop_on_and_loop_on_raises(tmp_path):
    """Step cannot have both stop_on and loop_on."""
    wf = tmp_path / "mywf.yaml"
    wf.write_text(
        "name: mywf\nsteps:\n"
        "  - agent: implementer\n    inputs: [__input__]\n"
        "  - agent: reviewer\n    inputs: [implementer]\n"
        "    output:\n"
        "      decision:\n"
        "        type: string\n"
        "        enum: [APPROVED, REJECTED, STOP]\n"
        "    stop_on: decision == STOP\n"
        "    loop_on: decision == REJECTED\n"
        "    loop_to: implementer\n"
    )
    from harness import load_workflow
    with pytest.raises(ValueError, match="stop_on.*loop_on"):
        load_workflow("mywf", workflows_dir=tmp_path)


def test_load_workflow_output_field_missing_type_raises(tmp_path):
    """output field without type raises ValueError."""
    wf = tmp_path / "mywf.yaml"
    wf.write_text(
        "name: mywf\nsteps:\n"
        "  - agent: agent1\n"
        "    inputs: [__input__]\n"
        "    output:\n"
        "      decision:\n"
        "        enum: [YES, NO]\n"
    )
    from harness import load_workflow
    with pytest.raises(ValueError, match="type"):
        load_workflow("mywf", workflows_dir=tmp_path)


def test_load_workflow_loop_on_with_output_validates_field(tmp_path):
    """loop_on with output validates that the field exists in the schema."""
    wf = tmp_path / "mywf.yaml"
    wf.write_text(
        "name: mywf\nsteps:\n"
        "  - agent: implementer\n    inputs: [__input__]\n"
        "  - agent: reviewer\n    inputs: [implementer]\n"
        "    output:\n"
        "      decision:\n"
        "        type: string\n"
        "        enum: [APPROVED, REJECTED]\n"
        "    loop_on: nonexistent == REJECTED\n"
        "    loop_to: implementer\n"
    )
    from harness import load_workflow
    with pytest.raises(ValueError, match="nonexistent"):
        load_workflow("mywf", workflows_dir=tmp_path)


def test_load_workflow_loop_on_with_output_validates_enum_value(tmp_path):
    """loop_on value must be in the enum if the field has one."""
    wf = tmp_path / "mywf.yaml"
    wf.write_text(
        "name: mywf\nsteps:\n"
        "  - agent: implementer\n    inputs: [__input__]\n"
        "  - agent: reviewer\n    inputs: [implementer]\n"
        "    output:\n"
        "      decision:\n"
        "        type: string\n"
        "        enum: [APPROVED, REJECTED]\n"
        "    loop_on: decision == INVALID\n"
        "    loop_to: implementer\n"
    )
    from harness import load_workflow
    with pytest.raises(ValueError, match="INVALID"):
        load_workflow("mywf", workflows_dir=tmp_path)


def test_load_workflow_loop_on_with_output_valid(tmp_path):
    """loop_on with output and valid field/value passes validation."""
    wf = tmp_path / "mywf.yaml"
    wf.write_text(
        "name: mywf\nsteps:\n"
        "  - agent: implementer\n    inputs: [__input__]\n"
        "  - agent: reviewer\n    inputs: [implementer]\n"
        "    output:\n"
        "      decision:\n"
        "        type: string\n"
        "        enum: [APPROVED, REJECTED]\n"
        "      feedback:\n"
        "        type: string\n"
        "    loop_on: decision == REJECTED\n"
        "    loop_to: implementer\n"
    )
    from harness import load_workflow
    steps = load_workflow("mywf", workflows_dir=tmp_path)["steps"]
    assert steps[1]["loop_on"] == "decision == REJECTED"
    assert steps[1]["output"] is not None


# --- Structured outputs: agent_loop interception tests ---


def test_agent_loop_intercepts_submit_result():
    """agent_loop returns structured result when submit_result tool is called."""
    from unittest.mock import patch
    from agent_openrouter import agent_loop
    from harness import build_submit_result_tool

    schema = {
        "decision": {"type": "string", "enum": ["APPROVED", "REJECTED"]},
        "feedback": {"type": "string"},
    }
    submit_tool = build_submit_result_tool(schema)

    chunks = [
        {
            "choices": [{
                "finish_reason": "tool_calls",
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_123",
                        "function": {
                            "name": "submit_result",
                            "arguments": '{"decision": "REJECTED", "feedback": "needs tests"}'
                        }
                    }]
                }
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.01}
        },
    ]

    with patch("agent_openrouter.call_api_streaming", return_value=iter(chunks)):
        messages = [{"role": "system", "content": "You are a reviewer."}]
        usage = agent_loop("Review this", messages, submit_result_schema=submit_tool)

    assert usage["result"] == {"decision": "REJECTED", "feedback": "needs tests"}
    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 50


def test_agent_loop_without_submit_result_schema_works_unchanged():
    """agent_loop without submit_result_schema works exactly as before."""
    from unittest.mock import patch
    from agent_openrouter import agent_loop

    chunks = [
        {
            "choices": [{
                "finish_reason": "stop",
                "delta": {"content": "Looks good. APPROVED"}
            }],
            "usage": {"prompt_tokens": 50, "completion_tokens": 20, "cost": 0.005}
        },
    ]

    with patch("agent_openrouter.call_api_streaming", return_value=iter(chunks)):
        messages = [{"role": "system", "content": "You are a reviewer."}]
        usage = agent_loop("Review this", messages)

    assert "result" not in usage
    assert usage["input_tokens"] == 50


# --- Structured outputs: run_pipeline control flow tests ---


def test_run_pipeline_stop_on_structured_result(tmp_path):
    """Pipeline exits early when stop_on condition matches structured result."""
    from unittest.mock import patch
    from harness import run_pipeline

    call_count = [0]

    def fake_agent_loop(user_message, messages, **kwargs):
        call_count[0] += 1
        messages.append({"role": "assistant", "content": "No cards found."})
        return {"result": {"status": "STOP", "context": ""}}

    steps = [
        {"agent": "agent1", "id": "agent1", "prompt": None, "inputs": ["__input__"], "output": {"status": {"type": "string", "enum": ["FOUND", "STOP"]}, "context": {"type": "string"}}, "stop_on": "status == STOP"},
        {"agent": "agent2", "id": "agent2", "prompt": None, "inputs": ["agent1"]},
    ]

    with patch("harness.load_agent", return_value=_agent_config()), \
         patch("harness.agent_loop", side_effect=fake_agent_loop), \
         patch("harness.build_mcp_clients", return_value=[]):
        run_pipeline(steps, "Do the thing", traces_dir=tmp_path)

    assert call_count[0] == 1


def test_run_pipeline_stop_on_structured_result_no_text_output(tmp_path):
    """Pipeline exits early on stop_on even when agent produces no text output."""
    from unittest.mock import patch
    from harness import run_pipeline

    call_count = [0]

    def fake_agent_loop(user_message, messages, **kwargs):
        call_count[0] += 1
        # No text content appended — only structured result via submit_result
        return {"result": {"status": "STOP", "context": ""}}

    steps = [
        {"agent": "agent1", "id": "agent1", "prompt": None, "inputs": ["__input__"], "output": {"status": {"type": "string", "enum": ["FOUND", "STOP"]}, "context": {"type": "string"}}, "stop_on": "status == STOP"},
        {"agent": "agent2", "id": "agent2", "prompt": None, "inputs": ["agent1"]},
    ]

    with patch("harness.load_agent", return_value=_agent_config()), \
         patch("harness.agent_loop", side_effect=fake_agent_loop), \
         patch("harness.build_mcp_clients", return_value=[]):
        run_pipeline(steps, "Do the thing", traces_dir=tmp_path)

    assert call_count[0] == 1


def test_run_pipeline_structured_result_used_as_output_when_no_text(tmp_path):
    """When agent produces no text but has a structured result, the result is passed to the next step."""
    from unittest.mock import patch
    from harness import run_pipeline

    captured_inputs = []

    def fake_agent_loop(user_message, messages, **kwargs):
        captured_inputs.append(user_message)
        if len(captured_inputs) == 1:
            # First step: structured result only, no text output
            return {"result": {"status": "FOUND", "context": "Fix the login bug"}}
        else:
            # Second step: normal text output
            messages.append({"role": "assistant", "content": "Plan created."})
            return {}

    steps = [
        {"agent": "agent1", "id": "agent1", "prompt": "Pick a card.", "inputs": ["__input__"], "output": {"status": {"type": "string", "enum": ["FOUND", "STOP"]}, "context": {"type": "string"}}, "stop_on": "status == STOP"},
        {"agent": "agent2", "id": "agent2", "prompt": None, "inputs": ["agent1"]},
    ]

    with patch("harness.load_agent", return_value=_agent_config()), \
         patch("harness.agent_loop", side_effect=fake_agent_loop), \
         patch("harness.build_mcp_clients", return_value=[]):
        run_pipeline(steps, "Do the thing", traces_dir=tmp_path)

    # The second step should receive the structured result as JSON, not the original command
    assert "Fix the login bug" in captured_inputs[1]
    assert "Do the thing" not in captured_inputs[1]


def test_run_pipeline_loop_on_structured_result(tmp_path):
    """Pipeline loops back when loop_on condition matches structured result."""
    from unittest.mock import patch
    from harness import run_pipeline

    call_log = []
    call_count = [0]

    def fake_load_agent(name, **kwargs):
        return _agent_config()

    def fake_agent_loop(user_message, messages, **kwargs):
        call_count[0] += 1
        call_log.append(call_count[0])
        if call_count[0] == 1:
            messages.append({"role": "assistant", "content": "impl v1"})
            return {}
        elif call_count[0] == 2:
            messages.append({"role": "assistant", "content": "Needs work."})
            return {"result": {"decision": "REJECTED", "feedback": "missing tests"}}
        elif call_count[0] == 3:
            messages.append({"role": "assistant", "content": "impl v2"})
            return {}
        else:
            messages.append({"role": "assistant", "content": "Looks good."})
            return {"result": {"decision": "APPROVED", "feedback": "all good"}}

    steps = [
        {"agent": "implementer", "id": "implementer", "prompt": None, "inputs": ["__input__", "reviewer"]},
        {"agent": "reviewer", "id": "reviewer", "prompt": None, "inputs": ["implementer"],
         "output": {"decision": {"type": "string", "enum": ["APPROVED", "REJECTED"]}, "feedback": {"type": "string"}},
         "loop_on": "decision == REJECTED", "loop_to": "implementer", "max_loops": 3},
    ]

    with patch("harness.load_agent", side_effect=fake_load_agent), \
         patch("harness.agent_loop", side_effect=fake_agent_loop), \
         patch("harness.build_mcp_clients", return_value=[]):
        run_pipeline(steps, "Fix the bug", traces_dir=tmp_path)

    assert call_count[0] == 4


def test_run_pipeline_structured_no_loop_when_condition_not_met(tmp_path):
    """Pipeline does not loop when structured result does not match loop_on."""
    from unittest.mock import patch
    from harness import run_pipeline

    call_count = [0]

    def fake_load_agent(name, **kwargs):
        return _agent_config()

    def fake_agent_loop(user_message, messages, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            messages.append({"role": "assistant", "content": "impl done"})
            return {}
        else:
            messages.append({"role": "assistant", "content": "Approved."})
            return {"result": {"decision": "APPROVED", "feedback": "great"}}

    steps = [
        {"agent": "implementer", "id": "implementer", "prompt": None, "inputs": ["__input__", "reviewer"]},
        {"agent": "reviewer", "id": "reviewer", "prompt": None, "inputs": ["implementer"],
         "output": {"decision": {"type": "string", "enum": ["APPROVED", "REJECTED"]}, "feedback": {"type": "string"}},
         "loop_on": "decision == REJECTED", "loop_to": "implementer", "max_loops": 3},
    ]

    with patch("harness.load_agent", side_effect=fake_load_agent), \
         patch("harness.agent_loop", side_effect=fake_agent_loop), \
         patch("harness.build_mcp_clients", return_value=[]):
        run_pipeline(steps, "Fix the bug", traces_dir=tmp_path)

    assert call_count[0] == 2


def test_run_pipeline_passes_submit_result_schema_to_agent_loop(tmp_path):
    """run_pipeline passes the submit_result_schema to agent_loop when step has output."""
    from unittest.mock import patch
    from harness import run_pipeline

    captured_kwargs = []

    def fake_agent_loop(user_message, messages, **kwargs):
        captured_kwargs.append(kwargs)
        messages.append({"role": "assistant", "content": "done"})
        return {"result": {"status": "FOUND", "context": "card info"}}

    steps = [
        {"agent": "agent1", "id": "agent1", "prompt": None, "inputs": ["__input__"],
         "output": {"status": {"type": "string"}, "context": {"type": "string"}}},
    ]

    with patch("harness.load_agent", return_value=_agent_config()), \
         patch("harness.agent_loop", side_effect=fake_agent_loop), \
         patch("harness.build_mcp_clients", return_value=[]):
        run_pipeline(steps, "Do the thing", traces_dir=tmp_path)

    assert "submit_result_schema" in captured_kwargs[0]
    assert captured_kwargs[0]["submit_result_schema"]["function"]["name"] == "submit_result"


def test_run_pipeline_no_schema_no_submit_result_kwarg(tmp_path):
    """run_pipeline passes submit_result_schema=None when step has no output."""
    from unittest.mock import patch
    from harness import run_pipeline

    captured_kwargs = []

    def fake_agent_loop(user_message, messages, **kwargs):
        captured_kwargs.append(kwargs)
        messages.append({"role": "assistant", "content": "done"})
        return {}

    steps = [
        {"agent": "agent1", "id": "agent1", "prompt": None, "inputs": ["__input__"]},
    ]

    with patch("harness.load_agent", return_value=_agent_config()), \
         patch("harness.agent_loop", side_effect=fake_agent_loop), \
         patch("harness.build_mcp_clients", return_value=[]):
        run_pipeline(steps, "Do the thing", traces_dir=tmp_path)

    assert captured_kwargs[0].get("submit_result_schema") is None
