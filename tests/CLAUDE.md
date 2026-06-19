# tests

pytest suite for the harness. Run with `pytest` from the repo root.

## Code in this directory

- `conftest.py` — adds the repo root to `sys.path` and force-disables Langfuse for the whole session (real `LANGFUSE_*` keys in `.env` would otherwise make span flushes hang)
- `test_harness.py` — the bulk of the suite: workflow/agent loading and validation, `run_pipeline` control flow (loops, `when`, `stop_on`, STOP), artifacts, traces, replay, and `submit_result` schema generation
- `test_trace.py` — `Trace`/`TraceEvent` save/load roundtrips, summary/detail formatting, and trace-event logging from `agent_loop`
