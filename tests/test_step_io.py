"""Unit tests for step_io.py — input assembly and output extraction."""

from step_io import extract_output, resolve_input


def test_resolve_input_single_source_passthrough():
    assert resolve_input(["plan"], {"plan": "the plan"}) == "the plan"


def test_resolve_input_multiple_sources_labeled():
    out = resolve_input(["plan", "research"], {"plan": "A", "research": "B"})
    assert out == "## Input: plan\nA\n\n## Input: research\nB"


def test_resolve_input_skips_unknown_refs():
    assert resolve_input(["plan", "missing"], {"plan": "A"}) == "## Input: plan\nA"


def test_resolve_input_empty():
    assert resolve_input([], {}) == ""


def test_extract_output_prefers_structured_result():
    assert extract_output({"decision": "APPROVED", "feedback": "ok"}, []) == "decision: APPROVED\nfeedback: ok"


def test_extract_output_falls_back_to_last_assistant_message():
    messages = [
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "ignored"},
        {"role": "assistant", "content": "last"},
    ]
    assert extract_output(None, messages) == "last"


def test_extract_output_none_when_no_assistant_content():
    assert extract_output(None, [{"role": "user", "content": "hi"}]) is None
