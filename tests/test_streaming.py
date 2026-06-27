"""Unit tests for streaming.accumulate_stream — the pure SSE delta folder."""

from streaming import accumulate_stream


def _text_chunk(text, finish=None, usage=None):
    chunk = {"choices": [{"finish_reason": finish, "delta": {"content": text}}]}
    if usage:
        chunk["usage"] = usage
    return chunk


def test_accumulates_text_across_chunks():
    result = accumulate_stream([_text_chunk("Hel"), _text_chunk("lo"), _text_chunk("!", finish="stop")])
    assert result.content == "Hello!"
    assert result.tool_calls is None
    assert result.finish_reason == "stop"


def test_on_text_called_per_delta():
    seen = []
    accumulate_stream([_text_chunk("a"), _text_chunk("b")], on_text=seen.append)
    assert seen == ["a", "b"]


def test_merges_tool_call_arguments_across_chunks():
    chunks = [
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "read_file", "arguments": '{"pa'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'th": "x"}'}}]}, "finish_reason": "tool_calls"}]},
    ]
    result = accumulate_stream(chunks)
    assert result.content is None
    assert result.tool_calls == [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "x"}'}}]
    assert result.finish_reason == "tool_calls"


def test_multiple_tool_calls_kept_in_index_order():
    chunks = [
        {"choices": [{"delta": {"tool_calls": [
            {"index": 1, "id": "b", "function": {"name": "two", "arguments": "{}"}},
            {"index": 0, "id": "a", "function": {"name": "one", "arguments": "{}"}},
        ]}}]},
    ]
    result = accumulate_stream(chunks)
    assert [tc["id"] for tc in result.tool_calls] == ["a", "b"]


def test_usage_captured_from_final_chunk():
    result = accumulate_stream([_text_chunk("hi", finish="stop", usage={"prompt_tokens": 10, "completion_tokens": 3, "cost": 0.002})])
    assert result.usage == {"prompt_tokens": 10, "completion_tokens": 3, "cost": 0.002}


def test_usage_only_chunk_without_choices():
    result = accumulate_stream([_text_chunk("hi", finish="stop"), {"usage": {"prompt_tokens": 5}}])
    assert result.content == "hi"
    assert result.usage == {"prompt_tokens": 5}


def test_empty_stream_yields_empty_result():
    result = accumulate_stream([])
    assert result.content is None
    assert result.tool_calls is None
    assert result.finish_reason is None
