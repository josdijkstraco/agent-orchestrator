"""Pure accumulation of an OpenAI-style streaming chat completion.

`accumulate_stream()` folds a sequence of SSE chunk dicts into the final text,
tool calls, usage, and finish reason. It has no network or global state, so the
fiddly per-index tool-call delta merging is unit-testable on plain dicts.
"""

from dataclasses import dataclass, field
from typing import Callable, Iterable


@dataclass
class StreamResult:
    content: str | None = None
    tool_calls: list[dict] | None = None
    usage: dict = field(default_factory=dict)
    finish_reason: str | None = None


def _merge_tool_call_delta(acc: dict[int, dict], deltas: list[dict]) -> None:
    """Merge a chunk's tool_call deltas into the per-index accumulator in place."""
    for tc in deltas:
        idx = tc.get("index")
        if idx is None:
            continue
        slot = acc.setdefault(idx, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
        if tc.get("id"):
            slot["id"] = tc["id"]
        fn = tc.get("function", {})
        if fn.get("name"):
            slot["function"]["name"] += fn["name"]
        if fn.get("arguments"):
            slot["function"]["arguments"] += fn["arguments"]


def accumulate_stream(chunks: Iterable[dict], on_text: Callable[[str], None] | None = None) -> StreamResult:
    """Fold streamed chunks into a StreamResult.

    `on_text`, if given, is called with each text delta as it arrives (used to
    stream output to the terminal). Tool-call argument fragments are concatenated
    by index; the last chunk carrying `usage` wins.
    """
    content_parts: list[str] = []
    tool_calls_acc: dict[int, dict] = {}
    result = StreamResult()

    for chunk in chunks:
        if not chunk.get("choices"):
            if chunk.get("usage"):
                result.usage = chunk["usage"]
            continue
        choice = chunk["choices"][0]
        result.finish_reason = choice.get("finish_reason") or result.finish_reason
        delta = choice.get("delta", {})

        if delta.get("content"):
            if on_text:
                on_text(delta["content"])
            content_parts.append(delta["content"])

        _merge_tool_call_delta(tool_calls_acc, delta.get("tool_calls", []))

        if chunk.get("usage"):
            result.usage = chunk["usage"]

    result.content = "".join(content_parts) or None
    result.tool_calls = [tool_calls_acc[i] for i in sorted(tool_calls_acc)] or None
    return result
