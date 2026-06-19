#!/usr/bin/env python3
"""Minimal coding agent — interactive REPL powered by Claude via OpenRouter."""

import json
import os
import sys
import threading
import time
import httpx

from dotenv import load_dotenv
from tools import ALL_TOOLS

load_dotenv()

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "qwen/qwen3.7-max"
AVAILABLE_MODELS = [
    "qwen/qwen3.7-max",
    "qwen/qwen3.6-plus",
    "moonshotai/kimi-k2.6",
    "z-ai/glm-5.1",
    "z-ai/glm-4.5-air:free",
    "z-ai/glm-5v-turbo",
    "qwen/qwen3-235b-a22b:free",
    "google/gemini-2.5-flash-preview",
    "deepseek/deepseek-chat-v3-0324:free",
    "openrouter/elephant-alpha",
    "google/gemma-4-31b-it",
]

MAX_TOKENS = 4096


class RequestCancelled(Exception):
    pass

API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    print("Error: OPENROUTER_API_KEY environment variable is required.")
    sys.exit(1)


# Convert Anthropic-format schemas to OpenAI-compatible format
def _to_openai_tool(tool) -> dict:
    s = tool.schema
    return {
        "type": "function",
        "function": {
            "name": s["name"],
            "description": s["description"],
            "parameters": s["input_schema"],
        },
    }

def execute_tool(name: str, params: dict, tool_handlers: dict, mcp_clients: list | None = None) -> str:
    handler = tool_handlers.get(name)
    if handler:
        try:
            return handler(params)
        except Exception as e:
            return f"Error: {e}"
    for client in (mcp_clients or []):
        if client.has_tool(name):
            try:
                return client.call_tool(name, params)
            except Exception as e:
                return f"Error: {e}"
    return f"Error: Unknown tool '{name}'"

def call_api_streaming(messages: list, tools: list, model: str = MODEL, cancel_event: threading.Event | None = None):
    """Yield parsed SSE data dicts from a streaming API call, with retry on overload."""
    max_retries = 5
    for attempt in range(max_retries):
        with httpx.stream(
            "POST",
            API_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": MAX_TOKENS,
                "tools": tools,
                "messages": messages,
                "stream": True,
            },
            timeout=300.0,
        ) as response:
            if response.status_code in (429, 529):
                delay = 2 ** attempt
                print(f"  [API overloaded, retrying in {delay}s...]")
                time.sleep(delay)
                continue
            response.raise_for_status()
            for line in response.iter_lines():
                if cancel_event and cancel_event.is_set():
                    raise RequestCancelled()
                if line.startswith("data: "):
                    payload = line[6:]
                    if payload == "[DONE]":
                        return
                    try:
                        yield json.loads(payload)
                    except json.JSONDecodeError:
                        # OpenRouter occasionally sends malformed chunks; skip silently.
                        continue
            return
    return


def agent_loop(
    user_message: str,
    messages: list,
    model: str = MODEL,
    cancel_event: threading.Event | None = None,
    mcp_clients: list | None = None,
    tools: list | None = None,
    trace: object | None = None,
    step_label: str | None = None,
    submit_result_schema: dict | None = None,
) -> dict:
    from langfuse_client import get_langfuse, null_ctx
    lf = get_langfuse()

    initial_len = len(messages)
    messages.append({"role": "user", "content": user_message})

    active_tool_list = tools if tools is not None else ALL_TOOLS
    tool_handlers = {t.name: t.handler for t in active_tool_list}
    active_tools = [_to_openai_tool(t) for t in active_tool_list]
    for client in (mcp_clients or []):
        active_tools.extend(client.tools)
    if submit_result_schema is not None:
        active_tools.append(submit_result_schema)

    total_input = 0
    total_output = 0
    total_cost = 0.0

    try:
        while True:
            content_parts: list[str] = []
            tool_calls_acc: dict[int, dict] = {}
            finish_reason = None
            usage: dict = {}

            messages_snapshot = list(messages)
            with (lf.start_as_current_observation(
                name="api-call",
                as_type="generation",
                model=model,
                model_parameters={"max_tokens": MAX_TOKENS},
                input=messages_snapshot,
            ) if lf else null_ctx()) as gen:
                for chunk in call_api_streaming(messages, active_tools, model, cancel_event):
                    if not chunk.get("choices"):
                        continue
                    choice = chunk["choices"][0]
                    finish_reason = choice.get("finish_reason") or finish_reason
                    delta = choice.get("delta", {})

                    # Accumulate and stream text content
                    if delta.get("content"):
                        print(delta["content"], end="", flush=True)
                        content_parts.append(delta["content"])

                    # Accumulate tool call deltas
                    for tc in delta.get("tool_calls", []):
                        idx = tc.get("index")
                        if idx is None:
                            continue
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": tc.get("id", ""),
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        acc = tool_calls_acc[idx]
                        if tc.get("id"):
                            acc["id"] = tc["id"]
                        fn = tc.get("function", {})
                        if fn.get("name"):
                            acc["function"]["name"] += fn["name"]
                        if fn.get("arguments"):
                            acc["function"]["arguments"] += fn["arguments"]

                    if chunk.get("usage"):
                        usage = chunk["usage"]

                if gen is not None:
                    gen.update(
                        output="".join(content_parts) or [tool_calls_acc[i] for i in sorted(tool_calls_acc)],
                        usage_details={
                            "input": usage.get("prompt_tokens", 0),
                            "output": usage.get("completion_tokens", 0),
                        },
                        cost_details={"total": usage.get("cost", 0.0)},
                    )

            if content_parts:
                print()  # newline after streamed content

            total_input += usage.get("prompt_tokens", 0)
            total_output += usage.get("completion_tokens", 0)
            total_cost += usage.get("cost", 0.0)

            if trace is not None:
                trace.log(step=step_label, event="api_call",
                          input_tokens=usage.get("prompt_tokens", 0),
                          output_tokens=usage.get("completion_tokens", 0),
                          cost=usage.get("cost", 0.0))

            content = "".join(content_parts) or None
            tool_calls = [tool_calls_acc[i] for i in sorted(tool_calls_acc)] or None
            message: dict = {"role": "assistant"}
            if content is not None:
                message["content"] = content
            if tool_calls is not None:
                message["tool_calls"] = tool_calls
            messages.append(message)

            if finish_reason == "stop":
                break

            if finish_reason == "tool_calls" and tool_calls:
                tool_results = []
                structured_result = None
                for tool_call in tool_calls:
                    name = tool_call["function"]["name"]
                    raw_args = tool_call["function"]["arguments"]
                    try:
                        params = json.loads(raw_args)
                    except json.JSONDecodeError as e:
                        print(f"  [Error: malformed tool call arguments for '{name}': {e}]")
                        print(f"  Raw arguments: {raw_args[:200]}")
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": f"[AGENT_ERROR] Failed to parse tool arguments for '{name}': {e}",
                        })
                        continue
                    if name == "submit_result" and submit_result_schema is not None:
                        print(f"  [submit_result] {params}")
                        structured_result = params
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": "Result accepted.",
                        })
                        continue
                    params_str = str(params)
                    if len(params_str) > 50:
                        params_str = params_str[:50] + "..."
                    print(f"  [Tool: {name}], params: {params_str}")
                    if trace is not None:
                        trace.log(step=step_label, event="tool_call", tool=name, params=params)
                    with (lf.start_as_current_observation(
                        name=f"tool:{name}",
                        as_type="tool",
                        input=params,
                    ) if lf else null_ctx()) as tool_obs:
                        result = execute_tool(name, params, tool_handlers, mcp_clients)
                        if tool_obs is not None:
                            tool_obs.update(
                                output=result[:2000],
                                level="ERROR" if result.startswith("Error:") else "DEFAULT",
                            )
                    if trace is not None:
                        from trace import _preview
                        trace.log(step=step_label, event="tool_result", tool=name,
                                  result_preview=_preview(result),
                                  error=result if result.startswith("Error:") else None)
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result,
                    })
                messages.extend(tool_results)
                if structured_result is not None:
                    return {"input_tokens": total_input, "output_tokens": total_output, "cost": total_cost, "result": structured_result}

    except RequestCancelled:
        del messages[initial_len:]
        return {"input_tokens": total_input, "output_tokens": total_output, "cost": total_cost, "cancelled": True}

    return {"input_tokens": total_input, "output_tokens": total_output, "cost": total_cost}
