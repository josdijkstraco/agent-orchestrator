#!/usr/bin/env python3
"""Agent loop powered by Claude (and other models) via OpenRouter.

`agent_loop()` drives one agent turn-by-turn: stream a completion, run any tool
calls, feed results back, repeat until the model stops or submits a structured
result. Streaming accumulation lives in `streaming.py`; this module owns the
turn orchestration, tool dispatch, tracing, and OpenRouter transport.
"""

import json
import os
import sys
import threading
import time

import httpx
from dotenv import load_dotenv

from langfuse_client import get_langfuse, null_ctx
from streaming import StreamResult, accumulate_stream
from tools import ALL_TOOLS

load_dotenv()

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "z-ai/glm-5.2"
AVAILABLE_MODELS = [
    "z-ai/glm-5.2",
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

API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    print("Error: OPENROUTER_API_KEY environment variable is required.")
    sys.exit(1)


class RequestCancelled(Exception):
    pass


def _to_openai_tool(tool) -> dict:
    """Convert an Anthropic-format Tool schema to OpenAI function-calling format."""
    s = tool.schema
    return {
        "type": "function",
        "function": {"name": s["name"], "description": s["description"], "parameters": s["input_schema"]},
    }


def execute_tool(name: str, params: dict, tool_handlers: dict, mcp_clients: list | None = None) -> str:
    """Run a built-in handler or MCP tool by name, returning its string output."""
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
            headers={"Authorization": f"Bearer {API_KEY}", "content-type": "application/json"},
            json={"model": model, "max_tokens": MAX_TOKENS, "tools": tools, "messages": messages, "stream": True},
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
    raise RuntimeError(f"API persistently overloaded after {max_retries} retries")


def _build_tool_context(tools: list | None, mcp_clients: list | None,
                        submit_result_schema: dict | None) -> tuple[dict, list]:
    """Return (tool_handlers, openai_tool_schemas) for the active tool set."""
    active_tool_list = tools if tools is not None else ALL_TOOLS
    tool_handlers = {t.name: t.handler for t in active_tool_list}
    schemas = [_to_openai_tool(t) for t in active_tool_list]
    for client in (mcp_clients or []):
        schemas.extend(client.tools)
    if submit_result_schema is not None:
        schemas.append(submit_result_schema)
    return tool_handlers, schemas


def _run_turn(messages: list, schemas: list, model: str,
              cancel_event: threading.Event | None, lf) -> StreamResult:
    """Stream one model response, recording it as a Langfuse generation."""
    snapshot = list(messages)
    span = (lf.start_as_current_observation(
        name="api-call", as_type="generation", model=model,
        model_parameters={"max_tokens": MAX_TOKENS}, input=snapshot,
    ) if lf else null_ctx())
    with span as gen:
        result = accumulate_stream(
            call_api_streaming(messages, schemas, model, cancel_event),
            on_text=lambda t: print(t, end="", flush=True),
        )
        if gen is not None:
            gen.update(
                output=result.content or result.tool_calls or [],
                usage_details={"input": result.usage.get("prompt_tokens", 0), "output": result.usage.get("completion_tokens", 0)},
                cost_details={"total": result.usage.get("cost", 0.0)},
            )
    if result.content:
        print()  # newline after streamed content
    return result


def _assistant_message(result: StreamResult) -> dict:
    """Build the assistant message to append from a streamed turn."""
    message: dict = {"role": "assistant"}
    if result.content is not None:
        message["content"] = result.content
    if result.tool_calls is not None:
        message["tool_calls"] = result.tool_calls
    return message


def _tool_result(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _run_tool_calls(tool_calls: list, tool_handlers: dict, mcp_clients: list | None,
                    submit_result_schema: dict | None, trace, step_label, lf) -> tuple[list, dict | None]:
    """Execute each tool call, returning (tool_result_messages, structured_result)."""
    from tracing import _preview

    tool_results: list = []
    structured_result: dict | None = None
    for tool_call in tool_calls:
        name = tool_call["function"]["name"]
        raw_args = tool_call["function"]["arguments"]
        try:
            params = json.loads(raw_args)
        except json.JSONDecodeError as e:
            print(f"  [Error: malformed tool call arguments for '{name}': {e}]")
            print(f"  Raw arguments: {raw_args[:200]}")
            tool_results.append(_tool_result(tool_call["id"], f"[AGENT_ERROR] Failed to parse tool arguments for '{name}': {e}"))
            continue

        if name == "submit_result" and submit_result_schema is not None:
            print(f"  [submit_result] {params}")
            structured_result = params
            tool_results.append(_tool_result(tool_call["id"], "Result accepted."))
            continue

        params_str = str(params)
        if len(params_str) > 50:
            params_str = params_str[:50] + "..."
        print(f"  [Tool: {name}], params: {params_str}")
        if trace is not None:
            trace.log(step=step_label, event="tool_call", tool=name, params=params)
        obs = (lf.start_as_current_observation(name=f"tool:{name}", as_type="tool", input=params) if lf else null_ctx())
        with obs as tool_obs:
            result = execute_tool(name, params, tool_handlers, mcp_clients)
            if tool_obs is not None:
                tool_obs.update(output=result[:2000], level="ERROR" if result.startswith("Error:") else "DEFAULT")
        if trace is not None:
            trace.log(step=step_label, event="tool_result", tool=name,
                      result_preview=_preview(result), error=result if result.startswith("Error:") else None)
        tool_results.append(_tool_result(tool_call["id"], result))
    return tool_results, structured_result


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
    max_turns: int = 20,
) -> dict:
    """Drive an agent turn-by-turn until it stops or submits a structured result.

    Returns a usage dict with input_tokens/output_tokens/cost, plus `result` when a
    structured result was submitted or `cancelled` when interrupted mid-stream.
    """
    lf = get_langfuse()
    initial_len = len(messages)
    messages.append({"role": "user", "content": user_message})
    tool_handlers, schemas = _build_tool_context(tools, mcp_clients, submit_result_schema)

    totals = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}

    def usage(**extra) -> dict:
        return {**totals, **extra}

    try:
        for _turn in range(max_turns):
            result = _run_turn(messages, schemas, model, cancel_event, lf)
            totals["input_tokens"] += result.usage.get("prompt_tokens", 0)
            totals["output_tokens"] += result.usage.get("completion_tokens", 0)
            totals["cost"] += result.usage.get("cost", 0.0)
            if trace is not None:
                trace.log(step=step_label, event="api_call",
                          input_tokens=result.usage.get("prompt_tokens", 0),
                          output_tokens=result.usage.get("completion_tokens", 0),
                          cost=result.usage.get("cost", 0.0))

            messages.append(_assistant_message(result))

            if result.finish_reason == "stop":
                break
            if result.finish_reason == "tool_calls" and result.tool_calls:
                tool_results, structured_result = _run_tool_calls(
                    result.tool_calls, tool_handlers, mcp_clients, submit_result_schema, trace, step_label, lf)
                messages.extend(tool_results)
                if structured_result is not None:
                    return usage(result=structured_result)
        else:
            raise RuntimeError(f"{step_label or 'agent'}: exceeded {max_turns} turns without end_turn")
    except RequestCancelled:
        del messages[initial_len:]
        return usage(cancelled=True)

    return usage()
