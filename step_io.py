"""Helpers for moving data into and out of a pipeline step.

Pure string/dict transformations: building a step's input from prior outputs,
interpolating prompt variables, extracting declared artifacts, generating the
`submit_result` tool schema, and reading a step's text output back out.
"""

import re

# Step id under which the workflow's initial command is stored.
INPUT_KEY = "__input__"


def substitute_prompt_vars(text: str, step_outputs: dict[str, str]) -> str:
    """Replace {name} placeholders in a step prompt.

    {step_id} expands to that step's output; {__input__} expands to the workflow's
    initial command. Unknown names are left unchanged so literal braces aren't
    consumed by accident.
    """
    def repl(m: re.Match) -> str:
        name = m.group(1).strip()
        if name in step_outputs:
            return step_outputs[name]
        return m.group(0)
    return re.sub(r"\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}", repl, text)


def parse_artifacts(raw_output: str, declared_outputs: dict[str, str]) -> dict[str, str]:
    """Extract declared artifacts from agent output.

    Looks for fenced code blocks tagged with the artifact name, e.g.:
        ```plan
        {"steps": [...]}
        ```
    Falls back to the full raw output if a block isn't found.
    """
    artifacts: dict[str, str] = {}
    for name in declared_outputs:
        pattern = rf'```{re.escape(name)}\s*\n(.*?)```'
        m = re.search(pattern, raw_output, re.DOTALL)
        artifacts[name] = m.group(1).strip() if m else raw_output
    return artifacts


def build_submit_result_tool(response_schema: dict) -> dict:
    """Generate an OpenAI-format tool definition from a response_schema dict."""
    properties = {}
    for field_name, field_def in response_schema.items():
        prop: dict = {"type": field_def["type"]}
        if "enum" in field_def:
            prop["enum"] = field_def["enum"]
        properties[field_name] = prop
    return {
        "type": "function",
        "function": {
            "name": "submit_result",
            "description": "Submit your final structured result for this step. You MUST call this tool when you have reached your conclusion.",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(response_schema.keys()),
            },
        },
    }


def resolve_input(input_ids: list[str], step_outputs: dict[str, str]) -> str:
    """Assemble a step's input from prior outputs.

    A single known source passes through verbatim; multiple sources are
    concatenated under '## Input: <id>' headers.
    """
    if len(input_ids) == 1 and input_ids[0] in step_outputs:
        return step_outputs[input_ids[0]]
    parts = [f"## Input: {ref}\n{step_outputs[ref]}" for ref in input_ids if ref in step_outputs]
    return "\n\n".join(parts)


def extract_output(structured_result: dict | None, messages: list) -> str | None:
    """The step's text output: a structured result rendered as lines, else the last assistant message."""
    if structured_result:
        return "\n".join(f"{k}: {v}" for k, v in structured_result.items())
    for msg in reversed(messages):
        if msg["role"] == "assistant" and msg.get("content"):
            return msg["content"]
    return None
