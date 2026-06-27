"""Agent loading.

`load_agent()` scans `agents/` for a YAML whose `name:` matches, resolves its
tool names to `Tool` objects, appends declared skills and an environment footer
to the system prompt, and validates that referenced MCP servers exist.
"""

import datetime
import os
from pathlib import Path
from typing import TypedDict

import yaml

from mcp_client import load_mcp_config
from skills_loader import append_skills
from tools import ALL_TOOLS, Tool

_HERE = Path(__file__).parent
_TOOL_MAP = {t.name: t for t in ALL_TOOLS}
_MCP_CONFIG = load_mcp_config()


class AgentConfig(TypedDict):
    prompt: str
    tools: list[Tool]
    tool_names: list[str]
    skill_names: list[str]
    mcp_names: list[str]
    model: str | None
    output: dict | None


def _append_environment(prompt: str) -> str:
    """Append current working directory and date to a system prompt."""
    cwd = os.getcwd()
    today = datetime.date.today().isoformat()
    suffix = f"\n\n## Environment\n- Working directory: {cwd}\n- Current date: {today}"
    return (prompt or "") + suffix


def _resolve_tools(name: str, raw_tools: list) -> tuple[list[str], list[Tool]]:
    """Map an agent's declared tool names to Tool objects, rejecting unknown names."""
    tool_names = [t if isinstance(t, str) else t["name"] for t in raw_tools]
    tools = []
    for tool_name in tool_names:
        if tool_name not in _TOOL_MAP:
            raise ValueError(f"Agent '{name}' references unknown tool '{tool_name}'")
        tools.append(_TOOL_MAP[tool_name])
    return tool_names, tools


def _resolve_mcp(name: str, raw_mcp: list) -> list[str]:
    """Extract MCP server names, rejecting any not present in .mcp.json."""
    mcp_names = [m["name"] if isinstance(m, dict) else m for m in raw_mcp]
    for mcp_name in mcp_names:
        if mcp_name not in _MCP_CONFIG:
            raise ValueError(f"Agent '{name}' references unknown MCP server '{mcp_name}'")
    return mcp_names


def load_agent(name: str, agents_dir: Path = _HERE / "agents") -> AgentConfig:
    """Scan agents_dir for a YAML whose name: field matches name.

    Returns an AgentConfig. Raises ValueError on an unknown agent, tool, or MCP server.
    """
    for path in sorted(agents_dir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text())
        if data.get("name") != name:
            continue
        tool_names, tools = _resolve_tools(name, data.get("tools", []))
        raw_skills = data.get("skills", [])
        skill_names = [s["name"] if isinstance(s, dict) else s for s in raw_skills]
        prompt = _append_environment(append_skills(data.get("prompt", ""), skill_names))
        mcp_names = _resolve_mcp(name, data.get("mcp", []))
        return {
            "prompt": prompt,
            "tools": tools,
            "model": data.get("model", None),
            "tool_names": tool_names,
            "skill_names": skill_names,
            "mcp_names": mcp_names,
            "output": data.get("output") or None,
        }
    raise ValueError(f"No agent named '{name}' found in {agents_dir}/")
