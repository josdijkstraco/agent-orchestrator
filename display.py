"""Terminal display helpers shared by the pipeline runner and the CLI."""

from agent_loader import AgentConfig


def agent_label(agent: AgentConfig) -> tuple[str, str, str]:
    """Comma-joined tool / skill / mcp names, each 'none' when empty."""
    return (
        ", ".join(agent["tool_names"]) or "none",
        ", ".join(agent["skill_names"]) or "none",
        ", ".join(agent["mcp_names"]) or "none",
    )


def print_agent_header(name: str, agent: AgentConfig, model: str | None = None, leading_newline: bool = False) -> None:
    tools_str, skills_str, mcp_str = agent_label(agent)
    prefix = "\n" if leading_newline else ""
    model_part = f"  model: {model}" if model is not None else ""
    print(f"{prefix}\033[1m[agent: {name}]\033[0m{model_part}  tools: {tools_str}  |  skills: {skills_str}  |  mcp: {mcp_str}")


def usage_totals(usage: dict) -> tuple[int, int, float]:
    """Pull (input_tokens, output_tokens, cost) out of an agent_loop usage dict."""
    return usage.get("input_tokens", 0), usage.get("output_tokens", 0), usage.get("cost", 0.0)
