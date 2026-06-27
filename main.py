#!/usr/bin/env python3
"""Interactive single-agent REPL — the general coding agent with all skills + MCP servers."""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    print("Error: OPENROUTER_API_KEY environment variable is required.")
    sys.exit(1)

from agent_openrouter import MODEL, agent_loop
from display import usage_totals
from mcp_client import build_all_mcp_clients
from repl import read_command, run_cancellable, select_model
from skills_loader import SKILLS, build_system_prompt


def _handle_mcp_command(parts: list[str], mcp_clients: list, mcp_map: dict) -> None:
    """Handle the /mcp command: list servers, or enable/disable one by name."""
    if len(parts) == 1:
        if not mcp_clients:
            print("No MCP servers configured.")
        for client in mcp_clients:
            print(f"  {client.name}: {'enabled' if client.enabled else 'disabled'}")
    elif len(parts) == 3 and parts[1] in ("enable", "disable"):
        client = mcp_map.get(parts[2])
        if client is None:
            print(f"MCP server '{parts[2]}' not found.")
        else:
            client.enabled = parts[1] == "enable"
            print(f"MCP '{client.name}' {'enabled' if client.enabled else 'disabled'}.")
    else:
        print("Usage: /mcp, /mcp enable <name>, /mcp disable <name>")


def main() -> None:
    if sys.stdout.isatty():
        print()

    mcp_clients = build_all_mcp_clients()
    mcp_map = {client.name: client for client in mcp_clients}
    for client in mcp_clients:
        print(f"  [{client.name}] tools: {', '.join(t['function']['name'] for t in client.tools)}")

    skill_names = list(SKILLS.keys())
    system_prompt = build_system_prompt(skill_names)
    print("Multi-Agent Harness (type 'exit' to quit, '/model' to switch, '/clear' to reset history, '/mcp' to toggle MCP servers)")
    if skill_names:
        print("Loaded skills: " + ", ".join(skill_names))

    messages: list = [{"role": "system", "content": system_prompt}]
    current_model = MODEL
    session_in = session_out = turns = 0
    session_cost = 0.0

    while True:
        try:
            user_input = read_command("> ", current_model, session_in, session_out, turns, session_cost)
        except (EOFError, KeyboardInterrupt):
            print()
            break

        command = user_input.strip()
        if command.lower() in ("exit", "quit"):
            break
        if command == "/model":
            current_model = select_model(current_model)
            continue
        if command == "/clear":
            messages[:] = [{"role": "system", "content": system_prompt}]
            session_in = session_out = turns = 0
            print("History cleared.")
            continue
        if command.startswith("/mcp"):
            _handle_mcp_command(command.split(), mcp_clients, mcp_map)
            continue
        if not command:
            continue

        usage = run_cancellable(lambda ce: agent_loop(
            user_input, messages, model=current_model, cancel_event=ce, mcp_clients=mcp_clients))
        if usage.get("cancelled"):
            print("\nRequest interrupted.")
            continue
        in_tok, out_tok, cost = usage_totals(usage)
        session_in += in_tok
        session_out += out_tok
        session_cost += cost
        turns = (len(messages) - 1) // 2

    for client in mcp_clients:
        client.close()


if __name__ == "__main__":
    main()
