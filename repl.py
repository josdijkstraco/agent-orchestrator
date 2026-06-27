"""Shared interactive-REPL helpers for the CLI entry points (main.py, harness.py).

`select_model` is the model picker; `run_cancellable` runs one agent turn in a
background thread with Escape-to-cancel and returns its usage dict.
"""

import threading
import warnings
from typing import Callable

from prompt_toolkit import prompt as pt_prompt

from agent_openrouter import AVAILABLE_MODELS
from repl_utils import COMMAND_COMPLETER, IS_TTY, status_text, watch_for_escape


def read_command(prompt_str: str, model: str, session_in: int, session_out: int,
                 turns: int, session_cost: float) -> str:
    """Read one REPL line, with completion + a live status bar when interactive."""
    if not IS_TTY:
        return input(prompt_str)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*CPR.*")
        return pt_prompt(
            prompt_str, completer=COMMAND_COMPLETER,
            bottom_toolbar=lambda: status_text(model, session_in, session_out, turns, session_cost),
            refresh_interval=0.5,
        )


def select_model(current: str) -> str:
    """Prompt the user to pick from AVAILABLE_MODELS, keeping the current one on blank/invalid input."""
    print("Available models:")
    for i, m in enumerate(AVAILABLE_MODELS, 1):
        marker = " *" if m == current else ""
        print(f"  {i}. {m}{marker}")
    try:
        choice = input("Pick a number (or press Enter to keep current): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return current
    if not choice:
        return current
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(AVAILABLE_MODELS):
            selected = AVAILABLE_MODELS[idx]
            print(f"Model set to: {selected}")
            return selected
        print("Invalid selection, keeping current model.")
    except ValueError:
        print("Invalid input, keeping current model.")
    return current


def run_cancellable(make_call: Callable[[threading.Event], dict]) -> dict:
    """Run `make_call(cancel_event)` in a thread, cancelling on Escape or Ctrl-C.

    Returns the usage dict it produced, or a cancelled placeholder if interrupted.
    """
    cancel_event = threading.Event()
    done_event = threading.Event()
    result: dict = {}

    def _run() -> None:
        result["usage"] = make_call(cancel_event)

    agent_thread = threading.Thread(target=_run, daemon=True)
    watcher_thread = threading.Thread(target=watch_for_escape, args=(cancel_event, done_event), daemon=True)
    watcher_thread.start()
    agent_thread.start()
    try:
        agent_thread.join()
    except KeyboardInterrupt:
        cancel_event.set()
        agent_thread.join()
    finally:
        done_event.set()
    return result.get("usage", {"input_tokens": 0, "output_tokens": 0, "cancelled": True})
