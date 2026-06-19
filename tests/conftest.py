"""Shared pytest fixtures.

Langfuse is force-disabled for the whole test session. `agent_openrouter` calls
`load_dotenv()` at import time, which loads the real `LANGFUSE_*` keys from `.env`.
With those keys present, `get_langfuse()` returns a live client whose span flush /
`lf_shutdown()` blocks trying to reach the Langfuse endpoint — hanging the suite.
No test exercises Langfuse, so we pin `get_langfuse()` to return None, which sends
`run_pipeline` down its null-context path and prevents any network I/O.
"""

import sys
from pathlib import Path

# Make the project root importable regardless of pytest's import mode.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import langfuse_client


@pytest.fixture(autouse=True)
def _disable_langfuse(monkeypatch):
    # `run_pipeline` does a local `from langfuse_client import get_langfuse`, so
    # patching the module attribute is enough — it resolves to this at call time.
    monkeypatch.setattr(langfuse_client, "get_langfuse", lambda: None)
    # Belt and suspenders: keep the real lazy singleton unset so lf_shutdown() and
    # any direct os.environ reads stay inert even if something bypasses the patch.
    monkeypatch.setattr(langfuse_client, "_instance", None, raising=False)
    monkeypatch.setattr(langfuse_client, "_checked", True, raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
