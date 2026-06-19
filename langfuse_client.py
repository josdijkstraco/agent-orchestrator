"""Lazy Langfuse singleton. Returns None if LANGFUSE_PUBLIC_KEY is not set."""
import os
from contextlib import nullcontext

_instance = None
_checked = False


def get_langfuse():
    global _instance, _checked
    if _checked:
        return _instance
    _checked = True
    if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
        try:
            from langfuse import Langfuse
            _instance = Langfuse()
        except Exception as exc:
            import sys
            print(f"[langfuse] init failed, observability disabled: {exc}", file=sys.stderr)
    return _instance


def lf_shutdown():
    if _instance is not None:
        _instance.shutdown()


null_ctx = nullcontext
