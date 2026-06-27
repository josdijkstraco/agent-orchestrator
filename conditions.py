"""Control-flow predicates for workflow steps.

Pure functions over text and structured-result dicts — no I/O, no harness state —
so each is unit-testable on its own. Used by `workflow.py` (load-time validation)
and `pipeline.py` (runtime `when` / `stop_on` / `loop_on` / STOP evaluation).
"""

import re

# A step's `when:` condition has the form 'PATTERN in step_id'.
_WHEN_RE = re.compile(r"^(.+?)\s+in\s+(\w+)$")
# A standalone "STOP" line (optionally with a trailing . or !) ends the pipeline.
_STOP_RE = re.compile(r"STOP[.!]?")


def parse_equality(expr: str) -> tuple[str, str] | None:
    """Split a 'field == value' expression into (field, value), or None if malformed."""
    parts = expr.split("==", 1)
    if len(parts) != 2:
        return None
    return parts[0].strip(), parts[1].strip()


def parse_when(expr: str) -> tuple[str, str] | None:
    """Split a 'PATTERN in step_id' expression into (pattern, ref_id), or None if malformed."""
    m = _WHEN_RE.match(expr)
    if not m:
        return None
    return m.group(1).strip(), m.group(2)


def eval_condition(expr: str, result: dict) -> bool:
    """Evaluate a 'field == value' expression against a structured result dict."""
    parsed = parse_equality(expr)
    if parsed is None:
        return False
    field, value = parsed
    return str(result.get(field, "")) == value


def token_present(pattern: str, text: str) -> bool:
    """True if pattern appears in text as a whole token, not embedded in a larger word.

    Word-boundary match so a `loop_on`/`when` pattern like 'APPROVED' doesn't fire
    inside 'UNAPPROVED'. This is the unstructured (text-output) control-flow path;
    prefer a structured `output` schema with `loop_on`/`stop_on` for new workflows.
    """
    return re.search(rf"(?<!\w){re.escape(pattern)}(?!\w)", text) is not None


def is_stop_signal(text: str) -> bool:
    """True when a step's text output ends with a standalone STOP line.

    Anchored to the final non-empty line so incidental prose — 'NONSTOP',
    'I did not STOP the process', 'STOP or continue?' — never halts the pipeline.
    Prefer a structured `stop_on` condition; this is the no-schema fallback.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return bool(lines) and _STOP_RE.fullmatch(lines[-1]) is not None


def when_skips(when_expr: str, step_outputs: dict[str, str]) -> bool:
    """True when a step's `when` condition is unmet and the step should be skipped.

    Only the 'PATTERN in step_id' form is recognised; a malformed expression never skips.
    """
    parsed = parse_when(when_expr)
    if parsed is None:
        return False
    pattern, ref_id = parsed
    return ref_id not in step_outputs or not token_present(pattern, step_outputs[ref_id])
