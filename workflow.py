"""Workflow loading and validation.

`load_workflow()` scans `workflows/` for a YAML whose `name:` matches, then
normalises and validates every step into a fully-populated `StepConfig`. All the
cross-step invariants (unique ids, backward-only loop targets, known references,
schema-consistent conditions) are enforced here at load time so `pipeline.py`
can assume well-formed steps.
"""

from pathlib import Path
from typing import NotRequired, TypedDict

import yaml

from conditions import parse_equality, parse_when
from step_io import INPUT_KEY

_HERE = Path(__file__).parent

# Loop-back cap applied when a step declares loop_on/loop_to but no max_loops.
DEFAULT_MAX_LOOPS = 3


class StepConfig(TypedDict):
    agent: str
    id: str
    prompt: NotRequired[str | None]
    inputs: NotRequired[list[str]]
    outputs: NotRequired[dict[str, str] | None]
    when: NotRequired[str | None]
    loop_on: NotRequired[str | None]
    loop_to: NotRequired[str | None]
    max_loops: NotRequired[int | None]
    output: NotRequired[dict | None]
    stop_on: NotRequired[str | None]


class WorkflowConfig(TypedDict):
    steps: list[StepConfig]


def _validate_loop(step_id: str, loop_on: str | None, loop_to: str | None,
                   when: str | None, seen_ids: set[str]) -> None:
    """Loop fields must come as a pair, point backwards, and not co-exist with `when`."""
    if (loop_on is None) != (loop_to is None):
        raise ValueError(f"Step '{step_id}' must have both loop_on and loop_to, or neither.")
    if loop_to is not None and loop_to not in seen_ids:
        raise ValueError(f"Step '{step_id}' loop_to='{loop_to}' must refer to an earlier step.")
    if loop_on is not None and when is not None:
        raise ValueError(f"Step '{step_id}' cannot have both loop_on and when.")


def _validate_references(step_id: str, inputs: list[str], when: str | None,
                         all_ids: set[str], all_artifact_names: set[str],
                         seen_ids: set[str], seen_artifact_names: set[str]) -> None:
    """Inputs and `when` must reference a known step id or artifact name."""
    for ref in inputs:
        if ref not in all_ids and ref not in all_artifact_names:
            raise ValueError(f"Step '{step_id}' inputs references unknown id '{ref}'.")
    if when is not None:
        parsed = parse_when(when)
        if parsed is None:
            raise ValueError(f"Step '{step_id}' when='{when}' must be 'PATTERN in step_id'.")
        ref_id = parsed[1]
        if ref_id not in seen_ids and ref_id not in seen_artifact_names:
            raise ValueError(f"Step '{step_id}' when references unknown id '{ref_id}'.")


def _validate_output(step_id: str, output: dict | None, stop_on: str | None,
                     loop_on: str | None) -> None:
    """An `output` schema must be well-formed and consistent with stop_on/loop_on."""
    if output is not None:
        for field_name, field_def in output.items():
            if "type" not in field_def:
                raise ValueError(f"Step '{step_id}' output field '{field_name}' must have a 'type'.")
            if "enum" in field_def and not isinstance(field_def["enum"], list):
                raise ValueError(f"Step '{step_id}' output field '{field_name}' enum must be a list.")
    if stop_on is not None and output is None:
        raise ValueError(f"Step '{step_id}' has stop_on but no output.")
    if stop_on is not None and loop_on is not None:
        raise ValueError(f"Step '{step_id}' cannot have both stop_on and loop_on.")
    if loop_on is not None and output is not None:
        parsed = parse_equality(loop_on)
        if parsed is None:
            raise ValueError(f"Step '{step_id}' loop_on must be 'field == value' when output is set.")
        field, value = parsed
        if field not in output:
            raise ValueError(f"Step '{step_id}' loop_on references unknown field '{field}' not in output.")
        field_def = output[field]
        if "enum" in field_def and value not in field_def["enum"]:
            raise ValueError(f"Step '{step_id}' loop_on value '{value}' is not in enum {field_def['enum']}.")


def _register_artifacts(step_id: str, outputs: dict[str, str] | None,
                        seen_ids: set[str], seen_artifact_names: set[str]) -> None:
    """Record declared artifact names, rejecting collisions with existing ids/names."""
    for artifact_name in (outputs or {}):
        if artifact_name in seen_artifact_names or artifact_name in seen_ids:
            raise ValueError(f"Step '{step_id}' output '{artifact_name}' conflicts with an existing id or artifact name.")
        seen_artifact_names.add(artifact_name)


def normalize_step(
    step: dict,
    seen_ids: set[str],
    seen_artifact_names: set[str],
    all_ids: set[str],
    all_artifact_names: set[str],
) -> StepConfig:
    """Validate a raw step dict and return it normalised to a full StepConfig.

    Declared artifact names are added to seen_artifact_names; the caller adds the
    returned step's id to seen_ids afterwards (so loop_to/when only see earlier steps).
    """
    step_agent = step["agent"]
    step_id = step.get("id") or step_agent
    if step_id in seen_ids:
        raise ValueError(
            f"Step id '{step_id}' is not unique. "
            f"When the same agent appears multiple times, each step must have an explicit 'id:'."
        )

    if "inputs" not in step:
        raise ValueError(
            f"Step '{step_id}' must declare 'inputs' explicitly. "
            f"Use 'inputs: [__input__]' for the workflow prompt, "
            f"'inputs: [<step_id>]' to consume a prior step, or 'inputs: []' for no input."
        )
    inputs: list[str] = list(step["inputs"]) if step["inputs"] is not None else []
    outputs: dict[str, str] | None = dict(step["outputs"]) if step.get("outputs") else None
    when: str | None = step.get("when") or None
    loop_on = step.get("loop_on") or None
    loop_to = step.get("loop_to") or None
    max_loops = step.get("max_loops")
    output: dict | None = step.get("output") or None
    stop_on: str | None = step.get("stop_on") or None

    _validate_loop(step_id, loop_on, loop_to, when, seen_ids)
    if loop_on is not None and max_loops is None:
        max_loops = DEFAULT_MAX_LOOPS
    _validate_references(step_id, inputs, when, all_ids, all_artifact_names, seen_ids, seen_artifact_names)
    _register_artifacts(step_id, outputs, seen_ids, seen_artifact_names)
    _validate_output(step_id, output, stop_on, loop_on)

    return {
        "agent": step_agent,
        "id": step_id,
        "prompt": step.get("prompt") or None,
        "inputs": inputs,
        "outputs": outputs,
        "when": when,
        "loop_on": loop_on,
        "loop_to": loop_to,
        "max_loops": max_loops,
        "output": output,
        "stop_on": stop_on,
    }


def _collect_names(raw_steps: list[dict]) -> tuple[set[str], set[str]]:
    """Pre-collect all step ids and artifact names so inputs can forward-reference
    a step that executes later (e.g. implementer consuming reviewer feedback on loop-back)."""
    all_ids: set[str] = {INPUT_KEY}
    all_artifact_names: set[str] = set()
    for step in raw_steps:
        all_ids.add(step.get("id") or step["agent"])
        for artifact_name in (step.get("outputs") or {}):
            all_artifact_names.add(artifact_name)
    return all_ids, all_artifact_names


def load_workflow(name: str, workflows_dir: Path = _HERE / "workflows") -> WorkflowConfig:
    """Scan workflows_dir for a YAML whose name: field matches name."""
    for path in sorted(workflows_dir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text())
        if data.get("name") != name:
            continue
        raw_steps = data.get("steps", [])
        all_ids, all_artifact_names = _collect_names(raw_steps)

        seen_ids: set[str] = set()
        seen_artifact_names: set[str] = set()
        steps: list[StepConfig] = []
        for step in raw_steps:
            normalized = normalize_step(step, seen_ids, seen_artifact_names, all_ids, all_artifact_names)
            steps.append(normalized)
            seen_ids.add(normalized["id"])
        return {"steps": steps}
    raise ValueError(f"No workflow named '{name}' found in {workflows_dir}/")
