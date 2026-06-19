# workflows

Workflow pipeline definitions, one YAML per file. `harness.py:load_workflow()` scans this directory and matches on the `name:` field. A workflow is an ordered list of `steps`, each naming an `agent` and declaring its `inputs:` explicitly (`__input__`, prior step ids, or named artifacts). Steps may add `prompt`, `output`/`stop_on`, `loop_on`/`loop_to`, `when`, and `outputs` (see the root README for the full schema).

## Code in this directory

- `example.yaml` — annotated plan → implement → review reference (also documents every step field inline)
- `pick-and-fix.yaml` — end-to-end: pull a Linear card → plan → implement ⇄ review loop → open PR → close card
- `multi-planner.yaml` — runs two planners on the same prompt, then a judge reconciles the differences
- `brainstorm.yaml` — single `brainstormer` step; `linear.yaml` — single `linear` step
- `when.yaml` — demo of `when:` conditional skipping (random ONE/TWO gates a hello step)
