# brainstorming

A skill that turns rough ideas into approved designs and specs through one-question-at-a-time dialogue, then writes the spec to `docs/superpowers/specs/`. Enforces a hard gate: no implementation before the user approves a design. Used by the `brainstormer` agent.

## Code in this directory

- `SKILL.md` — the skill instructions (frontmatter `name`/`description` is what `skills_loader.py` registers); defines the brainstorm checklist and process flow
- `visual-companion.md` — guide for when and how to use the browser visual companion
- `spec-document-reviewer-prompt.md` — subagent prompt template for reviewing a written spec before planning

## Subdirectories

- `scripts/` — Node server powering the browser visual companion
