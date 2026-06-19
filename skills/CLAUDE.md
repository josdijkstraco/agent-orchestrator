# skills

Skill plugins. Each `skills/<name>/SKILL.md` is discovered by `skills_loader.py`: its frontmatter `name`/`description` registers the skill, and an agent opts in by listing the name in its YAML `skills:` field. The loader appends the skill's path (not its body) to the agent's system prompt; the agent reads the full file on demand via `read_file`.

## Subdirectories

- `brainstorming/` — turns ideas into approved specs through dialogue; includes a browser visual companion
- `reverse/` — minimal example skill (reverses a string)
