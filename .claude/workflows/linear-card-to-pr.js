export const meta = {
  name: 'linear-card-to-pr',
  description: 'Fetch a Linear card, move it to In Progress, plan it, implement<->review until approved, mark the card Done, then open a PR',
  whenToUse: 'When you want to take a single Linear issue from backlog to an opened GitHub PR autonomously.',
  phases: [
    { title: 'Fetch', detail: 'load the Linear issue; if actionable, move it to In Progress' },
    { title: 'Plan',  detail: 'design the implementation (read-only Plan agent)' },
    { title: 'Build', detail: 'implement <-> review loop until the reviewer approves' },
    { title: 'Ship',  detail: 'branch, commit, push, open PR, move the card to Done' },
  ],
}

// ---- schemas: every agent that returns data is forced to match these ----
const CARD = {
  type: 'object',
  properties: {
    identifier: { type: 'string' },            // e.g. "ENG-123"
    title: { type: 'string' },
    description: { type: 'string' },
    url: { type: 'string' },
    state: { type: 'string' },
    suggestedBranch: { type: 'string' },        // Linear's git branch name, if available
  },
  required: ['identifier', 'title', 'description'],
}
const PLAN = {
  type: 'object',
  properties: {
    summary: { type: 'string' },
    steps: { type: 'array', items: { type: 'string' } },
    files: { type: 'array', items: { type: 'string' } },
    risks: { type: 'array', items: { type: 'string' } },
  },
  required: ['summary', 'steps'],
}
const REVIEW = {
  type: 'object',
  properties: {
    approved: { type: 'boolean' },
    blocking: { type: 'array', items: {
      type: 'object',
      properties: { title: { type: 'string' }, file: { type: 'string' }, detail: { type: 'string' } },
      required: ['title', 'detail'],
    }},
    summary: { type: 'string' },
  },
  required: ['approved', 'blocking', 'summary'],
}
const SHIP = {
  type: 'object',
  properties: { branch: { type: 'string' }, prUrl: { type: 'string' }, notes: { type: 'string' } },
  required: ['prUrl'],
}

// ---- inputs: resolve the target card from args (object, JSON string, or natural-language string) ----
// args reaches the script three ways and we must handle all of them:
//   1. a real object:            { card: 'AC-30', maxRounds: 5 }
//   2. a JSON-encoded string:    '{"card":"AC-30","maxRounds":5}'
//   3. a slash-command NL string: 'for card AC-30 and max iterations 5'
// We deliberately do NOT fall back to a placeholder card — guessing a card id is how the wrong
// issue gets touched. If no card can be resolved, the workflow stops with a usage message below.
function parseArgs(a) {
  if (a && typeof a === 'object') return a
  if (typeof a === 'string') {
    try { const o = JSON.parse(a); if (o && typeof o === 'object') return o } catch {}
    // free text: scrape a Linear identifier (e.g. AC-30) and an iteration count out of the words
    const id = a.match(/\b[A-Za-z][A-Za-z0-9]*-\d+\b/)
    const rounds = a.match(/\b(?:max|iterations?|rounds?)\D{0,12}?(\d+)/i)
    return { card: id ? id[0].toUpperCase() : undefined, maxRounds: rounds ? Number(rounds[1]) : undefined }
  }
  return {}
}
const _a = parseArgs(args)
const CARD_ID    = _a.card
const MAX_ROUNDS = _a.maxRounds ?? 5

// No card specified -> do not guess. Print usage and stop before doing any work.
if (!CARD_ID) {
  log(
    'linear-card-to-pr — no card specified; nothing to do.\n\n' +
    'USAGE\n' +
    '  Workflow({ name: "linear-card-to-pr", args: { card: "AC-30", maxRounds: 5 } })\n' +
    '  /linear-card-to-pr for card AC-30 and max iterations 5\n\n' +
    'ARGS\n' +
    '  card       (required)  Linear issue identifier, e.g. "AC-30"\n' +
    '  maxRounds  (optional)  implement<->review iterations before a draft PR is opened (default 5)\n\n' +
    'FLOW\n' +
    '  fetch the card -> move to In Progress -> plan -> implement<->review until approved -> open PR -> mark Done.'
  )
  return { status: 'usage', reason: 'no card specified', requiredArg: 'card (e.g. { card: "AC-30" })' }
}
const ATTRIB_COMMIT = 'Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>'
const ATTRIB_PR     = '🤖 Generated with [Claude Code](https://claude.com/claude-code)'
// All git/GitHub work goes through the local CLIs, never the GitHub MCP server.
const GH_ONLY = 'Use the local `git` and `gh` CLIs for ALL git and GitHub operations ' +
  '(create the branch, commit, push, and open the PR). Do NOT use the GitHub MCP server or its tools — ' +
  'authenticate via the already-logged-in `gh` CLI.'

// ================= 1. FETCH =================
phase('Fetch')
const card = await agent(
  `Use the Linear MCP to fetch issue "${CARD_ID}". ` +
  `Find the Linear tools first via ToolSearch (e.g. query "linear issue get" or "select:<tool>"). ` +
  `Return the issue's identifier, title, full description, url, current state, ` +
  `and Linear's suggested git branch name if one is provided.`,
  { label: `fetch:${CARD_ID}`, phase: 'Fetch', schema: CARD })
log(`Fetched ${card.identifier}: ${card.title}`)

// Decide whether to work on this card. If it's already finished or canceled, leave it untouched.
const cardState = (card.state ?? '').toLowerCase()
const alreadyClosed = ['done', 'completed', 'canceled', 'cancelled', 'duplicate', 'merged', 'released', 'closed']
  .some(s => cardState.includes(s))
if (alreadyClosed) {
  log(`${card.identifier} is "${card.state}" — nothing to do; not starting work.`)
  return { status: 'skipped', card: card.identifier, reason: `already ${card.state}` }
}

// Committed to the card -> move it to In Progress so the board reflects that work has started.
await agent(
  `Use the Linear MCP to move issue ${card.identifier} to its team's "In Progress" state ` +
  `(the workflow state whose type is "started"). Load the tools first via ToolSearch ` +
  `(e.g. "select:mcp__linear-server__list_issue_statuses,mcp__linear-server__save_issue"); ` +
  `list the team's statuses, pick the "started" one, and update the issue. ` +
  `If it is already In Progress, leave it as-is. Confirm the resulting state.`,
  { label: 'linear:in-progress', phase: 'Fetch' })
log(`${card.identifier} -> In Progress`)

// ================= 2. PLAN =================
phase('Plan')
const plan = await agent(
  `Plan the implementation for this Linear card. Read the relevant code in the repo FIRST, then plan.\n\n` +
  `${card.identifier} — ${card.title}\n\n${card.description}\n\n` +
  `Return: a one-paragraph summary, ordered implementation steps, the files you expect to change, and risks.`,
  { label: 'plan', phase: 'Plan', schema: PLAN, agentType: 'Plan', effort: 'high' })
log(`Plan ready: ${plan.steps.length} steps across ${(plan.files ?? []).length} file(s)`)

// ================= 3. BUILD (implement <-> review loop) =================
phase('Build')
let lastReview = null, round = 0, approved = false, prevSig = null
while (round < MAX_ROUNDS && !approved) {
  round++

  // implement: round 1 follows the plan; later rounds address ONLY the prior blocking findings
  const instruction = lastReview
    ? `Address ONLY these blocking review findings, editing the working tree in place. ` +
      `Do not rewrite unrelated code:\n${JSON.stringify(lastReview.blocking, null, 2)}`
    : `Implement this plan for ${card.identifier}, editing the working tree in place. ` +
      `Keep the change focused and idiomatic.\n\nPlan:\n${JSON.stringify(plan, null, 2)}\n\n` +
      `Card intent:\n${card.description}`
  await agent(instruction, { label: `implement:r${round}`, phase: 'Build' })

  // review: a fresh, strict skeptic reads the current diff against the card's intent
  lastReview = await agent(
    `Review the current uncommitted changes (run \`git diff\`) against the card's intent. Be strict. ` +
    `Set approved=true ONLY if there are no blocking correctness or security issues.\n\n` +
    `Card: ${card.identifier} — ${card.title}\n${card.description}`,
    { label: `review:r${round}`, phase: 'Build', schema: REVIEW, effort: 'high' })
  approved = lastReview.approved
  log(`Round ${round}: ${approved ? 'APPROVED' : `${lastReview.blocking.length} blocking issue(s)`}`)

  // oscillation guard: if the same findings recur, the implementer is stuck — stop
  const sig = JSON.stringify(lastReview.blocking.map(b => `${b.file ?? ''}:${b.title}`).sort())
  if (!approved && sig === prevSig) { log('No progress between rounds — stopping early.'); break }
  prevSig = sig
}

const branch = card.suggestedBranch || `fix/${card.identifier.toLowerCase()}`

// ================= 4. SHIP =================
phase('Ship')

// Not approved within the cap -> don't mark Done; open a DRAFT PR for a human to finish.
if (!approved) {
  const draft = await agent(
    `The change did NOT pass review after ${round} round(s). Open a DRAFT PR for human follow-up:\n` +
    `1. Create git branch "${branch}" (never commit to the default branch).\n` +
    `2. Stage and commit the WIP with a clear message referencing ${card.identifier}; ` +
    `end the commit message with:\n${ATTRIB_COMMIT}\n` +
    `3. Push to origin and run \`gh pr create --draft\`. In the body, explain the card, what was attempted, ` +
    `and the unresolved findings below; end the body with:\n${ATTRIB_PR}\n` +
    `Unresolved findings:\n${JSON.stringify(lastReview?.blocking ?? [], null, 2)}\n` +
    `Return the branch and PR url.\n\n${GH_ONLY}`,
    { label: 'ship:draft', phase: 'Ship', schema: SHIP })
  log(`Needs human attention — draft PR: ${draft.prUrl}`)
  return { status: 'needs-human', card: card.identifier, rounds: round, pr: draft }
}

// Approved -> open a real PR, then move the Linear card to Done.
const ship = await agent(
  `The change passed review. Ship it:\n` +
  `1. Create git branch "${branch}" (never commit to the default branch).\n` +
  `2. Stage and commit with a clear message referencing ${card.identifier}; ` +
  `end the commit message with:\n${ATTRIB_COMMIT}\n` +
  `3. Push to origin and open a PR with \`gh pr create\`. Write a clear title and a body that covers: ` +
  `what the card asked for, what changed and why, how it was verified, and the line "Closes ${card.identifier}". ` +
  `End the body with:\n${ATTRIB_PR}\n` +
  `4. Return the branch name and PR url.\n\n` +
  `Card: ${card.identifier} — ${card.title}\nReview summary: ${lastReview.summary}\n\n${GH_ONLY}`,
  { label: 'ship:pr', phase: 'Ship', schema: SHIP })

await agent(
  `Use the Linear MCP (load its tools via ToolSearch) to move issue ${card.identifier} to the "Done" state, ` +
  `and add a comment linking the merged work: ${ship.prUrl}`,
  { label: 'linear:done', phase: 'Ship' })

log(`Shipped ${card.identifier} -> ${ship.prUrl}`)
return { status: 'shipped', card: card.identifier, rounds: round, branch: ship.branch, pr: ship.prUrl }
