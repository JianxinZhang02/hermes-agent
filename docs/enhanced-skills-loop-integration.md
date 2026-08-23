# Enhanced Skills and Loop integration

This branch integrates the stable behavior from the Skills self-evolution and
Loop Engineering prototypes into the newer Hermes Memory/Knowledge codebase.
The migration is semantic: the current Agent, Session, Memory Manager,
Knowledge Manager, approval, provenance, gateway FIFO, and persistence
lifecycles remain authoritative.

## Skills self-evolution

Background review skill creation now uses a recurrence gate. A proposal from
one review window is stored in `~/.hermes/skills/.evidence.json`; it becomes a
real skill only after the same reusable class appears in the configured number
of independent review windows. Foreground/user-directed creates and edits to
existing skills are not recurrence-gated.

Configuration:

```yaml
skills:
  evidence_threshold: 2
  evidence_ttl_days: 14
```

Evidence writes are locked and atomic, stale candidates are pruned, and
bookkeeping fails open. Background-review payloads are scrubbed for
instance-specific figures. Pending candidates are visible through
`skills_list` only inside the review fork. The review prompt requires
class-level abstraction and forbids task/answer catalogs.

The bundled `skill_training` plugin supports `reference` and
`binary_enriched` training. Each item is answered blind, evaluated privately,
then followed by a one-sentence reflection. The evaluator's gold answer is not
shown to the initial worker. Runs emit audit artifacts and wait for only the
background reviews belonging to their agent.

## Goal Judge verification

The Goal Judge receives bounded, redacted evidence from tool calls created in
the current turn. When completion depends on a missing objective local-file
fact, its preliminary decision may request one internal `verify` action with
one to three `read_file` or `search_files` checks.

Verification is not an Agent tool turn. Checks are validated against the
current local workspace and run in spawned subprocesses with per-call and total
deadlines. Traversal, symlink escape, sensitive files, binary/media payloads,
unsafe globs, and unsupported tools are denied. A second Judge call makes the
final `done`, `continue`, or `wait` decision; recursive `verify` is rejected.
CLI, Gateway, and TUI use the same boundary.

Because tool evidence can contain sensitive application output, the
`goal_judge` endpoint and key are pinned to the main model trust boundary. A
different model may be selected only on that same API.

## Deliberately not productized

The Hybrid PlanIR scheduler in the external Loop Engineering experiment is not
installed in the normal Agent loop. Its experimental success path can bypass
Hermes finalization and persistence semantics. It should remain an A/B harness
until it re-enters normal finalization, Session persistence, interruption, and
Memory/Knowledge lifecycle handling on every path.

## Validation

Focused tests cover recurrence and review-window isolation, skill training,
non-interactive agent construction, bounded Goal evidence, verification path
security, judge routing, and CLI/Gateway/TUI goal regressions. Run:

```bash
python -m pytest -q tests/tools/test_skill_evidence.py \
  tests/plugins/test_skill_training_plugin.py \
  tests/hermes_cli/test_goal_verification.py \
  tests/agent/test_goal_judge_routing.py
```
