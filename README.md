# ABS Evolution

An independent local experiment engine for ABS. It learns from explicitly
submitted task receipts, proposes bounded changes, runs candidate programs in
an OS sandbox, and selects versions using separate training and acceptance
evidence. It does not install or update the ABS plugin.

## Scope

This first implementation delivers a supervised executable-tool experiment
path. Skills, workflow, memory and optimizer are supported artifact kinds;
their names do not establish that an evaluator can measure those capabilities.
The bundled evaluator currently measures Python programs with JSON input/output.
The example is synthetic and is not an ABS production defect or business test.

There is no background service, task interception, desktop integration,
automatic Git publishing, cross-host synchronization or production deployment.
No existing chat, automation, plugin cache, worktree or credential store is
modified. Existing desktop tasks do not automatically acquire registry leases.

## Run without installing dependencies

Use Python 3.12+ from this project directory. The core uses only the standard
library. Candidate execution currently requires macOS with a functioning
`sandbox-exec`; unsupported or unavailable isolation blocks experiments.
The engine never starts Docker or falls back to ordinary subprocess execution.

Seatbelt isolates filesystem and network access but does not enforce hard RAM
or aggregate scratch-storage quotas. Experiments therefore block by default.
`--allow-shared-resources` is an explicit opt-in for a supervised local run of
reviewed code that shares host resources. It is not suitable for unattended
model-generated code. A backend with enforced resource quotas is still required
before enabling automatic experiments.

```sh
PYTHONPATH=src python3 -m abs_evolution --help
PYTHONPATH=src python3 -m abs_evolution doctor
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The CLI accepts an explicit `--state /absolute/path/to/new-state` directory.
Use a separate local directory, not a source checkout, installed plugin cache,
shared network volume or credentials directory. Initialization rejects existing
unrelated content. No daemon or scheduled task is installed.

## A complete offline experiment

The files under `examples/acceptance/` deliberately model the difference between
an execution observation and an accepted outcome. The baseline incorrectly
accepts any nonempty evidence list. A reviewed fixture proposal repairs it.

```sh
PYTHONPATH=src python3 -m abs_evolution --state /absolute/path/to/new-state init
PYTHONPATH=src python3 -m abs_evolution --state /absolute/path/to/new-state snapshot import examples/acceptance/baseline --kind tool
```

Copy the returned snapshot ID into the next command. This initial selection is
an explicit operator choice, not a measured improvement:

```sh
PYTHONPATH=src python3 -m abs_evolution --state /absolute/path/to/new-state channel bootstrap SNAPSHOT_ID
PYTHONPATH=src python3 -m abs_evolution --state /absolute/path/to/new-state evolve --suite examples/acceptance/suite.json --provider file --proposal-file examples/acceptance/proposal.json --allow-shared-resources --promote
PYTHONPATH=src python3 -m abs_evolution --state /absolute/path/to/new-state status
```

`--promote` changes only this isolated registry's channel. It does not update
`main`, install a plugin or switch an existing task. Without the flag, an
eligible evaluation stays available for an explicit `channel promote` command.
The file provider exercises the real storage, sandbox, evaluation and promotion
path, but it does not demonstrate autonomous model-generated improvement.
The shared-resource opt-in above is for this small reviewed fixture, not an
approval to execute arbitrary proposals or run experiments in the background.

The acceptance set is consumed once after candidate training qualifies. To
conduct another final acceptance experiment, provide new holdout inputs rather
than repeatedly optimizing against the same answer key. Renaming a case or
suite, changing case order, or duplicating a case is not a new acceptance set.
Duplicate inputs within either split are rejected.

## Generate a proposal through OpenRouter

The OpenRouter provider reads `OPENROUTER_API_KEY` only from its process
environment. It does not discover credential files, copy keys into the project,
or create a key. Use your existing secret-manager injection when running it.
An explicit model ID is required; no model is silently selected or substituted.

The following command sends only the bundled synthetic source and its training
examples. It produces validated JSON proposal data without executing candidate
code, creating registry state or promoting a version:

```sh
PYTHONPATH=src python3 -m abs_evolution propose --provider openrouter --model openai/gpt-4.1-mini --request examples/acceptance/request.json --max-output-tokens 2048 --timeout-seconds 60
```

The output contains `proposal`, observed `usage`, and explicit execution/change
flags. A successful API response is not an accepted candidate. Review the
proposal before any separately authorized supervised evaluation. The fixture
request has a descriptive baseline ID; an actual experiment supplies a registry
snapshot ID and freshly observed training outcomes.

The adapter uses only the fixed OpenRouter HTTPS endpoint and strict JSON
schema output. It sends no tools, plugins, browser access or host-reading
capabilities. Routing requires parameter support, disallows data collection,
requests zero-data-retention endpoints and disables provider fallbacks. A
routing failure does not relax those settings. These are the documented
[OpenRouter routing controls](https://openrouter.ai/docs/guides/routing/provider-selection),
not an independent audit of the service's retention behavior.

Transport uses one bounded request with no redirects or automatic retry.
Tokens, model, generation ID and cost in credits are reported only when the
service provides valid values. A timeout does not prove the remote generation
was free or canceled; it is not retried. Token limits are not a dollar budget.
Review all selected input before a live request. Never send credentials,
private transcripts, unapproved business data or protected acceptance answers.

`evolve --provider openrouter` still blocks on the current Seatbelt backend.
Unlike a reviewed file proposal, model-generated code cannot use
`--allow-shared-resources` to bypass missing resource quotas. Proposal generation
works independently of that execution gate.

`--provider codex` remains blocked without starting a Codex process. The
inspected CLI cannot establish a verifiable zero-tool boundary. The offline
file provider still works and reports unknown model usage, not zero usage.
Separate processes or hosts do not by themselves isolate shared model quotas.

## Route with JEV before generating content

`--provider hybrid` asks JEV one typed Choice question over the reviewed
training request and an explicit pool of existing candidate proposals. Program
rules then either return a selected existing proposal, call the configured
OpenRouter generator once, or stop for insufficient evidence. JEV does not write
code or grant execution, promotion or publication permission.

This command can select the bundled synthetic fix without configuring an LLM:

```sh
PYTHONPATH=src python3 -m abs_evolution propose --provider hybrid --allow-external --request examples/acceptance/request.json --candidate-pool examples/acceptance/candidate-pool.json --timeout-seconds 60
```

It reads `TYPESAFE_API_KEY` from the process environment. `--allow-external`
asserts that the caller has reviewed and approved the input for TypeSafe and,
when configured, OpenRouter. The flag and secret detector do not establish
permission to send business records. No credentials are read from files.

To allow new-content generation, add `--model openai/gpt-4.1-mini` and provide
`OPENROUTER_API_KEY` through the existing secret manager. The model is not called
for a selected existing proposal. Omitting `--candidate-pool` supplies an empty
pool; it does not skip JEV or force generation. Without `--model`, a decision
that requires new content stops with `provider_blocked_generation_unavailable`.

The pool is a JSON list of at most 16 objects with `id`, `baseline_id`, and
`proposal`. Each proposal has the existing `summary`, `kind`, and `changes`
shape. Every entry must match the current request's baseline and pass local
path/content validation before any model request. Nothing is discovered from
chat history, other projects or the registry automatically.

JEV is pinned to `jev-1.13.0` by default; `--jev-model` selects a different
explicit version. Both `--min-choice-confidence` and
`--min-choice-probability` default to 0.8. These are provisional routing limits,
not validated accuracy estimates. An uncertain choice, invalid response or
service failure does not trigger an LLM fallback. The judge and optional
generator share one total deadline, with no automatic retry.

The output's `routing` field keeps the JEV distribution and observed usage
separate from the generator's usage. Input and candidate fingerprints identify
the decision context without copying raw input into the trace. A selected
candidate still needs real training and protected holdout evaluation.
`evolve --provider hybrid` requires hard resource quotas even for reuse and
cannot bypass the execution gate with `--allow-shared-resources`.

This follows TypeSafe's [Choice contract](https://docs.typesafe.ai/primitives/choice)
and [intent-routing pattern](https://docs.typesafe.ai/patterns/intent-routing).
Synthetic and offline checks establish the integration behavior, not model
quality, cost savings or a benefit on real ABS tasks.

## Protect foreground work

```sh
PYTHONPATH=src python3 -m abs_evolution --state /absolute/path/to/state task begin report-123
PYTHONPATH=src python3 -m abs_evolution --state /absolute/path/to/state task end report-123
```

Beginning a task returns its exact immutable snapshot path. A consumer must
actually use that path for its skills, references and scripts to receive the
pinning guarantee. Promotion and rollback affect only future task leases.
Old snapshots are retained. The first implementation intentionally has no
automatic garbage collection or task-lease expiry.

By default active registered foreground tasks prevent new experiments. An
experiment also checks between phases and cases and yields if a task appears.
An already running bounded case may take up to its configured timeout to exit;
this is not instantaneous preemption. Unregistered desktop activity is not
detected. If a process crashes, its uncertain running claim is not silently
expired and retried. This favors stopping optional learning over overlapping
possibly active work.

Task admission waits at most one second to acquire a busy registry write lock
and then returns `registry_busy`. Snapshot creation and verification can still
hold that lock. This is not a hard end-to-end latency guarantee. A future
desktop adapter must handle learning or lease errors without interrupting the
foreground task and must never silently change its selected version.

The policy bounds experiment concurrency, daily admission and reserved time,
case and experiment deadlines, output bytes, stored snapshots and receipts.
Reaching a limit blocks further learning instead of deleting task dependencies.
These are local reservations and wall-clock limits, not a billing guarantee.

Optional receipt collection or learning failures should be handled outside
the foreground task's success path. Mandatory business, authorization and
security checks must still stop protected actions when they fail.

## Opt-in experience

Submit compact receipts with `receipt record FILE.json`. The schema is limited
to identifiers, accepted/failed/unknown outcome, failure tags and opaque evidence
references. Unknown outcomes are not counted as successful tasks. Raw messages,
arbitrary metadata, URLs and local paths are not receipt fields.

```json
{
  "id": "receipt-001",
  "task_id": "report-123",
  "snapshot_id": "REPLACE_WITH_THE_PINNED_SNAPSHOT_ID",
  "outcome": "failed",
  "failure_tags": ["premature_acceptance"],
  "evidence_refs": ["evidence-001"]
}
```

`opportunities` groups repeated failures from distinct tasks and preserves their
evidence references. Its results are investigation candidates, not causal
diagnoses or permission to change every task's shared instructions.

## Trust and evidence boundaries

- The local operator controls policy, suite inputs/answers and the registry.
  Candidates are untrusted. They cannot read those files or alter acceptance
  thresholds through the experiment interface. Keep answer material outside
  imported snapshots. The engine rejects an accessible suite path or a complete
  copy of the selected suite in a snapshot, including reformatted JSON copies.
  This guard is not a detector for arbitrary encoded or fragmented answer keys.
- Snapshot hashes prove content identity, not correctness. Actual candidate
  programs execute under the sandbox and are checked against independent data.
- Candidate mutations are limited to operator-allowed relative paths. Symlinks,
  traversal and credential/runtime paths are rejected or excluded on import.
  Filename filtering is not a complete secret detector; source selection and
  sharing authorization remain operator responsibilities.
- Promotion binds the exact baseline, candidate, policy, suite and environment.
  It uses a transactional compare-and-swap. A changed policy or channel cannot
  reuse stale approval evidence.
- Infrastructure errors, malformed output and timeouts are separate from a
  valid program producing the wrong answer. Neither is silently called success.
- This is single-host local SQLite coordination. It is not a distributed lock,
  a defense against a malicious operator, or proof of desktop-session adoption.
- The current sandbox does not enforce hard RAM or aggregate scratch quotas.
  Its successful access-isolation probe is not evidence of unattended readiness.

See [the implementation contracts](docs/contracts.md) for schemas and lifecycle
rules, and [the implementation stages](docs/implementation.md) for the remaining
integration boundaries.

## Continuous integration

The validation workflow runs the offline suite on Ubuntu 24.04 with Python 3.12
and macOS 15 with Python 3.14. The macOS job requires a working Seatbelt probe
before running the real isolation and lifecycle tests. The Linux job verifies
that the unsupported execution backend fails closed; its macOS-only tests are
expected to skip. A green Linux job does not prove candidate execution isolation.

Both jobs build a wheel, install it into a fresh virtual environment, and run the
installed proposal-only CLI outside the checkout. No provider credentials are
configured, and no live model calls, deployment or publication is part of CI.
Passing these checks does not remove the hard-resource execution gate.

A separate Linux-only feasibility job runs fixed reviewed probes in a
digest-pinned official Python container, with no host mounts or injected keys.
It exercises OOM termination at 64 MiB (no swap), aggregate writes to an 8 MiB
scratch tmpfs and 1 MiB shared-memory mount, a 16-process ceiling, CPU throttling,
network denial, read-only root storage, and deadline/descendant cleanup.
Each probe removes its own container. This is **not** the engine's execution
backend, an audit against container escapes, or unattended-readiness evidence.
It runs only on disposable GitHub Linux runners; the Mac Docker daemon is not
started. No arbitrary candidate source is accepted by this probe script.
The limits follow Docker's [resource constraints](https://docs.docker.com/engine/containers/resource_constraints/)
and [tmpfs semantics](https://docs.docker.com/engine/storage/tmpfs/); tmpfs memory
also counts toward the container memory ceiling.

## Research used

The implementation is original and has no runtime dependency on these projects.
It uses the following independently checked mechanisms as design references:

- [recursive-improve](https://github.com/kayba-ai/recursive-improve/tree/9cf4b85ef6ef66d37ea3c452f49114ff7db1b1a3): trace-derived opportunities and bounded iterative comparisons.
- [EvoSkill](https://github.com/sentient-agi/EvoSkill/tree/36f6f04952293d7054145550c2b9f0b0411bff1c): separation of proposal, generation, evaluation and retained candidate versions.
- [Self-Improving Agents survey](https://github.com/selfimproving-agent/Awesome-Self-Improving-Agents/tree/a8f953dbabf6fcfe27b0988c707cb62030df35cd): prompt, memory, tool and whole-system evolution with independent evaluation.
- [FrontisAI survey](https://github.com/FrontisAI/Awesome-Self-Improving-Agents/tree/95a23e7f9b3a17384f5bec87dfbb2239ff1b9922): batch experience synthesis and team/meta-level evolution research.

Their benchmark results are not evidence for ABS. Shared-checkout mutation,
heuristic clean-success scoring and repeatedly consulted validation sets were
not adopted as production acceptance mechanisms.
