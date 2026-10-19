# ABS Evolution

An evidence-constrained continuous optimization engine for ABS. Explicit task
receipts can identify opportunities in quality, efficiency, methods and reuse,
including after successful tasks. Failures are one source of evidence, not a
prerequisite for improvement. The engine proposes bounded changes, runs candidate
programs in an OS sandbox, and selects versions using separate training and
acceptance evidence. It does not install or update the ABS plugin.

## Optimization goal

Define the desired outcome, observe the current version, propose an alternative,
compare both on independent cases, then retain a version with demonstrated gains.
Actual task adoption and continued measurement follow separately. A successful
task can still take unnecessary steps, cost too much, or reveal a reusable method.

Define "better" before generating candidates. Examples are lower measured cost
at unchanged quality, or better quality within the same budget. Authorization,
isolation and safety requirements are hard constraints, never weighted tradeoffs.
An observation starts an investigation; it does not prove that an alternative is
better or authorize a change to shared instructions.

The current executable evaluator measures exact JSON correctness only. It cannot
yet accept a same-quality candidate on cost or latency savings. Those objectives
need trusted measurements and corresponding promotion checks. Do not lower the
quality-gain threshold to present an unmeasured efficiency claim as a result.

## Semantic judgments use JEV

JEV owns semantic classification and proposal-route selection. Code owns exact
validation, counting, deadlines, budgets, permissions and evidence gates. A
generator writes proposed changes only when the selected route requires them.
Independent evaluation, not JEV's confidence, establishes measured gains.

Use `classify` for an observation that needs semantic interpretation. It sends
one reviewed request to JEV and selects a primary comparison hypothesis in
quality, efficiency, method or reuse. It can also return no supported opportunity,
insufficient evidence, or a deferred uncertain result. Success does not prevent
optimization; unknown task outcomes remain unknown.

```sh
PYTHONPATH=src python3 -m abs_evolution classify --request examples/optimization/request.json --allow-external
```

This command requires `TYPESAFE_API_KEY` in the process environment. It is a live
model command, not a dry run. The bundled input is synthetic; real inputs need
TypeSafe-specific sharing approval and reviewed, minimized evidence summaries.
Do not include transcripts, credentials, personal data or protected holdout cases.
The command does not read registry history or retrieve evidence on its own.

The output retains the Choice distribution, observed model/token usage, request
and judgment fingerprints, and opaque evidence references. It omits raw input
text. Confidence and selected probability thresholds default to 0.8 and remain
provisional; ties or low confidence defer without another model or heuristic
fallback. A service error is an error, not a negative classification.

`next_step` is advisory. No result creates a receipt, runs a candidate, changes a
channel or grants permission. `receipt record` preserves explicit supplied
observations; `opportunities` only aggregates them. Use `classify` when their
meaning needs judgment. Neither command silently infers labels from tag names.
The existing `hybrid` route below uses JEV to select reuse, generation or
insufficient evidence. Both stages follow TypeSafe's
[Choice](https://docs.typesafe.ai/primitives/choice) and
[confidence](https://docs.typesafe.ai/confidence) contracts.

For an experimental, non-exclusive assessment, use `assess` with the same reviewed
request. It leaves `classify` and the hybrid route unchanged:

```sh
PYTHONPATH=src python3 -m abs_evolution assess --request examples/optimization/request.json --allow-external
```

One API request contains a three-way Choice about whether an opportunity is worth
comparing, plus four independent [Noul](https://docs.typesafe.ai/primitives/noul)
questions for quality, efficiency, method and reuse. Several directions can be
supported together. Each Noul value is a probability of yes, not a gain estimate,
and the four values do not sum to one. Values at or above 0.8 are supported, at or
below 0.2 are not supported, and the middle remains uncertain. These thresholds
are provisional; the presence Choice retains the default 0.8 confidence and
probability gates. Uncertain presence never becomes a recommendation to compare
just because one direction is high. Conflicting confident answers defer for review.
Raw typed answers remain visible even when the combined assessment defers.

The same external-sharing opt-in, input limits and no-write boundaries apply.
Usage is recorded once for the entire batch. Changing from a six-way Choice to
separate questions changes the judgment being measured: higher values alone do
not establish better accuracy. Use fixed inputs, a pinned model and a predeclared
call budget when comparing question designs; evaluate correctness independently.

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
library. Default candidate execution uses macOS with a functioning
`sandbox-exec`; unsupported or unavailable isolation blocks experiments.
An explicit Linux Docker backend is described below. The engine never starts a
Docker daemon, pulls images, or falls back to ordinary subprocess execution.

Seatbelt isolates filesystem and network access but does not enforce hard RAM
or aggregate scratch-storage quotas. Experiments therefore block by default.
`--allow-shared-resources` is an explicit opt-in for a supervised local run of
reviewed code that shares host resources. It is not suitable for unattended
model-generated code. The Linux Docker backend enforces resource quotas, but
does not itself enable automatic experiments or desktop integration.

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
This remains a useful correctness regression, not a requirement that every
optimization begin with a defect.

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

### Explicit Linux execution with hard quotas

On a prepared Linux host, `doctor --sandbox docker` probes the existing local
Docker daemon and pinned image. `evolve --sandbox docker` selects that same
backend explicitly. The default remains Seatbelt, including on Linux, so a
missing explicit selection cannot silently start containers.

```sh
PYTHONPATH=src python3 -m abs_evolution doctor --sandbox docker
PYTHONPATH=src python3 -m abs_evolution --state /absolute/path/to/state evolve --sandbox docker --suite examples/acceptance/suite.json --provider file --proposal-file examples/acceptance/proposal.json
```

The backend requires the exact image pinned in `docker_sandbox.py` to be
prepared by the operator; it never downloads an image or starts a daemon.
Candidates receive only their snapshot, mounted read-only at `/candidate`,
and ephemeral scratch at `/tmp`. Scratch writes are discarded, not copied back
to the host `cwd`. Only isolated Python invocation is supported, not arbitrary
host executables. Registry state, other snapshots, the evaluator and protected
acceptance answers are not mounted.

Runtime admission must pass resource/isolation checks before the backend claims
hard limits. Candidate runs enforce memory, scratch, process, CPU, combined
output and execution-time bounds; container cleanup includes descendants.
This relies on the trusted Linux kernel and Docker daemon, not a VM-level
defense against kernel/container escapes. `unattended_ready` remains false:
desktop lifecycle integration, operational supervision and real ABS capability
evaluation are separate work.

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

Submit compact receipts with `receipt record FILE.json`. Keep the existing
outcome, failure tags and opaque evidence references. Optional `task_family`
and `optimization_signals` describe explicit observations from any outcome.
Each signal has a dimension of `quality`, `efficiency`, `method` or `reuse`, and
a safe identifier tag. Nonempty signals require a task family and evidence
references. No signal is inferred merely because a task succeeded or failed.
Unknown outcomes remain unknown. Raw messages, arbitrary metadata, URLs and
local paths are not receipt fields.

```json
{
  "id": "receipt-001",
  "task_id": "report-123",
  "snapshot_id": "REPLACE_WITH_THE_PINNED_SNAPSHOT_ID",
  "outcome": "accepted",
  "failure_tags": [],
  "task_family": "source-backed-summary",
  "optimization_signals": [
    {"dimension": "efficiency", "tag": "repeated_source_reads"}
  ],
  "evidence_refs": ["evidence-001"]
}
```

`opportunities` groups matching signals across distinct tasks within the same
family and dimension, preserving versions, receipts, evidence references and
observed outcomes. Two receipts from one task do not satisfy the default
two-task threshold. Accepted tasks can produce `needs_comparison` opportunities;
their observations are not measured gains. Legacy failure-only receipts remain
valid and produce `needs_investigation` results. The command is read-only and
does not start experiments, invoke a model or promote a version.

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
- Seatbelt does not enforce hard RAM or aggregate scratch quotas. The explicitly
  selected Linux Docker backend verifies and enforces these limits; neither
  backend alone establishes unattended readiness.

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

A separate Linux-only job first runs fixed reviewed feasibility probes in a
digest-pinned official Python container, with no host mounts or injected keys.
It exercises OOM termination at 64 MiB (no swap), aggregate writes to an 8 MiB
scratch tmpfs and 1 MiB shared-memory mount, a 16-process ceiling, CPU throttling,
network denial, read-only root storage, and deadline/descendant cleanup.
Each probe removes its own container. These independent probes are **not** an
audit against container escapes or unattended-readiness evidence.
It runs only on disposable GitHub Linux runners; the Mac Docker daemon is not
started. No arbitrary candidate source is accepted by this probe script.
The limits follow Docker's [resource constraints](https://docs.docker.com/engine/containers/resource_constraints/)
and [tmpfs semantics](https://docs.docker.com/engine/storage/tmpfs/); tmpfs memory
also counts toward the container memory ceiling.
The trusted memory probe observes a killed allocation child and positive
`max`, `oom` and `oom_kill` deltas in the container's
[kernel memory events](https://docs.kernel.org/admin-guide/cgroup-v2.html#memory).
Exit code 137 or Docker's `OOMKilled` field alone is not the admission proof.

The same job then tests the actual Docker backend and runs the public CLI
through offline evaluation, explicit promotion, execution of pinned old/new
versions, rollback and execution after rollback. A negative-control candidate
exceeds its real memory limit and must leave the channel unchanged. The scores
measure only the bundled synthetic fixture, not model-generated improvement or
real ABS skill quality. No model provider is called.

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
