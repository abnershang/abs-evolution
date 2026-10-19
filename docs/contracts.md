# Implementation contracts

This document defines the local engine API. It does not authorize installation
or declare that existing Codex conversations adopt a registry version.

## Storage and state

`Registry.initialize(root: Path, policy: Policy) -> Registry` creates a new state
root. `Registry(root: Path)` opens existing state without silently initializing.
All storage is beneath that explicitly selected root. No defaults point into
ABS, plugin caches, home directories or automation settings.

The registry owns SQLite transactions, immutable snapshots and channel pointers.
The trusted operator owns the root; candidates and generated programs do not.
There is one local machine per root. Do not put SQLite state on a shared/network
filesystem and do not claim distributed coordination.

Public methods return JSON-compatible dictionaries unless stated otherwise:

- `policy() -> Policy` and `update_policy(policy: Policy) -> None`.
- `import_snapshot(source: Path, *, kind="skill", parent_id=None,
  provenance=None) -> dict`: copy regular files, reject symlinks and unsafe
  names; exclude credential/runtime/cache paths. Return `id` (content hash),
  `file_count`, `byte_count`, `kind`, `parent_id`. Do not modify source.
- `create_candidate(parent_id: str, changes: dict[str, str | None], *,
  kind="skill", provenance=None) -> dict`: bounded UTF-8 replacements/deletions
  only at policy-allowed relative paths. No-op proposals are rejected.
- `snapshot_path(snapshot_id: str) -> Path`: verify the manifest and all content
  hashes before returning the immutable `files` directory. No symlinks.
- `snapshot_info(snapshot_id: str) -> dict` and `list_snapshots() -> list[dict]`.
- `bootstrap(snapshot_id: str, *, channel="stable") -> dict`: create an empty
  channel only; never overwrite an existing channel.
- `channel(channel="stable") -> dict`: return `snapshot_id` and `generation`.
- `begin_task(task_id: str, *, channel="stable") -> dict`: pin the channel
  snapshot as a persistent lease. Idempotent for the same active task. Return
  `task_id`, `snapshot_id`, `snapshot_path`, `status`. No automatic lease expiry.
- `end_task(task_id: str) -> dict` and `active_tasks() -> list[dict]`.
- `record_receipt(receipt: dict) -> dict`: strict bounded schema, opt-in only.
  Fields: `id`, `task_id`, `snapshot_id`, `outcome` (accepted/failed/unknown),
  `failure_tags` (list of safe identifiers), `evidence_refs` (opaque identifiers,
  not raw transcripts, URLs or paths). Optional `task_family` is a safe tag.
  Optional `optimization_signals` is a list of at most 16 unique
  `{"dimension":"quality|efficiency|method|reuse","tag":safe_identifier}`
  objects. Nonempty signals require a task family and nonempty evidence refs.
  Signals record hypotheses even when outcome is accepted; they are not scores,
  verified improvements or permission to act. Reject undocumented fields.
  Legacy payloads remain unchanged. Duplicate identical IDs are idempotent;
  conflicting reuse fails. `list_receipts(limit=100)`.
- `claim_experiment(*, baseline_id: str) -> dict`: in one transaction check
  foreground leases, concurrent claims, daily count/time-reservation budget;
  record a running claim with `id`, `policy_digest`, `baseline_id`. Reserve the
  full experiment deadline. Do not automatically expire/retry uncertain claims.
- `finish_experiment(experiment_id: str, *, status: str, elapsed_seconds: float,
  detail: dict | None = None) -> dict`: terminal status completed/failed/blocked.
  Retain original reservations conservatively for the day. Do not reassign a
  finished claim to another run. `experiment(experiment_id: str) -> dict`.
- `consume_holdout(holdout_digest: str, experiment_id: str) -> dict`: reserve a
  protected acceptance set once. Same-experiment retries are idempotent; another
  experiment cannot reuse that set. Never release after an uncertain/failed run.
  Hash normalized input/expected pairs, not changeable suite or case labels.
- `record_evaluation(experiment_id: str, evidence: dict) -> dict`: engine-only
  trusted write, no CLI for importing arbitrary user-supplied scores. Evidence
  contains `candidate_id`, `baseline_id`, `suite_digest`, `environment_digest`,
  `policy_digest`, `status`, `baseline_training`, `candidate_training`,
  `baseline_holdout`, `candidate_holdout`, `violations`, `errors`, `case_counts`.
  Every success rate is finite in [0,1]; counts are nonzero and exact; status
  must distinguish infrastructure failure from low quality. Return `id`.
- `promote(evaluation_id: str, *, expected_current: str, channel="stable") ->
  dict`: atomically revalidate evidence, current policy, candidate integrity and
  exact baseline/current channel. Require completed experiment, passed status,
  no errors/violations, minimum training/holdout success, no excess regression,
  required gain. Affect new leases only; audit the local pointer transition.
- `rollback(snapshot_id: str, *, expected_current: str, channel="stable") ->
  dict`: only to a snapshot previously accepted in that channel; CAS transition.
- `status() -> dict`: local channels, active leases/claims and bounded counts.

`summarize(receipts, *, min_occurrences=2)` groups observations by signal type,
task family, dimension and tag, using distinct task IDs for the threshold.
Explicit optimization signals can come from accepted, failed or unknown tasks;
outcome alone never creates a signal. Their status is `needs_comparison`.
Legacy failure tags still require a failed outcome and retain `failure_tag` and
`needs_investigation`. Every group retains provenance and `observed_outcomes`;
unknown does not become accepted. An absent family is null and is not pooled
with named families. Global `outcome_counts` count receipts, not distinct tasks.
This read-only summary neither establishes causation nor runs an experiment.

## Process isolation

`Sandbox.run(argv: list[str], *, cwd: Path, read_paths: list[Path],
write_paths: list[Path], stdin: str="", timeout_seconds: float=10,
max_output_bytes: int=262144) -> ProcessResult` uses an enforced OS backend.
macOS Seatbelt is the first supported backend. Network is denied; inherited
environment credentials, user startup files and unrelated filesystem roots are
not exposed. Reads include only specified paths and required system/runtime
libraries. Writes include only specified scratch paths. No shell interpolation.
No sandbox means `status="blocked"`; there is no unsandboxed fallback.
Timeout/output overflow kill the complete process group and return a typed
result. Separate stdout/stderr but enforce one combined byte limit.
`Sandbox.hard_resource_limits` is false for the current Seatbelt backend: hard
memory and aggregate scratch-storage quotas are not enforced. The engine must
block by default rather than confuse access isolation with resource isolation.

`DockerSandbox` is an explicit Linux-only alternative with the same run
signature but narrower invocation semantics: only isolated Python, one snapshot
read root, and one scratch directory equal to `cwd`. The snapshot is mounted
read-only at `/candidate`; scratch is an ephemeral container mount and is never
copied back to the host. Unsupported requests fail closed. The local Unix-socket
daemon and digest-pinned image must already exist; no implicit daemon startup,
pull, remote Docker endpoint or unsandboxed fallback is permitted. Runtime
admission verifies hard-resource controls before claiming capability. Limits
and the pinned image are part of the backend identity. Timeout/output overflow
must stop and remove the container, not merely its attached client process.
Neither Docker capability nor an accepted registry version means deployment or
unattended desktop readiness.

Python candidate invocation is `[sys.executable, "-I", "-S", entrypoint]`.
`read_paths` contains the snapshot only, never the registry database, evaluator,
suite files, expected answers or other versions. Candidate input is one JSON
value on stdin and output is one JSON value. Evaluation occurs in the trusted
parent process against expected values never passed to the candidate.

## Candidate provider

`CodexProvider(model: str | None = None).propose(request: dict, *,
timeout_seconds: float, max_output_bytes: int) -> ProposalResult` currently
raises `provider_blocked` without starting a process. The inspected CLI cannot
establish a verifiable zero-tool boundary. A future provider must receive only
selected candidate files and training examples, never holdout cases or runtime
credentials. It must have no host-reading or editing tools. A broad tool-enabled
coding session is not a fallback. The model proposes data, not host edits.

Output shape is `{"summary": str, "kind": "skill|tool|workflow|memory|optimizer",
"changes": [{"path": str, "content": str | null}]}`. Limit size and changes;
reject unknown fields, duplicate paths, unsafe paths and no-op content.
Record token counts only when observed; unavailable usage is null, never zero.
`FileProposalProvider(path: Path)` supports explicit reviewed/offline proposals
using the same contract. It is not a model-quality or autonomy proof.

`OpenRouterProvider(model: str, *, max_output_tokens=2048)` implements the same
proposal contract through the fixed OpenRouter Chat Completions endpoint.
The model ID is explicit. Read only `OPENROUTER_API_KEY` from the calling process;
do not resolve secret-manager references or read credential files in this class.
Transmit only the validated request as model input. Reject credential-like
material in any request field, not only the selected source files.

The provider sends a strict JSON schema and no tools or plugins. Routing sets
`require_parameters=true`, `data_collection="deny"`, `zdr=true`, and
`allow_fallbacks=false`. Do not redirect credentials, retry requests, change
models or relax privacy requirements after a failure. The trusted HTTP child
receives the request and the single key over stdin, never command arguments or
files; the parent enforces the total deadline and terminates a timed-out child.
Errors must not echo the key, source input or the remote error body.

The response must contain one completed assistant proposal, not a tool call,
refusal or truncated generation. Local shape/path/content validation still
applies. `ProposalResult` may include observed `model`, `generation_id` and
`cost_credits` alongside token counts. Missing observations remain null; do not
invent zero cost or convert credits to another unit.

`propose` exposes generation without candidate execution or registry writes.
OpenRouter declares `requires_hard_resource_limits=true`; the engine must not
apply a supervised file-proposal exception to unreviewed generated code.

Provider request shape is exactly `{"schema_version":1,"objective":str,
"baseline_id":str,"files":{relative_path:utf8text},"training_examples":[
{"id":str,"input":JSON,"expected":JSON,"actual":JSON,"passed":bool}],
"allowed_mutation_paths":[str],"max_changes":int}`. The engine owns provenance.
Unknown top-level fields are rejected. A task's nested JSON keys are data, not
provider permissions. Raw chat transcripts and holdout data are never inputs.

## Opportunity classification

`OpportunityJudge(judge, *, min_confidence=0.8, min_probability=0.8).assess(
request, *, timeout_seconds=30, max_output_bytes=65536)` uses the JEV adapter for
one primary-next-step Choice. Request fields are exactly `schema_version:1`,
`task_family`, `objective`, `observation`, `outcome` and `evidence`.
The family is a safe tag; outcome is accepted/failed/unknown. Objective and
observation are nonempty UTF-8 text bounded to 2000 and 4000 bytes respectively.
Evidence contains up to 16 unique `{id, summary}` records. IDs are opaque and
summaries are bounded to 4000 bytes each. The full request is at most 65536 bytes.
Empty evidence is allowed for an explicit insufficient-evidence judgment.
This request contains reviewed observations, never protected acceptance cases.

Choice options are `quality`, `efficiency`, `method`, `reuse`, `no_opportunity`
and `insufficient_evidence`. The four dimensions select a primary hypothesis,
not all applicable labels and not measured gains. Confident dimensions return
`status=classified, next_step=compare`; confident no-opportunity returns
`next_step=retain_baseline`. Insufficient evidence returns
`status=deferred, next_step=gather_evidence`. Ties or threshold misses preserve
the raw judgment but return `decision=uncertain, next_step=review_uncertainty`.
Service failures remain typed errors. No retry, generator fallback, evidence
retrieval or registry writes occur. CLI `classify` requires `--allow-external`
and works without registry state; a zero exit code only means a judgment was
returned, not that an experiment or adoption is authorized.

The output keeps the original observed outcome, evidence IDs, complete judgment
and usage metadata, configured thresholds, request fingerprint and rubric
fingerprint. It omits objective, observation and evidence summaries. Input and
output are credential-scanned and bounded. The request snapshot cannot change
if the caller mutates its input during judgment. Both classification and hybrid
routing share `validate_choice_result` for the external-result contract.

## Experimental split opportunity assessment

`OpportunityAssessment(judge, *, min_confidence=0.8, min_probability=0.8).assess(...)`
and CLI `assess` use the same request schema, external opt-in and no-registry
boundary as `classify`. The response is identified by
`assessment_mode=split_opportunity_v1`; it is not a replacement classification.
One batch contains `opportunity` as a Choice over `worth_comparing`,
`no_opportunity` and `insufficient_evidence`, plus independent Noul questions
`quality`, `efficiency`, `method` and `reuse`. Each question judges the same input,
never another answer. Method can coexist with quality/efficiency; reuse requires
an evidenced successful method and plausible transfer, not just an unfinished attempt.

`opportunity` preserves the Choice answer. `dimensions` records each
`probability_yes` and its status: supported at >=0.8, not_supported at <=0.2,
otherwise uncertain. All answers survive presence gating and need not agree.
`supported_dimensions` lists all supported labels, never a unique winner.
Usage and duration belong to the batch, not to individual answers.
Ties or presence threshold misses yield `review_uncertainty`; insufficient
evidence yields `gather_evidence`. Confident no-opportunity with a supported
dimension yields `review_conflict`, otherwise `retain_baseline`. Confident
worth-comparing with no supported dimension yields `review_dimensions`, otherwise
`compare`. Review/gather paths are deferred. All next steps are advisory.

`JevJudge.judge_batch(state, *, questions, timeout_seconds, max_output_bytes)`
accepts one to eight named Choice/Noul questions with nonempty string
instructions and string criteria. Noul criteria are exactly `true` and `false`.
The returned `BatchJudgment` has typed `ChoiceAnswer`/`NoulAnswer` entries and one
set of model/usage/duration metadata. Exact answer IDs, types and finite bounded
probabilities are required. `validate_batch_judgment` protects callers using
other judge implementations too. A malformed answer fails the entire batch;
there is no partial-result fallback or retry. Fixed endpoint, pinned model,
credential scanning, bounded worker isolation and deadline enforcement also
apply to batches. Neither this path nor `classify` generates candidate content.

## Typed judgment and conditional generation

`JevJudge(model="jev-1.13.0").choose(state, *, instructions, criteria,
timeout_seconds, max_output_bytes) -> ChoiceResult` sends one Choice question
to the fixed TypeSafe System One endpoint. It uses only `TYPESAFE_API_KEY` from
the process environment and the same bounded, credential-isolated HTTP
transport as OpenRouter. No redirects, configurable endpoints or automatic
retries are allowed. Validate the returned option, complete distribution,
finite probabilities summing to one, and confidence before consuming it.

`HybridProposalProvider(judge, generator=None, *, candidates=None,
min_confidence=0.8, min_probability=0.8)` preserves the provider request schema.
The judge receives exactly `{"request": request, "candidates": [...]}`.
Each candidate has `id`, `baseline_id`, and `proposal`; all must be bounded,
unique, valid and bound to the current baseline before the judge is called.
Candidate source remains data, not routing instructions or permissions.

The closed option set is `reuse_<id>` for every supplied candidate, `generate`,
and `insufficient_evidence`. Program-owned confidence/probability thresholds
are checked before routing. Confident reuse returns the exact validated
proposal and makes no generator call. Confident generation invokes the
configured generator once with the original request and remaining deadline.
Route selection follows the stated objective and training evidence, including
already-passing tasks. Failures are not required. Efficiency, method and reuse
are proposal hypotheses; the route does not establish acceptance or gain.
Missing generation capability, uncertainty or missing evidence blocks the
round without a substitute patch. Service/validation failures remain errors
and never trigger an implicit generator call.

`ProposalResult.routing` is an optional typed trace. `validated_routing` produces
bounded JSON containing the route, selected candidate ID, judge distribution
and usage, separate generator usage, and request/pool fingerprints. It contains
no raw task input or proposal text. CLI output exposes it as `routing`; the
engine validates and retains it in `provider_usage.routing`. Missing usage is
null. The generator `called` field records whether that branch ran; it is not
an inference from token counts.

The hybrid provider always requires hard resource limits. Model selection does
not establish candidate correctness or authorize registry promotion, host
changes or publication. Evaluate domain accuracy and thresholds separately
before adopting this routing for real tasks.

## Experiment engine

`EvolutionEngine(registry, sandbox, provider).run(suite_path: Path, *,
auto_promote=False, channel="stable", allow_shared_resources=False) -> dict`
owns the full cycle. It blocks execution when hard resource isolation is absent
unless the operator explicitly opts into a supervised shared-resource run of a
reviewed file proposal. A provider requiring hard resource limits cannot use
that exception.
This exception must not become an unattended-experiment default. It claims a
bounded experiment; evaluates the current baseline on training cases; proposes
one candidate using training examples; evaluates candidate training; then runs
one protected acceptance holdout for baseline and candidate. Holdout details are
not returned to the provider. Reserve the holdout only after training qualifies;
it is not a repeatedly consulted selection set. It records evidence, terminates the
claim and optionally promotes within the local registry. No call installs ABS.

Suite JSON version 1: `{"schema_version":1,"id":str,"mode":"python",
"entrypoint":"tools/solver.py","training":[{"id":str,"input":JSON,
"expected":JSON}],"holdout":[...]}`. Unique IDs and disjoint normalized inputs
across splits are required. Duplicate normalized inputs within a split are
rejected, even when their IDs differ. Holdout identity uses unique normalized
input/expected pairs and ignores case IDs and order.

Before execution, reject a selected suite inside an accessible snapshot and
complete copies of its raw or canonical JSON content within a snapshot. Apply
this check to both baseline and candidate snapshots. The operator must still
exclude oracle material from source imports; this check cannot recognize every
encoded or fragmented answer key.

Reject nonfinite JSON and unbounded cases. Python
candidates receive only input JSON. Exact JSON type/value equality is the first
supported oracle. A low score is a quality failure; process errors/timeouts or
invalid output are infrastructure/execution errors, not invented zero scores.
The suite fingerprint covers all cases/configuration. Record Python/platform
and sandbox identity. A monotonic overall deadline bounds generation and every
case; optional learning failure never bypasses business checks.

Foreground priority applies to leases registered with this engine, not to
unobserved desktop activity. Check before claiming and between generation/case
steps; yield by ending a claim blocked, without touching foreground tasks.
No production replay, daemon, schedule, autonomous deployment or live business
API call is part of this initial execution contract. An explicitly selected
OpenRouter proposal request is a model call, not a business-system operation.

Snapshot count and aggregate stored snapshot bytes are bounded by policy. Old
versions are never garbage-collected while tasks may still use them; reaching a
storage limit blocks further learning instead of deleting task dependencies.
