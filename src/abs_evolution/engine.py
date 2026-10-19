"""Bounded, evidence-gated orchestration for local evolution experiments.

The engine is the trusted parent of candidate execution: suite answers and
evaluation state stay outside the snapshot passed to the sandbox, and only
training examples are supplied to the proposal provider.
"""

from __future__ import annotations

from dataclasses import dataclass
import fnmatch
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import sys
import tempfile
import time
from typing import Any, Callable

from .contracts import Policy, ProposalResult, canonical_json, fingerprint
from .errors import EvolutionError


_SUITE_KEYS = {"schema_version", "id", "mode", "entrypoint", "training", "holdout"}
_CASE_KEYS = {"id", "input", "expected"}
_PROPOSAL_KEYS = {"summary", "kind", "changes"}
_CHANGE_KEYS = {"path", "content"}
_KINDS = {"skill", "tool", "workflow", "memory", "optimizer"}
_SAFE_CODE = re.compile(r"^[a-zA-Z0-9_.-]{1,80}$")
_SECRET_WORDS = (
    "secret", "credential", "password", "passwd", "private", "token",
    "cookie", "session", "auth",
)
_NON_CONTEXT_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache",
    "docs", "doc", "documentation", "tests", "test", "fixtures", "fixture",
    "examples", "evaluation", "evaluations", "eval", "cases", "answers",
    "holdout", "gold", "golden", "vendor", "node_modules",
}
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(?:api[_-]?key|secret(?:[_-]?key)?|access[_-]?token|"
    r"refresh[_-]?token|password|passwd|client[_-]?secret)\s*[:=]\s*"
    r"(?:b?['\"][^'\"\r\n]{8,}['\"]|[^\s#]{16,})"
)
_SECRET_TOKEN = re.compile(
    r"(?:AKIA[0-9A-Z]{16}|gh[opsu]_[A-Za-z0-9_]{20,}|"
    r"xox[baprs]-[A-Za-z0-9-]{16,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)"
)


class _RunProblem(Exception):
    """Internal, non-sensitive control-flow error with a stable public code."""

    def __init__(self, code: str, *, outcome: str = "failed", stage: str | None = None):
        self.code = code if _SAFE_CODE.fullmatch(code) else "experiment_failed"
        self.outcome = outcome
        self.stage = stage
        super().__init__(self.code)


@dataclass(frozen=True)
class _CaseOutcome:
    case_id: str
    input_value: Any
    expected: Any
    actual: Any
    passed: bool


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ValueError("non-finite JSON number")


def _strict_json_loads(raw: str) -> Any:
    return json.loads(raw, parse_constant=_reject_constant, object_pairs_hook=_strict_object)


def _json_size(value: Any) -> int:
    return len(canonical_json(value).encode("utf-8"))


def _safe_relative_path(raw: Any, label: str) -> PurePosixPath:
    if not isinstance(raw, str) or not raw or "\\" in raw or raw.startswith("/"):
        raise _RunProblem(f"invalid_{label}")
    parts = raw.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise _RunProblem(f"invalid_{label}")
    path = PurePosixPath(raw)
    if path.is_absolute() or path.as_posix() != raw:
        raise _RunProblem(f"invalid_{label}")
    return path


def _mutation_allowed(path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def _exact_json_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's bool/int or int/float coercions."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _exact_json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _exact_json_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, float) and not math.isfinite(left):
        return False
    return left == right


def _safe_error_code(exc: BaseException, fallback: str) -> str:
    code = getattr(exc, "code", None)
    if isinstance(code, str) and _SAFE_CODE.fullmatch(code):
        return code
    return fallback


def _blocking_code(code: str) -> bool:
    lowered = code.lower()
    return any(token in lowered for token in (
        "foreground", "lease", "busy", "limit", "budget", "unsupported",
        "sandbox_unavailable", "holdout_already_used", "provider_blocked",
        "resource_isolation_incomplete",
    ))


class EvolutionEngine:
    """Run one bounded local experiment against a pinned channel baseline."""

    def __init__(self, registry: Any, sandbox: Any, provider: Any):
        self.registry = registry
        self.sandbox = sandbox
        self.provider = provider

    def run(
        self,
        suite_path: Path,
        *,
        auto_promote: bool = False,
        channel: str = "stable",
        allow_shared_resources: bool = False,
    ) -> dict[str, Any]:
        started = time.monotonic()
        state: dict[str, Any] = {
            "outcome": "failed",
            "terminal": "failed",
            "reason": None,
            "stage": "setup",
            "experiment_id": None,
            "baseline_id": None,
            "candidate_id": None,
            "evaluation_id": None,
            "scores": {},
            "violations": [],
            "provider_usage": None,
            "promotion": None,
            "shared_host_resources": None,
            "finished": False,
        }
        policy: Policy | None = None
        deadline: float | None = None
        suite: dict[str, Any] | None = None
        suite_raw_digest: str | None = None
        suite_digest: str | None = None
        policy_digest: str | None = None
        claimed = False

        try:
            if type(auto_promote) is not bool:
                raise _RunProblem("invalid_auto_promote")
            if type(allow_shared_resources) is not bool:
                raise _RunProblem("invalid_allow_shared_resources")
            if not isinstance(channel, str) or not channel or len(channel) > 128:
                raise _RunProblem("invalid_channel")

            policy = self.registry.policy()
            if not isinstance(policy, Policy):
                raise _RunProblem("invalid_policy")
            hard_resource_limits = getattr(self.sandbox, "hard_resource_limits", False) is True
            requires_hard_resources = getattr(self.provider, "requires_hard_resource_limits", False) is True
            if not hard_resource_limits and (not allow_shared_resources or requires_hard_resources):
                raise _RunProblem(
                    "resource_isolation_incomplete", outcome="blocked", stage="resource_admission",
                )
            state["shared_host_resources"] = not hard_resource_limits
            deadline = started + policy.experiment_timeout_seconds
            state["stage"] = "suite_validation"
            suite, suite_raw_digest = self._load_suite(Path(suite_path), policy)
            suite_digest = fingerprint(suite)
            self._ensure_time(deadline, stage="suite_validation")

            if self._foreground_active(policy):
                raise _RunProblem("foreground_lease_active", outcome="blocked", stage="admission")

            channel_state = self.registry.channel(channel)
            baseline_id = channel_state.get("snapshot_id") if isinstance(channel_state, dict) else None
            if not isinstance(baseline_id, str) or not baseline_id:
                raise _RunProblem("channel_has_no_baseline", stage="admission")
            state["baseline_id"] = baseline_id

            baseline_path = Path(self.registry.snapshot_path(baseline_id))
            self._validate_suite_exclusion(
                Path(suite_path), baseline_path, suite, suite_raw_digest, policy, deadline,
            )
            self._validate_entrypoint(baseline_path, suite["entrypoint"])
            self._check_step(policy, deadline, stage="claim")

            state["stage"] = "claim"
            claim = self.registry.claim_experiment(baseline_id=baseline_id)
            if not isinstance(claim, dict):
                raise _RunProblem("invalid_experiment_claim", stage="claim")
            experiment_id = claim.get("id")
            if not isinstance(experiment_id, str) or not experiment_id:
                # The API promises an ID. Without it, a terminal write is not addressable.
                raise _RunProblem("invalid_experiment_claim", stage="claim")
            state["experiment_id"] = experiment_id
            claimed = True

            claim_policy_digest = claim.get("policy_digest")
            if not isinstance(claim_policy_digest, str) or not claim_policy_digest:
                raise _RunProblem("invalid_experiment_claim", stage="claim")
            policy_digest = claim_policy_digest
            if policy.digest != policy_digest:
                raise _RunProblem("policy_changed", outcome="blocked", stage="claim")
            if claim.get("baseline_id", baseline_id) != baseline_id:
                raise _RunProblem("baseline_claim_mismatch", outcome="blocked", stage="claim")

            state["stage"] = "admission_check"
            self._check_step(policy, deadline, stage="admission_check")
            if self.registry.channel(channel).get("snapshot_id") != baseline_id:
                raise _RunProblem("baseline_changed", outcome="blocked", stage="admission_check")

            baseline_path = Path(self.registry.snapshot_path(baseline_id))
            entrypoint = suite["entrypoint"]
            self._validate_entrypoint(baseline_path, entrypoint)
            sandbox_identity = self._sandbox_identity()
            if sandbox_identity == "unsupported":
                raise _RunProblem("sandbox_unavailable", outcome="blocked", stage="baseline_training")
            environment_digest = fingerprint({
                "python": {
                    "implementation": sys.implementation.name,
                    "version": list(sys.version_info[:3]),
                },
                "platform": platform.platform(),
                "sandbox": sandbox_identity,
            })

            state["stage"] = "baseline_training"
            baseline_training, baseline_train_outcomes = self._evaluate(
                baseline_path, entrypoint, suite["training"], policy, deadline,
                stage="baseline_training",
            )
            state["scores"]["baseline_training"] = baseline_training

            state["stage"] = "proposal"
            self._check_step(policy, deadline, stage="proposal")
            provider_request = self._provider_request(
                baseline_path, entrypoint, baseline_id, policy,
                baseline_train_outcomes,
            )
            self._ensure_time(deadline, stage="proposal")
            proposal_timeout = self._remaining(deadline, stage="proposal")
            try:
                proposal_result = self.provider.propose(
                    provider_request,
                    timeout_seconds=proposal_timeout,
                    max_output_bytes=policy.max_output_bytes,
                )
            except Exception as exc:
                raise _RunProblem(
                    _safe_error_code(exc, "provider_failed"),
                    outcome="blocked" if _blocking_code(_safe_error_code(exc, "provider_failed")) else "failed",
                    stage="proposal",
                ) from exc
            self._ensure_time(deadline, stage="proposal")
            proposal, usage = self._validate_proposal(proposal_result, baseline_path, policy)
            state["provider_usage"] = usage

            state["stage"] = "candidate_creation"
            self._check_step(policy, deadline, stage="candidate_creation")
            candidate = self.registry.create_candidate(
                baseline_id,
                proposal["changes"],
                kind=proposal["kind"],
                provenance={"experiment_id": experiment_id},
            )
            candidate_id = candidate.get("id") if isinstance(candidate, dict) else None
            if not isinstance(candidate_id, str) or not candidate_id:
                raise _RunProblem("invalid_candidate", stage="candidate_creation")
            if candidate_id == baseline_id:
                raise _RunProblem("candidate_matches_baseline", stage="candidate_creation")
            state["candidate_id"] = candidate_id

            state["stage"] = "candidate_isolation_check"
            self._check_step(policy, deadline, stage="candidate_isolation_check")
            candidate_path = Path(self.registry.snapshot_path(candidate_id))
            self._validate_suite_exclusion(
                Path(suite_path), candidate_path, suite, suite_raw_digest, policy, deadline,
            )
            self._validate_entrypoint(candidate_path, entrypoint)
            state["stage"] = "candidate_training"
            self._check_step(policy, deadline, stage="candidate_training")
            candidate_training, candidate_train_outcomes = self._evaluate(
                candidate_path, entrypoint, suite["training"], policy, deadline,
                stage="candidate_training",
            )
            state["scores"]["candidate_training"] = candidate_training

            training_violations = self._training_violations(
                baseline_training, candidate_training, policy,
            )
            if training_violations:
                state["violations"] = training_violations
                state["reason"] = training_violations[0]
                state["outcome"] = "quality_failed"
                state["terminal"] = "failed"
                state["stage"] = "candidate_training_gate"
                raise _RunProblem(
                    training_violations[0], outcome="quality_failed", stage="candidate_training_gate",
                )

            state["stage"] = "baseline_holdout"
            holdout_digest = self._holdout_digest(suite["holdout"])

            def consume_protected_holdout() -> None:
                self.registry.consume_holdout(holdout_digest, experiment_id)

            baseline_holdout, _baseline_holdout_outcomes = self._evaluate(
                baseline_path, entrypoint, suite["holdout"], policy, deadline,
                stage="baseline_holdout", before_first_case=consume_protected_holdout,
            )
            state["scores"]["baseline_holdout"] = baseline_holdout

            state["stage"] = "candidate_holdout"
            candidate_holdout, _candidate_holdout_outcomes = self._evaluate(
                candidate_path, entrypoint, suite["holdout"], policy, deadline,
                stage="candidate_holdout",
            )
            state["scores"]["candidate_holdout"] = candidate_holdout

            violations = self._quality_violations(
                baseline_training,
                candidate_training,
                baseline_holdout,
                candidate_holdout,
                policy,
            )
            state["violations"] = violations
            evidence_status = "quality_failed" if violations else "passed"
            state["stage"] = "record_evaluation"
            self._check_step(policy, deadline, stage="record_evaluation")
            evidence = {
                "candidate_id": candidate_id,
                "baseline_id": baseline_id,
                "suite_digest": suite_digest,
                "environment_digest": environment_digest,
                "policy_digest": policy_digest,
                "status": evidence_status,
                "baseline_training": baseline_training,
                "candidate_training": candidate_training,
                "baseline_holdout": baseline_holdout,
                "candidate_holdout": candidate_holdout,
                "violations": violations,
                "errors": [],
                "case_counts": {
                    "training": len(suite["training"]),
                    "holdout": len(suite["holdout"]),
                },
            }
            try:
                recorded = self.registry.record_evaluation(experiment_id, evidence)
            except Exception as exc:
                raise _RunProblem(
                    _safe_error_code(exc, "evaluation_write_failed"),
                    outcome="blocked" if _blocking_code(_safe_error_code(exc, "evaluation_write_failed")) else "failed",
                    stage="record_evaluation",
                ) from exc
            evaluation_id = recorded.get("id") if isinstance(recorded, dict) else None
            if not isinstance(evaluation_id, str) or not evaluation_id:
                raise _RunProblem("invalid_evaluation_record", stage="record_evaluation")
            state["evaluation_id"] = evaluation_id

            if violations:
                state["reason"] = "quality_gate_failed"
                state["outcome"] = "quality_failed"
                state["terminal"] = "failed"
            else:
                state["reason"] = None
                state["outcome"] = "completed"
                state["terminal"] = "completed"
                state["stage"] = "complete"

        except _RunProblem as problem:
            state["outcome"] = problem.outcome
            state["terminal"] = "blocked" if problem.outcome == "blocked" else "failed"
            state["reason"] = problem.code
            if problem.stage:
                state["stage"] = problem.stage
        except EvolutionError as exc:
            code = _safe_error_code(exc, "evolution_error")
            state["outcome"] = "blocked" if _blocking_code(code) else "failed"
            state["terminal"] = "blocked" if state["outcome"] == "blocked" else "failed"
            state["reason"] = code
        except Exception as exc:
            code = _safe_error_code(exc, "experiment_failed")
            state["outcome"] = "blocked" if _blocking_code(code) else "failed"
            state["terminal"] = "blocked" if state["outcome"] == "blocked" else "failed"
            state["reason"] = code
        finally:
            if claimed and not state["finished"]:
                elapsed = max(0.0, time.monotonic() - started)
                detail = self._finish_detail(state, suite_digest)
                try:
                    self.registry.finish_experiment(
                        state["experiment_id"],
                        status=state["terminal"],
                        elapsed_seconds=elapsed,
                        detail=detail,
                    )
                    state["finished"] = True
                except Exception as exc:
                    code = _safe_error_code(exc, "experiment_finish_failed")
                    state["outcome"] = "failed"
                    state["terminal"] = "failed"
                    state["reason"] = code

        if state["finished"] and state["terminal"] == "completed" and auto_promote:
            if policy is not None and deadline is not None:
                try:
                    self._ensure_time(deadline, stage="promotion")
                    if self._foreground_active(policy):
                        raise _RunProblem("foreground_lease_active", outcome="blocked", stage="promotion")
                    promoted = self.registry.promote(
                        state["evaluation_id"],
                        expected_current=state["baseline_id"],
                        channel=channel,
                    )
                    if not isinstance(promoted, dict):
                        raise _RunProblem("invalid_promotion_result", stage="promotion")
                    state["promotion"] = {
                        key: promoted[key]
                        for key in (
                            "channel", "snapshot_id", "generation",
                            "previous_snapshot_id", "evaluation_id",
                        )
                        if key in promoted
                    }
                except Exception as exc:
                    state["reason"] = _safe_error_code(exc, "promotion_failed")

        return {
            "status": state["outcome"],
            "experiment_id": state["experiment_id"],
            "baseline_id": state["baseline_id"],
            "candidate_id": state["candidate_id"],
            "evaluation_id": state["evaluation_id"],
            "scores": dict(state["scores"]),
            "provider_usage": state["provider_usage"],
            "auto_promoted": state["promotion"] is not None,
            "promotion": state["promotion"],
            "shared_host_resources": state["shared_host_resources"],
            "reason": state["reason"],
        }

    def _load_suite(self, path: Path, policy: Policy) -> tuple[dict[str, Any], str]:
        try:
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except OSError as exc:
            raise _RunProblem("suite_unavailable", stage="suite_validation") from exc
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_size > policy.max_snapshot_bytes:
                raise _RunProblem("invalid_suite_size", stage="suite_validation")
            with os.fdopen(descriptor, "rb") as source:
                descriptor = -1
                raw = source.read(policy.max_snapshot_bytes + 1)
            if len(raw) > policy.max_snapshot_bytes:
                raise _RunProblem("invalid_suite_size", stage="suite_validation")
        finally:
            if descriptor >= 0:
                os.close(descriptor)

        try:
            value = _strict_json_loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
            raise _RunProblem("invalid_suite_json", stage="suite_validation") from exc
        if not isinstance(value, dict) or set(value) != _SUITE_KEYS:
            raise _RunProblem("invalid_suite_schema", stage="suite_validation")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise _RunProblem("unsupported_suite_version", stage="suite_validation")
        if not isinstance(value["id"], str) or not value["id"] or len(value["id"]) > 128:
            raise _RunProblem("invalid_suite_id", stage="suite_validation")
        if value["mode"] != "python":
            raise _RunProblem("unsupported_suite_mode", stage="suite_validation")
        entrypoint = _safe_relative_path(value["entrypoint"], "entrypoint")
        if not isinstance(value["training"], list) or not isinstance(value["holdout"], list):
            raise _RunProblem("invalid_suite_cases", stage="suite_validation")
        if not value["training"] or not value["holdout"]:
            raise _RunProblem("empty_suite_split", stage="suite_validation")
        if len(value["training"]) + len(value["holdout"]) > policy.max_cases:
            raise _RunProblem("too_many_suite_cases", stage="suite_validation")

        seen_ids: set[str] = set()
        training_inputs: set[str] = set()
        holdout_inputs: set[str] = set()
        for split_name, cases in (("training", value["training"]), ("holdout", value["holdout"])):
            for case in cases:
                if not isinstance(case, dict) or set(case) != _CASE_KEYS:
                    raise _RunProblem("invalid_suite_case", stage="suite_validation")
                case_id = case["id"]
                if not isinstance(case_id, str) or not case_id or len(case_id) > 128 or case_id in seen_ids:
                    raise _RunProblem("invalid_suite_case_id", stage="suite_validation")
                seen_ids.add(case_id)
                try:
                    case_bytes = _json_size({"input": case["input"], "expected": case["expected"]})
                except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as exc:
                    raise _RunProblem("invalid_suite_case_json", stage="suite_validation") from exc
                if case_bytes > policy.max_file_bytes:
                    raise _RunProblem("suite_case_too_large", stage="suite_validation")
                normalized_input = canonical_json(case["input"])
                target = training_inputs if split_name == "training" else holdout_inputs
                if normalized_input in target:
                    raise _RunProblem("duplicate_suite_input", stage="suite_validation")
                target.add(normalized_input)
        if training_inputs & holdout_inputs:
            raise _RunProblem("suite_splits_overlap", stage="suite_validation")

        value["entrypoint"] = entrypoint.as_posix()
        return value, hashlib.sha256(raw).hexdigest()

    def _validate_suite_exclusion(
        self,
        suite_path: Path,
        snapshot_path: Path,
        suite: dict[str, Any],
        suite_raw_digest: str,
        policy: Policy,
        deadline: float,
    ) -> None:
        """Reject a known suite copy anywhere candidate-readable in a snapshot.

        This is an exact/canonical full-suite collision guard, not a general
        detector for manually encoded answer fragments. Keep evaluator suites
        and oracle material outside candidate snapshots by construction.
        """
        try:
            root = snapshot_path.resolve(strict=True)
            source = suite_path.resolve(strict=True)
            source.relative_to(root)
        except ValueError:
            pass
        except (OSError, RuntimeError) as exc:
            raise _RunProblem(
                "suite_isolation_unverified", outcome="blocked", stage="suite_isolation",
            ) from exc
        else:
            raise _RunProblem(
                "suite_snapshot_contamination", outcome="blocked", stage="suite_isolation",
            )

        try:
            canonical_suite = canonical_json(suite)
            root_info = root.lstat()
            if not stat.S_ISDIR(root_info.st_mode):
                raise OSError("snapshot path is not a directory")
            total_bytes = 0
            scan_errors: list[OSError] = []
            for directory, directory_names, file_names in os.walk(
                root, topdown=True, onerror=scan_errors.append, followlinks=False,
            ):
                self._ensure_time(deadline, stage="suite_isolation")
                directory_path = Path(directory)
                for name in list(directory_names):
                    child_dir = directory_path / name
                    if child_dir.is_symlink():
                        raise OSError("snapshot contains a directory symlink")
                for name in file_names:
                    self._ensure_time(deadline, stage="suite_isolation")
                    candidate_file = directory_path / name
                    info = candidate_file.lstat()
                    if not stat.S_ISREG(info.st_mode):
                        raise OSError("snapshot contains a non-regular file")
                    total_bytes += info.st_size
                    if total_bytes > policy.max_snapshot_bytes:
                        raise _RunProblem(
                            "suite_isolation_scan_limit", outcome="blocked", stage="suite_isolation",
                        )
                    descriptor = os.open(
                        candidate_file, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    )
                    try:
                        opened = os.fstat(descriptor)
                        if not stat.S_ISREG(opened.st_mode) or opened.st_size != info.st_size:
                            raise OSError("snapshot file changed during suite scan")
                        with os.fdopen(descriptor, "rb") as stream:
                            descriptor = -1
                            contents = stream.read(policy.max_snapshot_bytes + 1)
                    finally:
                        if descriptor >= 0:
                            os.close(descriptor)
                    if len(contents) != info.st_size:
                        raise OSError("snapshot file size changed during suite scan")
                    if hashlib.sha256(contents).hexdigest() == suite_raw_digest:
                        raise _RunProblem(
                            "suite_snapshot_contamination", outcome="blocked", stage="suite_isolation",
                        )
                    # Parse bounded files regardless of extension. For a larger
                    # JSON file, parse only when it has the complete suite's
                    # structural keys; this also catches renamed large copies.
                    suite_markers = (b'"schema_version"', b'"training"', b'"holdout"')
                    looks_suite_shaped = all(marker in contents for marker in suite_markers)
                    if len(contents) <= policy.max_file_bytes or looks_suite_shaped:
                        try:
                            decoded = contents.decode("utf-8")
                            parsed = _strict_json_loads(decoded)
                        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError):
                            continue
                        if self._contains_canonical_suite(parsed, canonical_suite):
                            raise _RunProblem(
                                "suite_snapshot_contamination", outcome="blocked", stage="suite_isolation",
                            )
            if scan_errors:
                raise scan_errors[0]
        except _RunProblem:
            raise
        except (OSError, RuntimeError, ValueError, RecursionError) as exc:
            raise _RunProblem(
                "suite_isolation_unverified", outcome="blocked", stage="suite_isolation",
            ) from exc

    @classmethod
    def _contains_canonical_suite(cls, value: Any, canonical_suite: str) -> bool:
        if isinstance(value, dict):
            if canonical_json(value) == canonical_suite:
                return True
            return any(cls._contains_canonical_suite(item, canonical_suite) for item in value.values())
        if isinstance(value, list):
            return any(cls._contains_canonical_suite(item, canonical_suite) for item in value)
        return False

    def _validate_entrypoint(self, snapshot_path: Path, entrypoint: str) -> None:
        relative = _safe_relative_path(entrypoint, "entrypoint")
        candidate = snapshot_path.joinpath(*relative.parts)
        try:
            info = candidate.lstat()
        except OSError as exc:
            raise _RunProblem("entrypoint_unavailable", stage="entrypoint_validation") from exc
        if not stat.S_ISREG(info.st_mode) or candidate.is_symlink():
            raise _RunProblem("entrypoint_unavailable", stage="entrypoint_validation")

    def _foreground_active(self, policy: Policy) -> bool:
        if not policy.yield_to_foreground:
            return False
        active = self.registry.active_tasks()
        return bool(active)

    def _check_step(self, policy: Policy, deadline: float, *, stage: str) -> None:
        self._ensure_time(deadline, stage=stage)
        if self._foreground_active(policy):
            raise _RunProblem("foreground_lease_active", outcome="blocked", stage=stage)
        self._ensure_time(deadline, stage=stage)

    @staticmethod
    def _ensure_time(deadline: float, *, stage: str) -> None:
        if deadline - time.monotonic() <= 0:
            raise _RunProblem("deadline_exceeded", stage=stage)

    @staticmethod
    def _remaining(deadline: float, *, stage: str) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _RunProblem("deadline_exceeded", stage=stage)
        return remaining

    def _sandbox_identity(self) -> str:
        identity = getattr(self.sandbox, "backend_identity", None)
        if isinstance(identity, str) and identity.strip():
            return identity.strip()
        return f"{type(self.sandbox).__module__}.{type(self.sandbox).__qualname__}"

    def _evaluate(
        self,
        snapshot_path: Path,
        entrypoint: str,
        cases: list[dict[str, Any]],
        policy: Policy,
        deadline: float,
        *,
        stage: str,
        before_first_case: Callable[[], None] | None = None,
    ) -> tuple[float, list[_CaseOutcome]]:
        outcomes: list[_CaseOutcome] = []
        successes = 0
        entry = snapshot_path.joinpath(*PurePosixPath(entrypoint).parts)
        for index, case in enumerate(cases):
            self._check_step(policy, deadline, stage=stage)
            if index == 0 and before_first_case is not None:
                try:
                    before_first_case()
                except Exception as exc:
                    code = _safe_error_code(exc, "holdout_consumption_failed")
                    outcome = "blocked" if _blocking_code(code) else "failed"
                    raise _RunProblem(code, outcome=outcome, stage=stage) from exc
            timeout = min(float(policy.case_timeout_seconds), self._remaining(deadline, stage=stage))
            stdin = canonical_json(case["input"]) + "\n"
            try:
                with tempfile.TemporaryDirectory(prefix="abs-evolution-case-") as scratch_name:
                    result = self.sandbox.run(
                        [sys.executable, "-I", "-S", str(entry)],
                        cwd=Path(scratch_name),
                        read_paths=[snapshot_path],
                        write_paths=[Path(scratch_name)],
                        stdin=stdin,
                        timeout_seconds=timeout,
                        max_output_bytes=policy.max_output_bytes,
                    )
            except Exception as exc:
                code = _safe_error_code(exc, "sandbox_execution_failed")
                raise _RunProblem(code, outcome="blocked" if _blocking_code(code) else "failed", stage=stage) from exc
            self._ensure_time(deadline, stage=stage)
            if getattr(result, "status", None) == "blocked":
                raise _RunProblem("sandbox_unavailable", outcome="blocked", stage=stage)
            if getattr(result, "status", None) != "passed" or getattr(result, "returncode", None) != 0:
                result_status = getattr(result, "status", None)
                reason = "sandbox_timed_out" if result_status == "timed_out" else (
                    "sandbox_output_limit" if result_status == "output_limit" else "sandbox_process_failed"
                )
                raise _RunProblem(reason, stage=stage)
            stdout = getattr(result, "stdout", None)
            if not isinstance(stdout, str):
                raise _RunProblem("candidate_output_invalid", stage=stage)
            try:
                if len(stdout.encode("utf-8")) > policy.max_output_bytes:
                    raise _RunProblem("candidate_output_limit", stage=stage)
                actual = _strict_json_loads(stdout)
                # Ensure every accepted value also has a canonical finite JSON representation.
                canonical_json(actual)
            except _RunProblem:
                raise
            except (json.JSONDecodeError, ValueError, TypeError, UnicodeEncodeError, RecursionError) as exc:
                raise _RunProblem("candidate_output_invalid", stage=stage) from exc
            passed = _exact_json_equal(actual, case["expected"])
            successes += int(passed)
            outcomes.append(_CaseOutcome(case["id"], case["input"], case["expected"], actual, passed))
        return successes / len(cases), outcomes

    def _provider_request(
        self,
        baseline_path: Path,
        entrypoint: str,
        baseline_id: str,
        policy: Policy,
        training: list[_CaseOutcome],
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "schema_version": 1,
            "objective": (
                "Improve observable behavior on the supplied training examples while preserving "
                "behavior that already passes. Stay within the allowed mutation paths; do not "
                "change the evaluator, sandbox, or operator acceptance rules."
            ),
            "baseline_id": baseline_id,
            "files": {},
            "training_examples": [],
            "allowed_mutation_paths": list(policy.allowed_mutation_paths),
            "max_changes": policy.max_changes,
        }
        limit = min(policy.max_output_bytes, policy.max_file_bytes)
        if _json_size(request) > limit:
            raise _RunProblem("provider_request_too_large", stage="proposal")

        file_paths = self._context_files(baseline_path, entrypoint, policy)
        ordered_training = sorted(training, key=lambda item: (item.passed, item.case_id))
        entrypoint_file = next(((path, content) for path, content in file_paths if path == entrypoint), None)
        if entrypoint_file is not None:
            request["files"][entrypoint_file[0]] = entrypoint_file[1]
            if _json_size(request) > limit:
                del request["files"][entrypoint_file[0]]

        def add_training_example(outcome: _CaseOutcome) -> bool:
            example = {
                "id": outcome.case_id,
                "input": outcome.input_value,
                "expected": outcome.expected,
                "actual": outcome.actual,
                "passed": outcome.passed,
            }
            request["training_examples"].append(example)
            if _json_size(request) > limit:
                request["training_examples"].pop()
                return False
            return True

        failures = [outcome for outcome in ordered_training if not outcome.passed]
        included_failure = False
        for outcome in failures:
            included_failure = add_training_example(outcome) or included_failure
        if failures and not included_failure:
            # Keep the failure evidence rather than silently asking the provider
            # to propose without any of the training failures it is meant to fix.
            request["files"].clear()
            for outcome in failures:
                included_failure = add_training_example(outcome) or included_failure
            if not included_failure:
                raise _RunProblem("training_context_too_large", stage="proposal")

        for outcome in ordered_training:
            if outcome.passed:
                add_training_example(outcome)
        for relative, content in file_paths:
            if relative in request["files"]:
                continue
            request["files"][relative] = content
            if _json_size(request) > limit:
                del request["files"][relative]
        if _json_size(request) > limit:
            raise _RunProblem("provider_request_too_large", stage="proposal")
        return request

    def _context_files(
        self,
        snapshot_path: Path,
        entrypoint: str,
        policy: Policy,
    ) -> list[tuple[str, str]]:
        relative_entry = _safe_relative_path(entrypoint, "entrypoint")
        prefix = relative_entry.parts[:-1]
        if prefix:
            context_root = snapshot_path.joinpath(*prefix)
            if not context_root.exists() or context_root.is_symlink():
                return []
            candidates = [path for path in context_root.rglob("*") if path.is_file()]
        else:
            candidates = [snapshot_path.joinpath(*relative_entry.parts)]

        selected: list[tuple[int, str, str]] = []
        for path in candidates:
            try:
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode) or info.st_size > policy.max_file_bytes:
                    continue
                relative = path.relative_to(snapshot_path).as_posix()
            except (OSError, ValueError):
                continue
            if not _mutation_allowed(relative, policy.allowed_mutation_paths):
                continue
            path_parts = PurePosixPath(relative).parts
            lowered_parts = [part.lower() for part in path_parts]
            if any(part in _NON_CONTEXT_DIRS for part in lowered_parts):
                continue
            if any(any(word in part for word in _SECRET_WORDS) for part in lowered_parts):
                continue
            name = path_parts[-1].lower()
            if name in {"readme.md", "license", "license.md", "changelog.md", "changes.md"}:
                continue
            if path.suffix.lower() not in {".py", ".md", ".txt"}:
                continue
            if name.startswith("test_") or name.endswith("_test.py"):
                continue
            try:
                raw = path.read_bytes()
                text = raw.decode("utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if "\x00" in text or _SECRET_ASSIGNMENT.search(text) or _SECRET_TOKEN.search(text):
                continue
            priority = 0 if relative == entrypoint else 1
            selected.append((priority, relative, text))
        selected.sort(key=lambda row: (row[0], row[1]))
        return [(path, content) for _priority, path, content in selected]

    def _validate_proposal(
        self,
        result: Any,
        baseline_path: Path,
        policy: Policy,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not isinstance(result, ProposalResult):
            raise _RunProblem("invalid_provider_result", stage="proposal")
        value = result.value
        if not isinstance(value, dict) or set(value) != _PROPOSAL_KEYS:
            raise _RunProblem("invalid_provider_proposal", stage="proposal")
        if not isinstance(value["summary"], str) or len(value["summary"].encode("utf-8")) > policy.max_output_bytes:
            raise _RunProblem("invalid_provider_proposal", stage="proposal")
        if not isinstance(value["kind"], str) or value["kind"] not in _KINDS or not isinstance(value["changes"], list):
            raise _RunProblem("invalid_provider_proposal", stage="proposal")
        if not value["changes"] or len(value["changes"]) > policy.max_changes:
            raise _RunProblem("invalid_provider_proposal", stage="proposal")

        changes: dict[str, str | None] = {}
        change_bytes = 0
        for change in value["changes"]:
            if not isinstance(change, dict) or set(change) != _CHANGE_KEYS:
                raise _RunProblem("invalid_provider_change", stage="proposal")
            path = _safe_relative_path(change["path"], "proposal_path").as_posix()
            if path in changes or not _mutation_allowed(path, policy.allowed_mutation_paths):
                raise _RunProblem("invalid_provider_path", stage="proposal")
            content = change["content"]
            if content is not None and not isinstance(content, str):
                raise _RunProblem("invalid_provider_content", stage="proposal")
            try:
                size = len(content.encode("utf-8")) if content is not None else 0
            except UnicodeEncodeError as exc:
                raise _RunProblem("invalid_provider_content", stage="proposal") from exc
            if size > policy.max_file_bytes:
                raise _RunProblem("provider_file_too_large", stage="proposal")
            change_bytes += size
            if change_bytes > policy.max_snapshot_bytes:
                raise _RunProblem("provider_changes_too_large", stage="proposal")
            baseline_file = baseline_path.joinpath(*PurePosixPath(path).parts)
            exists = baseline_file.exists()
            if exists and (baseline_file.is_symlink() or not baseline_file.is_file()):
                raise _RunProblem("invalid_baseline_file", stage="proposal")
            if content is None and not exists:
                raise _RunProblem("provider_noop", stage="proposal")
            if content is not None and exists:
                try:
                    current = baseline_file.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError) as exc:
                    raise _RunProblem("invalid_baseline_file", stage="proposal") from exc
                if current == content:
                    raise _RunProblem("provider_noop", stage="proposal")
            changes[path] = content

        usage: dict[str, Any] = {}
        for field_name in ("input_tokens", "output_tokens"):
            observed = getattr(result, field_name, None)
            if observed is not None and (isinstance(observed, bool) or not isinstance(observed, int) or observed < 0):
                raise _RunProblem("invalid_provider_usage", stage="proposal")
            usage[field_name] = observed
        for field_name in ("model", "generation_id"):
            observed = getattr(result, field_name, None)
            if observed is not None:
                if not isinstance(observed, str) or not observed or len(observed) > 256 or any(
                    ord(character) < 33 or ord(character) == 127 for character in observed
                ):
                    raise _RunProblem("invalid_provider_usage", stage="proposal")
                usage[field_name] = observed
        observed_cost = getattr(result, "cost_credits", None)
        if observed_cost is not None:
            if isinstance(observed_cost, bool) or not isinstance(observed_cost, (int, float)) or not math.isfinite(observed_cost) or observed_cost < 0:
                raise _RunProblem("invalid_provider_usage", stage="proposal")
            usage["cost_credits"] = observed_cost
        if result.routing is not None:
            from .contracts import validated_routing
            usage["routing"] = validated_routing(result.routing)
        return {"kind": value["kind"], "changes": changes}, usage

    @staticmethod
    def _training_violations(
        baseline_training: float,
        candidate_training: float,
        policy: Policy,
    ) -> list[str]:
        violations: list[str] = []
        if candidate_training < policy.min_training_success:
            violations.append("candidate_training_below_minimum")
        if candidate_training < baseline_training - policy.max_regression:
            violations.append("training_regression_exceeds_maximum")
        return violations

    @classmethod
    def _quality_violations(
        cls,
        baseline_training: float,
        candidate_training: float,
        baseline_holdout: float,
        candidate_holdout: float,
        policy: Policy,
    ) -> list[str]:
        violations = cls._training_violations(baseline_training, candidate_training, policy)
        if candidate_holdout < policy.min_holdout_success:
            violations.append("candidate_holdout_below_minimum")
        if candidate_holdout < baseline_holdout - policy.max_regression:
            violations.append("holdout_regression_exceeds_maximum")
        if candidate_holdout - baseline_holdout < policy.min_improvement:
            violations.append("holdout_gain_below_minimum")
        return violations

    @staticmethod
    def _holdout_digest(holdout_cases: list[dict[str, Any]]) -> str:
        pairs = {
            canonical_json({"input": case["input"], "expected": case["expected"]}): {
                "input": case["input"],
                "expected": case["expected"],
            }
            for case in holdout_cases
        }
        return fingerprint([pairs[key] for key in sorted(pairs)])

    @staticmethod
    def _finish_detail(state: dict[str, Any], suite_digest: str | None) -> dict[str, Any]:
        detail: dict[str, Any] = {
            "reason": state["reason"],
            "stage": state["stage"],
            "candidate_id": state["candidate_id"],
            "evaluation_id": state["evaluation_id"],
            "suite_digest": suite_digest,
            "scores": dict(state["scores"]),
            "violations": list(state["violations"]),
            "provider_usage": state["provider_usage"],
            "shared_host_resources": state["shared_host_resources"],
        }
        return detail
