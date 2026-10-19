"""Operator-owned limits and boundary value objects.

Candidate text and model output cannot change these limits. A policy change is
an explicit operator operation and invalidates prior promotion evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
import re
from typing import Any

from .errors import EvolutionError


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ChoiceResult:
    option: str
    probabilities: dict[str, float]
    confidence: float
    duration_seconds: float
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


def validate_choice_result(choice: ChoiceResult, expected_options: set[str]) -> None:
    """Validate a typed judge result before any caller consumes its decision."""
    def invalid(message: str) -> None:
        raise EvolutionError("provider_invalid_judgement", message)

    if not isinstance(choice, ChoiceResult):
        invalid("JEV returned an unsupported typed-choice result")
    if not isinstance(choice.option, str) or choice.option not in expected_options:
        invalid("JEV selected an option that was not supplied")
    probabilities = choice.probabilities
    if not isinstance(probabilities, dict) or set(probabilities) != expected_options:
        invalid("JEV probabilities must cover every supplied option exactly")
    checked: list[float] = []
    for probability in probabilities.values():
        if isinstance(probability, bool) or not isinstance(probability, (float, int)) or not 0 <= probability <= 1 or not math.isfinite(probability):
            invalid("JEV probabilities must be finite and bounded")
        checked.append(float(probability))
    if abs(math.fsum(checked) - 1.0) > 1e-6:
        invalid("JEV probabilities must sum to one")
    if probabilities[choice.option] != max(checked):
        invalid("JEV choice must have the highest probability")
    if isinstance(choice.confidence, bool) or not isinstance(choice.confidence, (float, int)) or not 0 <= choice.confidence <= 1 or not math.isfinite(choice.confidence):
        invalid("JEV confidence must be finite and bounded")
    if isinstance(choice.duration_seconds, bool) or not isinstance(choice.duration_seconds, (float, int)) or not 0 <= choice.duration_seconds <= 3600 or not math.isfinite(choice.duration_seconds):
        invalid("JEV duration must be finite and bounded")
    if choice.model is not None and (
        not isinstance(choice.model, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/+-]{0,255}", choice.model)
    ):
        invalid("JEV model metadata is invalid")
    for field, value in (("input_tokens", choice.input_tokens), ("output_tokens", choice.output_tokens)):
        if value is not None and (type(value) is not int or not 0 <= value <= 10_000_000):
            invalid(f"JEV {field} metadata is invalid")


@dataclass(frozen=True)
class ChoiceAnswer:
    option: str
    probabilities: dict[str, float]
    confidence: float


@dataclass(frozen=True)
class NoulAnswer:
    probability_yes: float


@dataclass(frozen=True)
class BatchJudgment:
    answers: dict[str, ChoiceAnswer | NoulAnswer]
    duration_seconds: float
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


def validate_batch_judgment(
    result: BatchJudgment,
    questions: dict[str, dict[str, Any]],
) -> None:
    """Validate one typed result against every requested Choice or Noul question."""
    def invalid(message: str) -> None:
        raise EvolutionError("provider_invalid_judgement", message)

    def bounded_text(value: Any, field: str) -> None:
        if type(value) is not str or not value.strip():
            invalid(f"{field} must be nonempty text")
        try:
            size = len(value.encode("utf-8", "strict"))
        except UnicodeError:
            invalid(f"{field} must be bounded UTF-8 text")
        if size > 8192:
            invalid(f"{field} must be bounded UTF-8 text")

    if type(result) is not BatchJudgment:
        invalid("JEV returned an unsupported batch judgment")
    if type(questions) is not dict or not 1 <= len(questions) <= 8:
        invalid("Batch questions must contain one to eight entries")
    for question_id, question in questions.items():
        if (
            type(question_id) is not str
            or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", question_id)
            or type(question) is not dict
            or set(question) != {"type", "instructions", "criteria"}
        ):
            invalid("Batch question IDs and shapes must match the supported schema")
        if type(question["type"]) is not str or question["type"] not in {"choice", "noul"}:
            invalid("Batch question type is unsupported")
        bounded_text(question["instructions"], "Question instructions")
        criteria = question["criteria"]
        if type(criteria) is not dict:
            invalid("Question criteria must be an object")
        if question["type"] == "choice":
            if not 1 <= len(criteria) <= 255:
                invalid("Choice criteria must contain one to 255 options")
        elif set(criteria) != {"true", "false"}:
            invalid("Noul criteria must define exactly true and false")
        for key, rubric in criteria.items():
            if type(key) is not str or not key.strip():
                invalid("Question criteria keys must be bounded text")
            try:
                if len(key.encode("utf-8", "strict")) > 256:
                    invalid("Question criteria keys must be bounded text")
            except UnicodeError:
                invalid("Question criteria keys must be bounded UTF-8 text")
            bounded_text(rubric, "Question criteria")

    answers = result.answers
    if type(answers) is not dict or set(answers) != set(questions):
        invalid("JEV answer IDs must exactly match the requested question IDs")
    for question_id, question in questions.items():
        answer = answers[question_id]
        criteria = question["criteria"]
        if question["type"] == "choice":
            if type(answer) is not ChoiceAnswer:
                invalid("Choice questions require ChoiceAnswer results")
            if type(answer.option) is not str or answer.option not in criteria:
                invalid("Choice answer must select one supplied option")
            probabilities = answer.probabilities
            if type(probabilities) is not dict or set(probabilities) != set(criteria):
                invalid("Choice probabilities must cover every supplied option exactly")
            checked: list[float] = []
            for probability in probabilities.values():
                if (
                    isinstance(probability, bool)
                    or not isinstance(probability, (int, float))
                    or not 0 <= probability <= 1
                    or not math.isfinite(probability)
                ):
                    invalid("Choice probabilities must be finite and bounded")
                checked.append(float(probability))
            if abs(math.fsum(checked) - 1.0) > 1e-6:
                invalid("Choice probabilities must sum to one")
            if probabilities[answer.option] != max(checked):
                invalid("Choice answer must have the highest probability")
            if (
                isinstance(answer.confidence, bool)
                or not isinstance(answer.confidence, (int, float))
                or not 0 <= answer.confidence <= 1
                or not math.isfinite(answer.confidence)
            ):
                invalid("Choice confidence must be finite and bounded")
        else:
            if type(answer) is not NoulAnswer:
                invalid("Noul questions require NoulAnswer results")
            probability = answer.probability_yes
            if (
                isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not 0 <= probability <= 1
                or not math.isfinite(probability)
            ):
                invalid("Noul yes-probability must be finite and between zero and one")

    duration = result.duration_seconds
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not 0 <= duration <= 3600 or not math.isfinite(duration):
        invalid("Batch duration must be finite and bounded")
    if result.model is not None and (
        type(result.model) is not str
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/+-]{0,255}", result.model)
    ):
        invalid("Batch model metadata is invalid")
    for field, value in (("input_tokens", result.input_tokens), ("output_tokens", result.output_tokens)):
        if value is not None and (type(value) is not int or not 0 <= value <= 10_000_000):
            invalid(f"Batch {field} metadata is invalid")


@dataclass(frozen=True)
class JudgeTrace:
    option: str
    probabilities: dict[str, float]
    confidence: float
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    duration_seconds: float


@dataclass(frozen=True)
class GeneratorTrace:
    called: bool
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    cost_credits: float | None
    duration_seconds: float | None
    generation_id: str | None


@dataclass(frozen=True)
class RoutingTrace:
    route: str
    selected_candidate_id: str | None
    judge: JudgeTrace
    generator: GeneratorTrace
    request_fingerprint: str
    candidate_pool_fingerprint: str


_TRACE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_TRACE_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/+-]{0,255}\Z")
_TRACE_GENERATION_ID = re.compile(r"[^\x00-\x20\x7f]{1,256}\Z")
_TRACE_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")
_TRACE_OPTIONS = {"generate", "insufficient_evidence"}
_MAX_ROUTING_BYTES = 65_536


def _trace_tokens(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or not 0 <= value <= 10_000_000:
        raise EvolutionError("invalid_routing_trace", f"{field} must be a bounded nonnegative integer or null")
    return value


def _trace_model(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _TRACE_MODEL.fullmatch(value):
        raise EvolutionError("invalid_routing_trace", f"{field} must be bounded model metadata or null")
    return value


def validated_routing(trace: RoutingTrace | dict[str, Any]) -> dict[str, Any]:
    """Return a bounded JSON-safe routing trace, rejecting untrusted metadata.

    This is the single serialization boundary used by both direct CLI proposals
    and persisted engine usage. The trace deliberately contains only routing
    decisions, provider metadata, and one-way fingerprints—never request text.
    """
    if isinstance(trace, RoutingTrace):
        value = asdict(trace)
    elif isinstance(trace, dict):
        value = trace
    else:
        raise EvolutionError("invalid_routing_trace", "Routing trace must be a routing record")

    expected = {
        "route", "selected_candidate_id", "judge", "generator",
        "request_fingerprint", "candidate_pool_fingerprint",
    }
    if set(value) != expected:
        raise EvolutionError("invalid_routing_trace", "Routing trace has an unsupported shape")
    if not isinstance(value["route"], str) or value["route"] not in {"reuse", "generate"}:
        raise EvolutionError("invalid_routing_trace", "Routing trace route is unsupported")
    candidate_id = value["selected_candidate_id"]
    if candidate_id is not None and (not isinstance(candidate_id, str) or not _TRACE_ID.fullmatch(candidate_id)):
        raise EvolutionError("invalid_routing_trace", "Selected candidate ID is invalid")
    if (value["route"] == "reuse") != (candidate_id is not None):
        raise EvolutionError("invalid_routing_trace", "Candidate selection does not match the route")

    judge = value["judge"]
    if not isinstance(judge, dict) or set(judge) != {
        "option", "probabilities", "confidence", "model", "input_tokens", "output_tokens", "duration_seconds"
    }:
        raise EvolutionError("invalid_routing_trace", "Judge trace has an unsupported shape")
    option = judge["option"]
    if not isinstance(option, str) or not (option in _TRACE_OPTIONS or re.fullmatch(r"reuse_[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", option)):
        raise EvolutionError("invalid_routing_trace", "Judge option is invalid")
    expected_option = f"reuse_{candidate_id}" if value["route"] == "reuse" else "generate"
    if option != expected_option:
        raise EvolutionError("invalid_routing_trace", "Judge option does not match the recorded route")
    probabilities = judge["probabilities"]
    if not isinstance(probabilities, dict) or not probabilities or len(probabilities) > 18:
        raise EvolutionError("invalid_routing_trace", "Judge probabilities must be a bounded object")
    checked_probabilities: dict[str, float] = {}
    for key, probability in probabilities.items():
        if not isinstance(key, str) or not (key in _TRACE_OPTIONS or re.fullmatch(r"reuse_[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", key)):
            raise EvolutionError("invalid_routing_trace", "Judge probability option is invalid")
        if isinstance(probability, bool) or not isinstance(probability, (float, int)) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise EvolutionError("invalid_routing_trace", "Judge probabilities must be finite values between zero and one")
        checked_probabilities[key] = float(probability)
    if option not in checked_probabilities or abs(sum(checked_probabilities.values()) - 1.0) > 1e-6:
        raise EvolutionError("invalid_routing_trace", "Judge probabilities must cover the selection and sum to one")
    max_probability = max(checked_probabilities.values())
    if checked_probabilities[option] != max_probability:
        raise EvolutionError("invalid_routing_trace", "Judge option must have the highest probability")
    confidence = judge["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (float, int)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise EvolutionError("invalid_routing_trace", "Judge confidence must be finite and between zero and one")
    judge_duration = judge["duration_seconds"]
    if isinstance(judge_duration, bool) or not isinstance(judge_duration, (float, int)) or not math.isfinite(judge_duration) or not 0 <= judge_duration <= 3600:
        raise EvolutionError("invalid_routing_trace", "Judge duration must be finite and bounded")
    checked_judge = {
        "option": option,
        "probabilities": checked_probabilities,
        "confidence": float(confidence),
        "model": _trace_model(judge["model"], "judge.model"),
        "input_tokens": _trace_tokens(judge["input_tokens"], "judge.input_tokens"),
        "output_tokens": _trace_tokens(judge["output_tokens"], "judge.output_tokens"),
        "duration_seconds": float(judge_duration),
    }

    generator = value["generator"]
    if not isinstance(generator, dict) or set(generator) != {
        "called", "model", "input_tokens", "output_tokens", "cost_credits", "duration_seconds", "generation_id"
    } or type(generator["called"]) is not bool:
        raise EvolutionError("invalid_routing_trace", "Generator trace has an unsupported shape")
    if generator["called"] != (value["route"] == "generate"):
        raise EvolutionError("invalid_routing_trace", "Generator usage does not match the route")
    cost = generator["cost_credits"]
    if cost is not None and (isinstance(cost, bool) or not isinstance(cost, (float, int)) or not math.isfinite(cost) or not 0 <= cost <= 1_000_000_000):
        raise EvolutionError("invalid_routing_trace", "Generator cost must be finite, bounded, and nonnegative")
    generator_duration = generator["duration_seconds"]
    if generator_duration is not None and (isinstance(generator_duration, bool) or not isinstance(generator_duration, (float, int)) or not math.isfinite(generator_duration) or not 0 <= generator_duration <= 3600):
        raise EvolutionError("invalid_routing_trace", "Generator duration must be finite and bounded")
    generation_id = generator["generation_id"]
    if generation_id is not None:
        if not isinstance(generation_id, str) or not _TRACE_GENERATION_ID.fullmatch(generation_id):
            raise EvolutionError("invalid_routing_trace", "Generator ID metadata is invalid")
        try:
            generation_id_bytes = generation_id.encode("utf-8", "strict")
        except UnicodeError as exc:
            raise EvolutionError("invalid_routing_trace", "Generator ID metadata is invalid") from exc
        if len(generation_id_bytes) > 256:
            raise EvolutionError("invalid_routing_trace", "Generator ID metadata exceeds the size limit")
    checked_generator = {
        "called": generator["called"],
        "model": _trace_model(generator["model"], "generator.model"),
        "input_tokens": _trace_tokens(generator["input_tokens"], "generator.input_tokens"),
        "output_tokens": _trace_tokens(generator["output_tokens"], "generator.output_tokens"),
        "cost_credits": float(cost) if cost is not None else None,
        "duration_seconds": float(generator_duration) if generator_duration is not None else None,
        "generation_id": generation_id,
    }
    if not generator["called"] and any(
        generator[field] is not None
        for field in ("model", "input_tokens", "output_tokens", "cost_credits", "duration_seconds", "generation_id")
    ):
        raise EvolutionError("invalid_routing_trace", "Uncalled generator usage must remain null")

    request_fingerprint = value["request_fingerprint"]
    pool_fingerprint = value["candidate_pool_fingerprint"]
    if not isinstance(request_fingerprint, str) or not _TRACE_FINGERPRINT.fullmatch(request_fingerprint):
        raise EvolutionError("invalid_routing_trace", "Request fingerprint is invalid")
    if not isinstance(pool_fingerprint, str) or not _TRACE_FINGERPRINT.fullmatch(pool_fingerprint):
        raise EvolutionError("invalid_routing_trace", "Candidate pool fingerprint is invalid")

    result = {
        "route": value["route"],
        "selected_candidate_id": candidate_id,
        "judge": checked_judge,
        "generator": checked_generator,
        "request_fingerprint": request_fingerprint,
        "candidate_pool_fingerprint": pool_fingerprint,
    }
    try:
        encoded = canonical_json(result).encode("utf-8", "strict")
    except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError) as exc:
        raise EvolutionError("invalid_routing_trace", "Routing trace must be finite JSON") from exc
    if len(encoded) > _MAX_ROUTING_BYTES:
        raise EvolutionError("invalid_routing_trace", "Routing trace exceeds the size limit")
    clean = json.loads(encoded.decode("utf-8"))
    # Import only at the serialization boundary to avoid making contracts
    # depend on provider implementations during module initialization.
    from .provider_security import contains_sensitive_material

    if contains_sensitive_material(clean):
        raise EvolutionError("invalid_routing_trace", "Routing metadata contains credential-like material")
    return clean


@dataclass(frozen=True)
class Policy:
    max_parallel_experiments: int = 1
    max_experiments_per_day: int = 12
    max_reserved_seconds_per_day: int = 1800
    experiment_timeout_seconds: int = 120
    case_timeout_seconds: int = 10
    max_output_bytes: int = 262144
    max_snapshot_files: int = 5000
    max_snapshot_bytes: int = 67108864
    max_snapshots: int = 64
    max_store_bytes: int = 1073741824
    max_file_bytes: int = 2097152
    max_receipts: int = 10000
    max_receipt_bytes: int = 16384
    max_cases: int = 100
    max_changes: int = 8
    min_training_success: float = 1.0
    min_holdout_success: float = 1.0
    min_improvement: float = 0.01
    max_regression: float = 0.0
    yield_to_foreground: bool = True
    allowed_mutation_paths: tuple[str, ...] = (
        "skills/*", "tools/*", "workflows/*", "memory/*", "optimizer/*"
    )

    def __post_init__(self) -> None:
        int_fields = [f.name for f in fields(self) if f.name.startswith("max_") and f.name != "max_regression"]
        int_fields += ["experiment_timeout_seconds", "case_timeout_seconds"]
        for name in int_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= 10_000_000_000:
                raise EvolutionError("invalid_policy", f"{name} must be a positive bounded integer")
        for name in ("min_training_success", "min_holdout_success", "min_improvement", "max_regression"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise EvolutionError("invalid_policy", f"{name} must be finite and between zero and one")
        if not isinstance(self.yield_to_foreground, bool):
            raise EvolutionError("invalid_policy", "yield_to_foreground must be boolean")
        if self.case_timeout_seconds > self.experiment_timeout_seconds:
            raise EvolutionError("invalid_policy", "A case timeout cannot exceed the experiment deadline")
        if self.max_file_bytes > self.max_snapshot_bytes:
            raise EvolutionError("invalid_policy", "A file limit cannot exceed the snapshot limit")
        if self.max_snapshot_bytes > self.max_store_bytes:
            raise EvolutionError("invalid_policy", "A snapshot limit cannot exceed the store limit")
        patterns = self.allowed_mutation_paths
        if not isinstance(patterns, tuple) or not patterns:
            raise EvolutionError("invalid_policy", "Mutation paths must be a nonempty list")
        for pattern in patterns:
            if not isinstance(pattern, str) or not pattern or pattern.startswith(("/", ".")) or ".." in pattern.split("/") or "\\" in pattern:
                raise EvolutionError("invalid_policy", "Mutation patterns must be safe relative paths")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Policy":
        if not isinstance(data, dict):
            raise EvolutionError("invalid_policy", "Policy must be an object")
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise EvolutionError("invalid_policy", f"Unknown policy fields: {', '.join(sorted(unknown))}")
        values = dict(data)
        if "allowed_mutation_paths" in values:
            raw = values["allowed_mutation_paths"]
            if not isinstance(raw, (list, tuple)):
                raise EvolutionError("invalid_policy", "Mutation paths must be a list")
            values["allowed_mutation_paths"] = tuple(raw)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def digest(self) -> str:
        return fingerprint(self.to_dict())


@dataclass(frozen=True)
class ProcessResult:
    status: str
    returncode: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    reason: str | None = None


@dataclass(frozen=True)
class ProposalResult:
    value: dict[str, Any]
    duration_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    model: str | None = None
    generation_id: str | None = None
    cost_credits: float | None = None
    routing: RoutingTrace | None = None
