"""JEV-routed reuse of reviewed candidates with optional proposal generation."""

from __future__ import annotations

import json
import math
import os
import re
import time
from typing import Any

from .contracts import (
    ChoiceResult,
    GeneratorTrace,
    JudgeTrace,
    ProposalResult,
    RoutingTrace,
    canonical_json,
    fingerprint,
    validated_routing,
)
from .errors import EvolutionError
from .providers import (
    _validate_call_limits,
    _validate_proposal,
    _validate_request,
)


_MAX_CANDIDATES = 16
_MAX_HYBRID_STATE_BYTES = 1_048_576
_MAX_JUDGE_OUTPUT_BYTES = 65_536
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SENSITIVE_ENV_KEYS = ("TYPESAFE_API_KEY", "OPENROUTER_API_KEY")


def _raise(code: str, message: str) -> None:
    raise EvolutionError(code, message)


class HybridProposalProvider:
    """Route among supplied baseline-bound proposals before optional generation.

    Reusing a candidate returns its existing, validated patch. Generating new
    proposal text is possible only after a confident JEV choice of ``generate``
    and only when an explicit generator has been configured.
    """

    # The generator can return executable proposal content, so this wrapper
    # must never weaken the engine's hard-resource admission gate.
    requires_hard_resource_limits = True

    def __init__(
        self,
        judge: Any,
        generator: Any | None = None,
        *,
        candidates: list[dict[str, Any]] | None = None,
        min_confidence: float = 0.8,
        min_probability: float = 0.8,
    ) -> None:
        if judge is None or not callable(getattr(judge, "choose", None)):
            _raise("invalid_provider_configuration", "Hybrid provider requires a typed-choice judge")
        if generator is not None and not callable(getattr(generator, "propose", None)):
            _raise("invalid_provider_configuration", "Hybrid generator must implement propose()")
        for name, value in (("min_confidence", min_confidence), ("min_probability", min_probability)):
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
                _raise("invalid_provider_limits", f"{name} must be finite and between zero and one")
        if candidates is None:
            candidate_list: list[Any] = []
        elif type(candidates) is list:
            candidate_list = candidates
        else:
            _raise("invalid_candidate_pool", "Candidate pool must be a JSON list")
        if len(candidate_list) > _MAX_CANDIDATES:
            _raise("invalid_candidate_pool", "Candidate pool exceeds the maximum of 16 entries")
        try:
            pool_json = canonical_json(candidate_list)
            pool_bytes = pool_json.encode("utf-8", "strict")
        except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError) as exc:
            raise EvolutionError("invalid_candidate_pool", "Candidate pool must contain bounded finite JSON") from exc
        if len(pool_bytes) > _MAX_HYBRID_STATE_BYTES:
            _raise("invalid_candidate_pool", "Candidate pool exceeds the bounded JSON size")
        self.judge = judge
        self.generator = generator
        # Snapshot caller-owned data at construction so later mutation cannot
        # change which alternatives the remote judge sees.
        self._candidate_pool_json = pool_json
        self.min_confidence = float(min_confidence)
        self.min_probability = float(min_probability)

    @staticmethod
    def _sensitive(value: Any) -> bool:
        from .provider_security import contains_sensitive_material

        if contains_sensitive_material(value):
            return True
        # Detect literal reuse of either configured credential before any
        # provider receives the request, including when the other provider is
        # the one being called first.
        for name in _SENSITIVE_ENV_KEYS:
            key = os.environ.get(name)
            if key and contains_sensitive_material(value, configured_api_key=key):
                return True
        return False

    def _validated_candidates(
        self,
        request: dict[str, Any],
        *,
        files: dict[str, str],
        patterns: tuple[str, ...],
        max_changes: int,
        max_output_bytes: int,
    ) -> list[dict[str, Any]]:
        try:
            raw_candidates = json.loads(self._candidate_pool_json)
        except (TypeError, ValueError, RecursionError) as exc:
            raise EvolutionError("invalid_candidate_pool", "Candidate pool snapshot could not be read") from exc
        seen: set[str] = set()
        validated: list[dict[str, Any]] = []
        for candidate in raw_candidates:
            if type(candidate) is not dict or set(candidate) != {"id", "baseline_id", "proposal"}:
                _raise("invalid_candidate_pool", "Each candidate must contain exactly id, baseline_id, and proposal")
            candidate_id = candidate["id"]
            if not isinstance(candidate_id, str) or not _SAFE_ID.fullmatch(candidate_id):
                _raise("invalid_candidate_pool", "Candidate IDs must be safe bounded identifiers")
            if candidate_id in seen:
                _raise("invalid_candidate_pool", "Candidate IDs must be unique")
            seen.add(candidate_id)
            if candidate["baseline_id"] != request["baseline_id"]:
                _raise("invalid_candidate_pool", "Every candidate must target the request's exact baseline")
            proposal = _validate_proposal(
                candidate["proposal"],
                files=files,
                patterns=patterns,
                max_changes=max_changes,
                max_output_bytes=max_output_bytes,
            )
            checked = {"id": candidate_id, "baseline_id": candidate["baseline_id"], "proposal": proposal}
            if self._sensitive(checked):
                _raise("provider_sensitive_input", "Credential-like key material is not valid candidate input")
            validated.append(checked)
        return validated

    @staticmethod
    def _validate_choice(choice: ChoiceResult, expected_options: set[str]) -> None:
        if not isinstance(choice, ChoiceResult):
            _raise("provider_invalid_judgement", "JEV returned an unsupported typed-choice result")
        if choice.option not in expected_options:
            _raise("provider_invalid_judgement", "JEV selected an option that was not supplied")
        probabilities = choice.probabilities
        if not isinstance(probabilities, dict) or set(probabilities) != expected_options:
            _raise("provider_invalid_judgement", "JEV probabilities must cover every supplied option exactly")
        checked: list[float] = []
        for probability in probabilities.values():
            if isinstance(probability, bool) or not isinstance(probability, (float, int)) or not math.isfinite(probability) or not 0 <= probability <= 1:
                _raise("provider_invalid_judgement", "JEV probabilities must be finite and bounded")
            checked.append(float(probability))
        if abs(sum(checked) - 1.0) > 1e-6:
            _raise("provider_invalid_judgement", "JEV probabilities must sum to one")
        maximum = max(checked)
        if probabilities[choice.option] != maximum:
            _raise("provider_invalid_judgement", "JEV choice must have the highest probability")
        if isinstance(choice.confidence, bool) or not isinstance(choice.confidence, (float, int)) or not math.isfinite(choice.confidence) or not 0 <= choice.confidence <= 1:
            _raise("provider_invalid_judgement", "JEV confidence must be finite and bounded")
        if isinstance(choice.duration_seconds, bool) or not isinstance(choice.duration_seconds, (float, int)) or not math.isfinite(choice.duration_seconds) or not 0 <= choice.duration_seconds <= 3600:
            _raise("provider_invalid_judgement", "JEV duration must be finite and bounded")
        if choice.model is not None and (
            not isinstance(choice.model, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/+-]{0,255}", choice.model)
        ):
            _raise("provider_invalid_judgement", "JEV model metadata is invalid")
        for field, value in (("input_tokens", choice.input_tokens), ("output_tokens", choice.output_tokens)):
            if value is not None and (type(value) is not int or not 0 <= value <= 10_000_000):
                _raise("provider_invalid_judgement", f"JEV {field} metadata is invalid")

    @staticmethod
    def _judge_trace(choice: ChoiceResult) -> JudgeTrace:
        return JudgeTrace(
            option=choice.option,
            probabilities=dict(choice.probabilities),
            confidence=choice.confidence,
            model=choice.model,
            input_tokens=choice.input_tokens,
            output_tokens=choice.output_tokens,
            duration_seconds=choice.duration_seconds,
        )

    def propose(
        self,
        request: dict[str, Any],
        *,
        timeout_seconds: float,
        max_output_bytes: int,
    ) -> ProposalResult:
        started = time.monotonic()
        try:
            timeout, output_limit = _validate_call_limits(timeout_seconds, max_output_bytes)
        except OverflowError as exc:
            raise EvolutionError("invalid_provider_limits", "Provider timeout must be positive and bounded") from exc
        deadline = started + timeout

        normalized_request, files, patterns, max_changes = _validate_request(request)
        if self._sensitive(normalized_request):
            _raise("provider_sensitive_input", "Credential-like key material is not valid provider input")
        candidates = self._validated_candidates(
            normalized_request,
            files=files,
            patterns=patterns,
            max_changes=max_changes,
            max_output_bytes=output_limit,
        )
        # Freeze the exact validated payload used to judge and fingerprint. The
        # judge and generator receive independent JSON copies, so mutations at
        # either provider boundary cannot alter the other call's input.
        try:
            normalized_request_json = canonical_json(normalized_request)
            state_json = canonical_json({"request": normalized_request, "candidates": candidates})
            state_bytes = state_json.encode("utf-8", "strict")
        except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError) as exc:
            raise EvolutionError("invalid_provider_request", "Hybrid state must be finite JSON") from exc
        if len(state_bytes) > _MAX_HYBRID_STATE_BYTES:
            _raise("provider_request_too_large", "Hybrid request and candidates exceed the bounded judge input")
        request_fingerprint = fingerprint(normalized_request)
        candidate_pool_fingerprint = fingerprint(candidates)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _raise("provider_timeout", "Hybrid provider exceeded the shared deadline before judgment")
        options = [*(f"reuse_{candidate['id']}" for candidate in candidates), "generate", "insufficient_evidence"]
        criteria = {
            **{
                f"reuse_{candidate['id']}": (
                    "Reuse this exact supplied proposal only if the training examples show it addresses "
                    "the observed failures while preserving the passing cases. Do not alter it."
                )
                for candidate in candidates
            },
            "generate": "New proposal content is needed because no supplied candidate is a sufficiently supported fit.",
            "insufficient_evidence": "The request and training evidence do not support either reuse or new proposal generation.",
        }
        choice = self.judge.choose(
            json.loads(state_json),
            instructions=(
                "Choose exactly one route using only the supplied request, its training examples, and all supplied candidates. "
                "A reuse option means the corresponding proposal must be returned exactly as supplied. Select generate "
                "only when new proposal content is justified by the training evidence. Select insufficient_evidence "
                "when evidence is inadequate. Do not invent candidates, inspect holdout data, or execute changes."
            ),
            criteria=criteria,
            timeout_seconds=remaining,
            max_output_bytes=min(output_limit, _MAX_JUDGE_OUTPUT_BYTES),
        )
        if time.monotonic() >= deadline:
            _raise("provider_timeout", "JEV judgment exceeded the shared provider deadline")
        self._validate_choice(choice, set(options))
        if choice.option == "insufficient_evidence":
            _raise("provider_blocked_uncertain", "JEV found insufficient evidence to select a proposal route")
        selected_probability = choice.probabilities[choice.option]
        if sum(probability == selected_probability for probability in choice.probabilities.values()) != 1:
            _raise("provider_blocked_uncertain", "JEV route is tied with another option")
        if choice.confidence < self.min_confidence or selected_probability < self.min_probability:
            _raise("provider_blocked_uncertain", "JEV route did not meet the configured confidence thresholds")

        candidate_id: str | None = None
        generator_trace = GeneratorTrace(
            called=False,
            model=None,
            input_tokens=None,
            output_tokens=None,
            cost_credits=None,
            duration_seconds=None,
            generation_id=None,
        )
        if choice.option.startswith("reuse_"):
            candidate_id = choice.option[len("reuse_"):]
            selected = next((item for item in candidates if item["id"] == candidate_id), None)
            if selected is None:
                _raise("provider_invalid_judgement", "JEV selected an unknown candidate")
            trace = RoutingTrace(
                route="reuse",
                selected_candidate_id=candidate_id,
                judge=self._judge_trace(choice),
                generator=generator_trace,
                request_fingerprint=request_fingerprint,
                candidate_pool_fingerprint=candidate_pool_fingerprint,
            )
            validated_routing(trace)
            result = ProposalResult(
                value=selected["proposal"],
                duration_seconds=time.monotonic() - started,
                routing=trace,
            )
            return result

        if choice.option != "generate":
            _raise("provider_invalid_judgement", "JEV selected an unsupported route")
        if self.generator is None:
            _raise("provider_blocked_generation_unavailable", "JEV selected generation but no generator is configured")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _raise("provider_timeout", "Hybrid provider exceeded the shared deadline before generation")
        # Validate every judge metadata field before a potentially paid
        # generator call. For a generation route, null generator usage is a
        # valid provisional state; the returned metadata is checked again once
        # generation completes.
        provisional_trace = RoutingTrace(
            route="generate",
            selected_candidate_id=None,
            judge=self._judge_trace(choice),
            generator=GeneratorTrace(
                called=True,
                model=None,
                input_tokens=None,
                output_tokens=None,
                cost_credits=None,
                duration_seconds=None,
                generation_id=None,
            ),
            request_fingerprint=request_fingerprint,
            candidate_pool_fingerprint=candidate_pool_fingerprint,
        )
        validated_routing(provisional_trace)
        # Supply the exact normalized values judged above, but as an independent
        # JSON object rather than the caller's mutable request instance.
        generated = self.generator.propose(
            json.loads(normalized_request_json),
            timeout_seconds=remaining,
            max_output_bytes=output_limit,
        )
        if time.monotonic() >= deadline:
            _raise("provider_timeout", "Proposal generation exceeded the shared provider deadline")
        if not isinstance(generated, ProposalResult) or generated.routing is not None:
            _raise("provider_invalid_response", "Generator returned an unsupported proposal result")
        proposal = _validate_proposal(
            generated.value,
            files=files,
            patterns=patterns,
            max_changes=max_changes,
            max_output_bytes=output_limit,
        )
        proposal = json.loads(canonical_json(proposal))
        if self._sensitive(proposal):
            _raise("provider_sensitive_output", "Generated proposal contained credential-like material")
        generator_trace = GeneratorTrace(
            called=True,
            model=generated.model,
            input_tokens=generated.input_tokens,
            output_tokens=generated.output_tokens,
            cost_credits=generated.cost_credits,
            duration_seconds=generated.duration_seconds,
            generation_id=generated.generation_id,
        )
        trace = RoutingTrace(
            route="generate",
            selected_candidate_id=None,
            judge=self._judge_trace(choice),
            generator=generator_trace,
            request_fingerprint=request_fingerprint,
            candidate_pool_fingerprint=candidate_pool_fingerprint,
        )
        validated_routing(trace)
        return ProposalResult(
            value=proposal,
            duration_seconds=time.monotonic() - started,
            input_tokens=generated.input_tokens,
            output_tokens=generated.output_tokens,
            model=generated.model,
            generation_id=generated.generation_id,
            cost_credits=generated.cost_credits,
            routing=trace,
        )
