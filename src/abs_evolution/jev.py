"""Bounded TypeSafe Choice adapter for narrow evidence-based routing."""

from __future__ import annotations

import math
import os
from pathlib import Path
import re
import time
from typing import Any

from ._jev_http import (
    MAX_API_KEY_BYTES,
    MAX_BODY_BYTES,
    MAX_RESPONSE_BYTES,
    MAX_WORKER_INPUT_BYTES,
)
from ._provider_transport import ProviderTransportError, run_bounded_worker
from .contracts import (
    BatchJudgment,
    ChoiceAnswer,
    ChoiceResult,
    NoulAnswer,
    canonical_json,
    validate_batch_judgment,
)
from .errors import EvolutionError
from .provider_security import contains_sensitive_material
from .providers import _strict_json_loads, _validate_call_limits, _validate_json_value


DEFAULT_MODEL = "jev-1.13.0"
_MODEL_ID = re.compile(r"jev-[0-9]+\.[0-9]+\.[0-9]+\Z")
_API_KEY = re.compile(r"[A-Za-z0-9._~-]{1,4096}\Z")
_MAX_OBSERVED_TOKENS = 10_000_000
_RESPONSE_METADATA_OVERHEAD = 64 * 1024
_MAX_BATCH_QUESTIONS = 8
_MAX_QUESTION_TEXT_BYTES = 8192
_QUESTION_ID = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}\Z")


def _raise(code: str, message: str) -> None:
    raise EvolutionError(code, message)


def _read_api_key() -> str:
    api_key = os.environ.get("TYPESAFE_API_KEY")
    if api_key is None or not api_key:
        _raise("provider_credentials_missing", "TYPESAFE_API_KEY is not set in the process environment")
    try:
        encoded = api_key.encode("ascii", "strict")
    except UnicodeError:
        _raise("provider_credentials_invalid", "TypeSafe credential is not a valid token")
    if len(encoded) > MAX_API_KEY_BYTES or not _API_KEY.fullmatch(api_key):
        _raise("provider_credentials_invalid", "TypeSafe credential is not a valid token")
    return api_key


def _validate_bounded_text(value: Any, *, field: str, max_bytes: int) -> None:
    if type(value) is not str or not value.strip():
        _raise("invalid_provider_request", f"{field} must be nonempty text")
    try:
        size = len(value.encode("utf-8", "strict"))
    except UnicodeError:
        _raise("invalid_provider_request", f"{field} must be valid UTF-8 text")
    if size > max_bytes:
        _raise("invalid_provider_request", f"{field} exceeds the bounded size")


def _validate_inputs(
    state: Any,
    *,
    instructions: Any,
    criteria: Any,
) -> tuple[dict[str, Any], str, dict[str, str]]:
    if type(state) is not dict:
        _raise("invalid_provider_request", "JEV state must be a JSON object")
    try:
        _validate_json_value(state, error_code="invalid_provider_request")
    except (RecursionError, OverflowError):
        _raise("invalid_provider_request", "JEV state must be finite bounded JSON")
    if type(instructions) is not str or not instructions.strip():
        _raise("invalid_provider_request", "JEV instructions must be nonempty text")
    if type(criteria) is not dict or not 1 <= len(criteria) <= 255:
        _raise("invalid_provider_request", "JEV Choice criteria must contain one to 255 options")
    if any(
        type(option) is not str
        or not option.strip()
        or type(description) is not str
        for option, description in criteria.items()
    ):
        _raise("invalid_provider_request", "JEV Choice options and descriptions must be bounded text")
    try:
        _validate_json_value(instructions, error_code="invalid_provider_request")
        _validate_json_value(criteria, error_code="invalid_provider_request")
        # Isolate the state and rubrics from mutation by a caller while the
        # subprocess request is being built.
        normalized_state = _strict_json_loads(canonical_json(state))
        normalized_criteria = _strict_json_loads(canonical_json(criteria))
    except EvolutionError:
        raise
    except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError):
        _raise("invalid_provider_request", "JEV request must be finite bounded JSON")
    return normalized_state, instructions, normalized_criteria


def _build_request_body(
    model: str,
    state: dict[str, Any],
    *,
    instructions: str,
    criteria: dict[str, str],
) -> tuple[str, bytes]:
    request_value = {
        "model": model,
        "state": state,
        "questions": {
            "decision": {
                "type": "choice",
                "instructions": instructions,
                "criteria": criteria,
            }
        },
    }
    try:
        request_text = canonical_json(request_value)
        encoded = request_text.encode("utf-8", "strict")
    except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError):
        _raise("invalid_provider_request", "JEV request could not be encoded safely")
    if len(encoded) > MAX_BODY_BYTES:
        _raise("provider_request_too_large", "JEV request exceeds the bounded input size")
    return request_text, encoded


def _execute_worker(
    input_bytes: bytearray,
    *,
    deadline: float,
    response_limit: int,
) -> bytes:
    try:
        worker_path = Path(__file__).with_name("_jev_http.py").resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        _raise("provider_worker_unavailable", "TypeSafe HTTP worker is unavailable")
    try:
        return run_bounded_worker(
            input_bytes,
            worker_path=worker_path,
            deadline=deadline,
            stdout_limit=response_limit + 3,
        )
    except ProviderTransportError as exc:
        if exc.reason == "timeout":
            _raise("provider_timeout", "TypeSafe request exceeded the provider deadline")
        if exc.reason == "output_limit":
            _raise("provider_output_limit", "TypeSafe response exceeded the bounded worker output")
        if exc.reason == "unavailable":
            _raise("provider_worker_unavailable", "TypeSafe HTTP worker could not be started")
        _raise("provider_worker_error", "TypeSafe HTTP worker failed safely")


def _worker_error(wire: bytes) -> None:
    try:
        lines = wire.decode("ascii", "strict").split("\n")
        if len(lines) != 4 or lines[0] != "ERR" or not lines[1].isascii():
            _raise("provider_worker_error", "TypeSafe HTTP worker returned an invalid status")
        error_code = lines[1]
        status = int(lines[2]) if lines[2] else None
        if status is not None and not 100 <= status <= 599:
            status = None
    except (UnicodeError, ValueError):
        _raise("provider_worker_error", "TypeSafe HTTP worker returned an invalid status")

    if error_code == "http_status":
        if status == 401:
            _raise("provider_auth_failed", "TypeSafe rejected the configured credential")
        if status == 403:
            _raise("provider_permission_denied", "TypeSafe denied permission for the configured account")
        if status == 400:
            _raise("provider_request_rejected", "TypeSafe rejected the typed-choice request")
        if status == 408:
            _raise("provider_timeout", "TypeSafe request exceeded the provider deadline")
        if status == 413:
            _raise("provider_request_too_large", "TypeSafe rejected the request as too large")
        if status == 415:
            _raise("provider_unsupported_media_type", "TypeSafe rejected the request media type")
        if status == 422:
            _raise("provider_question_invalid", "TypeSafe rejected the Choice question")
        if status == 429:
            _raise("provider_rate_limited", "TypeSafe rate limit was reached")
        if status is not None and 300 <= status < 400:
            _raise("provider_redirect_rejected", "TypeSafe redirected the fixed API request")
        if status == 529 or (status is not None and status >= 500):
            _raise("provider_unavailable", "TypeSafe is temporarily unavailable")
        _raise("provider_http_error", "TypeSafe returned a non-success HTTP status")
    if error_code == "network_timeout":
        _raise("provider_timeout", "TypeSafe request exceeded the provider deadline")
    if error_code == "network_error":
        _raise("provider_unavailable", "TypeSafe could not be reached")
    if error_code == "response_too_large":
        _raise("provider_output_limit", "TypeSafe response exceeded the bounded response limit")
    if error_code in {"invalid_content_type", "invalid_http_response"}:
        _raise("provider_invalid_response", "TypeSafe returned an invalid response envelope")
    _raise("provider_worker_error", "TypeSafe HTTP worker failed safely")


def _observed_tokens(usage: Any, field: str) -> int | None:
    if not isinstance(usage, dict):
        return None
    value = usage.get(field)
    if type(value) is int and 0 <= value <= _MAX_OBSERVED_TOKENS:
        return value
    return None


def _observed_model(value: Any, requested_model: str) -> str | None:
    if value is None:
        return None
    if type(value) is str and _MODEL_ID.fullmatch(value) and value == requested_model:
        return value
    _raise("provider_invalid_response", "TypeSafe returned unexpected model metadata")


def _parse_response(
    body: bytes,
    *,
    configured_api_key: str,
    criteria: dict[str, str],
    requested_model: str,
) -> tuple[str, dict[str, float], float, str | None, int | None, int | None]:
    try:
        text = body.decode("utf-8", "strict")
    except UnicodeError:
        _raise("provider_invalid_response", "TypeSafe returned invalid UTF-8 JSON")
    if contains_sensitive_material(text, configured_api_key):
        _raise("provider_sensitive_output", "TypeSafe response contained credential-like material")
    try:
        response = _strict_json_loads(text)
    except (ValueError, RecursionError):
        _raise("provider_invalid_response", "TypeSafe returned invalid JSON")
    if contains_sensitive_material(response, configured_api_key):
        _raise("provider_sensitive_output", "TypeSafe response contained credential-like material")
    try:
        _validate_json_value(response, error_code="provider_invalid_response")
    except (RecursionError, OverflowError):
        _raise("provider_invalid_response", "TypeSafe response exceeds the structural limit")
    if type(response) is not dict:
        _raise("provider_invalid_response", "TypeSafe returned an invalid response object")

    answers = response.get("answers")
    if type(answers) is not dict or set(answers) != {"decision"}:
        _raise("provider_invalid_response", "TypeSafe answers did not match the requested Choice")
    answer = answers["decision"]
    if type(answer) is not dict or answer.get("type") != "choice":
        _raise("provider_invalid_response", "TypeSafe returned an invalid Choice answer")

    choice = answer.get("choice")
    probabilities = answer.get("probabilities")
    confidence = answer.get("confidence")
    if type(choice) is not str or choice not in criteria:
        _raise("provider_invalid_response", "TypeSafe selected an option that was not supplied")
    if type(probabilities) is not dict or set(probabilities) != set(criteria):
        _raise("provider_invalid_response", "TypeSafe probabilities must cover every supplied option exactly")

    checked_probabilities: dict[str, float] = {}
    for option, probability in probabilities.items():
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            _raise("provider_invalid_response", "TypeSafe probabilities must be finite and bounded")
        checked_probabilities[option] = float(probability)
    if abs(math.fsum(checked_probabilities.values()) - 1.0) > 1e-6:
        _raise("provider_invalid_response", "TypeSafe probabilities must sum to one")

    highest = max(checked_probabilities.values())
    if checked_probabilities[choice] != highest:
        _raise("provider_invalid_response", "TypeSafe choice must be a highest-probability option")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        _raise("provider_invalid_response", "TypeSafe confidence must be finite and bounded")

    model = _observed_model(response.get("model"), requested_model)
    usage = response.get("usage")
    return (
        choice,
        checked_probabilities,
        float(confidence),
        model,
        _observed_tokens(usage, "input_tokens"),
        _observed_tokens(usage, "output_tokens"),
    )


def _validate_batch_inputs(
    state: Any,
    questions: Any,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if type(state) is not dict:
        _raise("invalid_provider_request", "JEV state must be a JSON object")
    try:
        _validate_json_value(state, error_code="invalid_provider_request")
    except (RecursionError, OverflowError):
        _raise("invalid_provider_request", "JEV state must be finite bounded JSON")
    if type(questions) is not dict or not 1 <= len(questions) <= _MAX_BATCH_QUESTIONS:
        _raise("invalid_provider_request", "JEV batch must contain one to eight questions")
    for question_id, question in questions.items():
        if type(question_id) is not str or not _QUESTION_ID.fullmatch(question_id):
            _raise("invalid_provider_request", "JEV question IDs must be safe bounded identifiers")
        if type(question) is not dict or set(question) != {"type", "instructions", "criteria"}:
            _raise("invalid_provider_request", "Each JEV question must contain exactly type, instructions, and criteria")
        question_type = question["type"]
        if type(question_type) is not str or question_type not in {"choice", "noul"}:
            _raise("invalid_provider_request", "JEV batch question type must be Choice or Noul")
        instructions = question["instructions"]
        _validate_bounded_text(
            instructions,
            field="JEV question instructions",
            max_bytes=_MAX_QUESTION_TEXT_BYTES,
        )
        criteria = question["criteria"]
        if type(criteria) is not dict:
            _raise("invalid_provider_request", "JEV question criteria must be an object")
        if question_type == "choice":
            if not 1 <= len(criteria) <= 255:
                _raise("invalid_provider_request", "Choice criteria must contain one to 255 options")
        elif set(criteria) != {"true", "false"}:
            _raise("invalid_provider_request", "Noul criteria must define exactly true and false")
        for option, description in criteria.items():
            _validate_bounded_text(option, field="JEV criteria key", max_bytes=256)
            _validate_bounded_text(
                description,
                field="JEV criteria text",
                max_bytes=_MAX_QUESTION_TEXT_BYTES,
            )
    try:
        _validate_json_value(questions, error_code="invalid_provider_request")
        normalized_state = _strict_json_loads(canonical_json(state))
        normalized_questions = _strict_json_loads(canonical_json(questions))
    except EvolutionError:
        raise
    except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError):
        _raise("invalid_provider_request", "JEV batch request must be finite bounded JSON")
    return normalized_state, normalized_questions


def _build_batch_request_body(
    model: str,
    state: dict[str, Any],
    questions: dict[str, dict[str, Any]],
) -> tuple[str, bytes]:
    request_value = {"model": model, "state": state, "questions": questions}
    try:
        request_text = canonical_json(request_value)
        encoded = request_text.encode("utf-8", "strict")
    except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError):
        _raise("invalid_provider_request", "JEV batch request could not be encoded safely")
    if len(encoded) > MAX_BODY_BYTES:
        _raise("provider_request_too_large", "JEV request exceeds the bounded input size")
    return request_text, encoded


def _parse_batch_response(
    body: bytes,
    *,
    configured_api_key: str,
    questions: dict[str, dict[str, Any]],
    requested_model: str,
) -> tuple[dict[str, ChoiceAnswer | NoulAnswer], str | None, int | None, int | None]:
    try:
        text = body.decode("utf-8", "strict")
    except UnicodeError:
        _raise("provider_invalid_response", "TypeSafe returned invalid UTF-8 JSON")
    if contains_sensitive_material(text, configured_api_key):
        _raise("provider_sensitive_output", "TypeSafe response contained credential-like material")
    try:
        response = _strict_json_loads(text)
    except (ValueError, RecursionError):
        _raise("provider_invalid_response", "TypeSafe returned invalid JSON")
    if contains_sensitive_material(response, configured_api_key):
        _raise("provider_sensitive_output", "TypeSafe response contained credential-like material")
    try:
        _validate_json_value(response, error_code="provider_invalid_response")
    except (RecursionError, OverflowError):
        _raise("provider_invalid_response", "TypeSafe response exceeds the structural limit")
    if type(response) is not dict or set(response) - {"answers", "model", "usage"}:
        _raise("provider_invalid_response", "TypeSafe returned an invalid batch response object")
    answers = response.get("answers")
    if type(answers) is not dict or set(answers) != set(questions):
        _raise("provider_invalid_response", "TypeSafe answer IDs must exactly match the requested questions")

    parsed: dict[str, ChoiceAnswer | NoulAnswer] = {}
    for question_id, question in questions.items():
        answer = answers[question_id]
        if type(answer) is not dict:
            _raise("provider_invalid_response", "TypeSafe returned an invalid typed answer")
        if question["type"] == "choice":
            if set(answer) != {"type", "choice", "probabilities", "confidence"} or answer["type"] != "choice":
                _raise("provider_invalid_response", "TypeSafe returned an invalid Choice answer shape")
            parsed[question_id] = ChoiceAnswer(
                option=answer["choice"],
                probabilities=answer["probabilities"],
                confidence=answer["confidence"],
            )
        else:
            if set(answer) != {"type", "noul"} or answer["type"] != "noul":
                _raise("provider_invalid_response", "TypeSafe returned an invalid Noul answer shape")
            parsed[question_id] = NoulAnswer(probability_yes=answer["noul"])

    model = _observed_model(response.get("model"), requested_model)
    usage = response.get("usage")
    return (
        parsed,
        model,
        _observed_tokens(usage, "input_tokens"),
        _observed_tokens(usage, "output_tokens"),
    )


class JevJudge:
    """Request one fixed-endpoint TypeSafe Choice judgment.

    The result is typed decision data only. It does not grant permissions,
    establish correctness, or authorize execution of the selected route.
    """

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        if type(model) is not str or not _MODEL_ID.fullmatch(model):
            _raise("invalid_provider_model", "TypeSafe model must be an explicit versioned Jev model ID")
        self.model = model

    def choose(
        self,
        state: dict[str, Any],
        *,
        instructions: str,
        criteria: dict[str, str],
        timeout_seconds: float,
        max_output_bytes: int,
    ) -> ChoiceResult:
        started = time.monotonic()
        try:
            timeout, output_limit = _validate_call_limits(timeout_seconds, max_output_bytes)
        except OverflowError:
            _raise("invalid_provider_limits", "Provider timeout must be positive and bounded")
        deadline = started + timeout
        normalized_state, safe_instructions, normalized_criteria = _validate_inputs(
            state,
            instructions=instructions,
            criteria=criteria,
        )
        request_text, request_bytes = _build_request_body(
            self.model,
            normalized_state,
            instructions=safe_instructions,
            criteria=normalized_criteria,
        )
        if contains_sensitive_material({
            "state": normalized_state,
            "instructions": safe_instructions,
            "criteria": normalized_criteria,
        }):
            _raise("provider_sensitive_input", "Credential-like key material is not valid JEV input")
        if time.monotonic() >= deadline:
            _raise("provider_timeout", "TypeSafe request exceeded the provider deadline")

        api_key = _read_api_key()
        worker_input = bytearray()
        try:
            if api_key in request_text or contains_sensitive_material(
                {"state": normalized_state, "instructions": safe_instructions, "criteria": normalized_criteria},
                api_key,
            ):
                _raise("provider_sensitive_input", "Configured credential material is not valid JEV input")
            response_limit = min(MAX_RESPONSE_BYTES, output_limit + _RESPONSE_METADATA_OVERHEAD)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _raise("provider_timeout", "TypeSafe request exceeded the provider deadline")
            envelope = {
                "api_key": api_key,
                "body": request_bytes.decode("utf-8", "strict"),
                "timeout_seconds": remaining,
                "max_response_bytes": response_limit,
            }
            try:
                worker_input.extend(canonical_json(envelope).encode("utf-8", "strict"))
            except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError):
                _raise("invalid_provider_request", "TypeSafe worker request could not be encoded safely")
            if len(worker_input) > MAX_WORKER_INPUT_BYTES:
                _raise("provider_request_too_large", "TypeSafe worker input exceeds the bounded size limit")
            del envelope

            try:
                wire = _execute_worker(
                    worker_input,
                    deadline=deadline,
                    response_limit=response_limit,
                )
            finally:
                for index in range(len(worker_input)):
                    worker_input[index] = 0
            if wire.startswith(b"ERR\n"):
                _worker_error(wire)
            if not wire.startswith(b"OK\n"):
                _raise("provider_worker_error", "TypeSafe HTTP worker returned an invalid response")
            parsed = _parse_response(
                wire[3:],
                configured_api_key=api_key,
                criteria=normalized_criteria,
                requested_model=self.model,
            )
            if time.monotonic() >= deadline:
                _raise("provider_timeout", "TypeSafe request exceeded the provider deadline")
            option, probabilities, confidence, model, input_tokens, output_tokens = parsed
            return ChoiceResult(
                option=option,
                probabilities=probabilities,
                confidence=confidence,
                duration_seconds=max(0.0, time.monotonic() - started),
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        finally:
            for index in range(len(worker_input)):
                worker_input[index] = 0
            del api_key

    def judge_batch(
        self,
        state: dict[str, Any],
        *,
        questions: dict[str, dict[str, Any]],
        timeout_seconds: float,
        max_output_bytes: int,
    ) -> BatchJudgment:
        """Ask one bounded mixed Choice/Noul request and return typed answers."""
        started = time.monotonic()
        try:
            timeout, output_limit = _validate_call_limits(timeout_seconds, max_output_bytes)
        except OverflowError:
            _raise("invalid_provider_limits", "Provider timeout must be positive and bounded")
        deadline = started + timeout
        normalized_state, normalized_questions = _validate_batch_inputs(state, questions)
        request_text, request_bytes = _build_batch_request_body(
            self.model,
            normalized_state,
            normalized_questions,
        )
        request_value = {
            "state": normalized_state,
            "questions": normalized_questions,
        }
        if contains_sensitive_material(request_value):
            _raise("provider_sensitive_input", "Credential-like key material is not valid JEV input")
        if time.monotonic() >= deadline:
            _raise("provider_timeout", "TypeSafe request exceeded the provider deadline")

        api_key = _read_api_key()
        worker_input = bytearray()
        try:
            if api_key in request_text or contains_sensitive_material(request_value, api_key):
                _raise("provider_sensitive_input", "Configured credential material is not valid JEV input")
            response_limit = min(MAX_RESPONSE_BYTES, output_limit + _RESPONSE_METADATA_OVERHEAD)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _raise("provider_timeout", "TypeSafe request exceeded the provider deadline")
            envelope = {
                "api_key": api_key,
                "body": request_bytes.decode("utf-8", "strict"),
                "timeout_seconds": remaining,
                "max_response_bytes": response_limit,
            }
            try:
                worker_input.extend(canonical_json(envelope).encode("utf-8", "strict"))
            except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError):
                _raise("invalid_provider_request", "TypeSafe worker request could not be encoded safely")
            if len(worker_input) > MAX_WORKER_INPUT_BYTES:
                _raise("provider_request_too_large", "TypeSafe worker input exceeds the bounded size limit")
            del envelope

            try:
                wire = _execute_worker(
                    worker_input,
                    deadline=deadline,
                    response_limit=response_limit,
                )
            finally:
                for index in range(len(worker_input)):
                    worker_input[index] = 0
            if wire.startswith(b"ERR\n"):
                _worker_error(wire)
            if not wire.startswith(b"OK\n"):
                _raise("provider_worker_error", "TypeSafe HTTP worker returned an invalid response")
            answers, model, input_tokens, output_tokens = _parse_batch_response(
                wire[3:],
                configured_api_key=api_key,
                questions=normalized_questions,
                requested_model=self.model,
            )
            result = BatchJudgment(
                answers=answers,
                duration_seconds=max(0.0, time.monotonic() - started),
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            try:
                validate_batch_judgment(result, normalized_questions)
            except EvolutionError as exc:
                raise EvolutionError(
                    "provider_invalid_response",
                    "TypeSafe returned an invalid batch judgment",
                ) from exc
            if time.monotonic() >= deadline:
                _raise("provider_timeout", "TypeSafe request exceeded the provider deadline")
            return result
        finally:
            for index in range(len(worker_input)):
                worker_input[index] = 0
            del api_key
