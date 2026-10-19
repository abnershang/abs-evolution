from __future__ import annotations

import copy
import io
import json
import os
from pathlib import Path
import time
import unittest
import urllib.error
import urllib.request
from unittest import mock

from abs_evolution import _jev_http
from abs_evolution.contracts import (
    BatchJudgment,
    ChoiceAnswer,
    ChoiceResult,
    NoulAnswer,
    canonical_json,
    validate_batch_judgment,
)
from abs_evolution.errors import EvolutionError
from abs_evolution.jev import DEFAULT_MODEL, JevJudge, _execute_worker


_TEST_KEY = "test-typesafe-key-not-a-real-credential"
_CRITERIA = {
    "reuse_candidate-a": "Reuse this exact reviewed proposal.",
    "generate": "Create new proposal content only when justified.",
    "insufficient_evidence": "The available evidence is not enough to decide.",
}


def state() -> dict:
    return {
        "request": {
            "baseline_id": "a" * 64,
            "training_examples": [
                {"id": "failed", "expected": 2, "actual": 1, "passed": False},
                {"id": "passed", "expected": "ok", "actual": "ok", "passed": True},
            ],
        },
        "candidates": [],
    }


def response(
    *,
    choice: str = "reuse_candidate-a",
    probabilities: dict | None = None,
    confidence: float = 0.7,
    model: str | None = DEFAULT_MODEL,
    usage: dict | None = None,
    answer: dict | None = None,
) -> dict:
    if probabilities is None:
        probabilities = {
            "reuse_candidate-a": 0.8,
            "generate": 0.15,
            "insufficient_evidence": 0.05,
        }
    selected_answer = {
        "type": "choice",
        "choice": choice,
        "probabilities": probabilities,
        "confidence": confidence,
    }
    if answer is not None:
        selected_answer = answer
    value = {"answers": {"decision": selected_answer}}
    if model is not None:
        value["model"] = model
    if usage is not None:
        value["usage"] = usage
    return value


def encode_response(value: dict | None = None) -> bytes:
    return canonical_json(response() if value is None else value).encode("utf-8")


def batch_questions() -> dict:
    return {
        "route": {
            "type": "choice",
            "instructions": "Select the supported route.",
            "criteria": {"reuse": "Reuse a matching proposal.", "generate": "New content is justified."},
        },
        "has_regression": {
            "type": "noul",
            "instructions": "Is there a training regression?",
            "criteria": {"true": "A regression is evidenced.", "false": "No regression is evidenced."},
        },
        "quality_opportunity": {
            "type": "noul",
            "instructions": "Is there a supported quality improvement opportunity?",
            "criteria": {"true": "A quality opportunity is supported.", "false": "No quality opportunity is supported."},
        },
    }


def batch_response(*, answers: dict | None = None, model: str | None = DEFAULT_MODEL, usage: dict | None = None) -> dict:
    value = {
        "answers": answers or {
            "route": {
                "type": "choice",
                "choice": "reuse",
                "probabilities": {"reuse": 0.8, "generate": 0.2},
                "confidence": 0.6,
            },
            "has_regression": {"type": "noul", "noul": 0.03},
            "quality_opportunity": {"type": "noul", "noul": 0.94},
        }
    }
    if model is not None:
        value["model"] = model
    if usage is not None:
        value["usage"] = usage
    return value


def encode_batch_response(value: dict | None = None) -> bytes:
    return canonical_json(batch_response() if value is None else value).encode("utf-8")


class _BinaryStream:
    def __init__(self, data: bytes = b"") -> None:
        self.buffer = io.BytesIO(data)


class _FakeResponse:
    def __init__(self, body: bytes, headers: dict[str, str], status: int = 200) -> None:
        self.body = body
        self.headers = headers
        self.status = status
        self.read_sizes: list[int] = []

    def getcode(self) -> int:
        return self.status

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        return self.body[:size]

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class JevJudgeTests(unittest.TestCase):
    def assert_code(self, code: str, call) -> EvolutionError:
        with self.assertRaises(EvolutionError) as caught:
            call()
        self.assertEqual(caught.exception.code, code)
        return caught.exception

    def call(
        self,
        raw_response: bytes,
        *,
        state_value: dict | None = None,
        instructions: str = "Choose the route supported by the provided evidence.",
        criteria: dict[str, str] | None = None,
        timeout: float = 5,
        output_limit: int = 8192,
        seen: dict | None = None,
    ) -> ChoiceResult:
        def worker(payload: bytearray, *, deadline: float, response_limit: int) -> bytes:
            if seen is not None:
                seen["envelope"] = json.loads(bytes(payload).decode("utf-8"))
                seen["deadline"] = deadline
                seen["response_limit"] = response_limit
            return raw_response if raw_response.startswith(b"ERR\n") else b"OK\n" + raw_response

        with mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": _TEST_KEY}, clear=True):
            with mock.patch("abs_evolution.jev._execute_worker", side_effect=worker):
                return JevJudge().choose(
                    state() if state_value is None else state_value,
                    instructions=instructions,
                    criteria=_CRITERIA if criteria is None else criteria,
                    timeout_seconds=timeout,
                    max_output_bytes=output_limit,
                )

    def call_batch(
        self,
        raw_response: bytes,
        *,
        questions: dict | None = None,
        timeout: float = 5,
        output_limit: int = 8192,
        seen: dict | None = None,
    ) -> BatchJudgment:
        def worker(payload: bytearray, *, deadline: float, response_limit: int) -> bytes:
            if seen is not None:
                seen["envelope"] = json.loads(bytes(payload).decode("utf-8"))
                seen["deadline"] = deadline
                seen["response_limit"] = response_limit
            return raw_response if raw_response.startswith(b"ERR\n") else b"OK\n" + raw_response

        with mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": _TEST_KEY}, clear=True):
            with mock.patch("abs_evolution.jev._execute_worker", side_effect=worker):
                return JevJudge().judge_batch(
                    state(),
                    questions=batch_questions() if questions is None else questions,
                    timeout_seconds=timeout,
                    max_output_bytes=output_limit,
                )

    def test_model_is_explicit_versioned_and_default_stays_pinned(self) -> None:
        self.assertEqual(JevJudge().model, "jev-1.13.0")
        self.assertEqual(JevJudge("jev-1.14.0").model, "jev-1.14.0")
        for model in ("jev-latest", "jev-preview", "", "Jev-1.13.0", "jev-1.13.0-suffix"):
            with self.subTest(model=model):
                self.assert_code("invalid_provider_model", lambda: JevJudge(model))

    def test_request_uses_one_typed_choice_at_fixed_model_and_returns_observed_values(self) -> None:
        seen: dict = {}
        result = self.call(
            encode_response(response(usage={"input_tokens": 132, "output_tokens": 29})),
            seen=seen,
        )
        self.assertEqual(result, ChoiceResult(
            option="reuse_candidate-a",
            probabilities={
                "reuse_candidate-a": 0.8,
                "generate": 0.15,
                "insufficient_evidence": 0.05,
            },
            confidence=0.7,
            duration_seconds=result.duration_seconds,
            model=DEFAULT_MODEL,
            input_tokens=132,
            output_tokens=29,
        ))
        self.assertGreaterEqual(result.duration_seconds, 0)
        self.assertLess(result.duration_seconds, 5)
        envelope = seen["envelope"]
        self.assertEqual(set(envelope), {"api_key", "body", "timeout_seconds", "max_response_bytes"})
        self.assertEqual(envelope["api_key"], _TEST_KEY)
        request_value = json.loads(envelope["body"])
        self.assertEqual(request_value["model"], DEFAULT_MODEL)
        self.assertEqual(request_value["state"], state())
        self.assertEqual(request_value["questions"], {
            "decision": {
                "type": "choice",
                "instructions": "Choose the route supported by the provided evidence.",
                "criteria": _CRITERIA,
            }
        })
        self.assertGreater(envelope["timeout_seconds"], 0)
        self.assertLessEqual(seen["response_limit"], _jev_http.MAX_RESPONSE_BYTES)

    def test_missing_usage_or_model_stays_unknown_not_zero_or_invented(self) -> None:
        result = self.call(encode_response(response(model=None)))
        self.assertIsNone(result.model)
        self.assertIsNone(result.input_tokens)
        self.assertIsNone(result.output_tokens)

    def test_missing_key_fails_closed_and_input_validation_happens_first(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("abs_evolution.jev._execute_worker") as worker:
                self.assert_code(
                    "invalid_provider_request",
                    lambda: JevJudge().choose(
                        [], instructions="Choose", criteria=_CRITERIA,
                        timeout_seconds=5, max_output_bytes=8192,
                    ),
                )
                self.assert_code(
                    "provider_credentials_missing",
                    lambda: JevJudge().choose(
                        state(), instructions="Choose", criteria=_CRITERIA,
                        timeout_seconds=5, max_output_bytes=8192,
                    ),
                )
                worker.assert_not_called()

    def test_invalid_choice_inputs_and_oversized_normalized_state_fail_before_worker(self) -> None:
        invalid_calls = [
            ("invalid_provider_request", {"x": float("inf")}, "Judge", _CRITERIA),
            ("invalid_provider_request", state(), "   ", _CRITERIA),
            ("invalid_provider_request", state(), "Judge", {}),
            ("invalid_provider_request", state(), "Judge", {"": "empty option"}),
            ("provider_request_too_large", {"blob": "x" * (4 * 1024 * 1024)}, "Judge", _CRITERIA),
        ]
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("abs_evolution.jev._execute_worker") as worker:
                for code, state_value, instructions, criteria in invalid_calls:
                    with self.subTest(code=code, state_type=type(state_value).__name__):
                        self.assert_code(
                            code,
                            lambda state_value=state_value, instructions=instructions, criteria=criteria: JevJudge().choose(
                                state_value,
                                instructions=instructions,
                                criteria=criteria,
                                timeout_seconds=5,
                                max_output_bytes=8192,
                            ),
                        )
                worker.assert_not_called()

    def test_sensitive_inputs_are_rejected_before_worker_and_configured_key_is_scanned(self) -> None:
        password_state = {"request": {"metadata": {"password": "synthetic-password-value"}}}
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("abs_evolution.jev._execute_worker") as worker:
                self.assert_code(
                    "provider_sensitive_input",
                    lambda: JevJudge().choose(
                        password_state, instructions="Judge", criteria=_CRITERIA,
                        timeout_seconds=5, max_output_bytes=8192,
                    ),
                )
                worker.assert_not_called()

        key_state = {"request": {"summary": _TEST_KEY}}
        with mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": _TEST_KEY}, clear=True):
            with mock.patch("abs_evolution.jev._execute_worker") as worker:
                self.assert_code(
                    "provider_sensitive_input",
                    lambda: JevJudge().choose(
                        key_state, instructions="Judge", criteria=_CRITERIA,
                        timeout_seconds=5, max_output_bytes=8192,
                    ),
                )
                worker.assert_not_called()

    def test_malformed_choice_shapes_and_inconsistent_confidence_are_rejected(self) -> None:
        malformed = [
            response(choice="not-supplied"),
            response(choice="generate"),
            response(probabilities={"reuse_candidate-a": 0.8, "generate": 0.2}),
            response(probabilities={"reuse_candidate-a": 0.7, "generate": 0.2, "insufficient_evidence": 0.05}),
            response(probabilities={"reuse_candidate-a": 1.01, "generate": 0, "insufficient_evidence": 0}),
            response(probabilities={"reuse_candidate-a": True, "generate": 0, "insufficient_evidence": 0}),
            response(confidence=-0.01),
            response(confidence=1.01),
            response(answer={"type": "noul", "choice": "reuse_candidate-a"}),
            {"answers": {"decision": response()["answers"]["decision"], "other": {}}},
            {"answers": {"decision": response()["answers"]["decision"]}, "model": "jev-1.14.0"},
        ]
        for value in malformed:
            with self.subTest(value=value):
                self.assert_code("provider_invalid_response", lambda: self.call(encode_response(value)))

        overflow_probability = b'{"model":"jev-1.13.0","answers":{"decision":{"type":"choice","choice":"reuse_candidate-a","probabilities":{"reuse_candidate-a":1e999,"generate":0,"insufficient_evidence":0},"confidence":1}}}'
        self.assert_code("provider_invalid_response", lambda: self.call(overflow_probability))

    def test_tied_highest_choice_is_accepted_for_higher_level_uncertainty_handling(self) -> None:
        result = self.call(encode_response(response(
            probabilities={"reuse_candidate-a": 0.5, "generate": 0.5, "insufficient_evidence": 0},
            confidence=0.25,
        )))
        self.assertEqual(result.option, "reuse_candidate-a")
        self.assertEqual(result.probabilities["reuse_candidate-a"], 0.5)

    def test_raw_and_decoded_response_are_scanned_for_credentials(self) -> None:
        raw_secret = encode_response({**response(), "metadata": "AKIA1234567890ABCDEF"})
        self.assert_code("provider_sensitive_output", lambda: self.call(raw_secret))

        encoded_key = "".join(f"\\u{ord(character):04x}" for character in _TEST_KEY)
        escaped_raw = encode_response({**response(), "metadata": _TEST_KEY}).decode("utf-8").replace(
            _TEST_KEY,
            encoded_key,
        ).encode("utf-8")
        self.assertNotIn(_TEST_KEY.encode("ascii"), escaped_raw)
        self.assert_code("provider_sensitive_output", lambda: self.call(escaped_raw))

    def test_invalid_usage_values_are_unknown_and_never_echoed(self) -> None:
        result = self.call(encode_response(response(usage={"input_tokens": -1, "output_tokens": True})))
        self.assertIsNone(result.input_tokens)
        self.assertIsNone(result.output_tokens)

    def test_http_error_statuses_are_sanitized_and_specific(self) -> None:
        cases = {
            301: "provider_redirect_rejected",
            400: "provider_request_rejected",
            401: "provider_auth_failed",
            403: "provider_permission_denied",
            408: "provider_timeout",
            413: "provider_request_too_large",
            415: "provider_unsupported_media_type",
            422: "provider_question_invalid",
            429: "provider_rate_limited",
            529: "provider_unavailable",
            503: "provider_unavailable",
        }
        for status, code in cases.items():
            wire = f"ERR\nhttp_status\n{status}\n".encode("ascii")
            with self.subTest(status=status):
                error = self.assert_code(code, lambda: self.call(wire))
                self.assertNotIn(_TEST_KEY, error.message)

    def test_real_worker_rejects_invalid_input_without_network(self) -> None:
        wire = _execute_worker(
            bytearray(b"not-json"),
            deadline=time.monotonic() + 5,
            response_limit=1024,
        )
        self.assertEqual(wire, b"ERR\ninvalid_worker_request\n\n")

    def test_batch_contract_rejects_oversized_text_and_huge_integers_at_both_boundaries(self) -> None:
        questions = batch_questions()
        valid_answers = {
            "route": ChoiceAnswer("reuse", {"reuse": 0.8, "generate": 0.2}, 0.6),
            "has_regression": NoulAnswer(0.03),
            "quality_opportunity": NoulAnswer(0.94),
        }
        huge = 10**1000

        # Direct typed-contract validation must reject values before math.isfinite
        # tries converting an enormous Python integer to float.
        malformed = []
        for field in ("probability", "confidence", "noul", "duration"):
            answers = dict(valid_answers)
            duration = 0.01
            if field == "probability":
                answers["route"] = ChoiceAnswer("reuse", {"reuse": huge, "generate": 0}, 0.6)
            elif field == "confidence":
                answers["route"] = ChoiceAnswer("reuse", {"reuse": 0.8, "generate": 0.2}, huge)
            elif field == "noul":
                answers["has_regression"] = NoulAnswer(huge)
            else:
                duration = huge
            malformed.append((field, BatchJudgment(answers, duration), questions))

        for field, result, requested in malformed:
            with self.subTest(field=field):
                with self.assertRaises(EvolutionError) as caught:
                    validate_batch_judgment(result, requested)
                self.assertEqual(caught.exception.code, "provider_invalid_judgement")

        for field in ("instructions", "criteria"):
            oversized_questions = copy.deepcopy(questions)
            if field == "instructions":
                oversized_questions["route"]["instructions"] = "x" * 8193
            else:
                oversized_questions["route"]["criteria"]["reuse"] = "x" * 8193
            with self.subTest(field=field):
                with self.assertRaises(EvolutionError) as caught:
                    validate_batch_judgment(BatchJudgment(valid_answers, 0.01), oversized_questions)
                self.assertEqual(caught.exception.code, "provider_invalid_judgement")

        # The real adapter boundary must convert huge JSON integers into its
        # typed malformed-response error, never leak OverflowError.
        wire_answers = batch_response()["answers"]
        wire_answers["route"]["probabilities"]["reuse"] = huge
        self.assert_code(
            "provider_invalid_response",
            lambda: self.call_batch(encode_batch_response(batch_response(answers=wire_answers))),
        )

        oversized_questions = batch_questions()
        oversized_questions["route"]["instructions"] = "x" * 8193
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("abs_evolution.jev._execute_worker") as worker:
                self.assert_code(
                    "invalid_provider_request",
                    lambda: JevJudge().judge_batch(
                        state(),
                        questions=oversized_questions,
                        timeout_seconds=5,
                        max_output_bytes=8192,
                    ),
                )
                worker.assert_not_called()

    def test_batch_parser_rejects_malformed_mixed_answers(self) -> None:
        valid = batch_response()["answers"]
        malformed: list[tuple[str, bytes]] = []

        answers = copy.deepcopy(valid)
        del answers["quality_opportunity"]
        malformed.append(("missing_question_id", encode_batch_response(batch_response(answers=answers))))

        answers = copy.deepcopy(valid)
        answers["unexpected"] = {"type": "noul", "noul": 0.5}
        malformed.append(("extra_question_id", encode_batch_response(batch_response(answers=answers))))

        answers = copy.deepcopy(valid)
        answers["route"] = {"type": "noul", "noul": 0.5}
        malformed.append(("wrong_choice_type", encode_batch_response(batch_response(answers=answers))))

        answers = copy.deepcopy(valid)
        del answers["route"]["confidence"]
        malformed.append(("wrong_choice_shape", encode_batch_response(batch_response(answers=answers))))

        answers = copy.deepcopy(valid)
        answers["has_regression"] = {
            "type": "choice",
            "choice": "true",
            "probabilities": {"true": 1.0, "false": 0.0},
            "confidence": 1.0,
        }
        malformed.append(("wrong_noul_type", encode_batch_response(batch_response(answers=answers))))

        answers = copy.deepcopy(valid)
        answers["has_regression"]["noul"] = True
        malformed.append(("boolean_noul", encode_batch_response(batch_response(answers=answers))))

        answers = copy.deepcopy(valid)
        answers["has_regression"]["noul"] = 1.01
        malformed.append(("out_of_range_noul", encode_batch_response(batch_response(answers=answers))))

        answers = copy.deepcopy(valid)
        answers["route"]["choice"] = "not-supplied"
        malformed.append(("unsupplied_choice", encode_batch_response(batch_response(answers=answers))))

        answers = copy.deepcopy(valid)
        answers["route"]["probabilities"] = {"reuse": 0.2, "generate": 0.8}
        malformed.append(("choice_not_highest", encode_batch_response(batch_response(answers=answers))))

        nonfinite_noul = encode_batch_response().replace(b'"noul":0.03', b'"noul":1e999', 1)
        malformed.append(("nonfinite_noul", nonfinite_noul))
        malformed.append(("mismatched_model", encode_batch_response(batch_response(model="jev-1.14.0"))))

        for name, body in malformed:
            with self.subTest(name=name):
                self.assert_code("provider_invalid_response", lambda body=body: self.call_batch(body))

    def test_batch_request_schema_and_sensitive_material_fail_closed(self) -> None:
        base = batch_questions()["route"]
        too_many = {f"q{index}": copy.deepcopy(base) for index in range(9)}
        malformed_questions = [
            ("empty_batch", {}),
            ("too_many_questions", too_many),
            ("unsafe_id", {"not a safe id": copy.deepcopy(base)}),
            ("non_string_id", {1: copy.deepcopy(base)}),
            ("extra_question_field", {"route": {**copy.deepcopy(base), "extra": True}}),
            ("missing_question_field", {"route": {"type": "choice", "instructions": "Choose."}}),
            ("unsupported_question_type", {"route": {**copy.deepcopy(base), "type": "binary"}}),
        ]
        with mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": _TEST_KEY}, clear=True):
            with mock.patch("abs_evolution.jev._execute_worker") as worker:
                for name, questions in malformed_questions:
                    with self.subTest(name=name):
                        self.assert_code(
                            "invalid_provider_request",
                            lambda questions=questions: JevJudge().judge_batch(
                                state(), questions=questions, timeout_seconds=5, max_output_bytes=8192,
                            ),
                        )

                secret_state = state()
                secret_state["request"]["password"] = "synthetic-password-value"
                self.assert_code(
                    "provider_sensitive_input",
                    lambda: JevJudge().judge_batch(
                        secret_state, questions=batch_questions(), timeout_seconds=5, max_output_bytes=8192,
                    ),
                )

                configured_key_state = state()
                configured_key_state["request"]["summary"] = _TEST_KEY
                secret_error = self.assert_code(
                    "provider_sensitive_input",
                    lambda: JevJudge().judge_batch(
                        configured_key_state, questions=batch_questions(), timeout_seconds=5, max_output_bytes=8192,
                    ),
                )
                self.assertNotIn(_TEST_KEY, secret_error.message)
                worker.assert_not_called()

        raw_output_secret = encode_batch_response({**batch_response(), "metadata": "AKIA1234567890ABCDEF"})
        self.assert_code("provider_sensitive_output", lambda: self.call_batch(raw_output_secret))

        escaped_output_secret = encode_batch_response({**batch_response(), "metadata": _TEST_KEY})
        encoded_key = "".join(f"\\u{ord(character):04x}" for character in _TEST_KEY).encode("ascii")
        escaped_output_secret = escaped_output_secret.replace(_TEST_KEY.encode("ascii"), encoded_key)
        self.assertNotIn(_TEST_KEY.encode("ascii"), escaped_output_secret)
        self.assert_code("provider_sensitive_output", lambda: self.call_batch(escaped_output_secret))

    def test_batch_timeout_rejects_late_worker_and_zeroes_credential_buffer(self) -> None:
        observed: dict[str, bytearray | bytes] = {}

        def late_worker(payload: bytearray, *, deadline: float, response_limit: int) -> bytes:
            observed["buffer"] = payload
            observed["before"] = bytes(payload)
            time.sleep(0.03)
            return b"OK\n" + encode_batch_response()

        with mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": _TEST_KEY}, clear=True):
            with mock.patch("abs_evolution.jev._execute_worker", side_effect=late_worker):
                self.assert_code(
                    "provider_timeout",
                    lambda: JevJudge().judge_batch(
                        state(), questions=batch_questions(), timeout_seconds=0.01, max_output_bytes=8192,
                    ),
                )

        before = observed["before"]
        buffer = observed["buffer"]
        self.assertIsInstance(before, bytes)
        self.assertIn(_TEST_KEY.encode("ascii"), before)
        self.assertIsInstance(buffer, bytearray)
        self.assertEqual(len(buffer), len(before))
        self.assertEqual(buffer, bytearray(len(before)))


class JevHttpWorkerTests(unittest.TestCase):
    def envelope(self, *, response_limit: int = 4096) -> dict:
        return {
            "api_key": _TEST_KEY,
            "body": canonical_json({"model": DEFAULT_MODEL}),
            "timeout_seconds": 3,
            "max_response_bytes": response_limit,
        }

    def run_worker(self, envelope: dict, response: _FakeResponse) -> bytes:
        opener = mock.Mock()
        opener.open.return_value = response
        fake_stdout = _BinaryStream()
        with mock.patch.object(_jev_http.urllib.request, "build_opener", return_value=opener) as build_opener:
            with mock.patch.object(
                _jev_http.sys,
                "stdin",
                _BinaryStream(canonical_json(envelope).encode("utf-8")),
            ):
                with mock.patch.object(_jev_http.sys, "stdout", fake_stdout):
                    self.assertEqual(_jev_http.main(), 0)
        self.assertEqual(build_opener.call_count, 1)
        return fake_stdout.buffer.getvalue()

    def test_worker_posts_to_fixed_tls_endpoint_with_no_proxy_or_redirect(self) -> None:
        body = encode_response()
        response = _FakeResponse(body, {"Content-Type": "application/json; charset=utf-8"})
        opener = mock.Mock()
        opener.open.return_value = response
        fake_stdout = _BinaryStream()
        envelope = self.envelope()

        with mock.patch.object(_jev_http.urllib.request, "build_opener", return_value=opener) as build_opener:
            with mock.patch.object(_jev_http.sys, "stdin", _BinaryStream(canonical_json(envelope).encode("utf-8"))):
                with mock.patch.object(_jev_http.sys, "stdout", fake_stdout):
                    self.assertEqual(_jev_http.main(), 0)

        output = fake_stdout.buffer.getvalue()
        self.assertEqual(output, b"OK\n" + body)
        request = opener.open.call_args.args[0]
        timeout = opener.open.call_args.kwargs["timeout"]
        self.assertEqual(request.full_url, "https://api.typesafe.ai/v1/systemone")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), f"Bearer {_TEST_KEY}")
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertEqual(timeout, 3.0)
        handlers = build_opener.call_args.args
        self.assertEqual(len(handlers), 3)
        self.assertIsInstance(handlers[0], urllib.request.ProxyHandler)
        self.assertEqual(handlers[0].proxies, {})
        self.assertIsInstance(handlers[1], _jev_http._NoRedirect)
        self.assertIsInstance(handlers[2], urllib.request.HTTPSHandler)
        self.assertEqual(response.read_sizes, [4097])

    def test_worker_refuses_redirect_and_oversized_declared_body(self) -> None:
        opener = mock.Mock()
        redirect_body = io.BytesIO(b"redirect response")
        opener.open.side_effect = urllib.error.HTTPError(
            _jev_http.ENDPOINT,
            302,
            "redirect",
            {},
            redirect_body,
        )
        fake_stdout = _BinaryStream()
        envelope = self.envelope()
        with mock.patch.object(_jev_http.urllib.request, "build_opener", return_value=opener):
            with mock.patch.object(_jev_http.sys, "stdin", _BinaryStream(canonical_json(envelope).encode("utf-8"))):
                with mock.patch.object(_jev_http.sys, "stdout", fake_stdout):
                    self.assertEqual(_jev_http.main(), 0)
        self.assertEqual(fake_stdout.buffer.getvalue(), b"ERR\nhttp_status\n302\n")
        self.assertTrue(redirect_body.closed)

        oversized = _FakeResponse(b"unused", {"Content-Type": "application/json", "Content-Length": "100"})
        wire = self.run_worker(self.envelope(response_limit=10), oversized)
        self.assertEqual(wire, b"ERR\nresponse_too_large\n\n")
        self.assertEqual(oversized.read_sizes, [])

    def test_worker_rejects_invalid_envelope_without_opening_network(self) -> None:
        fake_stdout = _BinaryStream()
        with mock.patch.object(_jev_http.urllib.request, "build_opener") as build_opener:
            with mock.patch.object(_jev_http.sys, "stdin", _BinaryStream(b"not-json")):
                with mock.patch.object(_jev_http.sys, "stdout", fake_stdout):
                    self.assertEqual(_jev_http.main(), 0)
        self.assertEqual(fake_stdout.buffer.getvalue(), b"ERR\ninvalid_worker_request\n\n")
        build_opener.assert_not_called()


if __name__ == "__main__":
    unittest.main()
