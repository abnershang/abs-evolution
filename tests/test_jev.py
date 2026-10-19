from __future__ import annotations

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
from abs_evolution.contracts import ChoiceResult, canonical_json
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
