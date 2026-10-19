from __future__ import annotations

import io
import json
import os
from pathlib import Path
import ssl
import subprocess
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from unittest import mock

from abs_evolution import _openrouter_http, _provider_transport
from abs_evolution.contracts import canonical_json
from abs_evolution.errors import EvolutionError
from abs_evolution.openrouter import OpenRouterProvider, _execute_worker


_TEST_KEY = "test-openrouter-key-not-a-real-credential"


def request() -> dict:
    return {
        "schema_version": 1,
        "objective": "Correct the observed training failure without changing passing cases.",
        "baseline_id": "b" * 64,
        "files": {"skills/solver.py": "def solve(value): return value\n"},
        "training_examples": [
            {
                "id": "failed-case",
                "input": {"n": 1},
                "expected": 2,
                "actual": 1,
                "passed": False,
            },
            {
                "id": "passed-case",
                "input": {"keep": True},
                "expected": {"ok": True},
                "actual": {"ok": True},
                "passed": True,
            },
        ],
        "allowed_mutation_paths": ["skills/*"],
        "max_changes": 2,
    }


def proposal() -> dict:
    return {
        "summary": "Correct the failing case while preserving the passing case.",
        "kind": "skill",
        "changes": [{"path": "skills/solver.py", "content": "def solve(value): return value + 1\n"}],
    }


def completion(
    proposal_value: dict | None = None,
    *,
    usage: dict | None = None,
    model: str = "openai/test-model",
    generation_id: str = "gen-test-123",
) -> bytes:
    payload = {
        "id": generation_id,
        "model": model,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": canonical_json(proposal() if proposal_value is None else proposal_value),
                },
            }
        ],
    }
    if usage is not None:
        payload["usage"] = usage
    return canonical_json(payload).encode("utf-8")


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


class OpenRouterProviderTests(unittest.TestCase):
    def assert_code(self, code: str, call) -> EvolutionError:
        with self.assertRaises(EvolutionError) as caught:
            call()
        self.assertEqual(caught.exception.code, code)
        return caught.exception

    def call(self, *, req: dict | None = None, timeout: float = 5, output_limit: int = 8192):
        return OpenRouterProvider("openai/test-model").propose(
            request() if req is None else req,
            timeout_seconds=timeout,
            max_output_bytes=output_limit,
        )

    def _mock_worker(self, raw_response: bytes, seen: dict | None = None):
        def run(payload: bytearray, *, deadline: float, response_limit: int) -> bytes:
            envelope = json.loads(bytes(payload))
            if seen is not None:
                seen["envelope"] = envelope
                seen["deadline"] = deadline
                seen["response_limit"] = response_limit
            return b"OK\n" + raw_response

        return run

    def test_proposal_request_is_strict_proposal_only_and_usage_is_observed(self) -> None:
        seen: dict = {}
        usage = {"prompt_tokens": 120, "completion_tokens": 48, "cost": 0.00042}
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": _TEST_KEY}):
            with mock.patch(
                "abs_evolution.openrouter._execute_worker",
                side_effect=self._mock_worker(completion(usage=usage), seen),
            ):
                result = self.call()

        envelope = seen["envelope"]
        body = json.loads(envelope["body"])
        self.assertEqual(set(envelope), {"api_key", "body", "timeout_seconds", "max_response_bytes"})
        self.assertEqual(envelope["api_key"], _TEST_KEY)
        self.assertNotIn(_TEST_KEY, envelope["body"])
        self.assertEqual(
            set(body),
            {"model", "messages", "stream", "max_tokens", "response_format", "provider"},
        )
        self.assertEqual(body["model"], "openai/test-model")
        self.assertIs(body["stream"], False)
        self.assertEqual(body["max_tokens"], 2048)
        self.assertNotIn("tools", body)
        self.assertNotIn("plugins", body)
        self.assertEqual(body["provider"], {
            "require_parameters": True,
            "data_collection": "deny",
            "zdr": True,
            "allow_fallbacks": False,
        })
        self.assertEqual(body["messages"][0]["role"], "system")
        self.assertEqual(body["messages"][1], {"role": "user", "content": canonical_json(request())})
        self.assertEqual(body["response_format"]["type"], "json_schema")
        schema = body["response_format"]["json_schema"]
        self.assertIs(schema["strict"], True)
        self.assertEqual(schema["schema"]["additionalProperties"], False)
        self.assertEqual(schema["schema"]["properties"]["changes"]["maxItems"], 2)
        self.assertEqual(result.value, proposal())
        self.assertEqual(result.input_tokens, 120)
        self.assertEqual(result.output_tokens, 48)
        self.assertEqual(result.model, "openai/test-model")
        self.assertEqual(result.generation_id, "gen-test-123")
        self.assertEqual(result.cost_credits, 0.00042)
        self.assertTrue(OpenRouterProvider.requires_hard_resource_limits)

    def test_key_is_required_and_sensitive_request_data_fails_before_worker(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("abs_evolution.openrouter._execute_worker") as worker:
                self.assert_code("provider_credentials_missing", self.call)
                worker.assert_not_called()

        sensitive = request()
        sensitive["training_examples"][0]["input"] = {"opaque": "sk-or-v1-" + "a" * 64}
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": _TEST_KEY}):
            with mock.patch("abs_evolution.openrouter._execute_worker") as worker:
                self.assert_code(
                    "provider_sensitive_input",
                    lambda: self.call(req=sensitive),
                )
                worker.assert_not_called()

    def test_nonempty_credential_fields_are_rejected_but_auth_and_empty_fields_are_allowed(self) -> None:
        sensitive = request()
        sensitive["training_examples"][0]["input"] = {"password": "synthetic-placeholder-value"}
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": _TEST_KEY}):
            with mock.patch("abs_evolution.openrouter._execute_worker") as worker:
                self.assert_code("provider_sensitive_input", lambda: self.call(req=sensitive))
                worker.assert_not_called()

        ordinary = request()
        ordinary["training_examples"][0]["input"] = {
            "auth": "ordinary task data",
            "password": "",
        }
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": _TEST_KEY}):
            with mock.patch(
                "abs_evolution.openrouter._execute_worker",
                side_effect=self._mock_worker(completion()),
            ):
                result = self.call(req=ordinary)
        self.assertEqual(result.value, proposal())

    def test_literal_configured_key_in_request_is_rejected_without_echo(self) -> None:
        sensitive = request()
        sensitive["objective"] = f"Please include {_TEST_KEY} in the candidate"
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": _TEST_KEY}):
            with mock.patch("abs_evolution.openrouter._execute_worker") as worker:
                error = self.assert_code(
                    "provider_sensitive_input",
                    lambda: self.call(req=sensitive),
                )
                worker.assert_not_called()
        self.assertNotIn(_TEST_KEY, str(error))

    def test_explicit_model_and_token_limits_are_enforced(self) -> None:
        for model in (
            "openai/test-model:online",
            "openrouter/auto",
            "openrouter/free",
            "openai/test-model,other/model",
            "../model",
            "",
        ):
            self.assert_code(
                "invalid_provider_model",
                lambda model=model: OpenRouterProvider(model),
            )
        for tokens in (True, 0, 8193, -1):
            self.assert_code(
                "invalid_provider_limits",
                lambda tokens=tokens: OpenRouterProvider("openai/test-model", max_output_tokens=tokens),
            )
        provider = OpenRouterProvider("openai/test-model", max_output_tokens=33)
        seen: dict = {}
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": _TEST_KEY}):
            with mock.patch(
                "abs_evolution.openrouter._execute_worker",
                side_effect=self._mock_worker(completion(), seen),
            ):
                provider.propose(request(), timeout_seconds=5, max_output_bytes=8192)
        body = json.loads(seen["envelope"]["body"])
        self.assertEqual(body["max_tokens"], 33)

    def test_http_200_errors_malformed_choices_tools_refusal_and_truncation_are_rejected(self) -> None:
        cases = [
            ({"error": {"message": _TEST_KEY}}, "provider_sensitive_output"),
            ({"choices": []}, "provider_invalid_response"),
            ({"choices": [{}, {}]}, "provider_invalid_response"),
            ({"choices": [{"finish_reason": "stop", "message": {"role": "user", "content": "{}"}}]}, "provider_invalid_response"),
            ({"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "tool_calls": [{"id": "call-1"}], "content": "{}"}}]}, "provider_invalid_response"),
            ({"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "function_call": {"name": "run", "arguments": "{}"}, "content": "{}"}}]}, "provider_invalid_response"),
            ({"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "refusal": "no", "content": "{}"}}]}, "provider_refusal"),
            ({"choices": [{"finish_reason": "length", "message": {"role": "assistant", "content": "{}"}}]}, "provider_truncated_response"),
        ]
        for response, expected_code in cases:
            raw = canonical_json(response).encode("utf-8")
            with self.subTest(expected_code=expected_code):
                with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": _TEST_KEY}):
                    with mock.patch(
                        "abs_evolution.openrouter._execute_worker",
                        side_effect=self._mock_worker(raw),
                    ):
                        error = self.assert_code(expected_code, self.call)
                self.assertNotIn(_TEST_KEY, str(error))

        for empty_tools in (None, []):
            response = json.loads(completion())
            response["choices"][0]["message"]["tool_calls"] = empty_tools
            with self.subTest(empty_tools=empty_tools):
                with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": _TEST_KEY}):
                    with mock.patch(
                        "abs_evolution.openrouter._execute_worker",
                        side_effect=self._mock_worker(canonical_json(response).encode("utf-8")),
                    ):
                        result = self.call()
                self.assertEqual(result.value, proposal())

    def test_http_error_statuses_are_mapped_without_body_details(self) -> None:
        cases = {
            302: "provider_redirect_rejected",
            400: "provider_request_rejected",
            401: "provider_auth_failed",
            402: "provider_credits_exhausted",
            403: "provider_permission_denied",
            408: "provider_timeout",
            429: "provider_rate_limited",
            503: "provider_routing_unavailable",
            504: "provider_unavailable",
        }
        for status, expected_code in cases.items():
            with self.subTest(status=status):
                with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": _TEST_KEY}):
                    with mock.patch(
                        "abs_evolution.openrouter._execute_worker",
                        return_value=f"ERR\nhttp_status\n{status}\n".encode("ascii"),
                    ):
                        error = self.assert_code(expected_code, self.call)
                self.assertNotIn(_TEST_KEY, str(error))

    def test_malformed_proposal_and_unobserved_or_invalid_usage_are_handled_safely(self) -> None:
        invalid = proposal()
        invalid["untrusted_extra"] = "no"
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": _TEST_KEY}):
            with mock.patch(
                "abs_evolution.openrouter._execute_worker",
                side_effect=self._mock_worker(completion(invalid)),
            ):
                self.assert_code("invalid_proposal", self.call)

            with mock.patch(
                "abs_evolution.openrouter._execute_worker",
                side_effect=self._mock_worker(completion(usage={
                    "prompt_tokens": True,
                    "completion_tokens": -1,
                    "cost": "0.01",
                })),
            ):
                result = self.call()
        self.assertIsNone(result.input_tokens)
        self.assertIsNone(result.output_tokens)
        self.assertIsNone(result.cost_credits)
        self.assertEqual(result.model, "openai/test-model")
        self.assertEqual(result.generation_id, "gen-test-123")

    def test_response_metadata_and_other_fields_never_echo_configured_or_credential_like_values(self) -> None:
        response_cases = (
            completion(generation_id=_TEST_KEY),
            completion(generation_id="password=synthetic-placeholder-value"),
            canonical_json({"api_key": "sensitive-value", "choices": []}).encode("utf-8"),
        )
        for index, response_bytes in enumerate(response_cases):
            with self.subTest(response_case=index):
                with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": _TEST_KEY}):
                    with mock.patch(
                        "abs_evolution.openrouter._execute_worker",
                        side_effect=self._mock_worker(response_bytes),
                    ):
                        error = self.assert_code("provider_sensitive_output", self.call)
                self.assertNotIn(_TEST_KEY, str(error))

    def test_unicode_escaped_configured_key_in_decoded_proposal_is_rejected(self) -> None:
        escaped_key = "".join(f"\\u{ord(character):04x}" for character in _TEST_KEY)
        inner_proposal = (
            '{"summary":"'
            + escaped_key
            + '","kind":"skill","changes":'
            + canonical_json(proposal()["changes"])
            + "}"
        )
        response = json.loads(completion())
        response["choices"][0]["message"]["content"] = inner_proposal
        raw_response = canonical_json(response).encode("utf-8")
        self.assertNotIn(_TEST_KEY, raw_response.decode("utf-8"))

        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": _TEST_KEY}):
            with mock.patch(
                "abs_evolution.openrouter._execute_worker",
                side_effect=self._mock_worker(raw_response),
            ):
                error = self.assert_code("provider_sensitive_output", self.call)
        self.assertNotIn(_TEST_KEY, str(error))

    def test_worker_posts_only_to_fixed_verified_https_endpoint_and_refuses_redirect(self) -> None:
        response_body = b'{"choices":[]}'
        response = _FakeResponse(
            response_body,
            {"Content-Type": "application/json; charset=utf-8", "Content-Length": str(len(response_body))},
        )
        opener = mock.Mock()
        opener.open.return_value = response
        envelope = {
            "api_key": _TEST_KEY,
            "body": "{\"model\":\"openai/test-model\"}",
            "timeout_seconds": 3,
            "max_response_bytes": 1024,
        }
        fake_stdin = _BinaryStream(canonical_json(envelope).encode("utf-8"))
        fake_stdout = _BinaryStream()
        with mock.patch.object(_openrouter_http.urllib.request, "build_opener", return_value=opener) as build:
            with mock.patch.object(_openrouter_http.sys, "stdin", fake_stdin):
                with mock.patch.object(_openrouter_http.sys, "stdout", fake_stdout):
                    self.assertEqual(_openrouter_http.main(), 0)

        frame = fake_stdout.buffer.getvalue()
        self.assertEqual(frame, b"OK\n" + response_body)
        request_object = opener.open.call_args.args[0]
        self.assertEqual(request_object.full_url, "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(request_object.get_method(), "POST")
        self.assertEqual(request_object.data, envelope["body"].encode("utf-8"))
        self.assertEqual(request_object.get_header("Authorization"), f"Bearer {_TEST_KEY}")
        handlers = build.call_args.args
        self.assertTrue(any(isinstance(handler, _openrouter_http._NoRedirect) for handler in handlers))
        self.assertTrue(any(isinstance(handler, urllib.request.ProxyHandler) and handler.proxies == {} for handler in handlers))
        tls_handler = next(handler for handler in handlers if isinstance(handler, urllib.request.HTTPSHandler))
        self.assertTrue(tls_handler._context.check_hostname)
        self.assertEqual(tls_handler._context.verify_mode, ssl.CERT_REQUIRED)
        self.assertEqual(response.read_sizes, [1025])

        redirect_body = io.BytesIO(b"ignored")
        redirect = urllib.error.HTTPError(
            _openrouter_http.ENDPOINT,
            302,
            "redirect",
            {},
            redirect_body,
        )
        opener.open.side_effect = redirect
        fake_stdout = _BinaryStream()
        with mock.patch.object(_openrouter_http.urllib.request, "build_opener", return_value=opener):
            with mock.patch.object(_openrouter_http.sys, "stdin", _BinaryStream(canonical_json(envelope).encode("utf-8"))):
                with mock.patch.object(_openrouter_http.sys, "stdout", fake_stdout):
                    self.assertEqual(_openrouter_http.main(), 0)
        self.assertEqual(fake_stdout.buffer.getvalue(), b"ERR\nhttp_status\n302\n")
        self.assertTrue(redirect_body.closed)

    def test_worker_rejects_oversized_http_response_without_reading_the_body(self) -> None:
        response = _FakeResponse(b"unused", {"Content-Type": "application/json", "Content-Length": "100"})
        opener = mock.Mock()
        opener.open.return_value = response
        envelope = {
            "api_key": _TEST_KEY,
            "body": "{}",
            "timeout_seconds": 3,
            "max_response_bytes": 10,
        }
        fake_stdout = _BinaryStream()
        with mock.patch.object(_openrouter_http.urllib.request, "build_opener", return_value=opener):
            with mock.patch.object(_openrouter_http.sys, "stdin", _BinaryStream(canonical_json(envelope).encode("utf-8"))):
                with mock.patch.object(_openrouter_http.sys, "stdout", fake_stdout):
                    self.assertEqual(_openrouter_http.main(), 0)
        self.assertEqual(fake_stdout.buffer.getvalue(), b"ERR\nresponse_too_large\n\n")
        self.assertEqual(response.read_sizes, [])

    def test_real_worker_rejects_invalid_input_without_network(self) -> None:
        wire = _execute_worker(
            bytearray(b"not-json"),
            deadline=time.monotonic() + 5,
            response_limit=1024,
        )
        self.assertEqual(wire, b"ERR\ninvalid_worker_request\n\n")

    def test_worker_deadline_kills_child_process_group_and_caps_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            interpreter = os.path.realpath(os.sys.executable)
            pid_file = root / "child.pid"
            slow_script = (
                "import os, pathlib, time\n"
                f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
                "time.sleep(30)\n"
            )
            real_popen = subprocess.Popen

            def run_script(script: str, **kwargs):
                return real_popen([interpreter, "-I", "-S", "-c", script], **kwargs)

            with mock.patch.object(_provider_transport.subprocess, "Popen", side_effect=lambda _args, **kwargs: run_script(slow_script, **kwargs)):
                error = self.assert_code(
                    "provider_timeout",
                    lambda: _execute_worker(
                        bytearray(b"synthetic input"),
                        deadline=time.monotonic() + 1,
                        response_limit=1024,
                    ),
                )
            self.assertEqual(error.code, "provider_timeout")
            child_pid = int(pid_file.read_text(encoding="ascii"))
            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)

            flood_script = "import os, time\nos.write(1, b'x' * 100000)\ntime.sleep(30)\n"
            with mock.patch.object(_provider_transport.subprocess, "Popen", side_effect=lambda _args, **kwargs: run_script(flood_script, **kwargs)):
                self.assert_code(
                    "provider_output_limit",
                    lambda: _execute_worker(
                        bytearray(b"synthetic input"),
                        deadline=time.monotonic() + 5,
                        response_limit=64,
                    ),
                )


if __name__ == "__main__":
    unittest.main()
