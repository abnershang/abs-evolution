"""Isolated stdlib HTTP worker for the OpenRouter proposal provider.

This file is executed directly with ``python -I -S``. Its only input is a
bounded JSON envelope on stdin and its only output is a bounded binary frame
on stdout. It has no configurable URL, proxy, redirect, retry, or logging path.
"""

from __future__ import annotations

import json
import math
import socket
import ssl
import sys
import urllib.error
import urllib.request


ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MAX_API_KEY_BYTES = 4096
MAX_BODY_BYTES = 4 * 1024 * 1024 + 32 * 1024
MAX_TIMEOUT_SECONDS = 3600.0
MAX_RESPONSE_BYTES = 65 * 1024 * 1024
MAX_WORKER_INPUT_BYTES = 32 * 1024 * 1024


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ValueError("non-finite number")


def _error_frame(code: str, status: int | None = None) -> bytes:
    if status is None:
        return f"ERR\n{code}\n\n".encode("ascii")
    if type(status) is not int or not 100 <= status <= 599:
        status = 0
    return f"ERR\n{code}\n{status}\n".encode("ascii")


def _read_envelope() -> tuple[str, bytes, float, int] | None:
    raw = sys.stdin.buffer.read(MAX_WORKER_INPUT_BYTES + 1)
    if not raw or len(raw) > MAX_WORKER_INPUT_BYTES:
        return None
    try:
        envelope = json.loads(
            raw.decode("utf-8", "strict"),
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError, RecursionError):
        return None
    if not isinstance(envelope, dict) or set(envelope) != {
        "api_key", "body", "timeout_seconds", "max_response_bytes"
    }:
        return None

    api_key = envelope["api_key"]
    body = envelope["body"]
    timeout = envelope["timeout_seconds"]
    response_limit = envelope["max_response_bytes"]
    if not isinstance(api_key, str) or not api_key:
        return None
    try:
        key_bytes = api_key.encode("ascii", "strict")
    except UnicodeError:
        return None
    if len(key_bytes) > MAX_API_KEY_BYTES or any(not 33 <= byte <= 126 for byte in key_bytes):
        return None
    if not isinstance(body, str):
        return None
    try:
        body_bytes = body.encode("utf-8", "strict")
    except UnicodeError:
        return None
    if not body_bytes or len(body_bytes) > MAX_BODY_BYTES:
        return None
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or not 0 < timeout <= MAX_TIMEOUT_SECONDS
    ):
        return None
    if type(response_limit) is not int or not 1 <= response_limit <= MAX_RESPONSE_BYTES:
        return None
    return api_key, body_bytes, float(timeout), response_limit


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


def _send(api_key: str, body: bytes, timeout: float, response_limit: int) -> bytes:
    context = ssl.create_default_context()
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _NoRedirect(),
        urllib.request.HTTPSHandler(context=context),
    )
    request = urllib.request.Request(
        ENDPOINT,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Accept-Encoding": "identity",
        },
        method="POST",
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            status = response.getcode()
            if status != 200:
                return _error_frame("http_status", status)
            content_type = response.headers.get("Content-Type", "")
            if content_type.split(";", 1)[0].strip().casefold() != "application/json":
                return _error_frame("invalid_content_type")
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    declared_length = int(content_length)
                except (TypeError, ValueError):
                    return _error_frame("invalid_http_response")
                if declared_length < 0:
                    return _error_frame("invalid_http_response")
                if declared_length > response_limit:
                    return _error_frame("response_too_large")
            response_body = response.read(response_limit + 1)
            if len(response_body) > response_limit:
                return _error_frame("response_too_large")
            return b"OK\n" + response_body
    except urllib.error.HTTPError as exc:
        status = exc.code
        try:
            exc.close()
        except Exception:
            pass
        return _error_frame("http_status", status)
    except (TimeoutError, socket.timeout):
        return _error_frame("network_timeout")
    except Exception:
        # Never return exception text: HTTP client errors can contain sensitive
        # request details, and remote response bodies are untrusted.
        return _error_frame("network_error")


def main() -> int:
    try:
        envelope = _read_envelope()
        if envelope is None:
            output = _error_frame("invalid_worker_request")
        else:
            api_key, body, timeout, response_limit = envelope
            output = _send(api_key, body, timeout, response_limit)
            del api_key
    except BaseException:
        # Suppress tracebacks so secrets or provider diagnostics never escape
        # through stderr if an unexpected worker failure occurs.
        output = _error_frame("worker_error")
    try:
        sys.stdout.buffer.write(output)
        sys.stdout.buffer.flush()
    except BaseException:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
