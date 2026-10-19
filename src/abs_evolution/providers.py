"""Candidate proposal providers.

The offline file provider is usable immediately. The Codex CLI provider fails
closed until the CLI offers a verifiable way to remove every model-side tool;
read-only sandboxing alone does not establish that boundary.
"""

from __future__ import annotations

import fnmatch
import json
import math
import os
import re
import stat
import time
from pathlib import Path, PurePosixPath
from typing import Any

from .contracts import ProposalResult, canonical_json
from .errors import EvolutionError


_REQUEST_KEYS = {
    "schema_version", "objective", "baseline_id", "files",
    "training_examples", "allowed_mutation_paths", "max_changes",
}
_EXAMPLE_KEYS = {"id", "input", "expected", "actual", "passed"}
_PROPOSAL_KEYS = {"summary", "kind", "changes"}
_CHANGE_KEYS = {"path", "content"}
_KINDS = {"skill", "tool", "workflow", "memory", "optimizer"}

_MAX_REQUEST_BYTES = 4 * 1024 * 1024
_MAX_OUTPUT_BYTES = 64 * 1024 * 1024
_MAX_FILE_BYTES = 2 * 1024 * 1024
_MAX_TOTAL_FILES = 128
_MAX_TRAINING_EXAMPLES = 100
_MAX_PATHS = 128
_MAX_CHANGES = 128
_MAX_JSON_DEPTH = 32
_MAX_JSON_NODES = 100_000
_MAX_TIMEOUT_SECONDS = 3600.0
_MAX_PATH_BYTES = 512
_MAX_SUMMARY_BYTES = 8192

_EXCLUDED_COMPONENTS = {
    ".git", ".hg", ".svn", ".codex", ".agents", ".ssh", ".aws",
    ".azure", ".config", ".cache", "cache", "caches", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".venv",
    "venv", "env", "node_modules", "runtime", "runtimes", "tmp", "temp",
    "logs", "log", "sessions", "secrets", "secret", "credentials",
    "credential", "private", "private_keys",
}
_EXCLUDED_NAMES = {
    ".netrc", ".npmrc", ".pypirc", "credentials", "credentials.json",
    "known_hosts", "authorized_keys", "id_rsa", "id_ed25519",
}
_EXCLUDED_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".keystore")
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(?:api[_-]?key|secret(?:[_-]?key)?|access[_-]?token|"
    r"refresh[_-]?token|password|passwd|client[_-]?secret)\s*[:=]\s*"
    r"(?:b?['\"][^'\"\r\n]{8,}['\"]|[^\s#]{16,})"
)
_SECRET_TOKEN = re.compile(
    r"(?:AKIA[0-9A-Z]{16}|gh[opsu]_[A-Za-z0-9_]{20,}|"
    r"xox[baprs]-[A-Za-z0-9-]{16,}|sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----)"
)


def _raise(code: str, message: str) -> None:
    raise EvolutionError(code, message)


def _is_excluded_path(value: str) -> bool:
    parts = value.split("/")
    if any(part.casefold() in _EXCLUDED_COMPONENTS for part in parts):
        return True
    name = parts[-1].casefold()
    return (
        name in _EXCLUDED_NAMES
        or name.startswith(".env")
        or name.endswith(_EXCLUDED_SUFFIXES)
        or any(token in name for token in ("credential", "secret", "access_token", "refresh_token"))
    )


def _safe_relative_path(
    value: Any,
    *,
    pattern: bool = False,
    error_code: str = "invalid_provider_request",
) -> str:
    if not isinstance(value, str) or not value:
        _raise(error_code, "Provider paths must be nonempty relative paths")
    try:
        encoded = value.encode("utf-8", "strict")
    except UnicodeError as exc:
        raise EvolutionError(error_code, "Provider paths must be valid UTF-8") from exc
    if (
        len(encoded) > _MAX_PATH_BYTES
        or "\x00" in value
        or "\\" in value
        or ":" in value
        or value.startswith("/")
    ):
        _raise(error_code, "Provider paths must be safe relative paths")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        _raise(error_code, "Provider paths must be canonical and remain relative")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        _raise(error_code, "Provider paths must be canonical and remain relative")
    if not pattern and _is_excluded_path(value):
        _raise("provider_sensitive_path", "Credential, runtime, and cache paths are not provider inputs")
    if pattern and _is_excluded_path(value.replace("*", "x").replace("?", "x")):
        _raise("provider_sensitive_path", "Credential, runtime, and cache paths are not provider permissions")
    return value


def _validate_json_value(
    value: Any,
    *,
    depth: int = 0,
    budget: list[int] | None = None,
    error_code: str = "invalid_provider_request",
) -> None:
    if budget is None:
        budget = [_MAX_JSON_NODES]
    budget[0] -= 1
    if budget[0] < 0 or depth > _MAX_JSON_DEPTH:
        _raise(error_code, "Provider JSON exceeds the structural limits")
    if value is None or type(value) is bool:
        return
    if type(value) is str:
        try:
            value.encode("utf-8", "strict")
        except UnicodeError as exc:
            raise EvolutionError(error_code, "Provider JSON strings must be valid UTF-8") from exc
        return
    if type(value) is int:
        if value.bit_length() > 256:
            _raise(error_code, "Provider JSON integer is too large")
        return
    if type(value) is float:
        if not math.isfinite(value):
            _raise(error_code, "Provider JSON numbers must be finite")
        return
    if type(value) is list:
        for item in value:
            _validate_json_value(item, depth=depth + 1, budget=budget, error_code=error_code)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                _raise(error_code, "Provider JSON object keys must be strings")
            _validate_json_value(key, depth=depth + 1, budget=budget, error_code=error_code)
            _validate_json_value(item, depth=depth + 1, budget=budget, error_code=error_code)
        return
    _raise(error_code, "Provider data must contain only JSON-compatible values")


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
    return left == right


def _validate_request(request: Any) -> tuple[dict[str, Any], dict[str, str], tuple[str, ...], int]:
    if not isinstance(request, dict) or set(request) != _REQUEST_KEYS:
        _raise("invalid_provider_request", "Provider request does not match schema version 1")
    _validate_json_value(request)
    if type(request["schema_version"]) is not int or request["schema_version"] != 1:
        _raise("invalid_provider_request", "Provider request schema version is unsupported")
    for field, limit in (("objective", 8192), ("baseline_id", 256)):
        value = request[field]
        if not isinstance(value, str) or not value.strip() or len(value.encode("utf-8")) > limit:
            _raise("invalid_provider_request", f"Provider {field} must be nonempty bounded text")

    files = request["files"]
    if not isinstance(files, dict) or len(files) > _MAX_TOTAL_FILES:
        _raise("invalid_provider_request", "Provider files must be a bounded path-to-text object")
    file_bytes = 0
    for raw_path, content in files.items():
        _safe_relative_path(raw_path)
        if not isinstance(content, str) or "\x00" in content:
            _raise("invalid_provider_request", "Provider files must contain UTF-8 text")
        try:
            size = len(content.encode("utf-8", "strict"))
        except UnicodeError as exc:
            raise EvolutionError("invalid_provider_request", "Provider files must contain UTF-8 text") from exc
        if size > _MAX_FILE_BYTES:
            _raise("invalid_provider_request", "A provider input file exceeds the size limit")
        if _SECRET_ASSIGNMENT.search(content) or _SECRET_TOKEN.search(content):
            _raise("provider_sensitive_input", "Credential-like key material is not provider input")
        file_bytes += size

    examples = request["training_examples"]
    if not isinstance(examples, list) or len(examples) > _MAX_TRAINING_EXAMPLES:
        _raise("invalid_provider_request", "Training examples must be a bounded list")
    for example in examples:
        if not isinstance(example, dict) or set(example) != _EXAMPLE_KEYS:
            _raise("invalid_provider_request", "Training examples must match the documented shape")
        if not isinstance(example["id"], str) or not example["id"].strip() or len(example["id"].encode("utf-8")) > 256:
            _raise("invalid_provider_request", "Training example IDs must be nonempty bounded text")
        if type(example["passed"]) is not bool:
            _raise("invalid_provider_request", "Training example passed values must be boolean")
        if example["passed"] != _exact_json_equal(example["actual"], example["expected"]):
            _raise("invalid_provider_request", "Training example outcome is inconsistent with its values")

    patterns = request["allowed_mutation_paths"]
    if not isinstance(patterns, list) or not patterns or len(patterns) > _MAX_PATHS:
        _raise("invalid_provider_request", "Allowed mutation paths must be a bounded nonempty list")
    checked_patterns = tuple(_safe_relative_path(item, pattern=True) for item in patterns)
    if len(set(checked_patterns)) != len(checked_patterns):
        _raise("invalid_provider_request", "Allowed mutation paths must not contain duplicates")

    max_changes = request["max_changes"]
    if type(max_changes) is not int or not 1 <= max_changes <= _MAX_CHANGES:
        _raise("invalid_provider_request", "max_changes must be a positive bounded integer")

    try:
        serialized = canonical_json(request).encode("utf-8", "strict")
    except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError) as exc:
        raise EvolutionError("invalid_provider_request", "Provider request is not finite JSON") from exc
    if len(serialized) > _MAX_REQUEST_BYTES or file_bytes > _MAX_REQUEST_BYTES:
        _raise("invalid_provider_request", "Provider request exceeds the size limit")
    # JSON round-tripping isolates the accepted payload from caller mutation.
    copied = json.loads(serialized.decode("utf-8"))
    return copied, copied["files"], checked_patterns, max_changes


def _validate_call_limits(timeout_seconds: Any, max_output_bytes: Any) -> tuple[float, int]:
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or not 0 < timeout_seconds <= _MAX_TIMEOUT_SECONDS
    ):
        _raise("invalid_provider_limits", "Provider timeout must be positive and bounded")
    if (
        type(max_output_bytes) is not int
        or not 0 < max_output_bytes <= _MAX_OUTPUT_BYTES
    ):
        _raise("invalid_provider_limits", "Provider output limit must be a positive bounded integer")
    return float(timeout_seconds), max_output_bytes


def _path_matches(path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def _validate_proposal(
    value: Any,
    *,
    files: dict[str, str],
    patterns: tuple[str, ...],
    max_changes: int,
    max_output_bytes: int,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _PROPOSAL_KEYS:
        _raise("invalid_proposal", "Proposal must contain exactly summary, kind, and changes")
    _validate_json_value(value, error_code="invalid_proposal")
    summary = value["summary"]
    if not isinstance(summary, str) or not summary.strip():
        _raise("invalid_proposal", "Proposal summary must be nonempty text")
    if len(summary.encode("utf-8", "strict")) > min(_MAX_SUMMARY_BYTES, max_output_bytes):
        _raise("invalid_proposal", "Proposal summary exceeds the size limit")
    kind = value["kind"]
    if not isinstance(kind, str) or kind not in _KINDS:
        _raise("invalid_proposal", "Proposal kind is unsupported")
    changes = value["changes"]
    if not isinstance(changes, list) or not changes or len(changes) > min(max_changes, _MAX_CHANGES):
        _raise("invalid_proposal", "Proposal changes must be a bounded nonempty list")

    seen: set[str] = set()
    total_content_bytes = 0
    for change in changes:
        if not isinstance(change, dict) or set(change) != _CHANGE_KEYS:
            _raise("invalid_proposal", "Each proposal change must contain exactly path and content")
        path = _safe_relative_path(change["path"], error_code="invalid_proposal")
        if path in seen:
            _raise("invalid_proposal", "Proposal paths must not be duplicated")
        seen.add(path)
        if not _path_matches(path, patterns):
            _raise("invalid_proposal", "Proposal path is outside the allowed mutation paths")
        content = change["content"]
        if content is not None:
            if not isinstance(content, str) or "\x00" in content:
                _raise("invalid_proposal", "Proposal content must be UTF-8 text or null")
            try:
                content_size = len(content.encode("utf-8", "strict"))
            except UnicodeError as exc:
                raise EvolutionError("invalid_proposal", "Proposal content must be valid UTF-8") from exc
            if content_size > _MAX_FILE_BYTES:
                _raise("invalid_proposal", "A proposal file exceeds the size limit")
            if _SECRET_ASSIGNMENT.search(content) or _SECRET_TOKEN.search(content):
                _raise("provider_sensitive_output", "Credential-like key material is not valid proposal content")
            if path in files and content == files[path]:
                _raise("provider_noop", "Proposal contains content identical to the selected baseline file")
            total_content_bytes += content_size
        elif path not in files:
            _raise("provider_noop", "Proposal deletes a path absent from the selected baseline files")
    if total_content_bytes > max_output_bytes:
        _raise("invalid_proposal", "Proposal content exceeds the output limit")
    try:
        encoded = canonical_json(value).encode("utf-8", "strict")
    except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError) as exc:
        raise EvolutionError("invalid_proposal", "Proposal is not finite JSON") from exc
    if len(encoded) > max_output_bytes:
        _raise("invalid_proposal", "Proposal exceeds the output limit")
    return value


def _strict_json_loads(text: str) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result

    def reject_constant(_value: str) -> None:
        raise ValueError("non-finite JSON number")

    return json.loads(text, object_pairs_hook=object_pairs, parse_constant=reject_constant)


class CodexProvider:
    """Fail-closed Codex adapter pending a verifiable zero-tool CLI mode.

    The current CLI feature switches disable important capabilities but do not
    provide a general, verifiable allowed-tools list. In particular, the core
    tool planner can register apply-patch from model metadata independently of
    the shell feature; read-only sandboxing is not a proof that every model-side
    reader is absent. Nor can this adapter prove that a persistent daemon will
    not be started by a future CLI version. It therefore starts no Codex process
    until that boundary is documented and verifiable; it never falls back to a
    broad coding session.
    """

    def __init__(self, model: str | None = None):
        if model is not None:
            if not isinstance(model, str) or not model.strip():
                _raise("invalid_provider_model", "Model name must be nonempty bounded text")
            try:
                model_bytes = model.encode("utf-8", "strict")
            except UnicodeError as exc:
                raise EvolutionError("invalid_provider_model", "Model name must be valid UTF-8") from exc
            if len(model_bytes) > 128:
                _raise("invalid_provider_model", "Model name must be nonempty bounded text")
        self.model = model

    def propose(
        self,
        request: dict[str, Any],
        *,
        timeout_seconds: float,
        max_output_bytes: int,
    ) -> ProposalResult:
        _validate_request(request)
        _validate_call_limits(timeout_seconds, max_output_bytes)
        _raise(
            "provider_blocked",
            "Codex CLI does not expose a verifiable zero-tool policy; use an explicit reviewed offline proposal instead.",
        )


class FileProposalProvider:
    """Load and validate an explicit, reviewed proposal JSON file."""

    def __init__(self, path: Path):
        if not isinstance(path, Path):
            _raise("invalid_proposal_file", "Proposal file must be an explicit Path")
        self.path = path

    def propose(
        self,
        request: dict[str, Any],
        *,
        timeout_seconds: float,
        max_output_bytes: int,
    ) -> ProposalResult:
        started = time.monotonic()
        _request, files, patterns, max_changes = _validate_request(request)
        timeout, output_limit = _validate_call_limits(timeout_seconds, max_output_bytes)
        deadline = started + timeout
        try:
            fd = os.open(self.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except OSError as exc:
            raise EvolutionError("invalid_proposal_file", "Proposal file is unavailable or unsafe") from exc
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode):
                _raise("invalid_proposal_file", "Proposal file must be a regular file")
            if metadata.st_size > output_limit:
                _raise("provider_output_limit", "Proposal file exceeds the output limit")
            with os.fdopen(fd, "rb", closefd=True) as stream:
                fd = -1
                chunks: list[bytes] = []
                total = 0
                while total <= output_limit:
                    if time.monotonic() >= deadline:
                        _raise("provider_timeout", "Proposal file exceeded the provider time limit")
                    chunk = stream.read(min(65536, output_limit + 1 - total))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total += len(chunk)
                raw = b"".join(chunks)
        finally:
            if fd >= 0:
                os.close(fd)
        if len(raw) > output_limit:
            _raise("provider_output_limit", "Proposal file exceeds the output limit")
        try:
            text = raw.decode("utf-8", "strict")
            value = _strict_json_loads(text)
        except (UnicodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
            raise EvolutionError("invalid_proposal_file", "Proposal file must be valid UTF-8 JSON") from exc
        proposal = _validate_proposal(
            value,
            files=files,
            patterns=patterns,
            max_changes=max_changes,
            max_output_bytes=output_limit,
        )
        if time.monotonic() >= deadline:
            _raise("provider_timeout", "Proposal file exceeded the provider time limit")
        return ProposalResult(
            value=proposal,
            duration_seconds=max(0.0, time.monotonic() - started),
            input_tokens=None,
            output_tokens=None,
        )
