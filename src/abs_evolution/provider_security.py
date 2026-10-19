"""Shared secret-scanning helpers for external provider boundaries."""

from __future__ import annotations

import re
from typing import Any

from .providers import _SECRET_ASSIGNMENT, _SECRET_TOKEN


_SENSITIVE_FIELD_LABEL = re.compile(
    r"(?i)(?:^|_)(?:api[_-]?key|api[_-]?secret|secret(?:[_-]?key)?|access[_-]?token|"
    r"refresh[_-]?token|client[_-]?secret|private[_-]?key|password|passwd|credential|"
    r"credentials|authorization|bearer|token)(?:$|_)"
)


def _has_nonempty_value(value: Any) -> bool:
    if value is None:
        return False
    if type(value) is bool:
        return value
    if type(value) in {str, list, dict}:
        return bool(value)
    if type(value) in {int, float}:
        return value != 0
    return True


def _is_sensitive_field_label(value: str) -> bool:
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", normalized)
    return _SENSITIVE_FIELD_LABEL.search(normalized) is not None


def contains_sensitive_material(value: Any, configured_api_key: str | None = None) -> bool:
    """Return whether JSON-like data contains credentials or credential fields.

    Traversal is iterative so untrusted nested payloads cannot exhaust the
    Python call stack. The function intentionally does not return matched data.
    """
    pending: list[Any] = [value]
    while pending:
        item = pending.pop()
        if type(item) is str:
            if (
                (configured_api_key is not None and configured_api_key in item)
                or _SECRET_ASSIGNMENT.search(item)
                or _SECRET_TOKEN.search(item)
            ):
                return True
        elif type(item) is dict:
            for key, child in item.items():
                if type(key) is str:
                    if (
                        configured_api_key is not None and configured_api_key in key
                    ) or _SECRET_ASSIGNMENT.search(key) or _SECRET_TOKEN.search(key):
                        return True
                    if _is_sensitive_field_label(key) and _has_nonempty_value(child):
                        return True
                    pending.append(key)
                pending.append(child)
        elif type(item) is list:
            pending.extend(item)
    return False
