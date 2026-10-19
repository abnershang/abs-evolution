"""Trusted local registry for immutable experiment snapshots and evidence."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import stat
import tempfile
from typing import Any, Iterator
import uuid

from .contracts import Policy, canonical_json, fingerprint
from .errors import EvolutionError


_SCHEMA_VERSION = "1"
_SNAPSHOT_ID = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SAFE_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_KINDS = {"skill", "tool", "workflow", "memory", "optimizer"}
_EVALUATION_STATUSES = {"passed", "quality_failed", "infrastructure_error", "blocked"}
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
_MAX_PROVENANCE_BYTES = 4096
_MAX_EVIDENCE_BYTES = 262144
_MAX_STATUS_ITEMS = 1000


def _error(code: str, message: str) -> None:
    raise EvolutionError(code, message)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _json(value: Any, code: str = "invalid_data") -> str:
    try:
        return canonical_json(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EvolutionError(code, "Value must be finite JSON-compatible data") from exc


def _decode_json(value: str) -> Any:
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError) as exc:
        raise EvolutionError("registry_corrupt", "Stored JSON is invalid") from exc


def _validate_kind(kind: str) -> str:
    if not isinstance(kind, str) or kind not in _KINDS:
        _error("invalid_kind", "Snapshot kind must be skill, tool, workflow, memory, or optimizer")
    return kind


def _validate_snapshot_id(snapshot_id: str) -> str:
    if not isinstance(snapshot_id, str) or not _SNAPSHOT_ID.fullmatch(snapshot_id):
        _error("invalid_snapshot_id", "Snapshot ID must be a lowercase SHA-256 digest")
    return snapshot_id


def _validate_identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value) or value in {".", ".."} or "://" in value:
        _error("invalid_identifier", f"{label} must be a bounded opaque identifier")
    return value


def _validate_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        _error("unsafe_path", "Snapshot paths must be nonempty bounded relative paths")
    try:
        value.encode("utf-8", "strict")
    except UnicodeError as exc:
        raise EvolutionError("unsafe_path", "Snapshot paths must be valid UTF-8") from exc
    if "\\" in value or "\x00" in value or ":" in value:
        _error("unsafe_path", "Snapshot path contains an unsafe character")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(part in {"", ".", ".."} for part in value.split("/")):
        _error("unsafe_path", "Snapshot paths must be canonical and remain relative")
    return value


def _is_excluded(relative: str) -> bool:
    parts = relative.split("/")
    if any(part.casefold() in _EXCLUDED_COMPONENTS for part in parts):
        return True
    name = parts[-1].casefold()
    if name in _EXCLUDED_NAMES or name.startswith(".env"):
        return True
    if name.endswith(_EXCLUDED_SUFFIXES):
        return True
    return any(token in name for token in ("credential", "secret", "access_token", "refresh_token"))


def _read_source_tree(source: Path, policy: Policy) -> dict[str, bytes]:
    """Read an untrusted tree through directory descriptors without following links."""
    if not isinstance(source, Path):
        _error("invalid_source", "Snapshot source must be a Path")
    if source.is_symlink():
        _error("unsafe_source", "Snapshot source cannot be a symlink")
    try:
        source = source.absolute()
        root_stat = source.stat(follow_symlinks=False)
    except OSError as exc:
        raise EvolutionError("invalid_source", "Snapshot source is unavailable") from exc
    if not stat.S_ISDIR(root_stat.st_mode):
        _error("invalid_source", "Snapshot source must be a directory")
    if source.name.casefold() in _EXCLUDED_COMPONENTS:
        _error("unsafe_source", "Credential, runtime, and cache roots cannot be imported")

    directory_flag = getattr(os, "O_DIRECTORY", 0)
    nofollow_flag = getattr(os, "O_NOFOLLOW", 0)
    try:
        root_fd = os.open(source, os.O_RDONLY | directory_flag | nofollow_flag)
    except OSError as exc:
        raise EvolutionError("unsafe_source", "Cannot safely open snapshot source") from exc

    entries: dict[str, bytes] = {}
    total_bytes = 0

    def walk(directory_fd: int, prefix: str) -> None:
        nonlocal total_bytes
        try:
            with os.scandir(directory_fd) as scanner:
                children = sorted(scanner, key=lambda item: item.name)
        except OSError as exc:
            raise EvolutionError("invalid_source", "Could not enumerate snapshot source") from exc
        for entry in children:
            name = entry.name
            try:
                name.encode("utf-8", "strict")
            except UnicodeError as exc:
                raise EvolutionError("unsafe_path", "Snapshot paths must be valid UTF-8") from exc
            relative = f"{prefix}/{name}" if prefix else name
            _validate_relative_path(relative)
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise EvolutionError("invalid_source", "Could not inspect snapshot source entry") from exc
            if stat.S_ISLNK(info.st_mode):
                _error("unsafe_source", f"Symlink is not allowed in snapshot source: {relative}")
            if _is_excluded(relative):
                continue
            if stat.S_ISDIR(info.st_mode):
                try:
                    child_fd = os.open(name, os.O_RDONLY | directory_flag | nofollow_flag, dir_fd=directory_fd)
                except OSError as exc:
                    raise EvolutionError("unsafe_source", f"Cannot safely open source directory: {relative}") from exc
                try:
                    walk(child_fd, relative)
                finally:
                    os.close(child_fd)
                continue
            if not stat.S_ISREG(info.st_mode):
                _error("unsafe_source", f"Only regular files are allowed: {relative}")
            if info.st_size > policy.max_file_bytes:
                _error("snapshot_too_large", f"File exceeds the policy limit: {relative}")
            try:
                file_fd = os.open(name, os.O_RDONLY | nofollow_flag, dir_fd=directory_fd)
            except OSError as exc:
                raise EvolutionError("unsafe_source", f"Cannot safely open source file: {relative}") from exc
            try:
                if not stat.S_ISREG(os.fstat(file_fd).st_mode):
                    _error("unsafe_source", f"Only regular files are allowed: {relative}")
                chunks: list[bytes] = []
                size = 0
                while True:
                    chunk = os.read(file_fd, min(65536, policy.max_file_bytes + 1 - size))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    size += len(chunk)
                    if size > policy.max_file_bytes:
                        _error("snapshot_too_large", f"File exceeds the policy limit: {relative}")
                content = b"".join(chunks)
            finally:
                os.close(file_fd)
            total_bytes += len(content)
            if total_bytes > policy.max_snapshot_bytes:
                _error("snapshot_too_large", "Snapshot exceeds the policy byte limit")
            entries[relative] = content
            if len(entries) > policy.max_snapshot_files:
                _error("snapshot_too_large", "Snapshot exceeds the policy file-count limit")
    try:
        walk(root_fd, "")
    finally:
        os.close(root_fd)
    return entries


def _snapshot_payload(entries: dict[str, bytes], kind: str, parent_id: str | None) -> dict[str, Any]:
    files = [
        {"path": name, "sha256": hashlib.sha256(content).hexdigest(), "size": len(content)}
        for name, content in sorted(entries.items())
    ]
    return {"schema_version": 1, "kind": kind, "parent_id": parent_id, "files": files}


def _snapshot_manifest(payload: dict[str, Any], snapshot_id: str) -> dict[str, Any]:
    return {
        **payload,
        "id": snapshot_id,
        "file_count": len(payload["files"]),
        "byte_count": sum(item["size"] for item in payload["files"]),
    }


class Registry:
    """SQLite-backed local registry; opening it never initializes missing state."""

    DB_NAME = "registry.sqlite3"

    @classmethod
    def initialize(cls, root: Path, policy: Policy) -> "Registry":
        if not isinstance(policy, Policy):
            _error("invalid_policy", "Registry initialization requires a Policy")
        root_path = cls._root_path(root)
        if root_path.exists():
            if root_path.is_symlink() or not root_path.is_dir():
                _error("invalid_root", "Registry root must be a real directory")
            try:
                if next(root_path.iterdir(), None) is not None:
                    _error("state_exists", "Registry initialization requires an empty root")
            except OSError as exc:
                raise EvolutionError("invalid_root", "Registry root cannot be inspected") from exc
        else:
            try:
                root_path.mkdir(parents=True)
            except OSError as exc:
                raise EvolutionError("invalid_root", "Registry root could not be created") from exc
        snapshots = root_path / "snapshots"
        try:
            snapshots.mkdir()
            db_path = root_path / cls.DB_NAME
            conn = sqlite3.connect(db_path, timeout=30.0, isolation_level=None)
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=FULL")
                conn.execute("PRAGMA foreign_keys=ON")
                conn.executescript(
                    """
                    BEGIN IMMEDIATE;
                    CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE snapshots (
                        id TEXT PRIMARY KEY, kind TEXT NOT NULL, parent_id TEXT,
                        manifest_json TEXT NOT NULL, provenance_json TEXT NOT NULL,
                        file_count INTEGER NOT NULL CHECK(file_count >= 0),
                        byte_count INTEGER NOT NULL CHECK(byte_count >= 0), created_at TEXT NOT NULL,
                        FOREIGN KEY(parent_id) REFERENCES snapshots(id)
                    );
                    CREATE TABLE channels (
                        name TEXT PRIMARY KEY, snapshot_id TEXT NOT NULL,
                        generation INTEGER NOT NULL CHECK(generation > 0), updated_at TEXT NOT NULL,
                        FOREIGN KEY(snapshot_id) REFERENCES snapshots(id)
                    );
                    CREATE TABLE accepted_snapshots (
                        channel TEXT NOT NULL, snapshot_id TEXT NOT NULL, accepted_at TEXT NOT NULL,
                        PRIMARY KEY(channel, snapshot_id),
                        FOREIGN KEY(channel) REFERENCES channels(name),
                        FOREIGN KEY(snapshot_id) REFERENCES snapshots(id)
                    );
                    CREATE TABLE channel_events (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT, channel TEXT NOT NULL,
                        action TEXT NOT NULL, from_id TEXT, to_id TEXT NOT NULL,
                        generation INTEGER NOT NULL, evaluation_id TEXT, created_at TEXT NOT NULL
                    );
                    CREATE TABLE tasks (
                        task_id TEXT PRIMARY KEY, channel TEXT NOT NULL, snapshot_id TEXT NOT NULL,
                        generation INTEGER NOT NULL, status TEXT NOT NULL CHECK(status IN ('active', 'ended')),
                        started_at TEXT NOT NULL, ended_at TEXT,
                        FOREIGN KEY(channel) REFERENCES channels(name),
                        FOREIGN KEY(snapshot_id) REFERENCES snapshots(id)
                    );
                    CREATE INDEX tasks_status_idx ON tasks(status, started_at);
                    CREATE TABLE receipts (id TEXT PRIMARY KEY, payload_json TEXT NOT NULL, created_at TEXT NOT NULL);
                    CREATE TABLE experiments (
                        id TEXT PRIMARY KEY, baseline_id TEXT NOT NULL, policy_digest TEXT NOT NULL,
                        status TEXT NOT NULL CHECK(status IN ('running', 'completed', 'failed', 'blocked')),
                        budget_day TEXT NOT NULL, reserved_seconds INTEGER NOT NULL CHECK(reserved_seconds > 0),
                        elapsed_seconds REAL, detail_json TEXT, created_at TEXT NOT NULL, finished_at TEXT,
                        FOREIGN KEY(baseline_id) REFERENCES snapshots(id)
                    );
                    CREATE INDEX experiments_status_idx ON experiments(status, created_at);
                    CREATE INDEX experiments_budget_idx ON experiments(budget_day);
                    CREATE TABLE holdout_uses (
                        digest TEXT PRIMARY KEY, experiment_id TEXT NOT NULL, consumed_at TEXT NOT NULL,
                        FOREIGN KEY(experiment_id) REFERENCES experiments(id)
                    );
                    CREATE TABLE evaluations (
                        id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL, candidate_id TEXT NOT NULL,
                        baseline_id TEXT NOT NULL, suite_digest TEXT NOT NULL, environment_digest TEXT NOT NULL,
                        policy_digest TEXT NOT NULL, status TEXT NOT NULL, evidence_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(experiment_id) REFERENCES experiments(id),
                        FOREIGN KEY(candidate_id) REFERENCES snapshots(id),
                        FOREIGN KEY(baseline_id) REFERENCES snapshots(id)
                    );
                    CREATE TABLE schema_info (version TEXT PRIMARY KEY);
                    INSERT INTO schema_info(version) VALUES ('1');
                    INSERT INTO settings(key, value) VALUES ('policy', 'null');
                    COMMIT;
                    """
                )
                conn.execute("UPDATE settings SET value=? WHERE key='policy'", (_json(policy.to_dict()),))
            finally:
                conn.close()
        except EvolutionError:
            raise
        except (OSError, sqlite3.Error) as exc:
            raise EvolutionError("initialize_failed", "Registry state could not be initialized") from exc
        return cls(root_path)

    @staticmethod
    def _root_path(root: Path) -> Path:
        if not isinstance(root, Path):
            _error("invalid_root", "Registry root must be an explicit Path")
        raw = root.expanduser().absolute()
        if raw.is_symlink():
            _error("invalid_root", "Registry root cannot be a symlink")
        try:
            return raw.resolve(strict=False)
        except OSError as exc:
            raise EvolutionError("invalid_root", "Registry root is invalid") from exc

    def __init__(self, root: Path):
        self.root = self._root_path(root)
        self._db_path = self.root / self.DB_NAME
        self._snapshots_path = self.root / "snapshots"
        if not self.root.is_dir() or self.root.is_symlink():
            _error("state_missing", "Registry root does not exist")
        if not self._db_path.is_file() or self._db_path.is_symlink():
            _error("state_missing", "Registry database does not exist; use initialize explicitly")
        if not self._snapshots_path.is_dir() or self._snapshots_path.is_symlink():
            _error("registry_corrupt", "Registry snapshot directory is missing or unsafe")
        try:
            conn = self._connect()
            try:
                version = conn.execute("SELECT version FROM schema_info").fetchone()
                if version is None or version["version"] != _SCHEMA_VERSION:
                    _error("unsupported_schema", "Registry schema version is unsupported")
                if conn.execute("SELECT value FROM settings WHERE key='policy'").fetchone() is None:
                    _error("registry_corrupt", "Registry policy is missing")
            finally:
                conn.close()
        except EvolutionError:
            raise
        except sqlite3.Error as exc:
            raise EvolutionError("registry_corrupt", "Registry database is missing required state") from exc

    def _connect(self, *, busy_timeout_ms: int = 30000) -> sqlite3.Connection:
        if self.root.is_symlink() or not self.root.is_dir():
            _error("unsafe_root", "Registry root is missing or unsafe")
        if self._db_path.is_symlink() or not self._db_path.is_file():
            _error("unsafe_root", "Registry database is missing or unsafe")
        if isinstance(busy_timeout_ms, bool) or not isinstance(busy_timeout_ms, int) or busy_timeout_ms < 0:
            _error("invalid_timeout", "SQLite busy timeout must be a nonnegative integer")
        try:
            conn = sqlite3.connect(self._db_path, timeout=busy_timeout_ms / 1000, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
            return conn
        except sqlite3.Error as exc:
            raise EvolutionError("registry_unavailable", "Registry database could not be opened") from exc

    @contextmanager
    def _transaction(
        self, *, write: bool = False, busy_timeout_ms: int = 30000
    ) -> Iterator[sqlite3.Connection]:
        conn = self._connect(busy_timeout_ms=busy_timeout_ms)
        try:
            conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield conn
            conn.execute("COMMIT")
        except sqlite3.OperationalError as exc:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            if "locked" in str(exc).casefold() or "busy" in str(exc).casefold():
                raise EvolutionError("registry_busy", "Registry is busy; retry the operation") from exc
            raise
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def _policy(self, conn: sqlite3.Connection) -> Policy:
        row = conn.execute("SELECT value FROM settings WHERE key='policy'").fetchone()
        if row is None:
            _error("registry_corrupt", "Registry policy is missing")
        try:
            return Policy.from_dict(_decode_json(row["value"]))
        except EvolutionError as exc:
            raise EvolutionError("registry_corrupt", "Stored registry policy is invalid") from exc

    def policy(self) -> Policy:
        with self._transaction() as conn:
            return self._policy(conn)

    def update_policy(self, policy: Policy) -> None:
        if not isinstance(policy, Policy):
            _error("invalid_policy", "Policy update requires a Policy")
        with self._transaction(write=True) as conn:
            conn.execute("UPDATE settings SET value=? WHERE key='policy'", (_json(policy.to_dict()),))

    def _snapshot_row(self, conn: sqlite3.Connection, snapshot_id: str) -> sqlite3.Row:
        _validate_snapshot_id(snapshot_id)
        row = conn.execute("SELECT * FROM snapshots WHERE id=?", (snapshot_id,)).fetchone()
        if row is None:
            _error("snapshot_not_found", "Snapshot does not exist")
        return row

    def _snapshot_dir(self, snapshot_id: str) -> Path:
        return self._snapshots_path / snapshot_id

    def _read_snapshot_tree(
        self,
        root: Path,
        *,
        max_file_bytes: int,
        max_snapshot_files: int,
        max_snapshot_bytes: int,
    ) -> dict[str, bytes]:
        limits = (max_file_bytes, max_snapshot_files, max_snapshot_bytes)
        if any(isinstance(limit, bool) or not isinstance(limit, int) or limit < 0 for limit in limits):
            _error("snapshot_corrupt", "Snapshot manifest limits are invalid")
        if root.is_symlink() or not root.is_dir():
            _error("snapshot_corrupt", "Snapshot files directory is missing or unsafe")
        try:
            root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
        except OSError as exc:
            raise EvolutionError("snapshot_corrupt", "Snapshot files directory cannot be opened safely") from exc
        entries: dict[str, bytes] = {}
        total_bytes = 0

        def walk(directory_fd: int, prefix: str) -> None:
            nonlocal total_bytes
            try:
                with os.scandir(directory_fd) as scanner:
                    children = sorted(scanner, key=lambda item: item.name)
            except OSError as exc:
                raise EvolutionError("snapshot_corrupt", "Snapshot files cannot be enumerated") from exc
            for entry in children:
                relative = f"{prefix}/{entry.name}" if prefix else entry.name
                _validate_relative_path(relative)
                try:
                    info = entry.stat(follow_symlinks=False)
                except OSError as exc:
                    raise EvolutionError("snapshot_corrupt", "Snapshot entry cannot be inspected") from exc
                if stat.S_ISLNK(info.st_mode) or _is_excluded(relative):
                    _error("snapshot_corrupt", "Snapshot contains an unsafe path")
                if stat.S_ISDIR(info.st_mode):
                    try:
                        child_fd = os.open(
                            entry.name,
                            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                            dir_fd=directory_fd,
                        )
                    except OSError as exc:
                        raise EvolutionError("snapshot_corrupt", "Snapshot directory cannot be opened safely") from exc
                    try:
                        walk(child_fd, relative)
                    finally:
                        os.close(child_fd)
                    continue
                if not stat.S_ISREG(info.st_mode) or info.st_size > max_file_bytes:
                    _error("snapshot_corrupt", "Snapshot contains a non-regular or oversized file")
                try:
                    file_fd = os.open(entry.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd)
                except OSError as exc:
                    raise EvolutionError("snapshot_corrupt", "Snapshot file cannot be opened safely") from exc
                try:
                    chunks: list[bytes] = []
                    size = 0
                    while True:
                        chunk = os.read(file_fd, min(65536, max_file_bytes + 1 - size))
                        if not chunk:
                            break
                        chunks.append(chunk)
                        size += len(chunk)
                        if size > max_file_bytes:
                            _error("snapshot_corrupt", "Snapshot file exceeds its manifest size bound")
                    content = b"".join(chunks)
                finally:
                    os.close(file_fd)
                entries[relative] = content
                total_bytes += len(content)
                if len(entries) > max_snapshot_files or total_bytes > max_snapshot_bytes:
                    _error("snapshot_corrupt", "Snapshot exceeds its manifest aggregate bounds")
        try:
            walk(root_fd, "")
        finally:
            os.close(root_fd)
        return entries

    def _verify_snapshot(self, conn: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
        snapshot_id = row["id"]
        if self._snapshots_path.is_symlink() or not self._snapshots_path.is_dir():
            _error("snapshot_corrupt", "Snapshot storage directory is missing or unsafe")
        directory = self._snapshot_dir(snapshot_id)
        if directory.is_symlink() or not directory.is_dir():
            _error("snapshot_corrupt", "Snapshot directory is missing or unsafe")
        manifest_path = directory / "manifest.json"
        files_path = directory / "files"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            _error("snapshot_corrupt", "Snapshot manifest is missing or unsafe")
        try:
            manifest_bytes = manifest_path.read_bytes()
            manifest_text = manifest_bytes.decode("utf-8")
            manifest = json.loads(manifest_text)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise EvolutionError("snapshot_corrupt", "Snapshot manifest cannot be read") from exc
        try:
            canonical_manifest = _json(manifest) + "\n"
        except EvolutionError as exc:
            raise EvolutionError("snapshot_corrupt", "Snapshot manifest contains invalid JSON values") from exc
        if not isinstance(manifest, dict) or canonical_manifest != manifest_text:
            _error("snapshot_corrupt", "Snapshot manifest is not canonical")
        if set(manifest) != {
            "schema_version", "kind", "parent_id", "files", "id", "file_count", "byte_count"
        }:
            _error("snapshot_corrupt", "Snapshot manifest fields are invalid")
        if manifest != _decode_json(row["manifest_json"]):
            _error("snapshot_corrupt", "Snapshot manifest disagrees with registry metadata")
        if manifest.get("id") != snapshot_id:
            _error("snapshot_corrupt", "Snapshot manifest ID does not match its address")
        if isinstance(manifest.get("schema_version"), bool) or manifest.get("schema_version") != 1:
            _error("snapshot_corrupt", "Snapshot manifest schema version is invalid")
        payload = {key: manifest.get(key) for key in ("schema_version", "kind", "parent_id", "files")}
        if fingerprint(payload) != snapshot_id:
            _error("snapshot_corrupt", "Snapshot manifest content hash is invalid")
        if manifest.get("kind") != row["kind"] or manifest.get("parent_id") != row["parent_id"]:
            _error("snapshot_corrupt", "Snapshot metadata disagrees with registry state")
        expected = manifest.get("files")
        if not isinstance(expected, list):
            _error("snapshot_corrupt", "Snapshot file manifest is invalid")
        expected_map: dict[str, tuple[str, int]] = {}
        for item in expected:
            if not isinstance(item, dict) or set(item) != {"path", "sha256", "size"}:
                _error("snapshot_corrupt", "Snapshot file manifest entry is invalid")
            name = _validate_relative_path(item["path"])
            if _is_excluded(name) or name in expected_map:
                _error("snapshot_corrupt", "Snapshot file manifest contains unsafe or duplicate paths")
            if not isinstance(item["sha256"], str) or not _SNAPSHOT_ID.fullmatch(item["sha256"]):
                _error("snapshot_corrupt", "Snapshot file digest is invalid")
            if isinstance(item["size"], bool) or not isinstance(item["size"], int) or item["size"] < 0:
                _error("snapshot_corrupt", "Snapshot file size is invalid")
            expected_map[name] = (item["sha256"], item["size"])
        file_count = manifest.get("file_count")
        byte_count = manifest.get("byte_count")
        if (
            isinstance(file_count, bool) or not isinstance(file_count, int) or file_count != len(expected_map)
            or isinstance(byte_count, bool) or not isinstance(byte_count, int)
            or byte_count != sum(size for _, size in expected_map.values())
            or file_count != row["file_count"] or byte_count != row["byte_count"]
        ):
            _error("snapshot_corrupt", "Snapshot aggregate counts are invalid")
        actual = self._read_snapshot_tree(
            files_path,
            max_file_bytes=max((size for _, size in expected_map.values()), default=0),
            max_snapshot_files=file_count,
            max_snapshot_bytes=byte_count,
        )
        if set(actual) != set(expected_map):
            _error("snapshot_corrupt", "Snapshot files do not match the manifest")
        for name, content in actual.items():
            digest, size = expected_map[name]
            if len(content) != size or hashlib.sha256(content).hexdigest() != digest:
                _error("snapshot_corrupt", f"Snapshot content hash mismatch: {name}")
        if file_count != len(actual) or byte_count != sum(map(len, actual.values())):
            _error("snapshot_corrupt", "Snapshot aggregate counts are invalid")
        return manifest

    def _write_snapshot(self, snapshot_id: str, entries: dict[str, bytes], manifest: dict[str, Any]) -> None:
        if self._snapshots_path.is_symlink() or not self._snapshots_path.is_dir():
            _error("unsafe_root", "Snapshot storage directory is unsafe")
        destination = self._snapshot_dir(snapshot_id)
        if destination.exists():
            _error("snapshot_orphan", "Snapshot address exists without registry metadata")
        stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=self._snapshots_path))
        renamed = False
        try:
            files_dir = stage / "files"
            files_dir.mkdir(mode=0o700)
            created_dirs: set[Path] = set()
            for relative, content in sorted(entries.items()):
                _validate_relative_path(relative)
                if _is_excluded(relative):
                    _error("unsafe_path", "Credential, runtime, and cache paths cannot be stored")
                file_path = files_dir.joinpath(*relative.split("/"))
                missing: list[Path] = []
                cursor = file_path.parent
                while cursor != files_dir and cursor not in created_dirs:
                    missing.append(cursor)
                    cursor = cursor.parent
                for directory in reversed(missing):
                    directory.mkdir(mode=0o700)
                    created_dirs.add(directory)
                fd = os.open(file_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                try:
                    view = memoryview(content)
                    while view:
                        written = os.write(fd, view)
                        view = view[written:]
                finally:
                    os.close(fd)
                os.chmod(file_path, 0o444)
            manifest_path = stage / "manifest.json"
            manifest_path.write_text(_json(manifest) + "\n", encoding="utf-8")
            os.chmod(manifest_path, 0o444)
            for directory in sorted(created_dirs, key=lambda item: len(item.parts), reverse=True):
                os.chmod(directory, 0o555)
            os.chmod(files_dir, 0o555)
            # macOS can require write permission on the directory being moved.
            # Seal its root after rename, before publishing registry metadata;
            # readers remain excluded by the enclosing write transaction.
            os.rename(stage, destination)
            renamed = True
            os.chmod(destination, 0o555)
        except OSError as exc:
            if renamed:
                self._remove_tree(destination)
            raise EvolutionError("snapshot_write_failed", "Immutable snapshot could not be written") from exc
        finally:
            if stage.exists():
                self._remove_tree(stage)

    @staticmethod
    def _remove_tree(path: Path) -> None:
        if path.is_symlink() or not path.exists():
            return
        for current, dirs, files in os.walk(path, topdown=False, followlinks=False):
            current_path = Path(current)
            for name in files:
                target = current_path / name
                if not target.is_symlink():
                    target.chmod(0o600)
            for name in dirs:
                target = current_path / name
                if not target.is_symlink():
                    target.chmod(0o700)
            current_path.chmod(0o700)
        shutil.rmtree(path)

    def _info(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"], "kind": row["kind"], "parent_id": row["parent_id"],
            "file_count": row["file_count"], "byte_count": row["byte_count"],
            "provenance": _decode_json(row["provenance_json"]), "created_at": row["created_at"],
        }

    def _store_entries(
        self,
        entries: dict[str, bytes],
        *,
        kind: str,
        parent_id: str | None,
        provenance: Any,
        candidate_changes: dict[str, str | None] | None = None,
    ) -> dict[str, Any]:
        kind = _validate_kind(kind)
        if parent_id is not None:
            _validate_snapshot_id(parent_id)
        provenance_json = _json(provenance, "invalid_provenance")
        if len(provenance_json.encode("utf-8")) > _MAX_PROVENANCE_BYTES:
            _error("invalid_provenance", "Provenance exceeds the bounded metadata limit")
        for name, content in entries.items():
            _validate_relative_path(name)
            if _is_excluded(name):
                _error("unsafe_path", "Credential, runtime, and cache paths cannot be stored")
            if not isinstance(content, bytes):
                _error("invalid_snapshot", f"Snapshot content must be bytes: {name}")
        payload = _snapshot_payload(entries, kind, parent_id)
        snapshot_id = fingerprint(payload)
        manifest = _snapshot_manifest(payload, snapshot_id)
        with self._transaction(write=True) as conn:
            policy = self._policy(conn)
            if candidate_changes is not None:
                if len(candidate_changes) > policy.max_changes:
                    _error("change_limit", "Candidate exceeds the current policy change-count limit")
                for path in candidate_changes:
                    safe_path = _validate_relative_path(path)
                    if _is_excluded(safe_path) or not any(
                        fnmatch.fnmatchcase(safe_path, pattern)
                        for pattern in policy.allowed_mutation_paths
                    ):
                        _error("path_not_allowed", f"Current policy does not allow candidate path: {safe_path}")
            if (
                len(entries) > policy.max_snapshot_files
                or sum(map(len, entries.values())) > policy.max_snapshot_bytes
                or any(len(content) > policy.max_file_bytes for content in entries.values())
            ):
                _error("snapshot_too_large", "Snapshot exceeds current policy limits")
            if parent_id is not None:
                parent = self._snapshot_row(conn, parent_id)
                self._verify_snapshot(conn, parent)
            existing = conn.execute("SELECT * FROM snapshots WHERE id=?", (snapshot_id,)).fetchone()
            if existing is not None:
                self._verify_snapshot(conn, existing)
                return self._info(existing)
            aggregate = conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(byte_count), 0) AS bytes FROM snapshots"
            ).fetchone()
            # Aggregate store admission uses immutable snapshot payload bytes;
            # manifest/database overhead remains separately bounded by metadata schema.
            if aggregate["n"] + 1 > policy.max_snapshots:
                _error("snapshot_count_limit", "Registry snapshot count limit has been reached")
            if aggregate["bytes"] + manifest["byte_count"] > policy.max_store_bytes:
                _error("store_size_limit", "Registry aggregate snapshot byte limit would be exceeded")
            if self._snapshot_dir(snapshot_id).exists():
                _error("snapshot_orphan", "Snapshot address exists without registry metadata")
            self._write_snapshot(snapshot_id, entries, manifest)
            timestamp = _now()
            conn.execute(
                "INSERT INTO snapshots(id, kind, parent_id, manifest_json, provenance_json, file_count, byte_count, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    snapshot_id, kind, parent_id, _json(manifest), provenance_json,
                    manifest["file_count"], manifest["byte_count"], timestamp,
                ),
            )
            row = conn.execute("SELECT * FROM snapshots WHERE id=?", (snapshot_id,)).fetchone()
            return self._info(row)

    def import_snapshot(
        self,
        source: Path,
        *,
        kind: str = "skill",
        parent_id: str | None = None,
        provenance: Any = None,
    ) -> dict[str, Any]:
        entries = _read_source_tree(source, self.policy())
        return self._store_entries(entries, kind=kind, parent_id=parent_id, provenance=provenance)

    def create_candidate(
        self,
        parent_id: str,
        changes: dict[str, str | None],
        *,
        kind: str = "skill",
        provenance: Any = None,
    ) -> dict[str, Any]:
        parent_id = _validate_snapshot_id(parent_id)
        kind = _validate_kind(kind)
        if not isinstance(changes, dict) or not changes:
            _error("invalid_changes", "Candidate changes must be a nonempty path-to-content object")
        with self._transaction() as conn:
            policy = self._policy(conn)
            parent = self._snapshot_row(conn, parent_id)
            parent_manifest = self._verify_snapshot(conn, parent)
            current = self._read_snapshot_tree(
                self._snapshot_dir(parent_id) / "files",
                max_file_bytes=max((item["size"] for item in parent_manifest["files"]), default=0),
                max_snapshot_files=parent_manifest["file_count"],
                max_snapshot_bytes=parent_manifest["byte_count"],
            )
            expected = {
                item["path"]: (item["sha256"], item["size"])
                for item in parent_manifest["files"]
            }
            if set(current) != set(expected) or any(
                len(content) != expected[name][1]
                or hashlib.sha256(content).hexdigest() != expected[name][0]
                for name, content in current.items()
            ):
                _error("snapshot_corrupt", "Snapshot content changed while preparing a candidate")
        if len(changes) > policy.max_changes:
            _error("change_limit", "Candidate exceeds the policy change-count limit")
        normalized: dict[str, str | None] = {}
        for path, content in changes.items():
            safe_path = _validate_relative_path(path)
            if _is_excluded(safe_path):
                _error("unsafe_path", "Credential, runtime, and cache paths cannot be changed")
            if not any(fnmatch.fnmatchcase(safe_path, pattern) for pattern in policy.allowed_mutation_paths):
                _error("path_not_allowed", f"Policy does not allow candidate path: {safe_path}")
            if content is not None and not isinstance(content, str):
                _error("invalid_changes", "Candidate replacement content must be UTF-8 text or null")
            normalized[safe_path] = content
        changed = False
        for path, content in normalized.items():
            if content is None:
                changed = changed or path in current
                current.pop(path, None)
                continue
            try:
                data = content.encode("utf-8", "strict")
            except UnicodeError as exc:
                raise EvolutionError("invalid_changes", "Candidate content must be valid UTF-8") from exc
            if len(data) > policy.max_file_bytes:
                _error("snapshot_too_large", f"Candidate file exceeds policy limit: {path}")
            changed = changed or current.get(path) != data
            current[path] = data
        if not changed:
            _error("no_op_candidate", "Candidate proposal does not change snapshot content")
        if not current:
            _error("empty_candidate", "Candidate snapshots must contain at least one file")
        if len(current) > policy.max_snapshot_files or sum(map(len, current.values())) > policy.max_snapshot_bytes:
            _error("snapshot_too_large", "Candidate exceeds snapshot policy limits")
        return self._store_entries(
            current,
            kind=kind,
            parent_id=parent_id,
            provenance=provenance,
            candidate_changes=normalized,
        )

    def snapshot_path(self, snapshot_id: str) -> Path:
        with self._transaction() as conn:
            row = self._snapshot_row(conn, snapshot_id)
            self._verify_snapshot(conn, row)
        return self._snapshot_dir(snapshot_id) / "files"

    def snapshot_info(self, snapshot_id: str) -> dict[str, Any]:
        with self._transaction() as conn:
            row = self._snapshot_row(conn, snapshot_id)
            self._verify_snapshot(conn, row)
            return self._info(row)

    def list_snapshots(self) -> list[dict[str, Any]]:
        with self._transaction() as conn:
            rows = conn.execute("SELECT * FROM snapshots ORDER BY created_at, id").fetchall()
            return [self._info(row) for row in rows]

    @staticmethod
    def _channel_name(channel: str) -> str:
        if not isinstance(channel, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", channel):
            _error("invalid_channel", "Channel name must be a bounded identifier")
        return channel

    @staticmethod
    def _channel_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "channel": row["name"],
            "snapshot_id": row["snapshot_id"],
            "generation": row["generation"],
            "updated_at": row["updated_at"],
        }

    def bootstrap(self, snapshot_id: str, *, channel: str = "stable") -> dict[str, Any]:
        _validate_snapshot_id(snapshot_id)
        channel = self._channel_name(channel)
        with self._transaction(write=True) as conn:
            snapshot = self._snapshot_row(conn, snapshot_id)
            self._verify_snapshot(conn, snapshot)
            exists = conn.execute("SELECT 1 FROM channels WHERE name=?", (channel,)).fetchone()
            if exists is not None:
                _error("channel_exists", "Bootstrap never overwrites an existing channel")
            timestamp = _now()
            conn.execute(
                "INSERT INTO channels(name, snapshot_id, generation, updated_at) VALUES (?, ?, 1, ?)",
                (channel, snapshot_id, timestamp),
            )
            conn.execute(
                "INSERT INTO accepted_snapshots(channel, snapshot_id, accepted_at) VALUES (?, ?, ?)",
                (channel, snapshot_id, timestamp),
            )
            conn.execute(
                "INSERT INTO channel_events(channel, action, from_id, to_id, generation, evaluation_id, created_at) "
                "VALUES (?, 'bootstrap', NULL, ?, 1, NULL, ?)",
                (channel, snapshot_id, timestamp),
            )
            row = conn.execute("SELECT * FROM channels WHERE name=?", (channel,)).fetchone()
            return self._channel_dict(row)

    def channel(self, channel: str = "stable") -> dict[str, Any]:
        channel = self._channel_name(channel)
        with self._transaction() as conn:
            row = conn.execute("SELECT * FROM channels WHERE name=?", (channel,)).fetchone()
            if row is None:
                _error("channel_not_found", "Channel has not been bootstrapped")
            return self._channel_dict(row)

    @staticmethod
    def _task_dict(row: sqlite3.Row, snapshot_path: Path | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "task_id": row["task_id"], "channel": row["channel"],
            "snapshot_id": row["snapshot_id"], "generation": row["generation"],
            "status": row["status"], "started_at": row["started_at"], "ended_at": row["ended_at"],
        }
        if snapshot_path is not None:
            result["snapshot_path"] = str(snapshot_path)
        return result

    def begin_task(self, task_id: str, *, channel: str = "stable") -> dict[str, Any]:
        task_id = _validate_identifier(task_id, "Task ID")
        channel = self._channel_name(channel)
        with self._transaction(write=True, busy_timeout_ms=1000) as conn:
            existing = conn.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            if existing is not None:
                if existing["status"] != "active":
                    _error("task_id_reused", "An ended task ID cannot be leased again")
                snapshot = self._snapshot_row(conn, existing["snapshot_id"])
                self._verify_snapshot(conn, snapshot)
                return self._task_dict(existing, self._snapshot_dir(existing["snapshot_id"]) / "files")
            current = conn.execute("SELECT * FROM channels WHERE name=?", (channel,)).fetchone()
            if current is None:
                _error("channel_not_found", "Channel has not been bootstrapped")
            snapshot = self._snapshot_row(conn, current["snapshot_id"])
            self._verify_snapshot(conn, snapshot)
            timestamp = _now()
            conn.execute(
                "INSERT INTO tasks(task_id, channel, snapshot_id, generation, status, started_at, ended_at) "
                "VALUES (?, ?, ?, ?, 'active', ?, NULL)",
                (task_id, channel, current["snapshot_id"], current["generation"], timestamp),
            )
            row = conn.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            return self._task_dict(row, self._snapshot_dir(row["snapshot_id"]) / "files")

    def end_task(self, task_id: str) -> dict[str, Any]:
        task_id = _validate_identifier(task_id, "Task ID")
        with self._transaction(write=True) as conn:
            row = conn.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            if row is None:
                _error("task_not_found", "Task lease does not exist")
            if row["status"] == "active":
                conn.execute(
                    "UPDATE tasks SET status='ended', ended_at=? WHERE task_id=? AND status='active'",
                    (_now(), task_id),
                )
                row = conn.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            return self._task_dict(row)

    def active_tasks(self) -> list[dict[str, Any]]:
        with self._transaction() as conn:
            rows = conn.execute("SELECT * FROM tasks WHERE status='active' ORDER BY started_at, task_id").fetchall()
            return [self._task_dict(row) for row in rows]

    @staticmethod
    def _validate_receipt(receipt: Any) -> tuple[str, str]:
        expected = {"id", "task_id", "snapshot_id", "outcome", "failure_tags", "evidence_refs"}
        optional = {"task_family", "optimization_signals"}
        if not isinstance(receipt, dict) or not expected.issubset(receipt) or set(receipt) - expected - optional:
            _error("invalid_receipt", "Receipt must contain exactly the documented fields")
        _validate_identifier(receipt["id"], "Receipt ID")
        _validate_identifier(receipt["task_id"], "Task ID")
        _validate_snapshot_id(receipt["snapshot_id"])
        if not isinstance(receipt["outcome"], str) or receipt["outcome"] not in {"accepted", "failed", "unknown"}:
            _error("invalid_receipt", "Receipt outcome must be accepted, failed, or unknown")
        for field in ("failure_tags", "evidence_refs"):
            value = receipt[field]
            if not isinstance(value, list) or len(value) > 100 or any(not isinstance(item, str) for item in value):
                _error("invalid_receipt", f"{field} must be a bounded list")
            if len(set(value)) != len(value):
                _error("invalid_receipt", f"{field} must contain unique string identifiers")
        if any(not _SAFE_TAG.fullmatch(item) for item in receipt["failure_tags"]):
            _error("invalid_receipt", "Failure tags must be safe identifiers")
        if any(
            not _IDENTIFIER.fullmatch(item) or item in {".", ".."} or "://" in item
            for item in receipt["evidence_refs"]
        ):
            _error("invalid_receipt", "Evidence references must be opaque identifiers, not URLs or paths")
        if "task_family" in receipt and (
            not isinstance(receipt["task_family"], str) or not _SAFE_TAG.fullmatch(receipt["task_family"])
        ):
            _error("invalid_receipt", "Task family must be a safe identifier")
        if "optimization_signals" in receipt:
            signals = receipt["optimization_signals"]
            if not isinstance(signals, list) or len(signals) > 16:
                _error("invalid_receipt", "Optimization signals must be a bounded list")
            seen: set[tuple[str, str]] = set()
            dimensions = {"quality", "efficiency", "method", "reuse"}
            for signal in signals:
                if not isinstance(signal, dict) or set(signal) != {"dimension", "tag"}:
                    _error("invalid_receipt", "Each optimization signal must contain exactly dimension and tag")
                dimension, tag = signal["dimension"], signal["tag"]
                if not isinstance(dimension, str) or dimension not in dimensions:
                    _error("invalid_receipt", "Optimization signal dimension is not supported")
                if not isinstance(tag, str) or not _SAFE_TAG.fullmatch(tag):
                    _error("invalid_receipt", "Optimization signal tag must be a safe identifier")
                pair = (dimension, tag)
                if pair in seen:
                    _error("invalid_receipt", "Optimization signals must be unique by dimension and tag")
                seen.add(pair)
            if signals and ("task_family" not in receipt or not receipt["evidence_refs"]):
                _error("invalid_receipt", "Optimization signals require task_family and evidence references")
        payload = _json(receipt, "invalid_receipt")
        return receipt["id"], payload

    def record_receipt(self, receipt: dict[str, Any]) -> dict[str, Any]:
        receipt_id, payload_json = self._validate_receipt(receipt)
        with self._transaction(write=True) as conn:
            policy = self._policy(conn)
            if len(payload_json.encode("utf-8")) > policy.max_receipt_bytes:
                _error("invalid_receipt", "Receipt exceeds the policy byte limit")
            task = conn.execute("SELECT * FROM tasks WHERE task_id=?", (receipt["task_id"],)).fetchone()
            if task is None or task["snapshot_id"] != receipt["snapshot_id"]:
                _error("receipt_task_mismatch", "Receipt must match a task's persistent snapshot pin")
            snapshot = self._snapshot_row(conn, receipt["snapshot_id"])
            self._verify_snapshot(conn, snapshot)
            existing = conn.execute("SELECT payload_json FROM receipts WHERE id=?", (receipt_id,)).fetchone()
            if existing is not None:
                if existing["payload_json"] != payload_json:
                    _error("receipt_id_conflict", "Receipt ID was already used for different content")
                return {"id": receipt_id, "receipt": _decode_json(existing["payload_json"]), "created": False}
            count = conn.execute("SELECT COUNT(*) AS n FROM receipts").fetchone()["n"]
            if count >= policy.max_receipts:
                _error("receipt_limit", "Registry receipt capacity has been reached")
            conn.execute(
                "INSERT INTO receipts(id, payload_json, created_at) VALUES (?, ?, ?)",
                (receipt_id, payload_json, _now()),
            )
            return {"id": receipt_id, "receipt": receipt, "created": True}

    def list_receipts(self, limit: int = 100) -> list[dict[str, Any]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            _error("invalid_limit", "Receipt limit must be a positive integer")
        with self._transaction() as conn:
            policy = self._policy(conn)
            if limit > policy.max_receipts:
                _error("invalid_limit", "Receipt limit exceeds the current policy maximum")
            rows = conn.execute("SELECT payload_json FROM receipts ORDER BY created_at, id LIMIT ?", (limit,)).fetchall()
            return [_decode_json(row["payload_json"]) for row in rows]

    @staticmethod
    def _experiment_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "baseline_id": row["baseline_id"],
            "policy_digest": row["policy_digest"],
            "status": row["status"],
            "budget_day": row["budget_day"],
            "reserved_seconds": row["reserved_seconds"],
            "elapsed_seconds": row["elapsed_seconds"],
            "detail": _decode_json(row["detail_json"]) if row["detail_json"] is not None else None,
            "created_at": row["created_at"],
            "finished_at": row["finished_at"],
        }

    def claim_experiment(self, *, baseline_id: str) -> dict[str, Any]:
        baseline_id = _validate_snapshot_id(baseline_id)
        with self._transaction(write=True) as conn:
            policy = self._policy(conn)
            baseline = self._snapshot_row(conn, baseline_id)
            self._verify_snapshot(conn, baseline)
            if policy.yield_to_foreground:
                active_task = conn.execute("SELECT task_id FROM tasks WHERE status='active' LIMIT 1").fetchone()
                if active_task is not None:
                    _error("foreground_active", "Experiment claim deferred while foreground task leases are active")
            active_claims = conn.execute(
                "SELECT COUNT(*) AS n FROM experiments WHERE status='running'"
            ).fetchone()["n"]
            if active_claims >= policy.max_parallel_experiments:
                _error("experiment_concurrency_limit", "Maximum concurrent experiment claims are already active")
            day = datetime.now(timezone.utc).date().isoformat()
            totals = conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(reserved_seconds), 0) AS seconds "
                "FROM experiments WHERE budget_day=?",
                (day,),
            ).fetchone()
            if totals["n"] >= policy.max_experiments_per_day:
                _error("daily_experiment_limit", "Daily experiment count limit has been reached")
            reserved = policy.experiment_timeout_seconds
            if totals["seconds"] + reserved > policy.max_reserved_seconds_per_day:
                _error("daily_time_budget", "Daily reserved experiment-time budget would be exceeded")
            experiment_id = str(uuid.uuid4())
            timestamp = _now()
            conn.execute(
                "INSERT INTO experiments(id, baseline_id, policy_digest, status, budget_day, reserved_seconds, "
                "elapsed_seconds, detail_json, created_at, finished_at) "
                "VALUES (?, ?, ?, 'running', ?, ?, NULL, NULL, ?, NULL)",
                (experiment_id, baseline_id, policy.digest, day, reserved, timestamp),
            )
            row = conn.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
            return self._experiment_dict(row)

    def finish_experiment(
        self,
        experiment_id: str,
        *,
        status: str,
        elapsed_seconds: float,
        detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _validate_identifier(experiment_id, "Experiment ID")
        if not isinstance(status, str) or status not in {"completed", "failed", "blocked"}:
            _error("invalid_experiment_status", "Terminal experiment status must be completed, failed, or blocked")
        try:
            elapsed_value = float(elapsed_seconds)
        except (TypeError, ValueError, OverflowError):
            _error("invalid_elapsed_time", "Elapsed time must be finite and nonnegative")
        if isinstance(elapsed_seconds, bool) or not isinstance(elapsed_seconds, (int, float)) or not math.isfinite(elapsed_value) or elapsed_value < 0:
            _error("invalid_elapsed_time", "Elapsed time must be finite and nonnegative")
        if detail is not None and not isinstance(detail, dict):
            _error("invalid_experiment_detail", "Experiment detail must be an object or null")
        detail_json = None if detail is None else _json(detail, "invalid_experiment_detail")
        with self._transaction(write=True) as conn:
            policy = self._policy(conn)
            if detail_json is not None and len(detail_json.encode("utf-8")) > policy.max_output_bytes:
                _error("invalid_experiment_detail", "Experiment detail exceeds the output byte limit")
            row = conn.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
            if row is None:
                _error("experiment_not_found", "Experiment claim does not exist")
            if row["status"] != "running":
                if (
                    row["status"] == status
                    and row["elapsed_seconds"] == elapsed_value
                    and row["detail_json"] == detail_json
                ):
                    return self._experiment_dict(row)
                _error("experiment_already_finished", "A terminal experiment claim cannot be reassigned")
            conn.execute(
                "UPDATE experiments SET status=?, elapsed_seconds=?, detail_json=?, finished_at=? "
                "WHERE id=? AND status='running'",
                (status, elapsed_value, detail_json, _now(), experiment_id),
            )
            row = conn.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
            return self._experiment_dict(row)

    def experiment(self, experiment_id: str) -> dict[str, Any]:
        _validate_identifier(experiment_id, "Experiment ID")
        with self._transaction() as conn:
            row = conn.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
            if row is None:
                _error("experiment_not_found", "Experiment claim does not exist")
            return self._experiment_dict(row)

    def consume_holdout(self, holdout_digest: str, experiment_id: str) -> dict[str, Any]:
        """Atomically burn one normalized holdout-data digest for one experiment."""
        self._validate_digest(holdout_digest, "holdout_digest")
        _validate_identifier(experiment_id, "Experiment ID")
        with self._transaction(write=True) as conn:
            existing = conn.execute(
                "SELECT experiment_id, consumed_at FROM holdout_uses WHERE digest=?",
                (holdout_digest,),
            ).fetchone()
            if existing is not None:
                if existing["experiment_id"] != experiment_id:
                    _error("holdout_already_used", "Protected holdout data has already been consumed")
                return {
                    "digest": holdout_digest,
                    "experiment_id": experiment_id,
                    "consumed_at": existing["consumed_at"],
                    "created": False,
                }
            experiment = conn.execute(
                "SELECT status FROM experiments WHERE id=?", (experiment_id,)
            ).fetchone()
            if experiment is None:
                _error("experiment_not_found", "Experiment claim does not exist")
            if experiment["status"] != "running":
                _error("experiment_not_running", "Holdout data can only be consumed by a running experiment")
            consumed_at = _now()
            conn.execute(
                "INSERT INTO holdout_uses(digest, experiment_id, consumed_at) VALUES (?, ?, ?)",
                (holdout_digest, experiment_id, consumed_at),
            )
            return {
                "digest": holdout_digest,
                "experiment_id": experiment_id,
                "consumed_at": consumed_at,
                "created": True,
            }

    @staticmethod
    def _validate_digest(value: Any, label: str) -> str:
        if not isinstance(value, str) or not _SNAPSHOT_ID.fullmatch(value):
            _error("invalid_evidence", f"{label} must be a lowercase SHA-256 digest")
        return value

    def _validate_evidence(self, evidence: Any, policy: Policy) -> dict[str, Any]:
        expected = {
            "candidate_id", "baseline_id", "suite_digest", "environment_digest", "policy_digest",
            "status", "baseline_training", "candidate_training", "baseline_holdout", "candidate_holdout",
            "violations", "errors", "case_counts",
        }
        if not isinstance(evidence, dict) or set(evidence) != expected:
            _error("invalid_evidence", "Evaluation evidence must contain exactly the documented fields")
        _validate_snapshot_id(evidence["candidate_id"])
        _validate_snapshot_id(evidence["baseline_id"])
        for field in ("suite_digest", "environment_digest", "policy_digest"):
            self._validate_digest(evidence[field], field)
        if not isinstance(evidence["status"], str) or evidence["status"] not in _EVALUATION_STATUSES:
            _error("invalid_evidence", "Evaluation status must distinguish quality and infrastructure outcomes")
        counts = evidence["case_counts"]
        if not isinstance(counts, dict) or set(counts) != {"training", "holdout"}:
            _error("invalid_evidence", "case_counts must contain exactly training and holdout")
        for count in counts.values():
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                _error("invalid_evidence", "Training and holdout case counts must be positive integers")
        if sum(counts.values()) > policy.max_cases:
            _error("invalid_evidence", "Evaluation case counts exceed the policy maximum")
        for field in ("baseline_training", "candidate_training", "baseline_holdout", "candidate_holdout"):
            score = evidence[field]
            if score is None and evidence["status"] in {"infrastructure_error", "blocked"}:
                continue
            try:
                score_value = float(score)
            except (TypeError, ValueError, OverflowError):
                _error("invalid_evidence", "Success rates must be finite values between zero and one")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score_value) or not 0 <= score_value <= 1:
                _error("invalid_evidence", "Success rates must be finite values between zero and one")
        for field in ("violations", "errors"):
            value = evidence[field]
            if not isinstance(value, list) or len(value) > 100:
                _error("invalid_evidence", f"{field} must be a bounded list")
            if any(not isinstance(item, str) or len(item) > 2048 for item in value):
                _error("invalid_evidence", f"{field} entries must be bounded strings")
        encoded = _json(evidence, "invalid_evidence")
        if len(encoded.encode("utf-8")) > _MAX_EVIDENCE_BYTES:
            _error("invalid_evidence", "Evaluation evidence exceeds the size limit")
        return evidence

    def record_evaluation(self, experiment_id: str, evidence: dict[str, Any]) -> dict[str, Any]:
        _validate_identifier(experiment_id, "Experiment ID")
        with self._transaction(write=True) as conn:
            policy = self._policy(conn)
            evidence = self._validate_evidence(evidence, policy)
            experiment = conn.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
            if experiment is None:
                _error("experiment_not_found", "Experiment claim does not exist")
            if evidence["baseline_id"] != experiment["baseline_id"]:
                _error("evidence_baseline_mismatch", "Evidence baseline differs from the experiment claim")
            if evidence["policy_digest"] != experiment["policy_digest"]:
                _error("evidence_policy_mismatch", "Evidence policy differs from the claim-time policy")
            baseline = self._snapshot_row(conn, evidence["baseline_id"])
            candidate = self._snapshot_row(conn, evidence["candidate_id"])
            self._verify_snapshot(conn, baseline)
            self._verify_snapshot(conn, candidate)
            if candidate["parent_id"] != evidence["baseline_id"]:
                _error("evidence_candidate_mismatch", "Candidate is not derived from the claimed baseline")
            if evidence["baseline_holdout"] is not None or evidence["candidate_holdout"] is not None:
                consumed = conn.execute(
                    "SELECT 1 FROM holdout_uses WHERE experiment_id=?",
                    (experiment_id,),
                ).fetchone()
                if consumed is None:
                    _error("holdout_not_consumed", "Holdout evidence requires a recorded one-use reservation")
            evidence_json = _json(evidence)
            if len(evidence_json.encode("utf-8")) > _MAX_EVIDENCE_BYTES:
                _error("invalid_evidence", "Evaluation evidence exceeds the size limit")
            evaluation_id = fingerprint({"experiment_id": experiment_id, "evidence": evidence})
            existing = conn.execute("SELECT evidence_json FROM evaluations WHERE id=?", (evaluation_id,)).fetchone()
            if existing is not None:
                if existing["evidence_json"] != evidence_json:
                    _error("evaluation_id_conflict", "Evaluation ID conflicts with stored evidence")
                return {"id": evaluation_id, "status": evidence["status"], "created": False}
            conn.execute(
                "INSERT INTO evaluations(id, experiment_id, candidate_id, baseline_id, suite_digest, "
                "environment_digest, policy_digest, status, evidence_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    evaluation_id, experiment_id, evidence["candidate_id"], evidence["baseline_id"],
                    evidence["suite_digest"], evidence["environment_digest"], evidence["policy_digest"],
                    evidence["status"], evidence_json, _now(),
                ),
            )
            return {"id": evaluation_id, "status": evidence["status"], "created": True}

    def _promotion_gate(
        self,
        conn: sqlite3.Connection,
        evaluation: sqlite3.Row,
        experiment: sqlite3.Row,
        evidence: dict[str, Any],
        current_id: str,
        policy: Policy,
    ) -> tuple[sqlite3.Row, sqlite3.Row]:
        self._validate_evidence(evidence, policy)
        if (
            evaluation["candidate_id"] != evidence["candidate_id"]
            or evaluation["baseline_id"] != evidence["baseline_id"]
            or evaluation["policy_digest"] != evidence["policy_digest"]
            or experiment["baseline_id"] != evidence["baseline_id"]
        ):
            _error("promotion_evidence_mismatch", "Stored evaluation metadata disagrees with its evidence")
        if experiment["status"] != "completed":
            _error("promotion_experiment_incomplete", "Promotion requires a completed experiment claim")
        if evidence["status"] != "passed":
            _error("promotion_evidence_not_passed", "Only passed evaluation evidence can be promoted")
        if evidence["errors"] or evidence["violations"]:
            _error("promotion_evidence_has_issues", "Promotion evidence contains errors or policy violations")
        if evidence["policy_digest"] != policy.digest or experiment["policy_digest"] != policy.digest:
            _error("promotion_policy_changed", "Policy changed after evaluation; fresh evidence is required")
        if evidence["baseline_id"] != current_id or experiment["baseline_id"] != current_id:
            _error("promotion_baseline_mismatch", "Evaluation baseline is not the current channel snapshot")
        if evaluation["baseline_id"] != current_id:
            _error("promotion_baseline_mismatch", "Stored evaluation baseline is not the current channel snapshot")
        counts = evidence["case_counts"]
        if not isinstance(counts, dict) or set(counts) != {"training", "holdout"}:
            _error("promotion_invalid_counts", "Promotion requires exact training and holdout case counts")
        if any(isinstance(counts[key], bool) or not isinstance(counts[key], int) or counts[key] <= 0 for key in ("training", "holdout")):
            _error("promotion_invalid_counts", "Promotion requires exact nonzero training and holdout counts")
        score_fields = ("baseline_training", "candidate_training", "baseline_holdout", "candidate_holdout")
        scores: dict[str, float] = {}
        for field in score_fields:
            score = evidence[field]
            try:
                score_value = float(score)
            except (TypeError, ValueError, OverflowError):
                _error("promotion_incomplete_scores", "Promotion requires four finite success rates")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score_value):
                _error("promotion_incomplete_scores", "Promotion requires four finite success rates")
            scores[field] = score_value
        if scores["candidate_training"] < policy.min_training_success:
            _error("promotion_training_threshold", "Candidate training success is below policy minimum")
        if scores["candidate_holdout"] < policy.min_holdout_success:
            _error("promotion_holdout_threshold", "Candidate holdout success is below policy minimum")
        training_delta = scores["candidate_training"] - scores["baseline_training"]
        holdout_delta = scores["candidate_holdout"] - scores["baseline_holdout"]
        if training_delta < -policy.max_regression or holdout_delta < -policy.max_regression:
            _error("promotion_regression", "Candidate exceeds the policy regression allowance")
        if holdout_delta < policy.min_improvement:
            _error("promotion_gain", "Candidate holdout gain is below policy minimum")
        baseline = self._snapshot_row(conn, current_id)
        candidate = self._snapshot_row(conn, evidence["candidate_id"])
        self._verify_snapshot(conn, baseline)
        self._verify_snapshot(conn, candidate)
        if candidate["parent_id"] != current_id:
            _error("promotion_candidate_mismatch", "Candidate is not derived from the current baseline")
        return baseline, candidate

    def promote(
        self,
        evaluation_id: str,
        *,
        expected_current: str,
        channel: str = "stable",
    ) -> dict[str, Any]:
        _validate_identifier(evaluation_id, "Evaluation ID")
        expected_current = _validate_snapshot_id(expected_current)
        channel = self._channel_name(channel)
        with self._transaction(write=True) as conn:
            policy = self._policy(conn)
            current = conn.execute("SELECT * FROM channels WHERE name=?", (channel,)).fetchone()
            if current is None:
                _error("channel_not_found", "Channel has not been bootstrapped")
            if current["snapshot_id"] != expected_current:
                _error("channel_conflict", "Channel moved since the caller observed its expected baseline")
            evaluation = conn.execute("SELECT * FROM evaluations WHERE id=?", (evaluation_id,)).fetchone()
            if evaluation is None:
                _error("evaluation_not_found", "Evaluation evidence does not exist")
            experiment = conn.execute(
                "SELECT * FROM experiments WHERE id=?", (evaluation["experiment_id"],)
            ).fetchone()
            if experiment is None:
                _error("registry_corrupt", "Evaluation has no experiment claim")
            evidence = _decode_json(evaluation["evidence_json"])
            self._promotion_gate(conn, evaluation, experiment, evidence, current["snapshot_id"], policy)
            next_generation = current["generation"] + 1
            timestamp = _now()
            changed = conn.execute(
                "UPDATE channels SET snapshot_id=?, generation=?, updated_at=? "
                "WHERE name=? AND snapshot_id=? AND generation=?",
                (
                    evaluation["candidate_id"], next_generation, timestamp, channel,
                    current["snapshot_id"], current["generation"],
                ),
            ).rowcount
            if changed != 1:
                _error("channel_conflict", "Channel changed during compare-and-swap")
            conn.execute(
                "INSERT OR IGNORE INTO accepted_snapshots(channel, snapshot_id, accepted_at) VALUES (?, ?, ?)",
                (channel, evaluation["candidate_id"], timestamp),
            )
            conn.execute(
                "INSERT INTO channel_events(channel, action, from_id, to_id, generation, evaluation_id, created_at) "
                "VALUES (?, 'promote', ?, ?, ?, ?, ?)",
                (
                    channel, current["snapshot_id"], evaluation["candidate_id"],
                    next_generation, evaluation_id, timestamp,
                ),
            )
            row = conn.execute("SELECT * FROM channels WHERE name=?", (channel,)).fetchone()
            return {**self._channel_dict(row), "previous_snapshot_id": current["snapshot_id"], "evaluation_id": evaluation_id}

    def rollback(
        self,
        snapshot_id: str,
        *,
        expected_current: str,
        channel: str = "stable",
    ) -> dict[str, Any]:
        snapshot_id = _validate_snapshot_id(snapshot_id)
        expected_current = _validate_snapshot_id(expected_current)
        channel = self._channel_name(channel)
        with self._transaction(write=True) as conn:
            current = conn.execute("SELECT * FROM channels WHERE name=?", (channel,)).fetchone()
            if current is None:
                _error("channel_not_found", "Channel has not been bootstrapped")
            if current["snapshot_id"] != expected_current:
                _error("channel_conflict", "Channel moved since the caller observed its expected baseline")
            if snapshot_id == current["snapshot_id"]:
                _error("rollback_noop", "Rollback target is already current")
            accepted = conn.execute(
                "SELECT 1 FROM accepted_snapshots WHERE channel=? AND snapshot_id=?",
                (channel, snapshot_id),
            ).fetchone()
            if accepted is None:
                _error("rollback_target_unaccepted", "Rollback target was never accepted in this channel")
            target = self._snapshot_row(conn, snapshot_id)
            self._verify_snapshot(conn, target)
            next_generation = current["generation"] + 1
            timestamp = _now()
            changed = conn.execute(
                "UPDATE channels SET snapshot_id=?, generation=?, updated_at=? "
                "WHERE name=? AND snapshot_id=? AND generation=?",
                (snapshot_id, next_generation, timestamp, channel, current["snapshot_id"], current["generation"]),
            ).rowcount
            if changed != 1:
                _error("channel_conflict", "Channel changed during compare-and-swap")
            conn.execute(
                "INSERT INTO channel_events(channel, action, from_id, to_id, generation, evaluation_id, created_at) "
                "VALUES (?, 'rollback', ?, ?, ?, NULL, ?)",
                (channel, current["snapshot_id"], snapshot_id, next_generation, timestamp),
            )
            row = conn.execute("SELECT * FROM channels WHERE name=?", (channel,)).fetchone()
            return {**self._channel_dict(row), "previous_snapshot_id": current["snapshot_id"]}

    def status(self) -> dict[str, Any]:
        with self._transaction() as conn:
            channels = [
                self._channel_dict(row)
                for row in conn.execute("SELECT * FROM channels ORDER BY name").fetchall()
            ]
            task_count = conn.execute("SELECT COUNT(*) AS n FROM tasks WHERE status='active'").fetchone()["n"]
            experiment_count = conn.execute(
                "SELECT COUNT(*) AS n FROM experiments WHERE status='running'"
            ).fetchone()["n"]
            tasks = conn.execute(
                "SELECT * FROM tasks WHERE status='active' ORDER BY started_at, task_id LIMIT ?",
                (_MAX_STATUS_ITEMS,),
            ).fetchall()
            experiments = conn.execute(
                "SELECT id, baseline_id, policy_digest, status, created_at FROM experiments "
                "WHERE status='running' ORDER BY created_at, id LIMIT ?",
                (_MAX_STATUS_ITEMS,),
            ).fetchall()
            counts = {
                "snapshots": conn.execute("SELECT COUNT(*) AS n FROM snapshots").fetchone()["n"],
                "channels": len(channels),
                "tasks": conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"],
                "receipts": conn.execute("SELECT COUNT(*) AS n FROM receipts").fetchone()["n"],
                "experiments": conn.execute("SELECT COUNT(*) AS n FROM experiments").fetchone()["n"],
                "evaluations": conn.execute("SELECT COUNT(*) AS n FROM evaluations").fetchone()["n"],
                "holdout_uses": conn.execute("SELECT COUNT(*) AS n FROM holdout_uses").fetchone()["n"],
            }
            return {
                "channels": channels,
                "active_tasks": [self._task_dict(row) for row in tasks],
                "active_task_count": task_count,
                "active_experiments": [dict(row) for row in experiments],
                "active_experiment_count": experiment_count,
                "counts": counts,
                "truncated": task_count > _MAX_STATUS_ITEMS or experiment_count > _MAX_STATUS_ITEMS,
            }
