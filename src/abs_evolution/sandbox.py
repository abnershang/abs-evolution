"""Fail-closed process isolation for untrusted candidate programs.

The first supported backend is macOS Seatbelt.  The sandbox has no fallback:
if Seatbelt is unavailable or its profile cannot be started, ``run`` returns a
blocked result without executing the requested command.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import sysconfig
import time
from typing import Iterable

from .contracts import ProcessResult


_MAX_OUTPUT_BYTES = 64 * 1024 * 1024
_MAX_TIMEOUT_SECONDS = 10_000_000_000
_KILL_WAIT_SECONDS = 0.25


class _SandboxSetupError(Exception):
    """A secure sandbox could not be described for the requested run."""


@dataclass(frozen=True)
class _PathRule:
    path: Path
    recursive: bool


def _quote_profile_path(path: Path) -> str:
    value = str(path)
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise _SandboxSetupError("A sandbox path contains unsupported control characters")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _existing_path(value: Path, *, expected: str) -> _PathRule:
    """Resolve one allowlist entry without granting its lexical parent."""
    try:
        raw = Path(value)
        if not raw.is_absolute():
            raise _SandboxSetupError("Sandbox allowlist paths must be absolute")
        if raw.is_symlink():
            raise _SandboxSetupError("Sandbox allowlist entries cannot be symlinks")
        resolved = raw.resolve(strict=True)
        if expected in {"read", "write"} and resolved == Path("/"):
            raise _SandboxSetupError("A filesystem-root allowlist entry is too broad")
        if expected == "directory":
            if not resolved.is_dir():
                raise _SandboxSetupError("A sandbox directory path is not a directory")
            return _PathRule(resolved, True)
        if expected == "read":
            if resolved.is_dir():
                return _PathRule(resolved, True)
            if not resolved.is_file():
                raise _SandboxSetupError("A sandbox read path is not a regular file or directory")
            return _PathRule(resolved, False)
        if expected == "write":
            if resolved.is_dir():
                return _PathRule(resolved, True)
            if resolved.is_file():
                return _PathRule(resolved, False)
            raise _SandboxSetupError("A sandbox write path is not a regular file or directory")
        raise AssertionError(f"unexpected path expectation: {expected}")
    except _SandboxSetupError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise _SandboxSetupError("A sandbox allowlist path could not be resolved safely") from exc


def _runtime_read_rules(executable: Path) -> list[_PathRule]:
    """Return narrow system and current-Python runtime read roots."""
    rules: list[_PathRule] = []
    for value in (Path("/usr/lib"), Path("/System/Library/Frameworks")):
        if value.exists():
            rules.append(_existing_path(value, expected="directory"))

    rules.append(_existing_path(executable, expected="read"))

    # The base prefix identifies the Python installation, unlike site-packages
    # or HOME.  -I -S excludes user/site package directories from import paths.
    base_prefix = Path(sys.base_prefix)
    if base_prefix.exists():
        rules.append(_existing_path(base_prefix, expected="directory"))

    stdlib = sysconfig.get_path("stdlib")
    if stdlib:
        stdlib_path = Path(stdlib)
        if stdlib_path.exists():
            rules.append(_existing_path(stdlib_path, expected="directory"))

    for entry in sys.path:
        if entry.endswith(".zip"):
            archive = Path(entry)
            if archive.exists():
                rules.append(_existing_path(archive, expected="read"))

    # Deduplicate while preserving the more specific rule for each path.
    unique: dict[tuple[str, bool], _PathRule] = {}
    for rule in rules:
        unique[(str(rule.path), rule.recursive)] = rule
    return list(unique.values())


def _seatbelt_profile(
    executable: Path,
    read_rules: Iterable[_PathRule],
    write_rules: Iterable[_PathRule],
    cwd: Path,
) -> str:
    lines = [
        "(version 1)",
        "(deny default)",
        "(deny process-fork)",
        "(allow sysctl-read)",
        f"(allow process-exec (literal {_quote_profile_path(executable)}))",
        "(deny network*)",
        # Python startup path lookup fails on this macOS host without root
        # directory read access. This permits listing only the root entries;
        # descendants still need a separate read rule. Ancestors of explicit
        # paths are metadata-only.
        '(allow file-read-data (literal "/"))',
    ]
    exec_paths = [executable]
    for rule in read_rules:
        operation = "subpath" if rule.recursive else "literal"
        lines.append(f"(allow file-read* ({operation} {_quote_profile_path(rule.path)}))")
    for rule in write_rules:
        operation = "subpath" if rule.recursive else "literal"
        lines.append(f"(allow file-write* ({operation} {_quote_profile_path(rule.path)}))")

    traversed_paths = {cwd, *exec_paths}
    traversed_paths.update(rule.path for rule in read_rules)
    traversed_paths.update(rule.path for rule in write_rules)
    metadata_paths: set[Path] = set()
    for path in traversed_paths:
        metadata_paths.add(path)
        metadata_paths.update(path.parents)
    metadata_paths.discard(Path("/"))
    for path in sorted(metadata_paths, key=lambda item: (len(item.parts), str(item))):
        lines.append(f"(allow file-read-metadata (literal {_quote_profile_path(path)}))")
    return "\n".join(lines)


def _is_positive_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if value <= 0 or value > _MAX_TIMEOUT_SECONDS:
        return False
    return isinstance(value, int) or math.isfinite(value)


class Sandbox:
    """Execute a command under a per-run macOS Seatbelt allowlist."""

    def __init__(self) -> None:
        sandbox_exec = Path("/usr/bin/sandbox-exec")
        if sys.platform == "darwin" and sandbox_exec.is_file() and os.access(sandbox_exec, os.X_OK):
            try:
                self._sandbox_exec: Path | None = sandbox_exec.resolve(strict=True)
            except OSError:
                self._sandbox_exec = None
        else:
            self._sandbox_exec = None

    @property
    def backend_identity(self) -> str:
        """Name the backend selected by this instance for experiment evidence."""
        return "macos-seatbelt" if self._sandbox_exec is not None else "unsupported"

    @property
    def hard_resource_limits(self) -> bool:
        """Seatbelt does not enforce memory or aggregate scratch-space quotas."""
        return False

    def run(
        self,
        argv: list[str],
        *,
        cwd: Path,
        read_paths: list[Path],
        write_paths: list[Path],
        stdin: str = "",
        timeout_seconds: float = 10,
        max_output_bytes: int = 262144,
    ) -> ProcessResult:
        """Run argv without a shell and enforce an explicit file/network policy."""
        started = time.monotonic()

        def blocked(reason: str, *, stderr: str = "") -> ProcessResult:
            return ProcessResult(
                status="blocked",
                returncode=None,
                stdout="",
                stderr=stderr,
                duration_seconds=max(0.0, time.monotonic() - started),
                reason=reason,
            )

        if self._sandbox_exec is None:
            return blocked("macOS Seatbelt is unavailable; candidate execution was not attempted")
        if not isinstance(argv, list) or not argv or any(not isinstance(arg, str) or "\0" in arg for arg in argv):
            return blocked("argv must be a nonempty list of NUL-free strings")
        if not isinstance(read_paths, list) or not isinstance(write_paths, list):
            return blocked("read_paths and write_paths must be lists")
        if not isinstance(stdin, str):
            return blocked("stdin must be text")
        if not _is_positive_finite_number(timeout_seconds):
            return blocked("timeout_seconds must be a positive finite number")
        if (
            isinstance(max_output_bytes, bool)
            or not isinstance(max_output_bytes, int)
            or not 0 < max_output_bytes <= _MAX_OUTPUT_BYTES
        ):
            return blocked("max_output_bytes is outside the supported positive bound")

        try:
            executable_input = Path(argv[0])
            if not executable_input.is_absolute():
                raise _SandboxSetupError("argv[0] must be an absolute executable path")
            executable = executable_input.resolve(strict=True)
            if not executable.is_file() or not os.access(executable, os.X_OK):
                raise _SandboxSetupError("argv[0] is not an executable regular file")

            # Homebrew's `sys.executable` is a launcher that forks the real
            # framework executable. Use that trusted binary directly so the
            # candidate does not need process-fork permission.
            python_launcher = Path(sys.executable).resolve(strict=True)
            python_app = Path(sys.base_prefix) / "Resources" / "Python.app" / "Contents" / "MacOS" / "Python"
            if executable == python_launcher and python_app.is_file():
                executable = python_app.resolve(strict=True)

            working_directory = _existing_path(Path(cwd), expected="directory").path
            read_rules = _runtime_read_rules(executable)
            read_rules.extend(_existing_path(path, expected="read") for path in read_paths)
            write_rules = [_existing_path(path, expected="write") for path in write_paths]
            profile = _seatbelt_profile(executable, read_rules, write_rules, working_directory)
            input_bytes = stdin.encode("utf-8", errors="strict")
        except (_SandboxSetupError, OSError, UnicodeError, TypeError, ValueError) as exc:
            reason = str(exc) if isinstance(exc, _SandboxSetupError) else "Sandbox inputs could not be encoded or resolved safely"
            return blocked(reason)

        deadline = started + float(timeout_seconds)
        sandbox_argv = [str(self._sandbox_exec), "-p", profile, "--", str(executable), *argv[1:]]
        scratch_directory = next((rule.path for rule in write_rules if rule.recursive), None)
        temporary_directory = str(scratch_directory) if scratch_directory is not None else "/nonexistent"
        environment = {
            "HOME": "/nonexistent",
            "PATH": "",
            "TMPDIR": temporary_directory,
            "TMP": temporary_directory,
            "TEMP": temporary_directory,
            "LC_ALL": "C",
            "TZ": "UTC",
            "PYTHONDONTWRITEBYTECODE": "1",
        }

        try:
            process = subprocess.Popen(
                sandbox_argv,
                cwd=working_directory,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                close_fds=True,
                start_new_session=True,
            )
        except OSError:
            return blocked("Seatbelt could not start the requested process")

        assert process.stdin is not None and process.stdout is not None and process.stderr is not None
        selector = selectors.DefaultSelector()
        stdout_buffer = bytearray()
        stderr_buffer = bytearray()
        total_output = 0
        input_offset = 0
        timed_out = False
        output_limited = False
        write_pipe = process.stdin
        pipe_failure = False

        try:
            os.set_blocking(write_pipe.fileno(), False)
            if input_bytes:
                selector.register(write_pipe, selectors.EVENT_WRITE, "stdin")
            else:
                write_pipe.close()
                write_pipe = None
            for stream, buffer in ((process.stdout, stdout_buffer), (process.stderr, stderr_buffer)):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, buffer)

            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                events = selector.select(remaining)
                if not events:
                    if time.monotonic() >= deadline:
                        timed_out = True
                        break
                    continue
                for key, _ in events:
                    if key.data == "stdin":
                        try:
                            count = os.write(write_pipe.fileno(), input_bytes[input_offset:input_offset + 65536])
                            input_offset += count
                            if input_offset == len(input_bytes):
                                selector.unregister(write_pipe)
                                write_pipe.close()
                                write_pipe = None
                        except BrokenPipeError:
                            selector.unregister(write_pipe)
                            write_pipe.close()
                            write_pipe = None
                        except BlockingIOError:
                            continue
                        except OSError:
                            pipe_failure = True
                            break
                    else:
                        stream = key.fileobj
                        buffer = key.data
                        try:
                            chunk = os.read(stream.fileno(), 65536)
                        except BlockingIOError:
                            continue
                        except OSError:
                            pipe_failure = True
                            break
                        if not chunk:
                            selector.unregister(stream)
                            stream.close()
                            continue
                        remaining_output = max_output_bytes - total_output
                        if remaining_output > 0:
                            buffer.extend(chunk[:remaining_output])
                            total_output += min(len(chunk), remaining_output)
                        if len(chunk) > remaining_output:
                            output_limited = True
                            break
                if output_limited:
                    break
                if pipe_failure:
                    break

            if timed_out or output_limited or pipe_failure:
                self._kill_process_group(process)
            try:
                process.wait(timeout=_KILL_WAIT_SECONDS if timed_out or output_limited or pipe_failure else max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                if not output_limited and not pipe_failure:
                    timed_out = True
                self._kill_process_group(process)
                try:
                    process.wait(timeout=_KILL_WAIT_SECONDS)
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            self._kill_process_group(process)
            try:
                process.wait(timeout=_KILL_WAIT_SECONDS)
            except subprocess.TimeoutExpired:
                pass
            raise
        finally:
            for key in list(selector.get_map().values()):
                try:
                    selector.unregister(key.fileobj)
                except Exception:
                    pass
                try:
                    key.fileobj.close()
                except OSError:
                    pass
            selector.close()
            if write_pipe is not None:
                try:
                    write_pipe.close()
                except OSError:
                    pass

        returncode = process.returncode
        stderr_text = stderr_buffer.decode("utf-8", errors="replace")
        stdout_text = stdout_buffer.decode("utf-8", errors="replace")
        duration = max(0.0, time.monotonic() - started)
        if timed_out:
            return ProcessResult("timed_out", returncode, stdout_text, stderr_text, duration, "hard execution deadline exceeded")
        if output_limited:
            return ProcessResult("output_limit", returncode, stdout_text, stderr_text, duration, "combined stdout and stderr exceeded the byte limit")
        if pipe_failure:
            return ProcessResult("failed", returncode, stdout_text, stderr_text, duration, "sandbox process I/O failed")
        if returncode != 0 and stderr_text.startswith("sandbox-exec:"):
            return ProcessResult("blocked", returncode, stdout_text, stderr_text, duration, "Seatbelt rejected process startup or its profile")
        if returncode == 0:
            return ProcessResult("passed", returncode, stdout_text, stderr_text, duration)
        return ProcessResult("failed", returncode, stdout_text, stderr_text, duration, "candidate process exited unsuccessfully")

    @staticmethod
    def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            try:
                process.kill()
            except OSError:
                pass
