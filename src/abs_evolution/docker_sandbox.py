"""Fail-closed Linux Docker execution for untrusted evolution candidates.

This backend accepts only the engine's isolated Python entrypoint call (or
the fixed doctor smoke command), requires an existing local Linux Docker
daemon and an already-loaded pinned image, and never falls back to host code.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import time
import uuid
from typing import Any
from urllib.parse import urlsplit

from .contracts import ProcessResult


# Keep this identical to scripts/check_linux_limits.py.
IMAGE = "docker.io/library/python@sha256:2b4f19dae3a777dfc3b76730bda1e82e1f66ab2a2686fa93ca78edbfb4f04ffe"
IMAGE_ARCH = "amd64"
PYTHON = "/usr/local/bin/python"

MEMORY_BYTES = 64 * 1024 * 1024
SCRATCH_BYTES = 8 * 1024 * 1024
SHM_BYTES = 1024 * 1024
PIDS_LIMIT = 16
CPU_QUOTA = 50_000
CPU_PERIOD = 100_000
CPU_NANOS = 500_000_000

MAX_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_INPUT_BYTES = 1024 * 1024
MAX_TIMEOUT_SECONDS = 10_000_000_000
MAX_SNAPSHOT_ENTRIES = 100_000
CAPABILITY_TIMEOUT_SECONDS = 12.0
CAPABILITY_COMMAND_TIMEOUT_SECONDS = 3.0
CLEANUP_GRACE_SECONDS = 3.0
_KILL_WAIT_SECONDS = 0.25
_OWNER_LABEL = "com.openai.abs-evolution.owner"
_SAFE_ENV = (
    "HOME=/nonexistent",
    "PATH=/usr/local/bin:/usr/bin:/bin",
    "TMPDIR=/tmp",
    "TMP=/tmp",
    "TEMP=/tmp",
    "LC_ALL=C",
    "TZ=UTC",
    "PYTHONDONTWRITEBYTECODE=1",
)
_HEX_ID = re.compile(r"[0-9a-f]{64}\Z")


class _SandboxInputError(ValueError):
    """An input cannot be mapped without widening access."""


def _positive_finite(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if value <= 0 or value > MAX_TIMEOUT_SECONDS:
        return False
    return isinstance(value, int) or math.isfinite(value)


def _safe_path_string(value: Path) -> str:
    raw = str(value)
    if any(ord(character) < 32 or ord(character) == 127 for character in raw):
        raise _SandboxInputError("Sandbox paths cannot contain control characters")
    # Docker's --mount argument uses commas as field separators.
    if "," in raw:
        raise _SandboxInputError("Snapshot paths cannot contain commas")
    return raw


def _absolute_directory(value: Path, *, name: str) -> Path:
    try:
        raw = Path(value)
        if not raw.is_absolute() or raw.is_symlink():
            raise _SandboxInputError(f"{name} must be an absolute, non-symlink directory")
        resolved = raw.resolve(strict=True)
        if resolved == Path("/") or not resolved.is_dir():
            raise _SandboxInputError(f"{name} must be a non-root directory")
        _safe_path_string(resolved)
        return resolved
    except _SandboxInputError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise _SandboxInputError(f"{name} could not be resolved safely") from exc


def _validate_snapshot_tree(root: Path) -> None:
    """Reject links and special files so the bind cannot follow an escape."""
    seen = 0

    def on_error(_error: OSError) -> None:
        raise _SandboxInputError("Snapshot tree could not be inspected safely")

    try:
        for directory, directories, files in os.walk(root, topdown=True, followlinks=False, onerror=on_error):
            current = Path(directory)
            for name in [*directories, *files]:
                seen += 1
                if seen > MAX_SNAPSHOT_ENTRIES:
                    raise _SandboxInputError("Snapshot contains too many filesystem entries")
                child = current / name
                _safe_path_string(child)
                metadata = child.lstat()
                if stat.S_ISLNK(metadata.st_mode):
                    raise _SandboxInputError("Snapshot cannot contain symbolic links")
                if name in directories and not stat.S_ISDIR(metadata.st_mode):
                    raise _SandboxInputError("Snapshot contains a non-directory path component")
                if name in files and not stat.S_ISREG(metadata.st_mode):
                    raise _SandboxInputError("Snapshot can contain only regular files and directories")
    except _SandboxInputError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise _SandboxInputError("Snapshot tree could not be inspected safely") from exc


def _docker_endpoint() -> str:
    """Resolve only a local Unix socket, never TCP or a Docker context."""
    supplied = os.environ.get("DOCKER_HOST")
    if supplied is None or supplied == "":
        supplied = "unix:///var/run/docker.sock"
    try:
        parsed = urlsplit(supplied)
        if parsed.scheme != "unix" or parsed.netloc or parsed.query or parsed.fragment:
            raise _SandboxInputError("DockerSandbox requires a local Unix Docker endpoint")
        if not parsed.path.startswith("/") or "%" in parsed.path:
            raise _SandboxInputError("Docker Unix endpoint path is invalid")
        endpoint_path = Path(parsed.path).resolve(strict=True)
        if not stat.S_ISSOCK(endpoint_path.stat().st_mode):
            raise _SandboxInputError("Docker endpoint is not a Unix socket")
        return f"unix://{endpoint_path}"
    except _SandboxInputError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise _SandboxInputError("A local Unix Docker socket is unavailable") from exc


_CONTROL_AND_SCRATCH_PROBE = r'''import errno, os, pathlib, socket, sys
from pathlib import Path
p = Path
assert sys.version_info[:2] == (3, 12)
assert os.getuid() == 65534 and os.getgid() == 65534
status = p('/proc/self/status').read_text()
assert 'NoNewPrivs:\t1' in status
assert 'CapEff:\t0000000000000000' in status
assert 'Seccomp:\t2' in status
cgroup = p('/sys/fs/cgroup')
assert (cgroup / 'cgroup.controllers').is_file()
assert (cgroup / 'memory.max').read_text().strip() == str(64 * 1024 * 1024)
assert (cgroup / 'memory.swap.max').read_text().strip() == '0'
assert (cgroup / 'pids.max').read_text().strip() == '16'
assert (cgroup / 'cpu.max').read_text().split() == ['50000', '100000']
assert os.listdir('/sys/class/net') == ['lo']
assert not p('/var/run/docker.sock').exists()
assert not p('/run/docker.sock').exists()
assert not p('/github/workspace').exists()
assert not any(k in os.environ for k in ('OPENROUTER_API_KEY', 'TYPESAFE_API_KEY', 'GITHUB_TOKEN'))
for name in ('/etc/abs-evolution-probe', '/usr/local/abs-evolution-probe'):
    try:
        p(name).write_text('not allowed')
    except OSError as exc:
        assert exc.errno == errno.EROFS, exc
    else:
        raise AssertionError('container root filesystem is writable')
sock = socket.socket()
sock.settimeout(1)
try:
    try:
        sock.connect(('192.0.2.1', 443))
    except OSError as exc:
        assert exc.errno == errno.ENETUNREACH, exc
    else:
        raise AssertionError('container network is reachable')
finally:
    sock.close()
for root, limit in (('/tmp', 8 * 1024 * 1024), ('/dev/shm', 1024 * 1024)):
    written = 0
    try:
        for index in range(160):
            with open(f'{root}/{index}', 'wb', buffering=0) as stream:
                written += stream.write(b'x' * (64 * 1024))
    except OSError as exc:
        assert exc.errno == errno.ENOSPC, exc
    else:
        raise AssertionError(f'aggregate scratch quota did not stop writes: {root}')
    assert 0 < written <= limit, written
'''

_CPU_PROBE = r'''import pathlib, time
def stats():
    return dict(line.split() for line in pathlib.Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())
before = int(stats()['nr_throttled'])
end = time.monotonic() + 1.2
while time.monotonic() < end:
    pass
assert int(stats()['nr_throttled']) > before, 'CPU quota did not throttle'
'''

_MEMORY_PROBE = r'''import os, signal, sys
from pathlib import Path
cgroup = Path('/sys/fs/cgroup')
if (cgroup / 'memory.max').read_text().strip() != str(64 * 1024 * 1024):
    raise AssertionError('unexpected memory.max')
def events():
    return {
        key: int(value)
        for key, value in (line.split() for line in (cgroup / 'memory.events').read_text().splitlines())
    }
before = events()
pid = os.fork()
if pid == 0:
    blocks = []
    for _ in range(256):
        blocks.append(bytearray(1024 * 1024))
    os._exit(42)
_, status = os.waitpid(pid, 0)
after = events()
if not os.WIFSIGNALED(status) or os.WTERMSIG(status) != signal.SIGKILL:
    raise AssertionError('memory-limit child was not killed')
if any(after.get(key, 0) - before.get(key, 0) <= 0 for key in ('max', 'oom', 'oom_kill')):
    raise AssertionError('memory.events did not confirm a kernel OOM kill')
sys.stdout.write('kernel-oom-verified\n')
'''

_PIDS_PROBE = r'''import errno, os, signal, time
children = []
denied = False
try:
    for _ in range(32):
        try:
            pid = os.fork()
        except OSError as exc:
            assert exc.errno == errno.EAGAIN, exc
            denied = True
            break
        if pid == 0:
            time.sleep(10)
            os._exit(0)
        children.append(pid)
    assert denied and 0 < len(children) < 16, len(children)
finally:
    for pid in children:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
'''

# This fixed prelude checks effective controls in every candidate container
# immediately before replacing itself with the candidate process.
_ENTRYPOINT_GUARD = r'''import errno, os, pathlib, re, sys
from pathlib import Path
def fail():
    os.write(2, b'ABS_SANDBOX_ADMISSION_FAILED\n')
    os._exit(125)
try:
    if sys.version_info[:2] != (3, 12) or os.getuid() != 65534 or os.getgid() != 65534:
        fail()
    status = Path('/proc/self/status').read_text()
    if 'NoNewPrivs:\t1' not in status or 'CapEff:\t0000000000000000' not in status or 'Seccomp:\t2' not in status:
        fail()
    cgroup = Path('/sys/fs/cgroup')
    if not (cgroup / 'cgroup.controllers').is_file():
        fail()
    if (cgroup / 'memory.max').read_text().strip() != str(64 * 1024 * 1024):
        fail()
    if (cgroup / 'memory.swap.max').read_text().strip() != '0':
        fail()
    if (cgroup / 'pids.max').read_text().strip() != '16':
        fail()
    if (cgroup / 'cpu.max').read_text().split() != ['50000', '100000']:
        fail()
    if os.listdir('/sys/class/net') != ['lo']:
        fail()
    mounts = [line.split() for line in Path('/proc/mounts').read_text().splitlines()]
    if not any(len(fields) >= 4 and fields[1] == '/candidate' and 'ro' in fields[3].split(',') for fields in mounts):
        fail()
    def tmpfs_limit(mountpoint, maximum):
        for fields in mounts:
            if len(fields) >= 4 and fields[1].replace('\\040', ' ') == mountpoint and fields[2] == 'tmpfs':
                options = fields[3].split(',')
                raw_size = next((item.split('=', 1)[1] for item in options if item.startswith('size=')), '')
                match = re.fullmatch(r'([0-9]+)([kKmMgG]?)', raw_size)
                if match is None:
                    return False
                scale = {'': 1, 'k': 1024, 'm': 1024 ** 2, 'g': 1024 ** 3}[match.group(2).lower()]
                size = int(match.group(1)) * scale
                stats = os.statvfs(mountpoint)
                total = stats.f_blocks * stats.f_frsize
                return size == maximum and total <= maximum and total > maximum - 65536 and {'rw', 'noexec', 'nosuid', 'nodev'}.issubset(options)
        return False
    if not tmpfs_limit('/tmp', 8 * 1024 * 1024) or not tmpfs_limit('/dev/shm', 1024 * 1024):
        fail()
    for name in ('/etc/abs-evolution-guard', '/usr/local/abs-evolution-guard'):
        try:
            Path(name).write_text('not allowed')
        except OSError as exc:
            if exc.errno != errno.EROFS:
                fail()
        else:
            fail()
    entry = Path(sys.argv[1]).resolve(strict=True)
    entry.relative_to('/candidate')
    if not entry.is_file():
        fail()
    os.execv(sys.executable, [sys.executable, '-I', '-S', str(entry), *sys.argv[2:]])
except BaseException:
    fail()
'''


class DockerSandbox:
    """Run Python entrypoints in a bounded local Docker container."""

    def __init__(self) -> None:
        self._docker_path = shutil.which("docker")
        self._endpoint: str | None = None
        self._endpoint_error: str | None = None
        self._capability: bool | None = None
        self._capability_reason = "Docker capability has not been checked"
        self._capability_deadline = 0.0
        try:
            self._endpoint = _docker_endpoint()
        except _SandboxInputError as exc:
            self._endpoint_error = str(exc)

    @property
    def backend_identity(self) -> str:
        """Include the exact pinned image and hard limits in evidence."""
        return (
            f"docker-local-unix:image={IMAGE};arch=linux/{IMAGE_ARCH};"
            f"memory={MEMORY_BYTES};swap=0;scratch={SCRATCH_BYTES};shm={SHM_BYTES};"
            f"pids={PIDS_LIMIT};cpu={CPU_QUOTA}/{CPU_PERIOD};network=none;root=readonly"
        )

    @property
    def hard_resource_limits(self) -> bool:
        """Report true only after bounded runtime admission has passed."""
        return self.ensure_capability()

    @property
    def capability_reason(self) -> str:
        self.ensure_capability()
        return self._capability_reason

    def ensure_capability(self) -> bool:
        """Run a bounded set of fixed real probes once; cache failure closed."""
        if self._capability is not None:
            return self._capability
        self._capability = False
        if sys.platform != "linux":
            self._capability_reason = "DockerSandbox requires Linux; candidate execution was not attempted"
            return False
        if not self._docker_path:
            self._capability_reason = "Docker CLI is unavailable; candidate execution was not attempted"
            return False
        if self._endpoint is None:
            self._capability_reason = self._endpoint_error or "Local Docker socket is unavailable"
            return False

        self._capability_deadline = time.monotonic() + CAPABILITY_TIMEOUT_SECONDS
        try:
            info = self._docker_json(["info", "--format={{json .}}"], timeout=self._cap_timeout())
            if info.get("OSType") != "linux" or str(info.get("CgroupVersion")) != "2":
                raise RuntimeError("Docker must use Linux with cgroup v2")
            images = self._docker_json(["image", "inspect", IMAGE], timeout=self._cap_timeout())
            if not isinstance(images, list) or len(images) != 1:
                raise RuntimeError("Pinned Python image is not available locally")
            image = images[0]
            if image.get("Os") != "linux" or image.get("Architecture") != IMAGE_ARCH:
                raise RuntimeError("Pinned Python image must be Linux/amd64")
            if not self._run_capability_probes():
                raise RuntimeError(self._capability_reason)
            self._capability_reason = "Linux cgroup v2 Docker runtime passed bounded resource-isolation probes"
            self._capability = True
            return True
        except (OSError, RuntimeError, ValueError, TypeError, subprocess.TimeoutExpired) as exc:
            self._capability_reason = str(exc) or "Docker runtime admission failed closed"
            self._capability = False
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
        """Execute only the engine's Python call shape; never use host fallback."""
        started = time.monotonic()

        def blocked(reason: str) -> ProcessResult:
            return ProcessResult(
                "blocked", None, "", "", max(0.0, time.monotonic() - started), reason,
            )

        try:
            parsed = self._validate_call(
                argv, cwd=cwd, read_paths=read_paths, write_paths=write_paths,
                stdin=stdin, timeout_seconds=timeout_seconds,
                max_output_bytes=max_output_bytes,
            )
        except _SandboxInputError as exc:
            return blocked(str(exc))
        if not self.ensure_capability():
            return blocked(self._capability_reason)

        # The caller's budget includes candidate container create, inspect,
        # start, output streaming, and stop. Admission is a separate cached gate.
        deadline = time.monotonic() + float(timeout_seconds)
        name = "abs-evolution-" + uuid.uuid4().hex
        owner = uuid.uuid4().hex
        container_id: str | None = None
        cli_process: subprocess.Popen[bytes] | None = None
        stdout_buffer = bytearray()
        stderr_buffer = bytearray()
        timed_out = False
        output_limited = False
        io_failed = False
        create_timed_out = False
        container_state: dict[str, Any] | None = None
        attach_returncode: int | None = None
        setup_error = ""
        try:
            create = self._docker_call(
                self._create_args(name, owner, parsed["command"], snapshot=parsed["snapshot"]),
                timeout=self._remaining(deadline),
            )
            if create.returncode != 0:
                setup_error = "Docker could not create a bounded candidate container"
            else:
                candidate_id = create.stdout.strip()
                if not _HEX_ID.fullmatch(candidate_id):
                    setup_error = "Docker returned an invalid container identity"
                else:
                    container_id = candidate_id
                    inspect_status, inspected = self._inspect(container_id, timeout=self._remaining(deadline))
                    if inspect_status != "present" or inspected is None or not self._verify_container(inspected, owner, parsed["snapshot"]):
                        setup_error = "Created container did not retain the required isolation settings"
                    elif time.monotonic() >= deadline:
                        timed_out = True
                    else:
                        cli_process = subprocess.Popen(
                            self._docker_argv(["start", "--attach", "--interactive", container_id]),
                            cwd="/", env=self._docker_env(), stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            bufsize=0, close_fds=True, start_new_session=True,
                        )
                        stdout_buffer, stderr_buffer, timed_out, output_limited, io_failed = self._collect(
                            cli_process, input_bytes=parsed["input_bytes"],
                            deadline=deadline, max_output_bytes=max_output_bytes,
                        )
                        attach_returncode = cli_process.returncode
                        if timed_out or output_limited or io_failed:
                            self._kill_container(container_id)
                        elif cli_process.returncode is None:
                            self._kill_container(container_id)
                            self._kill_cli_group(cli_process)
                        state_status, state_record = self._inspect(container_id, timeout=CLEANUP_GRACE_SECONDS)
                        if state_status == "present" and state_record is not None and isinstance(state_record.get("State"), dict):
                            container_state = state_record["State"]
        except subprocess.TimeoutExpired:
            create_timed_out = container_id is None
            timed_out = True
        except (OSError, ValueError, TypeError):
            setup_error = "Docker sandbox execution could not be started safely"
        finally:
            if cli_process is not None and cli_process.poll() is None:
                self._kill_container(container_id)
                self._kill_cli_group(cli_process)
            cleanup_ok = self._cleanup_container(name, owner, container_id)

        duration = max(0.0, time.monotonic() - started)
        stdout_text = stdout_buffer.decode("utf-8", errors="replace")
        stderr_text = stderr_buffer.decode("utf-8", errors="replace")
        if not cleanup_ok:
            return ProcessResult(
                "blocked", None, stdout_text, stderr_text, duration,
                "Docker container cleanup could not be verified; execution was not accepted",
            )
        if setup_error:
            return ProcessResult("blocked", None, stdout_text, stderr_text, duration, setup_error)
        if create_timed_out or timed_out:
            return ProcessResult("timed_out", None, stdout_text, stderr_text, duration, "hard execution deadline exceeded")
        if output_limited:
            return ProcessResult("output_limit", None, stdout_text, stderr_text, duration, "combined stdout and stderr exceeded the byte limit")
        if io_failed:
            return ProcessResult("failed", None, stdout_text, stderr_text, duration, "sandbox process I/O failed")
        if container_state is None:
            return ProcessResult("blocked", None, stdout_text, stderr_text, duration, "Docker container exit state could not be verified")
        if (
            container_state.get("Status") != "exited"
            or container_state.get("Running") is not False
            or type(container_state.get("ExitCode")) is not int
            or attach_returncode is None
            or attach_returncode != container_state.get("ExitCode")
        ):
            return ProcessResult("blocked", None, stdout_text, stderr_text, duration, "Docker attach did not confirm a terminal container exit")
        returncode = container_state.get("ExitCode")
        if type(returncode) is not int:
            return ProcessResult("blocked", None, stdout_text, stderr_text, duration, "Docker returned an invalid container exit status")
        if container_state.get("OOMKilled") is True:
            return ProcessResult("failed", returncode, stdout_text, stderr_text, duration, "Docker reported an OOM in the candidate container")
        if b"ABS_SANDBOX_ADMISSION_FAILED" in stderr_buffer:
            return ProcessResult("blocked", returncode, stdout_text, stderr_text, duration, "effective Docker isolation check failed")
        if returncode == 0:
            return ProcessResult("passed", returncode, stdout_text, stderr_text, duration)
        return ProcessResult("failed", returncode, stdout_text, stderr_text, duration, "candidate process exited unsuccessfully")

    def _validate_call(
        self,
        argv: list[str],
        *,
        cwd: Path,
        read_paths: list[Path],
        write_paths: list[Path],
        stdin: str,
        timeout_seconds: float,
        max_output_bytes: int,
    ) -> dict[str, Any]:
        if not isinstance(argv, list) or not argv or any(not isinstance(arg, str) or "\0" in arg for arg in argv):
            raise _SandboxInputError("argv must be a nonempty list of NUL-free strings")
        if len(argv) > 5 or sum(len(arg.encode("utf-8", errors="strict")) for arg in argv) > 64 * 1024:
            raise _SandboxInputError("argv exceeds the supported bounded Python invocation")
        if not isinstance(read_paths, list) or len(read_paths) != 1 or not isinstance(write_paths, list) or len(write_paths) != 1:
            raise _SandboxInputError("exactly one snapshot read path and one scratch write path are required")
        if not isinstance(stdin, str):
            raise _SandboxInputError("stdin must be text")
        if not _positive_finite(timeout_seconds):
            raise _SandboxInputError("timeout_seconds must be positive and finite")
        if isinstance(max_output_bytes, bool) or not isinstance(max_output_bytes, int) or not 0 < max_output_bytes <= MAX_OUTPUT_BYTES:
            raise _SandboxInputError("max_output_bytes is outside the supported positive bound")
        try:
            input_bytes = stdin.encode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise _SandboxInputError("stdin must be valid UTF-8 text") from exc
        if len(input_bytes) > MAX_INPUT_BYTES:
            raise _SandboxInputError("stdin exceeds the supported one-megabyte bound")

        try:
            executable_input = Path(argv[0])
            if not executable_input.is_absolute() or executable_input.resolve(strict=True) != Path(sys.executable).resolve(strict=True):
                raise _SandboxInputError("argv[0] must be this Python interpreter")
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            if isinstance(exc, _SandboxInputError):
                raise
            raise _SandboxInputError("Python interpreter path is unavailable") from exc
        if argv[1:3] != ["-I", "-S"]:
            raise _SandboxInputError("only isolated Python -I -S execution is supported")

        snapshot = _absolute_directory(read_paths[0], name="snapshot")
        scratch = _absolute_directory(write_paths[0], name="scratch")
        working = _absolute_directory(Path(cwd), name="cwd")
        if scratch != working:
            raise _SandboxInputError("write_paths must contain exactly cwd")
        _validate_snapshot_tree(snapshot)

        if len(argv) == 5 and argv[3] == "-c":
            if argv[4] != "print('sandbox-ready')":
                raise _SandboxInputError("only the fixed doctor -c command is supported")
            return {
                "snapshot": snapshot,
                "command": [PYTHON, "-I", "-S", "-c", argv[4]],
                "input_bytes": input_bytes,
            }
        if len(argv) != 4:
            raise _SandboxInputError("only a single isolated Python entrypoint is supported")

        entry_input = Path(argv[3])
        if not entry_input.is_absolute() or entry_input.is_symlink():
            raise _SandboxInputError("candidate entrypoint must be an absolute regular file in snapshot")
        try:
            entry = entry_input.resolve(strict=True)
            relative = entry.relative_to(snapshot)
        except (OSError, RuntimeError, ValueError) as exc:
            raise _SandboxInputError("candidate entrypoint must remain inside the read-only snapshot") from exc
        if not entry.is_file() or entry.is_symlink():
            raise _SandboxInputError("candidate entrypoint must be a regular file")
        container_entry = Path("/candidate", relative.as_posix()).as_posix()
        command = [PYTHON, "-I", "-S", "-c", _ENTRYPOINT_GUARD, container_entry]
        return {"snapshot": snapshot, "command": command, "input_bytes": input_bytes}

    def _docker_env(self) -> dict[str, str]:
        assert self._docker_path is not None
        return {
            "PATH": f"{Path(self._docker_path).parent}:/usr/bin:/bin",
            "HOME": "/nonexistent",
            "DOCKER_CONFIG": "/nonexistent",
            "LC_ALL": "C",
            "TZ": "UTC",
        }

    def _docker_argv(self, arguments: list[str]) -> list[str]:
        assert self._docker_path is not None and self._endpoint is not None
        return [self._docker_path, "--host", self._endpoint, *arguments]

    def _docker_call(self, arguments: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
        if self._docker_path is None or self._endpoint is None:
            raise RuntimeError("Docker CLI or local Unix socket is unavailable")
        if timeout <= 0:
            raise subprocess.TimeoutExpired(self._docker_argv(arguments), 0)
        return subprocess.run(
            self._docker_argv(arguments), cwd="/", env=self._docker_env(),
            stdin=subprocess.DEVNULL, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout, check=False,
            close_fds=True,
        )

    def _docker_json(self, arguments: list[str], *, timeout: float) -> Any:
        result = self._docker_call(arguments, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError("Docker daemon query failed closed")
        try:
            return json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError) as exc:
            raise RuntimeError("Docker daemon returned invalid capability data") from exc

    def _cap_timeout(self) -> float:
        remaining = self._capability_deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired("Docker runtime admission", CAPABILITY_TIMEOUT_SECONDS)
        return min(CAPABILITY_COMMAND_TIMEOUT_SECONDS, remaining)

    def _base_create_args(self, name: str, owner: str, command: list[str], *, snapshot: Path | None) -> list[str]:
        args = [
            "create", "--pull=never", "--name", name,
            "--label", f"{_OWNER_LABEL}={owner}",
            "--network=none", "--ipc=private", "--cgroupns=private",
            "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges=true",
            "--user=65534:65534", f"--memory={MEMORY_BYTES}", f"--memory-swap={MEMORY_BYTES}",
            "--oom-kill-disable=false", "--cpus=0.5", f"--pids-limit={PIDS_LIMIT}",
            f"--shm-size={SHM_BYTES}",
            f"--tmpfs=/tmp:rw,noexec,nosuid,nodev,size={SCRATCH_BYTES},mode=1777",
            "--ulimit=nofile=64:64", "--ulimit=core=0:0", "--log-driver=none",
            "--workdir=/tmp", "--interactive", "--attach=stdin", "--attach=stdout", "--attach=stderr",
        ]
        args.extend(f"--env={entry}" for entry in _SAFE_ENV)
        if snapshot is not None:
            args.extend(["--mount", f"type=bind,source={snapshot},target=/candidate,readonly"])
        args.extend([IMAGE, *command])
        return args

    def _create_args(self, name: str, owner: str, command: list[str], *, snapshot: Path | None) -> list[str]:
        return self._base_create_args(name, owner, command, snapshot=snapshot)

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired("Docker candidate execution", 0)
        return remaining

    def _container_presence(self, identifier: str, *, is_name: bool, timeout: float) -> bool | None:
        if is_name:
            arguments = [
                "ps", "--all", "--no-trunc", "--format={{.Names}}",
                "--filter", f"name=^{identifier}$",
            ]
        else:
            arguments = [
                "ps", "--all", "--no-trunc", "--format={{.ID}}",
                "--filter", f"id={identifier}",
            ]
        result = self._docker_call(arguments, timeout=timeout)
        if result.returncode != 0:
            return None
        return identifier in result.stdout.splitlines()

    def _inspect(self, identifier: str, *, timeout: float) -> tuple[str, dict[str, Any] | None]:
        """Return present/absent/error; a daemon error is never absence proof."""
        result = self._docker_call(["inspect", "--format={{json .}}", identifier], timeout=timeout)
        if result.returncode == 0:
            try:
                value = json.loads(result.stdout)
            except (json.JSONDecodeError, TypeError):
                return "error", None
            return ("present", value) if isinstance(value, dict) else ("error", None)
        error_text = (result.stderr + result.stdout).casefold()
        if "no such object" not in error_text and "no such container" not in error_text:
            return "error", None
        is_name = not bool(_HEX_ID.fullmatch(identifier))
        present = self._container_presence(identifier, is_name=is_name, timeout=timeout)
        if present is None:
            return "error", None
        return ("present", None) if present else ("absent", None)

    @staticmethod
    def _verify_container(record: dict[str, Any], owner: str, snapshot: Path | None) -> bool:
        try:
            config = record["Config"]
            host = record["HostConfig"]
            labels = config["Labels"]
            image_names = {IMAGE, IMAGE.removeprefix("docker.io/"), IMAGE.removeprefix("docker.io/library/")}
            if labels.get(_OWNER_LABEL) != owner or config.get("Image") not in image_names:
                return False
            if config.get("User") != "65534:65534" or not config.get("OpenStdin"):
                return False
            if host.get("Memory") != MEMORY_BYTES or host.get("MemorySwap") != MEMORY_BYTES:
                return False
            if host.get("PidsLimit") != PIDS_LIMIT or host.get("ShmSize") != SHM_BYTES:
                return False
            if host.get("NanoCpus") != CPU_NANOS:
                return False
            if host.get("ReadonlyRootfs") is not True or host.get("NetworkMode") != "none":
                return False
            if host.get("Privileged") is True or host.get("OomKillDisable") is True:
                return False
            if host.get("CapDrop") != ["ALL"]:
                return False
            if not set(host.get("SecurityOpt") or []).intersection({"no-new-privileges", "no-new-privileges:true", "no-new-privileges=true"}):
                return False
            if host.get("PidMode") not in ("", "private") or host.get("IpcMode") not in ("", "private"):
                return False
            if host.get("CgroupnsMode") not in ("", "private"):
                return False
            if host.get("LogConfig", {}).get("Type") != "none":
                return False
            tmpfs = host.get("Tmpfs") or {}
            if "/tmp" not in tmpfs or f"size={SCRATCH_BYTES}" not in tmpfs["/tmp"]:
                return False
            if host.get("Binds") or host.get("VolumesFrom") or host.get("PortBindings"):
                return False
            actual_binds = [item for item in record.get("Mounts", []) if item.get("Type") in ("bind", "volume")]
            if snapshot is None:
                if actual_binds:
                    return False
            else:
                if len(actual_binds) != 1:
                    return False
                mount = actual_binds[0]
                if (
                    mount.get("Type") != "bind" or mount.get("Source") != str(snapshot)
                    or mount.get("Destination") != "/candidate" or mount.get("RW") is not False
                ):
                    return False
            return True
        except (KeyError, TypeError, AttributeError):
            return False

    def _run_capability_probes(self) -> bool:
        probes = (
            (_CONTROL_AND_SCRATCH_PROBE, "normal"),
            (_CPU_PROBE, "normal"),
            (_PIDS_PROBE, "normal"),
            (_MEMORY_PROBE, "kernel_oom"),
        )
        for index, (source, expected) in enumerate(probes, 1):
            self._capability_reason = f"Docker resource probe {index} did not complete"
            remaining = self._capability_deadline - time.monotonic()
            if remaining <= 0 or not self._run_capability_probe(source, expected=expected, timeout=remaining):
                return False
        return True

    def _run_capability_probe(self, source: str, *, expected: str, timeout: float) -> bool:
        name = "abs-evolution-admission-" + uuid.uuid4().hex
        owner = uuid.uuid4().hex
        container_id: str | None = None
        ok = False
        try:
            create = self._docker_call(
                self._base_create_args(name, owner, [PYTHON, "-I", "-S", "-c", source], snapshot=None),
                timeout=min(CAPABILITY_COMMAND_TIMEOUT_SECONDS, timeout, self._cap_timeout()),
            )
            if create.returncode != 0:
                self._capability_reason = "Fixed probe container creation failed: " + create.stderr[-512:]
            if create.returncode == 0 and _HEX_ID.fullmatch(create.stdout.strip()):
                container_id = create.stdout.strip()
                inspect_status, record = self._inspect(
                    container_id,
                    timeout=min(CAPABILITY_COMMAND_TIMEOUT_SECONDS, timeout, self._cap_timeout()),
                )
                if inspect_status == "present" and record is not None and self._verify_container(record, owner, snapshot=None):
                    start = self._docker_call(
                        ["start", "--attach", "--interactive", container_id],
                        timeout=min(CAPABILITY_COMMAND_TIMEOUT_SECONDS, timeout, self._cap_timeout()),
                    )
                    exit_status, exited = self._inspect(
                        container_id,
                        timeout=min(CAPABILITY_COMMAND_TIMEOUT_SECONDS, timeout, self._cap_timeout()),
                    )
                    if exit_status == "present" and exited is not None and isinstance(exited.get("State"), dict):
                        state = exited["State"]
                        terminal = (
                            state.get("Status") == "exited"
                            and state.get("Running") is False
                            and type(state.get("ExitCode")) is int
                            and start.returncode == state.get("ExitCode")
                        )
                        if expected == "normal":
                            ok = terminal and state.get("ExitCode") == 0 and state.get("OOMKilled") is False
                        elif expected == "kernel_oom":
                            ok = (
                                terminal
                                and state.get("ExitCode") == 0
                                and start.stdout == "kernel-oom-verified\n"
                                and start.stderr == ""
                            )
                        if not ok:
                            self._capability_reason = (
                                f"Fixed {expected} probe failed with exit {state.get('ExitCode')}: "
                                + start.stderr[-512:]
                            )
                elif record is not None:
                    # Only trusted fixed-probe settings, never candidate data or environment.
                    host = record.get("HostConfig", {})
                    detail = {key: host.get(key) for key in (
                        "Memory", "MemorySwap", "PidsLimit", "ShmSize", "NanoCpus",
                        "ReadonlyRootfs", "NetworkMode", "Privileged", "OomKillDisable",
                        "CapDrop", "SecurityOpt", "PidMode", "IpcMode", "CgroupnsMode",
                        "LogConfig", "Tmpfs",
                    )}
                    detail["Image"] = record.get("Config", {}).get("Image")
                    self._capability_reason = "Fixed probe configuration rejected: " + json.dumps(detail)[:2048]
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired):
            ok = False
        if not self._cleanup_container(
            name, owner, container_id, grace=min(CLEANUP_GRACE_SECONDS, max(0.1, timeout)),
        ):
            return False
        return ok

    def _cleanup_container(
        self,
        name: str,
        owner: str,
        container_id: str | None,
        *,
        grace: float = CLEANUP_GRACE_SECONDS,
    ) -> bool:
        """Remove only this UUID/label-owned object, then verify it is absent."""
        if self._docker_path is None or self._endpoint is None:
            return False
        deadline = time.monotonic() + max(0.1, grace)
        target = container_id
        while time.monotonic() < deadline:
            remaining = max(0.1, deadline - time.monotonic())
            try:
                status, record = self._inspect(target or name, timeout=min(1.0, remaining))
            except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired):
                return False
            if status == "error":
                if target is None:
                    return False
                # A successful create response is an ownership receipt for this
                # immutable ID; removing that exact ID remains safe if inspect
                # itself becomes unavailable.
                try:
                    removed = self._docker_call(
                        ["rm", "--force", target],
                        timeout=min(1.0, max(0.1, deadline - time.monotonic())),
                    )
                    if removed.returncode != 0:
                        return False
                    id_status, _ = self._inspect(target, timeout=min(1.0, max(0.1, deadline - time.monotonic())))
                    if id_status != "absent":
                        return False
                    name_status, named = self._inspect(name, timeout=min(1.0, max(0.1, deadline - time.monotonic())))
                    if name_status == "absent":
                        return True
                    if name_status != "present" or named is None:
                        return False
                    named_config = named.get("Config")
                    named_labels = named_config.get("Labels") if isinstance(named_config, dict) else None
                    return not isinstance(named_labels, dict) or named_labels.get(_OWNER_LABEL) != owner
                except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired):
                    return False
            if status == "absent":
                if target is None:
                    # A create request may still finish after its client timed out.
                    time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
                    continue
                try:
                    name_status, named = self._inspect(name, timeout=min(1.0, max(0.1, deadline - time.monotonic())))
                except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired):
                    return False
                if name_status == "absent":
                    return True
                if name_status != "present" or named is None:
                    return False
                named_config = named.get("Config")
                named_labels = named_config.get("Labels") if isinstance(named_config, dict) else None
                if not isinstance(named_labels, dict) or named_labels.get(_OWNER_LABEL) != owner:
                    # Our known immutable ID is gone; a newly occupying foreign
                    # name is not ours and must not be removed.
                    return True
                target = None
                record = named
            if record is None:
                return False
            config = record.get("Config") if record is not None else None
            labels = config.get("Labels") if isinstance(config, dict) else None
            current_id = record.get("Id") if record is not None else None
            if not isinstance(labels, dict) or labels.get(_OWNER_LABEL) != owner:
                # A UUID collision is not ours; never remove another container.
                return container_id is None
            if not isinstance(current_id, str) or not _HEX_ID.fullmatch(current_id):
                return False
            try:
                removed = self._docker_call(
                    ["rm", "--force", current_id],
                    timeout=min(1.0, max(0.1, deadline - time.monotonic())),
                )
                if removed.returncode != 0:
                    continue
                id_status, _ = self._inspect(current_id, timeout=min(1.0, max(0.1, deadline - time.monotonic())))
                if id_status != "absent":
                    continue
                late_status, late = self._inspect(name, timeout=min(1.0, max(0.1, deadline - time.monotonic())))
                if late_status == "absent":
                    return True
                if late_status != "present" or late is None:
                    return False
                late_config = late.get("Config")
                late_labels = late_config.get("Labels") if isinstance(late_config, dict) else None
                if not isinstance(late_labels, dict) or late_labels.get(_OWNER_LABEL) != owner:
                    return container_id is None or current_id == container_id
                target = None
            except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired):
                return False
        return False

    def _kill_container(self, container_id: str | None) -> None:
        if container_id is None or self._docker_path is None or self._endpoint is None:
            return
        try:
            self._docker_call(["kill", container_id], timeout=1.0)
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired):
            pass

    @staticmethod
    def _kill_cli_group(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            try:
                process.kill()
            except OSError:
                pass
        try:
            process.wait(timeout=_KILL_WAIT_SECONDS)
        except subprocess.TimeoutExpired:
            pass

    def _collect(
        self,
        process: subprocess.Popen[bytes],
        *,
        input_bytes: bytes,
        deadline: float,
        max_output_bytes: int,
    ) -> tuple[bytearray, bytearray, bool, bool, bool]:
        assert process.stdin is not None and process.stdout is not None and process.stderr is not None
        selector = selectors.DefaultSelector()
        stdout_buffer = bytearray()
        stderr_buffer = bytearray()
        total = 0
        input_offset = 0
        timed_out = False
        output_limited = False
        io_failed = False
        input_pipe = process.stdin
        try:
            os.set_blocking(input_pipe.fileno(), False)
            if input_bytes:
                selector.register(input_pipe, selectors.EVENT_WRITE, "stdin")
            else:
                input_pipe.close()
                input_pipe = None
            for stream, buffer in ((process.stdout, stdout_buffer), (process.stderr, stderr_buffer)):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, buffer)

            while selector.get_map() or process.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                if not selector.get_map():
                    try:
                        process.wait(timeout=min(remaining, 0.1))
                    except subprocess.TimeoutExpired:
                        pass
                    continue
                events = selector.select(min(remaining, 0.1))
                if not events:
                    if process.poll() is not None and not selector.get_map():
                        break
                    continue
                for key, _ in events:
                    if key.data == "stdin":
                        try:
                            count = os.write(input_pipe.fileno(), input_bytes[input_offset:input_offset + 65536])
                            if count <= 0:
                                io_failed = True
                                break
                            input_offset += count
                            if input_offset == len(input_bytes):
                                selector.unregister(input_pipe)
                                input_pipe.close()
                                input_pipe = None
                        except BrokenPipeError:
                            selector.unregister(input_pipe)
                            input_pipe.close()
                            input_pipe = None
                        except BlockingIOError:
                            continue
                        except OSError:
                            io_failed = True
                            break
                    else:
                        stream = key.fileobj
                        buffer = key.data
                        try:
                            chunk = os.read(stream.fileno(), 65536)
                        except BlockingIOError:
                            continue
                        except OSError:
                            io_failed = True
                            break
                        if not chunk:
                            selector.unregister(stream)
                            stream.close()
                            continue
                        left = max_output_bytes - total
                        if left > 0:
                            buffer.extend(chunk[:left])
                            total += min(len(chunk), left)
                        if len(chunk) > left:
                            output_limited = True
                            break
                if output_limited or io_failed:
                    break

            try:
                process.wait(timeout=_KILL_WAIT_SECONDS if timed_out or output_limited or io_failed else max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                if not output_limited and not io_failed:
                    timed_out = True
                self._kill_cli_group(process)
        except BaseException:
            self._kill_cli_group(process)
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
            if input_pipe is not None:
                try:
                    input_pipe.close()
                except OSError:
                    pass
        return stdout_buffer, stderr_buffer, timed_out, output_limited, io_failed
