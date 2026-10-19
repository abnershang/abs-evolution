"""Bounded subprocess transport shared by isolated provider HTTP workers."""

from __future__ import annotations

import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time


_MAX_WORKER_STDERR_BYTES = 4096
_KILL_WAIT_SECONDS = 0.25


class ProviderTransportError(Exception):
    """A sanitized failure from the isolated worker boundary."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


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


def run_bounded_worker(
    input_bytes: bytearray,
    *,
    worker_path: Path,
    deadline: float,
    stdout_limit: int,
    stderr_limit: int = _MAX_WORKER_STDERR_BYTES,
    kill_wait_seconds: float = _KILL_WAIT_SECONDS,
) -> bytes:
    """Run a fixed worker with bounded I/O and an absolute monotonic deadline."""
    try:
        worker = worker_path.resolve(strict=True)
        selector = selectors.DefaultSelector()
    except (OSError, RuntimeError, ValueError):
        raise ProviderTransportError("unavailable") from None

    try:
        process = subprocess.Popen(
            [sys.executable, "-I", "-S", str(worker)],
            cwd="/",
            env={},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            close_fds=True,
            start_new_session=True,
        )
    except OSError:
        selector.close()
        raise ProviderTransportError("unavailable") from None

    assert process.stdin is not None and process.stdout is not None and process.stderr is not None
    stdout_buffer = bytearray()
    stderr_buffer = bytearray()
    input_offset = 0
    timed_out = False
    output_limited = False
    pipe_failure = False
    write_pipe = process.stdin
    try:
        os.set_blocking(write_pipe.fileno(), False)
        os.set_blocking(process.stdout.fileno(), False)
        os.set_blocking(process.stderr.fileno(), False)
        if input_bytes:
            selector.register(write_pipe, selectors.EVENT_WRITE, "stdin")
        else:
            write_pipe.close()
            write_pipe = None
        selector.register(process.stdout, selectors.EVENT_READ, stdout_buffer)
        selector.register(process.stderr, selectors.EVENT_READ, stderr_buffer)

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
                        count = os.write(
                            write_pipe.fileno(),
                            input_bytes[input_offset:input_offset + 65536],
                        )
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
                    continue

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
                limit = stdout_limit if buffer is stdout_buffer else stderr_limit
                remaining_bytes = limit - len(buffer)
                if len(chunk) > remaining_bytes:
                    if remaining_bytes > 0:
                        buffer.extend(chunk[:remaining_bytes])
                    output_limited = True
                    break
                buffer.extend(chunk)
            if output_limited or pipe_failure:
                break

        if timed_out or output_limited or pipe_failure:
            _kill_process_group(process)
        try:
            wait_time = min(kill_wait_seconds, max(0.0, deadline - time.monotonic()))
            process.wait(
                timeout=wait_time
                if timed_out or output_limited or pipe_failure
                else max(0.0, deadline - time.monotonic())
            )
        except subprocess.TimeoutExpired:
            if not output_limited and not pipe_failure:
                timed_out = True
            _kill_process_group(process)
            try:
                process.wait(timeout=kill_wait_seconds)
            except subprocess.TimeoutExpired:
                pass
    except (KeyboardInterrupt, SystemExit):
        _kill_process_group(process)
        try:
            process.wait(timeout=kill_wait_seconds)
        except subprocess.TimeoutExpired:
            pass
        raise
    except Exception:
        _kill_process_group(process)
        try:
            process.wait(timeout=kill_wait_seconds)
        except subprocess.TimeoutExpired:
            pass
        raise ProviderTransportError("failure") from None
    finally:
        for key in list(selector.get_map().values()):
            try:
                selector.unregister(key.fileobj)
            except Exception:
                pass
            try:
                key.fileobj.close()
            except Exception:
                pass
        selector.close()
        if write_pipe is not None:
            try:
                write_pipe.close()
            except Exception:
                pass

    if timed_out:
        raise ProviderTransportError("timeout")
    if output_limited:
        raise ProviderTransportError("output_limit")
    if pipe_failure or process.returncode != 0:
        raise ProviderTransportError("failure")
    return bytes(stdout_buffer)
