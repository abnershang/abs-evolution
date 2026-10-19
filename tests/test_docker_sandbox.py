from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import textwrap
import time
import unittest

from abs_evolution.docker_sandbox import DockerSandbox


_OWNER_LABEL = "com.openai.abs-evolution.owner"
_DOCKER_TESTS_ENABLED = sys.platform == "linux" and os.environ.get("ABS_EVOLUTION_DOCKER_TESTS") == "1"


class DockerSandboxFailClosedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_context = tempfile.TemporaryDirectory(prefix="abs-evolution-docker-test-")
        self.root = Path(self.temp_context.name).resolve()
        self.snapshot = self.root / "candidate"
        self.snapshot.mkdir()
        self.scratch = self.root / "scratch"
        self.scratch.mkdir()

    def tearDown(self) -> None:
        self.temp_context.cleanup()

    def _run(self, path: Path):
        return DockerSandbox().run(
            [sys.executable, "-I", "-S", str(path)],
            cwd=self.scratch,
            read_paths=[self.snapshot],
            write_paths=[self.scratch],
            timeout_seconds=1,
            max_output_bytes=4096,
        )

    @unittest.skipIf(sys.platform == "linux", "non-Linux fail-closed contract")
    def test_non_linux_never_executes_candidate_or_claims_hard_limits(self) -> None:
        marker = self.root / "host-execution-marker"
        entrypoint = self.snapshot / "candidate.py"
        entrypoint.write_text(
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('executed')\n",
            encoding="utf-8",
        )

        sandbox = DockerSandbox()
        result = self._run(entrypoint)

        self.assertEqual(result.status, "blocked")
        self.assertIn("requires Linux", result.reason or "")
        self.assertFalse(marker.exists(), "candidate must not run unsandboxed")
        self.assertFalse(sandbox.hard_resource_limits)

    def test_symlinked_snapshot_entry_is_rejected_before_docker_admission(self) -> None:
        outside = self.root / "outside.py"
        outside.write_text("print('must not execute')\n", encoding="utf-8")
        (self.snapshot / "candidate.py").write_text("print('candidate')\n", encoding="utf-8")
        (self.snapshot / "escape").symlink_to(outside)

        result = self._run(self.snapshot / "candidate.py")

        self.assertEqual(result.status, "blocked")
        self.assertIn("symbolic links", result.reason or "")

    def test_entrypoint_outside_snapshot_is_rejected(self) -> None:
        outside = self.root / "outside.py"
        outside.write_text("print('must not execute')\n", encoding="utf-8")

        result = self._run(outside)

        self.assertEqual(result.status, "blocked")
        self.assertIn("inside the read-only snapshot", result.reason or "")


@unittest.skipUnless(
    _DOCKER_TESTS_ENABLED,
    "real Linux Docker contracts require ABS_EVOLUTION_DOCKER_TESTS=1",
)
class DockerSandboxRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sandbox = DockerSandbox()
        if not cls.sandbox.ensure_capability():
            raise AssertionError(f"opt-in Linux Docker admission failed: {cls.sandbox.capability_reason}")
        if not cls.sandbox.backend_identity.startswith("docker-local-unix:image="):
            raise AssertionError("backend identity must name the digest-pinned local runtime")

    def setUp(self) -> None:
        self.temp_context = tempfile.TemporaryDirectory(prefix="abs-evolution-docker-runtime-")
        self.root = Path(self.temp_context.name).resolve()
        self.snapshot = self.root / "candidate"
        self.snapshot.mkdir()
        self.scratch = self.root / "scratch"
        self.scratch.mkdir()
        self.secret = self.root / "host-only-secret.txt"
        self.secret.write_text("host-only", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_context.cleanup()

    def _owned_container_ids(self) -> set[str]:
        result = self.sandbox._docker_call(
            ["ps", "--all", "--no-trunc", "--quiet", "--filter", f"label={_OWNER_LABEL}"],
            timeout=4,
        )
        self.assertEqual(result.returncode, 0, "Docker container inventory must be readable")
        return set(result.stdout.splitlines())

    def _run(
        self,
        source: str,
        *,
        stdin: str = "",
        timeout_seconds: float = 5,
        max_output_bytes: int = 4096,
    ):
        entrypoint = self.snapshot / "candidate.py"
        entrypoint.write_text(textwrap.dedent(source), encoding="utf-8")
        return self.sandbox.run(
            [sys.executable, "-I", "-S", str(entrypoint)],
            cwd=self.scratch,
            read_paths=[self.snapshot],
            write_paths=[self.scratch],
            stdin=stdin,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )

    def _assert_no_owned_container_leak(self, before: set[str]) -> None:
        self.assertEqual(self._owned_container_ids(), before, "sandbox must remove its container and descendants")

    def test_normal_json_execution_is_read_only_networkless_and_discards_tmp_writes(self) -> None:
        before = self._owned_container_ids()
        payload = json.dumps({"host_secret_path": str(self.secret)})
        source = """
            import errno, json, os, socket, sys
            from pathlib import Path
            data = json.loads(sys.stdin.read())
            result = {}
            result['uid'] = os.getuid()
            try:
                (Path('/candidate') / 'candidate.py').write_text('overwrite')
                result['snapshot_write'] = 'allowed'
            except OSError:
                result['snapshot_write'] = 'denied'
            Path('/tmp/transient-output').write_text('discard this')
            try:
                Path(data['host_secret_path']).read_text()
                result['host_secret'] = 'visible'
            except OSError:
                result['host_secret'] = 'absent'
            result['host_key_present'] = 'OPENROUTER_API_KEY' in os.environ
            sock = socket.socket()
            sock.settimeout(0.5)
            try:
                sock.connect(('192.0.2.1', 443))
                result['network'] = 'allowed'
            except OSError as exc:
                result['network'] = 'denied'
                result['network_errno'] = exc.errno
            finally:
                sock.close()
            print(json.dumps(result, sort_keys=True))
        """
        old_value = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = "test-only-not-a-credential"
        try:
            result = self._run(source, stdin=payload)
        finally:
            if old_value is None:
                os.environ.pop("OPENROUTER_API_KEY", None)
            else:
                os.environ["OPENROUTER_API_KEY"] = old_value

        self.assertEqual(result.status, "passed", result)
        self.assertEqual(
            json.loads(result.stdout),
            {
                "host_key_present": False,
                "host_secret": "absent",
                "network": "denied",
                "network_errno": 101,
                "snapshot_write": "denied",
                "uid": 65534,
            },
        )
        self.assertFalse((self.scratch / "transient-output").exists())
        self.assertEqual(self.secret.read_text(encoding="utf-8"), "host-only")
        self._assert_no_owned_container_leak(before)

    def test_aggregate_scratch_and_process_count_limits_are_enforced(self) -> None:
        before = self._owned_container_ids()
        result = self._run(
            """
            import errno, json, os, signal, time
            def fill(root, limit):
                written = 0
                try:
                    for index in range(256):
                        with open(f'{root}/{index}', 'wb', buffering=0) as stream:
                            written += stream.write(b'x' * (64 * 1024))
                except OSError as exc:
                    return exc.errno == errno.ENOSPC and 0 < written <= limit
                return False
            children = []
            denied = False
            try:
                for _ in range(32):
                    try:
                        pid = os.fork()
                    except OSError as exc:
                        denied = exc.errno == errno.EAGAIN
                        break
                    if pid == 0:
                        time.sleep(30)
                        os._exit(0)
                    children.append(pid)
                print(json.dumps({
                    'tmp_limited': fill('/tmp', 8 * 1024 * 1024),
                    'shm_limited': fill('/dev/shm', 1024 * 1024),
                    'pids_limited': denied and 0 < len(children) < 16,
                }, sort_keys=True))
            finally:
                for pid in children:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ChildProcessError:
                        pass
                    except ProcessLookupError:
                        pass
                    try:
                        os.waitpid(pid, 0)
                    except ChildProcessError:
                        pass
            """
        )

        self.assertEqual(result.status, "passed", result)
        self.assertEqual(json.loads(result.stdout), {"pids_limited": True, "shm_limited": True, "tmp_limited": True})
        self._assert_no_owned_container_leak(before)

    def test_memory_overrun_is_failed_and_never_accepted(self) -> None:
        before = self._owned_container_ids()
        result = self._run(
            """
            blocks = []
            for _ in range(256):
                blocks.append(bytearray(1024 * 1024))
            raise AssertionError('64 MiB memory quota was not enforced')
            """,
            timeout_seconds=5,
        )

        self.assertEqual(result.status, "failed", result)
        self.assertEqual(result.returncode, 137)
        self._assert_no_owned_container_leak(before)

    def test_combined_output_limit_is_enforced_and_container_is_removed(self) -> None:
        before = self._owned_container_ids()
        result = self._run(
            """
            import os
            chunk = b'x' * 1024
            while True:
                os.write(1, chunk)
                os.write(2, chunk)
            """,
            timeout_seconds=3,
            max_output_bytes=257,
        )

        self.assertEqual(result.status, "output_limit", result)
        self.assertLessEqual(
            len(result.stdout.encode("utf-8")) + len(result.stderr.encode("utf-8")),
            257,
        )
        self.assertGreater(len(result.stdout) + len(result.stderr), 0)
        self._assert_no_owned_container_leak(before)

    def test_deadline_cleans_descendant_after_candidate_closes_stdio(self) -> None:
        before = self._owned_container_ids()
        started = time.monotonic()
        result = self._run(
            """
            import os, time
            child = os.fork()
            if child == 0:
                os.close(0)
                os.close(1)
                os.close(2)
                time.sleep(30)
                os._exit(0)
            os.write(1, str(child).encode() + b'\\n')
            os.close(0)
            os.close(1)
            os.close(2)
            time.sleep(30)
            """,
            timeout_seconds=0.75,
        )

        self.assertEqual(result.status, "timed_out", result)
        self.assertLess(time.monotonic() - started, 4)
        self.assertTrue(result.stdout.strip().isdigit(), result.stdout)
        self._assert_no_owned_container_leak(before)


if __name__ == "__main__":
    unittest.main()
