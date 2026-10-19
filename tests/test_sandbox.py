from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

from abs_evolution.sandbox import Sandbox


@unittest.skipUnless(
    sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").is_file(),
    "macOS Seatbelt is required for process-isolation tests",
)
class SeatbeltSandboxTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_context = tempfile.TemporaryDirectory(prefix="abs-evolution-sandbox-")
        self.root = Path(self.temp_context.name).resolve()
        self.snapshot = self.root / "candidate"
        self.scratch = self.root / "scratch"
        self.outside = self.root / "outside"
        self.snapshot.mkdir()
        self.scratch.mkdir()
        self.outside.mkdir()
        self.sandbox = Sandbox()
        self.assertEqual(self.sandbox.backend_identity, "macos-seatbelt")

    def tearDown(self) -> None:
        self.temp_context.cleanup()

    def _write_candidate(self, source: str) -> Path:
        entrypoint = self.snapshot / "candidate.py"
        entrypoint.write_text(source, encoding="utf-8")
        return entrypoint

    def _run_script(
        self,
        entrypoint: Path,
        *,
        timeout_seconds: float = 3,
        max_output_bytes: int = 4096,
        extra_args: list[str] | None = None,
    ):
        return self.sandbox.run(
            [sys.executable, "-I", "-S", str(entrypoint), *(extra_args or [])],
            cwd=self.scratch,
            read_paths=[self.snapshot],
            write_paths=[self.scratch],
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )

    def test_seatbelt_does_not_claim_hard_memory_or_aggregate_scratch_quotas(self) -> None:
        self.assertFalse(self.sandbox.hard_resource_limits)

    def test_explicit_scratch_is_writable_but_outside_paths_and_network_are_denied(self) -> None:
        secret = self.outside / "secret.txt"
        secret.write_text("not for the candidate", encoding="utf-8")
        outside_write = self.outside / "candidate-write.txt"
        allowed_write = self.scratch / "result.txt"
        symlink = self.snapshot / "escape"
        symlink.symlink_to(self.outside, target_is_directory=True)
        entrypoint = self._write_candidate(
            "import json, os, socket, sys\n"
            "from pathlib import Path\n"
            "secret, outside_write, allowed_write = map(Path, sys.argv[1:])\n"
            "result = {}\n"
            "try:\n"
            "    allowed_write.write_text('scratch-ok', encoding='utf-8')\n"
            "    result['scratch_write'] = 'allowed'\n"
            "except OSError:\n"
            "    result['scratch_write'] = 'denied'\n"
            "try:\n"
            "    secret.read_text(encoding='utf-8')\n"
            "    result['outside_read'] = 'allowed'\n"
            "except OSError:\n"
            "    result['outside_read'] = 'denied'\n"
            "try:\n"
            "    os.listdir(secret.parent)\n"
            "    result['outside_list'] = 'allowed'\n"
            "except OSError:\n"
            "    result['outside_list'] = 'denied'\n"
            "try:\n"
            "    (Path(__file__).parent / 'escape' / 'secret.txt').read_text(encoding='utf-8')\n"
            "    result['symlink_escape_read'] = 'allowed'\n"
            "except OSError:\n"
            "    result['symlink_escape_read'] = 'denied'\n"
            "try:\n"
            "    outside_write.write_text('escaped', encoding='utf-8')\n"
            "    result['outside_write'] = 'allowed'\n"
            "except OSError:\n"
            "    result['outside_write'] = 'denied'\n"
            "try:\n"
            "    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
            "    sock.bind(('127.0.0.1', 0))\n"
            "    result['network_bind'] = 'allowed'\n"
            "except OSError:\n"
            "    result['network_bind'] = 'denied'\n"
            "print(json.dumps(result, sort_keys=True))\n"
        )

        result = self._run_script(entrypoint, extra_args=[str(secret), str(outside_write), str(allowed_write)])

        self.assertEqual(result.status, "passed", result)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(
            json.loads(result.stdout),
            {
                "network_bind": "denied",
                "outside_list": "denied",
                "outside_read": "denied",
                "outside_write": "denied",
                "scratch_write": "allowed",
                "symlink_escape_read": "denied",
            },
        )
        self.assertEqual(allowed_write.read_text(encoding="utf-8"), "scratch-ok")
        self.assertFalse(outside_write.exists())

    def test_deadline_kills_the_candidate_group_and_candidate_cannot_fork(self) -> None:
        process_pid_file = self.scratch / "candidate.pid"
        entrypoint = self._write_candidate(
            "import json, os, subprocess, sys, time\n"
            "from pathlib import Path\n"
            "Path(sys.argv[1]).write_text(str(os.getpid()), encoding='utf-8')\n"
            "try:\n"
            "    os.fork()\n"
            "    fork_status = 'allowed'\n"
            "except OSError:\n"
            "    fork_status = 'denied'\n"
            "try:\n"
            "    subprocess.run([sys.executable, '-I', '-S', '-c', 'pass'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1)\n"
            "    spawn_status = 'allowed'\n"
            "except OSError:\n"
            "    spawn_status = 'denied'\n"
            "print(json.dumps({'fork': fork_status, 'spawn': spawn_status}), flush=True)\n"
            "time.sleep(30)\n"
        )

        result = self._run_script(
            entrypoint,
            timeout_seconds=0.75,
            extra_args=[str(process_pid_file)],
        )

        self.assertEqual(result.status, "timed_out", result)
        self.assertLess(result.duration_seconds, 2.0)
        self.assertEqual(json.loads(result.stdout), {"fork": "denied", "spawn": "denied"})
        self.assertTrue(process_pid_file.exists(), "candidate did not start before the deadline")
        process_pid = int(process_pid_file.read_text(encoding="utf-8"))
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                os.kill(process_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.01)
        else:
            self.fail(f"sandbox process {process_pid} remained alive after timeout")

    def test_combined_stdout_and_stderr_output_limit_is_enforced(self) -> None:
        entrypoint = self._write_candidate(
            "import os\n"
            "chunk = b'x' * 1024\n"
            "while True:\n"
            "    os.write(1, chunk)\n"
            "    os.write(2, chunk)\n"
        )

        result = self._run_script(entrypoint, timeout_seconds=3, max_output_bytes=257)

        self.assertEqual(result.status, "output_limit", result)
        self.assertLessEqual(len(result.stdout.encode("utf-8")) + len(result.stderr.encode("utf-8")), 257)
        self.assertGreater(len(result.stdout) + len(result.stderr), 0)
        self.assertIsNotNone(result.returncode)


if __name__ == "__main__":
    unittest.main()
