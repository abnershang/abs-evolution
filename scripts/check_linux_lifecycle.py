"""Exercise the public CLI and actual version consumers on disposable Linux CI.

Synthetic offline proposal only; no provider calls, installed ABS or host writes
from candidates. The independent suite remains outside all candidate mounts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from abs_evolution.docker_sandbox import DockerSandbox


PROJECT = Path(__file__).resolve().parents[1]
EXAMPLE = PROJECT / "examples" / "acceptance"


def command(state: Path, *args: str, expected: int = 0) -> dict:
    env = dict(os.environ)
    for key in ("OPENROUTER_API_KEY", "TYPESAFE_API_KEY"):
        env.pop(key, None)
    result = subprocess.run(
        [sys.executable, "-m", "abs_evolution", "--state", str(state), *args],
        cwd=PROJECT, env=env, capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == expected, result.stderr + result.stdout
    return json.loads(result.stdout)


def initialize(state: Path) -> str:
    command(state, "init")
    baseline = command(state, "snapshot", "import", str(EXAMPLE / "baseline"), "--kind", "tool")["id"]
    command(state, "channel", "bootstrap", baseline)
    return baseline


def consume(sandbox: DockerSandbox, lease: dict, scratch: Path) -> dict:
    snapshot = Path(lease["snapshot_path"])
    result = sandbox.run(
        [sys.executable, "-I", "-S", str(snapshot / "tools" / "acceptance.py")],
        cwd=scratch, read_paths=[snapshot], write_paths=[scratch],
        # A build is not accepted delivery, even if its own success is true.
        stdin=json.dumps({"goal": "delivery", "records": [
            {"goal": "delivery", "kind": "build", "accepted": True},
        ]}), timeout_seconds=10, max_output_bytes=4096,
    )
    assert result.status == "passed", result
    return json.loads(result.stdout)


def main() -> None:
    if sys.platform != "linux" or os.environ.get("GITHUB_ACTIONS") != "true":
        raise SystemExit("Run only on disposable GitHub Linux runners")
    with tempfile.TemporaryDirectory(prefix="abs-closed-loop-") as raw:
        root = Path(raw)
        state = root / "registry"
        baseline = initialize(state)
        default = command(state, "doctor", expected=2)
        assert default["backend"] == "unsupported" and default["hard_resource_limits"] is False
        doctor = command(state, "doctor", "--sandbox", "docker")
        assert doctor["hard_resource_limits"] is True and doctor["unattended_ready"] is False
        experiment = command(
            state, "evolve", "--sandbox", "docker", "--suite", str(EXAMPLE / "suite.json"),
            "--provider", "file", "--proposal-file", str(EXAMPLE / "proposal.json"),
        )
        assert experiment["evaluation_id"] is not None, experiment
        assert experiment["shared_host_resources"] is False, experiment
        assert experiment["scores"] == {
            "baseline_training": 0.5, "candidate_training": 1.0,
            "baseline_holdout": 0.25, "candidate_holdout": 1.0,
        }, experiment
        candidate = experiment["candidate_id"]
        assert candidate != baseline
        old = command(state, "task", "begin", "old-task")
        sandbox = DockerSandbox()
        assert sandbox.hard_resource_limits
        scratch = root / "scratch"
        scratch.mkdir()
        assert consume(sandbox, old, scratch) == {"accepted": True}
        command(state, "channel", "promote", experiment["evaluation_id"], "--expected-current", baseline)
        new = command(state, "task", "begin", "new-task")
        assert new["snapshot_id"] == candidate and old["snapshot_id"] == baseline
        assert consume(sandbox, new, scratch) == {"accepted": False}
        old_again = command(state, "task", "begin", "old-task")
        assert old_again["snapshot_id"] == baseline
        assert consume(sandbox, old_again, scratch) == {"accepted": True}
        command(state, "channel", "rollback", baseline, "--expected-current", candidate)
        after = command(state, "task", "begin", "after-rollback")
        new_again = command(state, "task", "begin", "new-task")
        assert after["snapshot_id"] == baseline and new_again["snapshot_id"] == candidate
        assert consume(sandbox, after, scratch) == {"accepted": True}
        assert consume(sandbox, new_again, scratch) == {"accepted": False}
        for task in ("old-task", "new-task", "after-rollback"):
            command(state, "task", "end", task)
        print(json.dumps({"check": "closed-loop", "status": "passed",
                          "baseline_id": baseline, "candidate_id": candidate,
                          "scores": experiment["scores"], "actual_leased_execution": True,
                          "promotion_verified": True, "rollback_verified": True}), flush=True)

        # A real OOM in candidate training must never publish a version.
        failed_state = root / "oom-registry"
        failed_baseline = initialize(failed_state)
        proposal = root / "oom-proposal.json"
        proposal.write_text(json.dumps({
            "summary": "Reviewed negative-control allocation fixture", "kind": "tool",
            "changes": [{"path": "tools/acceptance.py", "content":
                         "blocks = [bytearray(1024*1024) for _ in range(256)]\n"}],
        }), encoding="utf-8")
        failed = command(
            failed_state, "evolve", "--sandbox", "docker", "--suite", str(EXAMPLE / "suite.json"),
            "--provider", "file", "--proposal-file", str(proposal), "--promote", expected=2,
        )
        assert failed["reason"] == "sandbox_process_failed", failed
        assert failed["promotion"] is None, failed
        unchanged = command(failed_state, "task", "begin", "after-oom")
        assert unchanged["snapshot_id"] == failed_baseline
        command(failed_state, "task", "end", "after-oom")
        print(json.dumps({"check": "oom-does-not-promote", "status": "passed",
                          "reason": failed["reason"], "baseline_unchanged": True}), flush=True)
    print(json.dumps({"status": "passed", "provider": "file", "model_called": False,
                      "deployment_changed": False, "unattended_ready": False}), flush=True)


if __name__ == "__main__":
    main()
