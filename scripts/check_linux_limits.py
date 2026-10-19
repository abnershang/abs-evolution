"""CI-only feasibility proof, not a candidate execution backend.

Runs only the fixed, reviewed probes below on a disposable Linux runner.
Never accepts candidate code, mounts host paths, or changes engine readiness.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import uuid


# Official python:3.12-slim linux/amd64, resolved 2026-10-19 (3.12.15).
IMAGE = "docker.io/library/python@sha256:2b4f19dae3a777dfc3b76730bda1e82e1f66ab2a2686fa93ca78edbfb4f04ffe"


def docker(*args: str, timeout: float = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=timeout,
        check=True,
    )


def probe(label: str, code: str, *, kernel_oom: bool = False, deadline: bool = False) -> None:
    name = "abs-limits-" + uuid.uuid4().hex
    created = False
    try:
        docker(
            "create", "--name", name, "--pull=never", "--network=none",
            "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--user=65534:65534", "--memory=64m", "--memory-swap=64m",
            "--cpus=0.5", "--pids-limit=16", "--shm-size=1m",
            "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=8m,mode=1777",
            "--ulimit=nofile=64:64", "--ulimit=core=0:0", "--log-driver=none",
            "--workdir=/tmp", IMAGE, "python", "-I", "-S", "-c",
            textwrap.dedent(code),
        )
        created = True
        timed_out = False
        output = ""
        try:
            output = docker("start", "--attach", name, timeout=3 if deadline else 20).stdout
        except subprocess.TimeoutExpired:
            timed_out = True
            docker("kill", name)
        state = json.loads(docker("inspect", "--format={{json .State}}", name).stdout)
        if deadline:
            assert timed_out and not state["Running"] and state["ExitCode"] == 137, state
        elif kernel_oom:
            assert not timed_out and state["ExitCode"] == 0 and output.strip() == "kernel-oom-verified", state
        else:
            assert not timed_out and state["ExitCode"] == 0 and not state["OOMKilled"], state
    finally:
        if created:
            docker("rm", "--force", name)
            remaining = docker("ps", "--all", "--quiet", "--filter", f"name=^{name}$").stdout
            assert not remaining.strip(), "Probe container was not removed"
    print(json.dumps({"probe": label, "status": "passed", "container_removed": True}), flush=True)


def main() -> None:
    if sys.platform != "linux" or os.environ.get("GITHUB_ACTIONS") != "true":
        raise SystemExit("This probe is restricted to disposable GitHub Linux CI runners")
    info = json.loads(docker("info", "--format={{json .}}").stdout)
    assert info["OSType"] == "linux" and info["CgroupVersion"] == "2", "Linux cgroup v2 required"
    docker("pull", IMAGE, timeout=120)
    probe("access-and-cgroup-controls", """
        import errno, os, pathlib, socket
        p = pathlib.Path
        assert os.getuid() == 65534
        assert p('/sys/fs/cgroup/memory.max').read_text().strip() == str(64*1024*1024)
        assert p('/sys/fs/cgroup/memory.swap.max').read_text().strip() == '0'
        assert p('/sys/fs/cgroup/pids.max').read_text().strip() == '16'
        assert p('/sys/fs/cgroup/cpu.max').read_text().strip() == '50000 100000'
        status = p('/proc/self/status').read_text()
        assert 'NoNewPrivs:\\t1' in status and 'CapEff:\\t0000000000000000' in status
        assert 'Seccomp:\\t2' in status
        assert not p('/var/run/docker.sock').exists()
        assert not p('/github/workspace').exists()
        assert not any(k in os.environ for k in ('OPENROUTER_API_KEY', 'TYPESAFE_API_KEY', 'GITHUB_TOKEN'))
        for path in ('/etc/abs-probe', '/usr/local/abs-probe'):
            try:
                p(path).write_text('must fail')
            except OSError as exc:
                assert exc.errno == errno.EROFS, exc
            else:
                raise AssertionError('root filesystem is writable')
        assert os.listdir('/sys/class/net') == ['lo']
        sock = socket.socket()
        sock.settimeout(1)
        try:
            sock.connect(('192.0.2.1', 443))
        except OSError as exc:
            assert exc.errno == errno.ENETUNREACH, exc
        else:
            raise AssertionError('external network reachable')
        finally:
            sock.close()
        p('/tmp/small').write_bytes(b'works')
        assert p('/tmp/small').read_bytes() == b'works'
    """)
    probe("aggregate-scratch-and-shared-memory", """
        import errno, pathlib
        for root, limit in (('/tmp', 8*1024*1024), ('/dev/shm', 1024*1024)):
            written = 0
            try:
                for i in range(160):
                    # Many individually small files must share one mount quota.
                    with open(f'{root}/{i}', 'wb', buffering=0) as stream:
                        written += stream.write(b'x' * (64*1024))
            except OSError as exc:
                assert exc.errno == errno.ENOSPC, exc
            else:
                raise AssertionError('aggregate write quota did not stop writes')
            assert 0 < written <= limit, written
    """)
    probe("hard-memory-oom", """
        import os, signal
        from pathlib import Path
        cgroup = Path('/sys/fs/cgroup')
        assert (cgroup / 'memory.max').read_text().strip() == str(64*1024*1024)
        def events():
            return {key: int(value) for key, value in
                    (line.split() for line in (cgroup / 'memory.events').read_text().splitlines())}
        before = events()
        child = os.fork()
        if child == 0:
            blocks = [bytearray(1024*1024) for _ in range(256)]
            os._exit(1)
        _, status = os.waitpid(child, 0)
        after = events()
        assert os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGKILL, status
        assert all(after[key] > before[key] for key in ('max', 'oom', 'oom_kill')), (before, after)
        print('kernel-oom-verified')
    """, kernel_oom=True)
    probe("process-count", """
        import errno, os, signal, time
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
                    time.sleep(15)
                    os._exit(0)
                children.append(pid)
            assert denied and 0 < len(children) < 16, len(children)
        finally:
            for pid in children:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
    """)
    probe("cpu-throttling", """
        import pathlib, time
        def stats():
            return dict(line.split() for line in pathlib.Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())
        before = int(stats()['nr_throttled'])
        end = time.monotonic() + 2
        while time.monotonic() < end:
            pass
        assert int(stats()['nr_throttled']) > before, 'CPU quota did not throttle'
    """)
    probe("deadline-and-descendant-cleanup", """
        import os, time
        os.fork()
        time.sleep(60)
    """, deadline=True)
    print(json.dumps({
        "status": "passed", "scope": "ci-resource-feasibility-only", "image": IMAGE,
        "engine_backend_integrated": False, "unattended_ready": False,
        "model_called": False, "deployment_changed": False,
    }), flush=True)


if __name__ == "__main__":
    main()
