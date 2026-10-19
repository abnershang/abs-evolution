"""Check the installed CLI with synthetic data, outside the source checkout."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: check_installed.py /path/to/examples/acceptance")
    examples = Path(sys.argv[1]).resolve(strict=True)
    environment = dict(os.environ)
    for key in ("PYTHONPATH", "OPENROUTER_API_KEY", "TYPESAFE_API_KEY"):
        environment.pop(key, None)
    with tempfile.TemporaryDirectory(prefix="abs-evolution-installed-") as raw:
        root = Path(raw).resolve()
        state = root / "state"
        process = subprocess.run(
            [sys.executable, "-I", "-m", "abs_evolution", "--state", str(state),
             "propose", "--provider", "file", "--request", str(examples / "request.json"),
             "--proposal-file", str(examples / "proposal.json")],
            cwd=root, env=environment, capture_output=True, text=True, timeout=15,
        )
        if process.returncode != 0:
            raise SystemExit(f"Installed proposal command failed with exit {process.returncode}")
        receipt = json.loads(process.stdout)
        if receipt["status"] != "proposed":
            raise SystemExit("Installed proposal command did not produce a proposal")
        expected = json.loads((examples / "proposal.json").read_text(encoding="utf-8"))
        if receipt["proposal"] != expected:
            raise SystemExit("Installed package changed the supplied fixture proposal")
        for key in ("candidate_executed", "registry_changed", "deployment_changed"):
            if receipt[key] is not False:
                raise SystemExit(f"Installed proposal command reported an unexpected side effect: {key}")
        if state.exists():
            raise SystemExit("Proposal-only execution created registry state")
    print("Installed package: proposal-only CLI passed without registry or deployment changes.")


if __name__ == "__main__":
    main()
