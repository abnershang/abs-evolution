"""User-command contracts, including real sandboxed lifecycle integration."""

from __future__ import annotations

import json
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


PROJECT = Path(__file__).resolve().parents[1]
EXAMPLE = PROJECT / "examples" / "acceptance"


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="abs-evolution-cli-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.state = self.root / "state"

    def command(self, *arguments, expected_code=0, with_state=True):
        env = dict(os.environ)
        env.pop("OPENROUTER_API_KEY", None)
        env.pop("TYPESAFE_API_KEY", None)
        env["PYTHONPATH"] = str(PROJECT / "src")
        state_arguments = ["--state", str(self.state)] if with_state else []
        result = subprocess.run(
            [sys.executable, "-m", "abs_evolution", *state_arguments, *map(str, arguments)],
            cwd=PROJECT, env=env, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, expected_code, result.stderr + result.stdout)
        stream = result.stdout if result.stdout.strip() else result.stderr
        return json.loads(stream)

    def test_read_command_does_not_initialize_or_mutate_missing_state(self):
        value = self.command("status", expected_code=2)
        self.assertEqual(value["error"]["code"], "state_missing")
        self.assertFalse(self.state.exists())

    def test_proposal_only_command_does_not_require_or_create_registry_state(self):
        value = self.command(
            "propose", "--provider", "file", "--request", EXAMPLE / "request.json",
            "--proposal-file", EXAMPLE / "proposal.json", with_state=False,
        )
        self.assertEqual(value["status"], "proposed")
        self.assertEqual(value["proposal"]["changes"][0]["path"], "tools/acceptance.py")
        self.assertFalse(value["candidate_executed"])
        self.assertFalse(value["registry_changed"])
        self.assertIsNone(value["usage"]["input_tokens"])
        self.assertIsNone(value["usage"]["cost_credits"])
        self.assertFalse(self.state.exists())

    def test_openrouter_requires_explicit_model_and_reports_missing_credentials(self):
        missing_model = self.command(
            "propose", "--provider", "openrouter", "--request", EXAMPLE / "request.json",
            expected_code=2, with_state=False,
        )
        self.assertEqual(missing_model["error"]["code"], "model_required")
        missing_key = self.command(
            "propose", "--provider", "openrouter", "--model", "openai/test-model",
            "--request", EXAMPLE / "request.json", expected_code=2, with_state=False,
        )
        self.assertEqual(missing_key["error"]["code"], "provider_credentials_missing")
        self.assertFalse(self.state.exists())

    def test_openrouter_cannot_override_missing_resource_limits_for_unreviewed_code(self):
        self.command("init")
        baseline = self.command("snapshot", "import", EXAMPLE / "baseline", "--kind", "tool")["id"]
        self.command("channel", "bootstrap", baseline)
        for provider, extras in (("openrouter", []), ("hybrid", ["--allow-external"])):
            with self.subTest(provider=provider):
                result = self.command(
                    "evolve", "--provider", provider, "--model", "openai/test-model",
                    "--suite", EXAMPLE / "suite.json", "--allow-shared-resources", *extras,
                    expected_code=2,
                )
                self.assertEqual(result["status"], "blocked")
                self.assertEqual(result["reason"], "resource_isolation_incomplete")
                self.assertIsNone(result["experiment_id"])
                self.assertIsNone(result["candidate_id"])

    def test_hybrid_cli_requires_external_assertion_and_rejects_misplaced_options(self):
        denied = self.command(
            "propose", "--provider", "hybrid", "--request", EXAMPLE / "request.json",
            expected_code=2, with_state=False,
        )
        self.assertEqual(denied["error"]["code"], "provider_blocked_external_not_allowed")
        misplaced = self.command(
            "propose", "--provider", "file", "--proposal-file", EXAMPLE / "proposal.json",
            "--request", EXAMPLE / "request.json", "--jev-model", "jev-1.13.0",
            expected_code=2, with_state=False,
        )
        self.assertEqual(misplaced["error"]["code"], "invalid_arguments")
        fifo = self.root / "candidate-pipe"
        os.mkfifo(fifo)
        nonregular = self.command(
            "propose", "--provider", "hybrid", "--allow-external",
            "--request", EXAMPLE / "request.json", "--candidate-pool", fifo,
            expected_code=2, with_state=False,
        )
        self.assertEqual(nonregular["error"]["code"], "invalid_candidate_pool")
        self.assertFalse(self.state.exists())

    def test_hybrid_cli_selects_supplied_candidate_and_reports_separate_judge_usage(self):
        from abs_evolution.cli import main
        from abs_evolution.contracts import ChoiceResult

        request = json.loads((EXAMPLE / "request.json").read_text())
        proposal = json.loads((EXAMPLE / "proposal.json").read_text())
        pool = self.root / "pool.json"
        pool.write_text(json.dumps([{
            "id": "reviewed-fix", "baseline_id": request["baseline_id"], "proposal": proposal,
        }]))
        choice = ChoiceResult(
            option="reuse_reviewed-fix",
            probabilities={"reuse_reviewed-fix": 1.0, "generate": 0.0, "insufficient_evidence": 0.0},
            confidence=1.0, duration_seconds=0.01,
            model="jev-1.13.0", input_tokens=100, output_tokens=20,
        )
        stream = io.StringIO()
        with mock.patch("abs_evolution.jev.JevJudge.choose", return_value=choice) as judge, \
                mock.patch("abs_evolution.openrouter.OpenRouterProvider.propose") as generator, \
                contextlib.redirect_stdout(stream):
            exit_code = main([
                "propose", "--provider", "hybrid", "--allow-external",
                "--request", str(EXAMPLE / "request.json"), "--candidate-pool", str(pool),
                "--model", "openai/test-model", "--jev-model", "jev-1.13.0",
                "--min-choice-confidence", "0.9", "--min-choice-probability", "0.9",
            ])
        self.assertEqual(exit_code, 0)
        value = json.loads(stream.getvalue())
        self.assertEqual(value["proposal"], proposal)
        self.assertEqual(value["routing"]["route"], "reuse")
        self.assertEqual(value["routing"]["selected_candidate_id"], "reviewed-fix")
        self.assertEqual(value["routing"]["judge"]["input_tokens"], 100)
        self.assertFalse(value["routing"]["generator"]["called"])
        self.assertIsNone(value["usage"]["model"])
        self.assertIsNone(value["usage"]["cost_credits"])
        self.assertFalse(value["candidate_executed"])
        self.assertFalse(value["registry_changed"])
        self.assertFalse(value["deployment_changed"])
        self.assertFalse(self.state.exists())
        judge.assert_called_once()
        generator.assert_not_called()

    def test_bounded_receipt_and_public_opportunity_command(self):
        self.command("init")
        baseline = self.command("snapshot", "import", EXAMPLE / "baseline", "--kind", "tool")["id"]
        self.command("channel", "bootstrap", baseline)
        for number in (1, 2):
            task_id = f"task-{number}"
            self.command("task", "begin", task_id)
            self.command("task", "end", task_id)
            receipt = self.root / f"receipt-{number}.json"
            receipt.write_text(json.dumps({
                "id": f"receipt-{number}", "task_id": task_id, "snapshot_id": baseline,
                "outcome": "failed", "failure_tags": ["premature_acceptance"],
                "evidence_refs": [f"evidence-{number}"],
            }), encoding="utf-8")
            self.command("receipt", "record", receipt)
        output = self.command("opportunities")
        self.assertEqual(output["opportunities"][0]["distinct_tasks"], 2)
        self.assertEqual(output["opportunities"][0]["status"], "needs_investigation")

    @unittest.skipUnless(sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").exists(),
                         "requires the actual macOS sandbox backend")
    def test_real_experiment_promotion_and_rollback_preserve_running_task_versions(self):
        self.command("init")
        baseline = self.command("snapshot", "import", EXAMPLE / "baseline", "--kind", "tool")["id"]
        self.command("channel", "bootstrap", baseline)
        blocked = self.command(
            "evolve", "--suite", EXAMPLE / "suite.json", "--provider", "file",
            "--proposal-file", EXAMPLE / "proposal.json", expected_code=2,
        )
        self.assertEqual(blocked["status"], "blocked")
        self.assertEqual(blocked["reason"], "resource_isolation_incomplete")
        self.assertIsNone(blocked["candidate_id"])
        experiment = self.command(
            "evolve", "--suite", EXAMPLE / "suite.json", "--provider", "file",
            "--proposal-file", EXAMPLE / "proposal.json", "--allow-shared-resources",
        )
        self.assertIsNotNone(experiment["evaluation_id"])
        self.assertEqual(experiment["scores"], {
            "baseline_training": 0.5, "candidate_training": 1.0,
            "baseline_holdout": 0.25, "candidate_holdout": 1.0,
        })
        candidate = experiment["candidate_id"]
        self.assertNotEqual(candidate, baseline)
        old_lease = self.command("task", "begin", "old-task")
        self.assertEqual(old_lease["snapshot_id"], baseline)
        old_source = Path(old_lease["snapshot_path"]) / "tools" / "acceptance.py"
        original_bytes = old_source.read_bytes()
        self.command("channel", "promote", experiment["evaluation_id"], "--expected-current", baseline)
        new_lease = self.command("task", "begin", "new-task")
        self.assertEqual(new_lease["snapshot_id"], candidate)
        self.assertEqual(self.command("task", "begin", "old-task")["snapshot_id"], baseline)
        self.assertEqual(old_source.read_bytes(), original_bytes)
        self.command("channel", "rollback", baseline, "--expected-current", candidate)
        self.assertEqual(self.command("task", "begin", "new-task")["snapshot_id"], candidate)
        self.assertEqual(self.command("task", "begin", "after-rollback")["snapshot_id"], baseline)
        self.assertEqual((EXAMPLE / "baseline" / "tools" / "acceptance.py").read_bytes(), original_bytes)


if __name__ == "__main__":
    unittest.main()
