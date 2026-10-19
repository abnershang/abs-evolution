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

    @unittest.skipIf(sys.platform == "linux", "non-Linux fail-closed selection contract")
    def test_explicit_docker_on_unsupported_host_never_falls_back_to_seatbelt(self):
        doctor = self.command("doctor", "--sandbox", "docker", expected_code=2, with_state=False)
        self.assertEqual(doctor["status"], "blocked")
        self.assertFalse(doctor["hard_resource_limits"])
        self.assertFalse(doctor["unattended_ready"])
        self.assertFalse(doctor["model_called"])
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

    def test_classification_requires_reviewed_external_input_and_never_initializes_state(self):
        request_path = PROJECT / "examples" / "optimization" / "request.json"
        denied = self.command("classify", "--request", request_path, expected_code=2)
        self.assertEqual(denied["error"]["code"], "provider_blocked_external_not_allowed")
        missing_key = self.command("classify", "--request", request_path, "--allow-external", expected_code=2)
        self.assertEqual(missing_key["error"]["code"], "provider_credentials_missing")
        malformed = self.root / "duplicate-keys.json"
        malformed.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
        invalid = self.command("classify", "--request", malformed, "--allow-external", expected_code=2)
        self.assertEqual(invalid["error"]["code"], "invalid_opportunity_request")
        self.assertFalse(self.state.exists())

    def test_assessment_requires_reviewed_external_input_without_creating_state(self):
        request_path = PROJECT / "examples" / "optimization" / "request.json"
        denied = self.command("assess", "--request", request_path, expected_code=2)
        self.assertEqual(denied["error"]["code"], "provider_blocked_external_not_allowed")
        missing_key = self.command("assess", "--request", request_path, "--allow-external", expected_code=2)
        self.assertEqual(missing_key["error"]["code"], "provider_credentials_missing")
        malformed = self.root / "assessment-duplicate-keys.json"
        malformed.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
        invalid = self.command("assess", "--request", malformed, "--allow-external", expected_code=2)
        self.assertEqual(invalid["error"]["code"], "invalid_opportunity_request")
        self.assertFalse(self.state.exists())

    def test_assessment_cli_keeps_nonexclusive_directions_and_single_batch_usage(self):
        from abs_evolution.cli import main

        request_path = PROJECT / "examples" / "optimization" / "request.json"
        wire = b"OK\n" + json.dumps({
            "model": "jev-1.13.0",
            "answers": {
                "opportunity": {"type": "choice", "choice": "worth_comparing", "confidence": 0.9,
                                "probabilities": {"worth_comparing": 0.95, "no_opportunity": 0.03,
                                                  "insufficient_evidence": 0.02}},
                "quality": {"type": "noul", "noul": 0.1},
                "efficiency": {"type": "noul", "noul": 0.92},
                "method": {"type": "noul", "noul": 0.88},
                "reuse": {"type": "noul", "noul": 0.55},
            },
            "usage": {"input_tokens": 160, "output_tokens": 90},
        }).encode()
        stream = io.StringIO()
        with mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": "fixture-assessment-key"}, clear=True), \
                mock.patch("abs_evolution.jev._execute_worker", return_value=wire) as transport, \
                contextlib.redirect_stdout(stream):
            code = main(["--state", str(self.state), "assess", "--request", str(request_path), "--allow-external"])
        self.assertEqual(code, 0)
        output = json.loads(stream.getvalue())
        self.assertEqual(output["decision"], "worth_comparing")
        self.assertEqual(output["supported_dimensions"], ["efficiency", "method"])
        self.assertEqual(output["dimensions"]["reuse"], {"probability_yes": 0.55, "status": "uncertain"})
        self.assertEqual(output["next_step"], "compare")
        self.assertEqual(output["usage"]["input_tokens"], 160)
        self.assertNotIn("input_tokens", output["opportunity"])
        self.assertNotIn("observation", output)
        for field in ("candidate_executed", "registry_changed", "deployment_changed"):
            self.assertFalse(output[field])
        self.assertFalse(self.state.exists())
        transport.assert_called_once()

    def test_classify_cli_uses_real_jev_adapter_without_executing_or_persisting(self):
        from abs_evolution.cli import main

        request_path = PROJECT / "examples" / "optimization" / "request.json"
        sent_requests = []

        def respond(worker_input, **_kwargs):
            # Substitute only the network boundary, not classification or parsing.
            envelope = json.loads(bytes(worker_input))
            sent_requests.append(json.loads(envelope["body"]))
            return b"OK\n" + json.dumps({
                "model": "jev-1.13.0",
                "answers": {"decision": {
                    "type": "choice", "choice": "efficiency", "confidence": 1.0,
                    "probabilities": {"quality": 0.0, "efficiency": 1.0, "method": 0.0,
                                      "reuse": 0.0, "no_opportunity": 0.0, "insufficient_evidence": 0.0},
                }},
                "usage": {"input_tokens": 100, "output_tokens": 20},
            }).encode()

        stream = io.StringIO()
        with mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": "fixture-classification-key"}, clear=True), \
                mock.patch("abs_evolution.jev._execute_worker", side_effect=respond) as transport, \
                mock.patch("abs_evolution.openrouter.OpenRouterProvider.propose") as generator, \
                contextlib.redirect_stdout(stream):
            exit_code = main([
                "--state", str(self.state), "classify", "--request", str(request_path), "--allow-external",
            ])
        self.assertEqual(exit_code, 0)
        output = json.loads(stream.getvalue())
        self.assertEqual(output["dimension"], "efficiency")
        self.assertEqual(output["next_step"], "compare")
        self.assertEqual(output["observed_outcome"], "accepted")
        self.assertEqual(output["judgment"]["input_tokens"], 100)
        self.assertFalse(output["candidate_executed"])
        self.assertFalse(output["registry_changed"])
        self.assertFalse(output["deployment_changed"])
        self.assertEqual(sent_requests[0]["state"]["request"], json.loads(request_path.read_text()))
        self.assertEqual(sent_requests[0]["questions"]["decision"]["type"], "choice")
        self.assertNotIn("observation", output)
        self.assertFalse(self.state.exists())
        transport.assert_called_once()
        generator.assert_not_called()

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
            signal_task_id = f"signal-task-{number}"
            self.command("task", "begin", signal_task_id)
            self.command("task", "end", signal_task_id)
            signal_receipt = self.root / f"signal-receipt-{number}.json"
            signal_receipt.write_text(json.dumps({
                "id": f"signal-receipt-{number}", "task_id": signal_task_id, "snapshot_id": baseline,
                "outcome": "accepted", "failure_tags": [], "evidence_refs": [f"signal-evidence-{number}"],
                "task_family": "support",
                "optimization_signals": [{"dimension": "efficiency", "tag": "fewer_steps"}],
            }), encoding="utf-8")
            self.command("receipt", "record", signal_receipt)
        output = self.command("opportunities")
        failure, optimization = output["opportunities"]
        self.assertEqual(failure["failure_tag"], "premature_acceptance")
        self.assertEqual(failure["distinct_tasks"], 2)
        self.assertEqual(failure["status"], "needs_investigation")
        self.assertEqual(optimization["tag"], "fewer_steps")
        self.assertEqual(optimization["signal_type"], "optimization")
        self.assertEqual(optimization["observed_outcomes"], ["accepted"])
        self.assertEqual(optimization["status"], "needs_comparison")
        state = self.command("status")
        self.assertEqual(state["counts"]["experiments"], 0)
        self.assertEqual(state["channels"][0]["snapshot_id"], baseline)

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
