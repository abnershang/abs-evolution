"""Behavioral tests for experiment orchestration at its injected boundaries.

The sandbox here is deliberately a mock: it returns fixed process results and
never executes candidate source. Real OS isolation is covered by sandbox tests
and the repository's separate end-to-end acceptance run.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from abs_evolution.contracts import ChoiceResult, Policy, ProcessResult, ProposalResult
from abs_evolution.errors import EvolutionError
from abs_evolution.engine import EvolutionEngine
from abs_evolution.registry import Registry


_TRAINING = [
    {
        "id": "repair-known-defect",
        "input": {"case": "repair"},
        "expected": {"value": "fixed"},
    },
    {
        "id": "preserve-working-path",
        "input": {"case": "preserve", "auth": "ordinary-input-field"},
        "expected": {"value": "good"},
    },
]
_HOLDOUT = [
    {
        "id": "protected-answer-not-for-provider",
        "input": {"case": "protected", "marker": "holdout-input-secret"},
        "expected": {"value": "holdout-good", "answer_marker": "holdout-answer-secret"},
    }
]


class _MockSandbox:
    """Return controlled ProcessResults without executing the candidate file."""

    backend_identity = "unit-test-mock-boundary"
    # This test double does not execute code. The flag only lets orchestration
    # tests reach later stages; it is not evidence about the real sandbox.
    hard_resource_limits = True

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.fail_call: int | None = None
        self.candidate_training_failure = False
        self.boolean_case = False

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
        index = len(self.calls)
        call = {
            "argv": list(argv),
            "cwd": cwd,
            "read_paths": list(read_paths),
            "write_paths": list(write_paths),
            "stdin": stdin,
            "timeout_seconds": timeout_seconds,
            "max_output_bytes": max_output_bytes,
        }
        self.calls.append(call)
        if self.fail_call == index:
            return ProcessResult("timed_out", None, "", "", timeout_seconds, "mock timeout")

        source = Path(argv[-1]).read_text(encoding="utf-8")
        is_candidate = "CANDIDATE_MARKER" in source
        item = json.loads(stdin)
        case_name = item.get("case")
        if case_name == "repair":
            if self.boolean_case:
                output = {"value": 1}
            elif is_candidate and not self.candidate_training_failure:
                output = {"value": "fixed"}
            else:
                output = {"value": "wrong"}
        elif case_name == "preserve":
            output = {"value": "good"}
        elif case_name == "protected":
            output = (
                {"value": "holdout-good", "answer_marker": "holdout-answer-secret"}
                if is_candidate else {"value": "wrong"}
            )
        elif case_name == "protected-alt":
            output = (
                {"value": "holdout-alt", "answer_marker": "holdout-alt-secret"}
                if is_candidate else {"value": "wrong"}
            )
        else:
            output = None
        return ProcessResult(
            "passed", 0,
            json.dumps(output, separators=(",", ":"), allow_nan=False),
            "", 0.001,
        )


class _MockProvider:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []
        self.timeouts: list[float] = []
        self.error: EvolutionError | None = None
        self.extra_changes: list[dict[str, object]] = []
        self.observed_usage: dict[str, object] = {}

    def propose(
        self,
        request: dict[str, object],
        *,
        timeout_seconds: float,
        max_output_bytes: int,
    ) -> ProposalResult:
        self.requests.append(json.loads(json.dumps(request)))
        self.timeouts.append(timeout_seconds)
        if self.error is not None:
            raise self.error
        changes = [{"path": "tools/solver.py", "content": "# CANDIDATE_MARKER\n"}]
        changes.extend(self.extra_changes)
        return ProposalResult(
            {
                "summary": "A reviewed test proposal.",
                "kind": "tool",
                "changes": changes,
            },
            duration_seconds=0.001,
            **self.observed_usage,
        )


class EvolutionEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="abs-evolution-engine-test-")
        self.root = Path(self.temp.name)
        self.policy = Policy()
        self.registry = Registry.initialize(self.root / "registry", self.policy)
        self.source = self.root / "baseline"
        (self.source / "tools").mkdir(parents=True)
        (self.source / "tools" / "solver.py").write_text("# BASELINE_MARKER\n", encoding="utf-8")
        (self.source / "tools" / "helper.py").write_text("def helper(): return 'safe'\n", encoding="utf-8")
        (self.source / "tools" / "docs").mkdir()
        (self.source / "tools" / "docs" / "unrelated.md").write_text(
            "unrelated document that must not be sent to the provider\n", encoding="utf-8",
        )
        (self.source / "tools" / "credentials.py").write_text(
            "API_KEY = 'a-real-looking-secret-value-12345'\n", encoding="utf-8",
        )
        imported = self.registry.import_snapshot(self.source, kind="tool")
        self.baseline_id = imported["id"]
        self.registry.bootstrap(self.baseline_id, channel="stable")
        self.sandbox = _MockSandbox()
        self.provider = _MockProvider()
        self.engine = EvolutionEngine(self.registry, self.sandbox, self.provider)
        self.suite_path = self.root / "suite.json"
        self.write_suite(_TRAINING, _HOLDOUT)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_suite(self, training: list[dict[str, object]], holdout: list[dict[str, object]]) -> None:
        self.suite_path.write_text(
            json.dumps({
                "schema_version": 1,
                "id": "engine-unit-suite",
                "mode": "python",
                "entrypoint": "tools/solver.py",
                "training": training,
                "holdout": holdout,
            }),
            encoding="utf-8",
        )

    def test_full_cycle_sends_only_training_and_records_aggregate_evidence(self) -> None:
        from abs_evolution.hybrid import HybridProposalProvider

        judge = mock.Mock()
        judge.choose.return_value = ChoiceResult(
            option="generate", probabilities={"generate": 1.0, "insufficient_evidence": 0.0},
            confidence=1.0, duration_seconds=0.002, model="jev-1.13.0",
            input_tokens=80, output_tokens=10,
        )
        self.engine = EvolutionEngine(
            self.registry, self.sandbox, HybridProposalProvider(judge, self.provider),
        )
        self.provider.observed_usage = {
            "input_tokens": 120,
            "output_tokens": 40,
            "model": "test/model-v1",
            "generation_id": "test-generation-1",
            "cost_credits": 0.0002,
        }
        result = self.engine.run(self.suite_path, auto_promote=True)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["scores"], {
            "baseline_training": 0.5,
            "candidate_training": 1.0,
            "baseline_holdout": 0.0,
            "candidate_holdout": 1.0,
        })
        self.assertTrue(result["auto_promoted"])
        self.assertFalse(result["shared_host_resources"])
        usage = result["provider_usage"]
        self.assertEqual({key: value for key, value in usage.items() if key != "routing"}, self.provider.observed_usage)
        self.assertEqual(usage["routing"]["route"], "generate")
        self.assertEqual(usage["routing"]["judge"]["input_tokens"], 80)
        self.assertTrue(usage["routing"]["generator"]["called"])
        self.assertEqual(usage["routing"]["generator"]["model"], "test/model-v1")
        self.assertIsNotNone(result["evaluation_id"])
        self.assertEqual(result["promotion"]["snapshot_id"], result["candidate_id"])
        self.assertEqual(result["promotion"]["channel"], "stable")
        self.assertEqual(result["promotion"]["previous_snapshot_id"], self.baseline_id)
        self.assertEqual(result["promotion"]["evaluation_id"], result["evaluation_id"])
        self.assertEqual(self.registry.channel("stable")["snapshot_id"], result["candidate_id"])
        self.assertEqual(self.registry.experiment(result["experiment_id"])["status"], "completed")

        request = self.provider.requests[0]
        self.assertEqual(set(request), {
            "schema_version", "objective", "baseline_id", "files",
            "training_examples", "allowed_mutation_paths", "max_changes",
        })
        encoded_request = json.dumps(request, sort_keys=True)
        judge.choose.assert_called_once()
        judge_state = judge.choose.call_args.args[0]
        self.assertEqual(judge_state["request"], request)
        self.assertEqual(judge_state["candidates"], [])
        self.assertNotIn("protected-answer-not-for-provider", encoded_request)
        self.assertNotIn("holdout-input-secret", encoded_request)
        self.assertNotIn("holdout-answer-secret", encoded_request)
        self.assertNotIn(str(self.suite_path), encoded_request)
        self.assertIn("tools/solver.py", request["files"])
        self.assertIn("tools/helper.py", request["files"])
        self.assertNotIn("tools/credentials.py", request["files"])
        self.assertNotIn("tools/docs/unrelated.md", request["files"])
        examples = request["training_examples"]
        self.assertEqual({item["id"] for item in examples}, {"repair-known-defect", "preserve-working-path"})
        self.assertTrue(any("auth" in item["input"] for item in examples))

        # The mocked sandbox receives raw input only; it does not execute the
        # script or receive expected answers. Real isolation is verified elsewhere.
        for call in self.sandbox.calls:
            passed_input = json.loads(call["stdin"])
            self.assertIn("case", passed_input)
            self.assertNotIn("expected", passed_input)
            self.assertEqual(len(call["read_paths"]), 1)
            self.assertEqual(len(call["write_paths"]), 1)
        detail = self.registry.experiment(result["experiment_id"])["detail"]
        self.assertNotIn("holdout-answer-secret", json.dumps(detail, sort_keys=True))
        self.assertEqual(detail["provider_usage"], result["provider_usage"])

    def test_training_quality_failure_keeps_protected_holdout_unused(self) -> None:
        self.sandbox.candidate_training_failure = True
        result = self.engine.run(self.suite_path)

        self.assertEqual(result["status"], "quality_failed")
        self.assertEqual(result["reason"], "candidate_training_below_minimum")
        self.assertEqual(result["scores"], {"baseline_training": 0.5, "candidate_training": 0.5})
        self.assertIsNone(result["evaluation_id"])
        self.assertEqual(self.registry.experiment(result["experiment_id"])["status"], "failed")
        self.assertEqual(len(self.sandbox.calls), 4)  # baseline + candidate training only

        # The same exact protected cases remain available for one later attempt.
        self.sandbox.candidate_training_failure = False
        second = self.engine.run(self.suite_path)
        self.assertEqual(second["status"], "completed")

    def test_process_timeout_is_infrastructure_failure_not_zero_quality(self) -> None:
        self.sandbox.fail_call = 0
        result = self.engine.run(self.suite_path)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "sandbox_timed_out")
        self.assertEqual(result["scores"], {})
        self.assertIsNone(result["candidate_id"])
        self.assertIsNone(result["evaluation_id"])
        self.assertEqual(self.provider.requests, [])
        experiment = self.registry.experiment(result["experiment_id"])
        self.assertEqual(experiment["status"], "failed")
        self.assertEqual(experiment["detail"]["scores"], {})

    def test_boolean_and_number_are_not_equal_json_values(self) -> None:
        self.sandbox.boolean_case = True
        training = [{
            "id": "typed-json-comparison",
            "input": {"case": "repair"},
            "expected": {"value": True},
        }]
        self.write_suite(training, _HOLDOUT)

        result = self.engine.run(self.suite_path)

        self.assertEqual(result["status"], "quality_failed")
        self.assertEqual(result["scores"], {"baseline_training": 0.0, "candidate_training": 0.0})
        self.assertIsNone(result["evaluation_id"])
        self.assertEqual(len(self.sandbox.calls), 2)

    def test_existing_foreground_lease_blocks_before_claim(self) -> None:
        self.registry.begin_task("foreground-check", channel="stable")
        try:
            result = self.engine.run(self.suite_path)
        finally:
            self.registry.end_task("foreground-check")

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["reason"], "foreground_lease_active")
        self.assertIsNone(result["experiment_id"])
        self.assertEqual(self.provider.requests, [])
        self.assertEqual(self.sandbox.calls, [])

    def test_invalid_overlapping_splits_do_not_claim_or_call_provider(self) -> None:
        overlapping = [
            {"id": "train", "input": {"case": "same"}, "expected": 0},
        ]
        holdout = [
            {"id": "holdout", "input": {"case": "same"}, "expected": 1},
        ]
        self.write_suite(overlapping, holdout)

        result = self.engine.run(self.suite_path)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "suite_splits_overlap")
        self.assertIsNone(result["experiment_id"])
        self.assertEqual(self.provider.requests, [])
        self.assertEqual(self.sandbox.calls, [])

    def test_duplicate_normalized_inputs_within_a_split_are_rejected(self) -> None:
        duplicate = dict(_HOLDOUT[0], id="same-input-new-label")
        self.write_suite(_TRAINING, [_HOLDOUT[0], duplicate])

        result = self.engine.run(self.suite_path)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "duplicate_suite_input")
        self.assertIsNone(result["experiment_id"])
        self.assertEqual(self.provider.requests, [])
        self.assertEqual(self.sandbox.calls, [])

    def test_suite_path_and_full_copies_inside_snapshot_are_rejected(self) -> None:
        suite_bytes = self.suite_path.read_bytes()
        parsed_suite = json.loads(suite_bytes)
        variants = ("suite_path", "raw_copy", "canonical_copy")

        for index, variant in enumerate(variants):
            with self.subTest(variant=variant):
                source = self.root / f"contaminated-{variant}"
                (source / "tools").mkdir(parents=True)
                (source / "tools" / "solver.py").write_text("# BASELINE_MARKER\n", encoding="utf-8")
                if variant == "suite_path":
                    (source / "suite.json").write_bytes(suite_bytes)
                elif variant == "raw_copy":
                    (source / "renamed.payload").write_bytes(suite_bytes)
                else:
                    (source / "renamed.payload").write_text(
                        json.dumps(parsed_suite, sort_keys=True, indent=2), encoding="utf-8",
                    )
                imported = self.registry.import_snapshot(source, kind="tool")
                channel = f"contaminated-{index}"
                self.registry.bootstrap(imported["id"], channel=channel)
                if variant == "suite_path":
                    selected_suite = self.registry.snapshot_path(imported["id"]) / "suite.json"
                else:
                    selected_suite = self.suite_path

                result = self.engine.run(selected_suite, channel=channel)

                self.assertEqual(result["status"], "blocked")
                self.assertEqual(result["reason"], "suite_snapshot_contamination")
                self.assertIsNone(result["experiment_id"])
                self.assertEqual(self.provider.requests, [])
                self.assertEqual(self.sandbox.calls, [])

    def test_candidate_snapshot_copy_of_suite_is_rejected_before_candidate_training(self) -> None:
        self.provider.extra_changes = [{
            "path": "tools/suite-copy.json",
            "content": self.suite_path.read_text(encoding="utf-8"),
        }]

        result = self.engine.run(self.suite_path)

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["reason"], "suite_snapshot_contamination")
        self.assertIsNotNone(result["candidate_id"])
        self.assertIsNone(result["evaluation_id"])
        self.assertEqual(len(self.sandbox.calls), 2)  # baseline training only
        self.assertEqual(self.registry.status()["counts"]["holdout_uses"], 0)

    def test_holdout_identity_ignores_ids_and_order_for_one_use(self) -> None:
        holdout = [
            _HOLDOUT[0],
            {
                "id": "second-protected-case",
                "input": {"case": "protected-alt", "marker": "another-input-secret"},
                "expected": {"value": "holdout-alt", "answer_marker": "holdout-alt-secret"},
            },
        ]
        self.write_suite(_TRAINING, holdout)
        first = self.engine.run(self.suite_path)
        self.assertEqual(first["status"], "completed")

        relabeled_reversed = [
            dict(holdout[1], id="new-label-two"),
            dict(holdout[0], id="new-label-one"),
        ]
        self.write_suite(_TRAINING, relabeled_reversed)
        second = self.engine.run(self.suite_path)

        self.assertEqual(second["status"], "blocked")
        self.assertEqual(second["reason"], "holdout_already_used")
        self.assertIsNone(second["evaluation_id"])
        self.assertEqual(self.registry.experiment(second["experiment_id"])["status"], "blocked")

    def test_provider_blocked_is_not_reported_as_ordinary_failure(self) -> None:
        self.provider.error = EvolutionError("provider_blocked", "provider capability boundary is unavailable")

        result = self.engine.run(self.suite_path)

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["reason"], "provider_blocked")
        self.assertIsNone(result["candidate_id"])
        self.assertEqual(self.registry.experiment(result["experiment_id"])["status"], "blocked")

    def test_shared_host_resource_use_requires_explicit_opt_in(self) -> None:
        self.sandbox.hard_resource_limits = False

        blocked = self.engine.run(self.suite_path)

        self.assertEqual(blocked["status"], "blocked")
        self.assertEqual(blocked["reason"], "resource_isolation_incomplete")
        self.assertIsNone(blocked["experiment_id"])
        self.assertIsNone(blocked["shared_host_resources"])
        self.assertEqual(self.provider.requests, [])
        self.assertEqual(self.sandbox.calls, [])

        opted_in = self.engine.run(self.suite_path, allow_shared_resources=True)
        self.assertEqual(opted_in["status"], "completed")
        self.assertTrue(opted_in["shared_host_resources"])


if __name__ == "__main__":
    unittest.main()
