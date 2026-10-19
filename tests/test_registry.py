from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from abs_evolution.contracts import Policy
from abs_evolution.errors import EvolutionError
from abs_evolution.registry import Registry


class RegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / "registry"
        self.policy = Policy(
            min_training_success=0.75,
            min_holdout_success=0.75,
            min_improvement=0.1,
            max_regression=0.0,
        )
        self.registry = Registry.initialize(self.root, self.policy)
        self.source = self.base / "source"
        (self.source / "skills").mkdir(parents=True)
        (self.source / "skills" / "solver.py").write_text("def solve(x): return x\n", encoding="utf-8")
        self.baseline = self.registry.import_snapshot(self.source, provenance={"source": "test fixture"})
        self.registry.bootstrap(self.baseline["id"])

    def tearDown(self) -> None:
        self.temp.cleanup()

    def assert_code(self, code: str, call) -> EvolutionError:
        with self.assertRaises(EvolutionError) as caught:
            call()
        self.assertEqual(caught.exception.code, code)
        return caught.exception

    def candidate(self, body: str = "def solve(x): return x + 1\n") -> dict:
        return self.registry.create_candidate(
            self.baseline["id"],
            {"skills/solver.py": body},
            provenance={"review": "offline"},
        )

    def claim_and_evaluate(
        self,
        candidate_id: str,
        *,
        status: str = "passed",
        baseline_training: float | None = 0.8,
        candidate_training: float | None = 1.0,
        baseline_holdout: float | None = 0.6,
        candidate_holdout: float | None = 0.8,
        finish: bool = True,
        holdout_digest: str | None = None,
    ) -> tuple[dict, dict]:
        claim = self.registry.claim_experiment(baseline_id=self.baseline["id"])
        if holdout_digest is not None:
            self.registry.consume_holdout(holdout_digest, claim["id"])
        evidence = {
            "candidate_id": candidate_id,
            "baseline_id": self.baseline["id"],
            "suite_digest": "a" * 64,
            "environment_digest": "b" * 64,
            "policy_digest": claim["policy_digest"],
            "status": status,
            "baseline_training": baseline_training,
            "candidate_training": candidate_training,
            "baseline_holdout": baseline_holdout,
            "candidate_holdout": candidate_holdout,
            "violations": [],
            "errors": [],
            "case_counts": {"training": 2, "holdout": 2},
        }
        evaluation = self.registry.record_evaluation(claim["id"], evidence)
        if finish:
            self.registry.finish_experiment(claim["id"], status="completed", elapsed_seconds=1.25)
        return claim, evaluation

    def test_open_does_not_initialize_missing_state(self) -> None:
        missing = self.base / "not-created"
        self.assert_code("state_missing", lambda: Registry(missing))
        self.assertFalse(missing.exists())

    def test_foreground_task_begin_fails_fast_when_registry_writer_is_busy(self) -> None:
        connection = sqlite3.connect(self.root / Registry.DB_NAME, timeout=0, isolation_level=None)
        connection.execute("BEGIN IMMEDIATE")
        started = time.monotonic()
        try:
            self.assert_code("registry_busy", lambda: self.registry.begin_task("busy-task"))
        finally:
            connection.execute("ROLLBACK")
            connection.close()
        self.assertLess(time.monotonic() - started, 5)

    def test_import_excludes_credential_runtime_and_cache_paths(self) -> None:
        source = self.base / "filtered"
        (source / "skills").mkdir(parents=True)
        (source / "runtime").mkdir()
        (source / ".cache").mkdir()
        (source / "secrets").mkdir()
        (source / "skills" / "safe.md").write_text("safe", encoding="utf-8")
        (source / ".env").write_text("not captured", encoding="utf-8")
        (source / "runtime" / "state.json").write_text("not captured", encoding="utf-8")
        (source / ".cache" / "build").write_text("not captured", encoding="utf-8")
        (source / "secrets" / "token.txt").write_text("not captured", encoding="utf-8")

        result = self.registry.import_snapshot(source)
        files = self.registry.snapshot_path(result["id"])
        self.assertEqual(result["file_count"], 1)
        self.assertEqual((files / "skills" / "safe.md").read_text(encoding="utf-8"), "safe")
        self.assertFalse((files / ".env").exists())
        self.assertFalse((files / "runtime").exists())
        self.assertTrue((source / ".env").exists(), "Import must not alter source content")

    def test_import_rejects_symlink_and_snapshot_tamper_is_detected(self) -> None:
        source = self.base / "linked"
        source.mkdir()
        outside = self.base / "outside.txt"
        outside.write_text("outside", encoding="utf-8")
        try:
            (source / "escape.txt").symlink_to(outside)
        except (OSError, NotImplementedError):
            self.skipTest("Symlinks are unavailable on this filesystem")
        self.assert_code("unsafe_source", lambda: self.registry.import_snapshot(source))

        files = self.registry.snapshot_path(self.baseline["id"])
        target = files / "skills" / "solver.py"
        target.chmod(0o644)
        target.write_text("tampered\n", encoding="utf-8")
        self.assert_code("snapshot_corrupt", lambda: self.registry.snapshot_path(self.baseline["id"]))

    def test_policy_tightening_preserves_old_task_snapshot_but_limits_new_imports(self) -> None:
        pinned = self.registry.begin_task("pinned-before-tightening")
        self.assertEqual(pinned["snapshot_id"], self.baseline["id"])

        self.registry.update_policy(Policy(
            max_file_bytes=1,
            max_snapshot_bytes=1,
            max_snapshot_files=1,
            max_store_bytes=1,
        ))
        snapshot_path = self.registry.snapshot_path(self.baseline["id"])
        self.assertEqual(
            (snapshot_path / "skills" / "solver.py").read_text(encoding="utf-8"),
            "def solve(x): return x\n",
        )
        self.assertEqual(
            self.registry.begin_task("pinned-before-tightening")["snapshot_id"],
            self.baseline["id"],
        )

        oversized = self.base / "oversized-under-new-policy"
        (oversized / "skills").mkdir(parents=True)
        (oversized / "skills" / "solver.py").write_text("too large", encoding="utf-8")
        self.assert_code("snapshot_too_large", lambda: self.registry.import_snapshot(oversized))

    def test_candidates_are_bounded_content_addressed_and_noop_is_rejected(self) -> None:
        first = self.candidate()
        second = self.candidate()
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(first["parent_id"], self.baseline["id"])
        self.assert_code(
            "no_op_candidate",
            lambda: self.registry.create_candidate(
                self.baseline["id"], {"skills/solver.py": "def solve(x): return x\n"}
            ),
        )
        self.assert_code(
            "unsafe_path",
            lambda: self.registry.create_candidate(self.baseline["id"], {"skills/../escape": "bad"}),
        )
        self.assert_code(
            "path_not_allowed",
            lambda: self.registry.create_candidate(self.baseline["id"], {"README.md": "outside policy"}),
        )

    def test_candidate_rechecks_current_mutation_policy_at_storage_admission(self) -> None:
        scenarios = (
            (
                "allowed-path tightening",
                Policy(allowed_mutation_paths=("docs/*",)),
                {"skills/solver.py": "def solve(x): return x + 2\n"},
                "path_not_allowed",
            ),
            (
                "change-count tightening",
                Policy(max_changes=1),
                {
                    "skills/solver.py": "def solve(x): return x + 2\n",
                    "skills/extra.py": "def extra(): return True\n",
                },
                "change_limit",
            ),
        )
        for label, tightened_policy, changes, expected_error in scenarios:
            with self.subTest(label=label):
                self.registry.update_policy(self.policy)
                store_entries = self.registry._store_entries

                def tighten_then_store(entries, **kwargs):
                    Registry(self.root).update_policy(tightened_policy)
                    return store_entries(entries, **kwargs)

                with patch.object(self.registry, "_store_entries", side_effect=tighten_then_store):
                    self.assert_code(
                        expected_error,
                        lambda: self.registry.create_candidate(self.baseline["id"], changes),
                    )
                self.assertEqual(
                    [item["id"] for item in self.registry.list_snapshots()],
                    [self.baseline["id"]],
                )

    def test_task_pin_survives_promotion_and_rollback_is_accepted_only(self) -> None:
        self.registry.update_policy(Policy(
            min_training_success=0.75,
            min_holdout_success=0.75,
            min_improvement=0.1,
            max_regression=0.0,
            yield_to_foreground=False,
        ))
        candidate = self.candidate()
        pinned = self.registry.begin_task("task-1")
        self.assertEqual(pinned["snapshot_id"], self.baseline["id"])
        self.assertEqual(self.registry.begin_task("task-1")["snapshot_id"], self.baseline["id"])

        _, evaluation = self.claim_and_evaluate(
            candidate["id"], holdout_digest="c" * 64
        )
        promoted = self.registry.promote(evaluation["id"], expected_current=self.baseline["id"])
        self.assertEqual(promoted["snapshot_id"], candidate["id"])
        self.assertEqual(self.registry.channel()["generation"], 2)

        old_task = self.registry.begin_task("task-1")
        new_task = self.registry.begin_task("task-2")
        self.assertEqual(old_task["snapshot_id"], self.baseline["id"])
        self.assertEqual(new_task["snapshot_id"], candidate["id"])
        self.assertEqual(Path(old_task["snapshot_path"]), self.registry.snapshot_path(self.baseline["id"]))
        self.assert_code(
            "channel_conflict",
            lambda: self.registry.promote(evaluation["id"], expected_current=self.baseline["id"]),
        )

        rolled_back = self.registry.rollback(
            self.baseline["id"], expected_current=candidate["id"]
        )
        self.assertEqual(rolled_back["snapshot_id"], self.baseline["id"])
        self.assertEqual(rolled_back["generation"], 3)
        self.assertEqual(self.registry.begin_task("task-3")["snapshot_id"], self.baseline["id"])
        self.registry.end_task("task-1")
        self.assert_code("task_id_reused", lambda: self.registry.begin_task("task-1"))

    def test_unaccepted_snapshot_cannot_be_rollback_target(self) -> None:
        candidate = self.candidate()
        self.assert_code(
            "rollback_target_unaccepted",
            lambda: self.registry.rollback(candidate["id"], expected_current=self.baseline["id"]),
        )

    def test_receipts_are_strict_bounded_and_idempotent(self) -> None:
        task = self.registry.begin_task("receipt-task")
        receipt = {
            "id": "receipt-1",
            "task_id": task["task_id"],
            "snapshot_id": task["snapshot_id"],
            "outcome": "unknown",
            "failure_tags": ["downstream_unknown"],
            "evidence_refs": ["sha256:0123456789abcdef"],
        }
        first = self.registry.record_receipt(receipt)
        again = self.registry.record_receipt(receipt)
        self.assertTrue(first["created"])
        self.assertFalse(again["created"])
        self.assertEqual(len(self.registry.list_receipts()), 1)
        self.assertEqual(self.registry.list_receipts()[0], receipt)

        signal_receipt = {
            **receipt, "id": "receipt-signal", "outcome": "accepted", "failure_tags": [],
            "task_family": "support", "optimization_signals": [
                {"dimension": "efficiency", "tag": "fewer_steps"},
                {"dimension": "reuse", "tag": "fewer_steps"},
            ],
        }
        self.assertTrue(self.registry.record_receipt(signal_receipt)["created"])
        reopened = Registry(self.root)
        self.assertEqual(reopened.record_receipt(signal_receipt)["created"], False)
        self.assertEqual(next(item for item in reopened.list_receipts() if item["id"] == "receipt-signal"),
                         signal_receipt)
        empty_signals = {**receipt, "id": "receipt-empty-signals", "optimization_signals": []}
        self.assertTrue(reopened.record_receipt(empty_signals)["created"])
        self.assertEqual(next(item for item in reopened.list_receipts()
                              if item["id"] == "receipt-empty-signals"), empty_signals)

        conflict = {**receipt, "outcome": "failed"}
        self.assert_code("receipt_id_conflict", lambda: self.registry.record_receipt(conflict))
        self.assert_code(
            "invalid_receipt",
            lambda: self.registry.record_receipt({**receipt, "evidence_refs": ["https://example.com/secret"]}),
        )
        self.assert_code(
            "invalid_receipt",
            lambda: self.registry.record_receipt({**receipt, "unexpected": "field"}),
        )
        valid_signal = {**receipt, "task_family": "support", "optimization_signals": [
            {"dimension": "quality", "tag": "clearer"}
        ]}
        invalid_signals = [
            {**valid_signal, "optimization_signals": "quality"},
            {**valid_signal, "optimization_signals": [{"dimension": "other", "tag": "clearer"}]},
            {**valid_signal, "optimization_signals": [{"dimension": "quality", "tag": "bad tag"}]},
            {**valid_signal, "optimization_signals": [{"dimension": "quality", "tag": "clearer", "extra": 1}]},
            {**valid_signal, "optimization_signals": [{"dimension": "quality", "tag": "clearer"}] * 2},
            {**valid_signal, "optimization_signals": [
                {"dimension": "quality", "tag": "tag"}
            ] * 17},
            {**receipt, "task_family": "not safe"},
            {**receipt, "optimization_signals": [{"dimension": "reuse", "tag": "x"}]},
            {**valid_signal, "evidence_refs": []},
        ]
        for invalid in invalid_signals:
            with self.subTest(receipt=invalid):
                self.assert_code("invalid_receipt", lambda invalid=invalid: self.registry.record_receipt(invalid))

    def test_foreground_and_daily_budget_denials_do_not_reassign_claims(self) -> None:
        constrained = Policy(
            experiment_timeout_seconds=120,
            case_timeout_seconds=10,
            max_experiments_per_day=1,
            max_reserved_seconds_per_day=120,
        )
        self.registry.update_policy(constrained)
        self.registry.begin_task("foreground")
        self.assert_code(
            "foreground_active",
            lambda: self.registry.claim_experiment(baseline_id=self.baseline["id"]),
        )
        self.registry.end_task("foreground")
        claim = self.registry.claim_experiment(baseline_id=self.baseline["id"])
        reopened = Registry(self.root)
        self.assertEqual(reopened.experiment(claim["id"])["status"], "running")
        self.assert_code(
            "experiment_concurrency_limit",
            lambda: reopened.claim_experiment(baseline_id=self.baseline["id"]),
        )
        reopened.finish_experiment(claim["id"], status="blocked", elapsed_seconds=0.2)
        self.assert_code(
            "daily_experiment_limit",
            lambda: self.registry.claim_experiment(baseline_id=self.baseline["id"]),
        )

    def test_holdout_reservation_is_one_use_across_experiments(self) -> None:
        self.registry.update_policy(Policy(max_parallel_experiments=2))
        first = self.registry.claim_experiment(baseline_id=self.baseline["id"])
        digest = "d" * 64
        initial = self.registry.consume_holdout(digest, first["id"])
        retry = self.registry.consume_holdout(digest, first["id"])
        self.assertTrue(initial["created"])
        self.assertFalse(retry["created"])

        second = self.registry.claim_experiment(baseline_id=self.baseline["id"])
        self.assert_code(
            "holdout_already_used",
            lambda: Registry(self.root).consume_holdout(digest, second["id"]),
        )
        self.registry.finish_experiment(first["id"], status="blocked", elapsed_seconds=1)
        self.registry.finish_experiment(second["id"], status="blocked", elapsed_seconds=1)
        self.assertFalse(self.registry.consume_holdout(digest, first["id"])["created"])

    def test_promotion_requires_completed_passed_evidence_and_current_policy(self) -> None:
        candidate = self.candidate()
        claim, evaluation = self.claim_and_evaluate(
            candidate["id"], finish=False, holdout_digest="e" * 64
        )
        self.assert_code(
            "promotion_experiment_incomplete",
            lambda: self.registry.promote(evaluation["id"], expected_current=self.baseline["id"]),
        )
        self.registry.finish_experiment(claim["id"], status="completed", elapsed_seconds=1)

        self.registry.update_policy(Policy(
            min_training_success=0.75,
            min_holdout_success=0.75,
            min_improvement=0.11,
            max_regression=0.0,
        ))
        self.assert_code(
            "promotion_policy_changed",
            lambda: self.registry.promote(evaluation["id"], expected_current=self.baseline["id"]),
        )

    def test_promotion_rejects_insufficient_holdout_gain_and_infrastructure_scores(self) -> None:
        candidate = self.candidate()
        _, weak = self.claim_and_evaluate(
            candidate["id"],
            baseline_holdout=0.66,
            candidate_holdout=0.75,
            holdout_digest="f" * 64,
        )
        self.assert_code(
            "promotion_gain",
            lambda: self.registry.promote(weak["id"], expected_current=self.baseline["id"]),
        )

        self.registry.update_policy(self.policy)
        claim = self.registry.claim_experiment(baseline_id=self.baseline["id"])
        evidence = {
            "candidate_id": candidate["id"],
            "baseline_id": self.baseline["id"],
            "suite_digest": "a" * 64,
            "environment_digest": "b" * 64,
            "policy_digest": claim["policy_digest"],
            "status": "infrastructure_error",
            "baseline_training": None,
            "candidate_training": None,
            "baseline_holdout": None,
            "candidate_holdout": None,
            "violations": [],
            "errors": ["sandbox unavailable"],
            "case_counts": {"training": 2, "holdout": 2},
        }
        infrastructure = self.registry.record_evaluation(claim["id"], evidence)
        self.registry.finish_experiment(claim["id"], status="completed", elapsed_seconds=1)
        self.assert_code(
            "promotion_evidence_not_passed",
            lambda: self.registry.promote(infrastructure["id"], expected_current=self.baseline["id"]),
        )

    def test_aggregate_snapshot_capacity_is_atomic_and_never_gc_old_versions(self) -> None:
        policy = Policy(
            experiment_timeout_seconds=120,
            case_timeout_seconds=10,
            max_snapshot_bytes=4,
            max_file_bytes=4,
            max_snapshot_files=5,
            max_snapshots=2,
            max_store_bytes=6,
        )
        registry = Registry.initialize(self.base / "small-store", policy)
        initial = self.base / "small-source"
        (initial / "skills").mkdir(parents=True)
        (initial / "skills" / "x").write_text("four", encoding="utf-8")
        first = registry.import_snapshot(initial)
        next_source = self.base / "small-next"
        (next_source / "skills").mkdir(parents=True)
        (next_source / "skills" / "x").write_text("new", encoding="utf-8")
        self.assert_code("store_size_limit", lambda: registry.import_snapshot(next_source))
        self.assertEqual([item["id"] for item in registry.list_snapshots()], [first["id"]])

    def test_concurrent_snapshot_imports_share_aggregate_count_admission(self) -> None:
        policy = Policy(max_snapshots=2)
        root = self.base / "count-limited-store"
        first_registry = Registry.initialize(root, policy)
        (self.base / "count-source-0" / "skills").mkdir(parents=True)
        (self.base / "count-source-0" / "skills" / "x").write_text("baseline", encoding="utf-8")
        first_registry.import_snapshot(self.base / "count-source-0")

        sources = []
        for index in (1, 2):
            source = self.base / f"count-source-{index}"
            (source / "skills").mkdir(parents=True)
            (source / "skills" / "x").write_text(f"variant-{index}", encoding="utf-8")
            sources.append(source)
        registries = (Registry(root), Registry(root))
        barrier = threading.Barrier(2)

        def import_one(args: tuple[Registry, Path]) -> str:
            registry, source = args
            barrier.wait(timeout=5)
            try:
                registry.import_snapshot(source)
                return "created"
            except EvolutionError as exc:
                return exc.code

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(import_one, zip(registries, sources)))
        self.assertCountEqual(outcomes, ["created", "snapshot_count_limit"])
        self.assertEqual(len(first_registry.list_snapshots()), 2)

    def test_concurrent_claims_are_serialized_by_independent_registry_connections(self) -> None:
        first_registry = Registry(self.root)
        second_registry = Registry(self.root)
        barrier = threading.Barrier(2)

        def try_claim(registry: Registry) -> str:
            barrier.wait(timeout=5)
            try:
                return registry.claim_experiment(baseline_id=self.baseline["id"])["id"]
            except EvolutionError as exc:
                return exc.code

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(try_claim, (first_registry, second_registry)))
        successes = [value for value in outcomes if value != "experiment_concurrency_limit"]
        denials = [value for value in outcomes if value == "experiment_concurrency_limit"]
        self.assertEqual(len(successes), 1, outcomes)
        self.assertEqual(len(denials), 1, outcomes)
        self.assertEqual(len(self.registry.status()["active_experiments"]), 1)


if __name__ == "__main__":
    unittest.main()
