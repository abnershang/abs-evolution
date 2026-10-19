import unittest

from abs_evolution.opportunities import summarize


class OpportunitiesTests(unittest.TestCase):
    def test_only_distinct_failed_tasks_create_a_research_candidate(self):
        def receipt(identifier, task, outcome, tag="wrong_acceptance"):
            return {"id": identifier, "task_id": task, "snapshot_id": "v1", "outcome": outcome,
                    "failure_tags": [tag], "evidence_refs": ["evidence-" + identifier]}
        records = [receipt("r1", "t1", "failed"), receipt("r2", "t1", "failed"),
                   receipt("r3", "t2", "failed"), receipt("r4", "t3", "unknown", "other"),
                   receipt("r5", "t4", "accepted", "other")]
        result = summarize(records)
        self.assertEqual(result["outcome_counts"], {"accepted": 1, "failed": 3, "unknown": 1})
        self.assertEqual(len(result["opportunities"]), 1)
        item = result["opportunities"][0]
        self.assertEqual(item["failure_tag"], "wrong_acceptance")
        self.assertEqual(item["distinct_tasks"], 2)
        self.assertEqual(item["receipt_ids"], ["r1", "r2", "r3"])
        self.assertEqual(item["status"], "needs_investigation")

    def test_explicit_optimization_signals_are_separate_by_family_and_dimension(self):
        def receipt(identifier, task, family, dimension, tag, outcome="accepted"):
            return {"id": identifier, "task_id": task, "snapshot_id": "v1", "outcome": outcome,
                    "failure_tags": [], "evidence_refs": ["evidence-" + identifier],
                    "task_family": family,
                    "optimization_signals": [{"dimension": dimension, "tag": tag}]}

        records = [receipt("r1", "t1", "support", "efficiency", "fewer_steps"),
                   receipt("r2", "t1", "support", "efficiency", "fewer_steps"),
                   receipt("r3", "t2", "support", "efficiency", "fewer_steps", "unknown"),
                   receipt("r4", "t3", "billing", "efficiency", "fewer_steps"),
                   receipt("r5", "t4", "support", "quality", "fewer_steps")]
        result = summarize(records)
        optimization = [item for item in result["opportunities"] if item["signal_type"] == "optimization"]
        self.assertEqual(len(optimization), 1)
        item = optimization[0]
        self.assertEqual((item["task_family"], item["dimension"], item["tag"]),
                         ("support", "efficiency", "fewer_steps"))
        self.assertEqual(item["distinct_tasks"], 2)
        self.assertEqual(item["observed_outcomes"], ["accepted", "unknown"])
        self.assertEqual(item["status"], "needs_comparison")
        self.assertNotIn("failure_tag", item)
        self.assertEqual(item["receipt_ids"], ["r1", "r2", "r3"])
        self.assertEqual(summarize(records[:2])["opportunities"], [])

    def test_no_signal_is_invented_and_explicit_signal_does_not_need_failed_outcome(self):
        def base(identifier, outcome):
            return {"id": identifier, "task_id": identifier, "snapshot_id": "v1", "outcome": outcome,
                    "failure_tags": [], "evidence_refs": ["evidence-" + identifier]}
        self.assertEqual(summarize([base("accepted", "accepted"), base("failed", "failed"),
                                    base("unknown", "unknown")], min_occurrences=1)["opportunities"], [])
        explicit = {**base("explicit", "accepted"), "task_family": "support",
                    "optimization_signals": [{"dimension": "reuse", "tag": "shared_step"}]}
        item = summarize([explicit], min_occurrences=1)["opportunities"][0]
        self.assertEqual(item["observed_outcomes"], ["accepted"])
        self.assertEqual(item["signal_type"], "optimization")


if __name__ == "__main__":
    unittest.main()
