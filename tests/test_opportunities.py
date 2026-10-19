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


if __name__ == "__main__":
    unittest.main()
