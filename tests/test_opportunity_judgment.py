from __future__ import annotations

import copy
import os
import time
import unittest
from unittest import mock

from abs_evolution.contracts import ChoiceResult, fingerprint
from abs_evolution.errors import EvolutionError
from abs_evolution.opportunity_judgment import MAX_REQUEST_BYTES, OpportunityJudge


OPTIONS = ("quality", "efficiency", "method", "reuse", "no_opportunity", "insufficient_evidence")


def request(*, outcome: str = "accepted") -> dict:
    return {
        "schema_version": 1,
        "task_family": "reporting",
        "objective": "Prepare a weekly result summary.",
        "observation": "The accepted summary took longer than expected.",
        "outcome": outcome,
        "evidence": [{"id": "run:001", "summary": "The observed run completed and was accepted."}],
    }


def choice(option: str, *, confidence: float = 0.94, probabilities: dict[str, float] | None = None) -> ChoiceResult:
    if probabilities is None:
        probabilities = {item: (0.94 if item == option else 0.012) for item in OPTIONS}
    return ChoiceResult(option, probabilities, confidence, 0.03, "jev-1.13.0", 42, 7)


class ControlledJudge:
    def __init__(self, result: ChoiceResult | Exception, *, delay: float = 0, mutate: object | None = None) -> None:
        self.result = result
        self.delay = delay
        self.mutate = mutate
        self.calls: list[dict] = []

    def choose(self, state, *, instructions, criteria, timeout_seconds, max_output_bytes):
        self.calls.append({"state": copy.deepcopy(state), "instructions": instructions,
                           "criteria": criteria, "timeout_seconds": timeout_seconds,
                           "max_output_bytes": max_output_bytes})
        if isinstance(self.mutate, dict):
            self.mutate["evidence"][0]["summary"] = "caller changed during call"
        if self.delay:
            time.sleep(self.delay)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class OpportunityJudgeTests(unittest.TestCase):
    def assert_code(self, code: str, call):
        with self.assertRaises(EvolutionError) as caught:
            call()
        self.assertEqual(caught.exception.code, code)
        return caught.exception

    def test_each_choice_maps_to_the_expected_advisory_state(self):
        expected = {
            "quality": ("classified", "quality", "compare"),
            "efficiency": ("classified", "efficiency", "compare"),
            "method": ("classified", "method", "compare"),
            "reuse": ("classified", "reuse", "compare"),
            "no_opportunity": ("classified", None, "retain_baseline"),
            "insufficient_evidence": ("deferred", None, "gather_evidence"),
        }
        for option, (status, dimension, next_step) in expected.items():
            with self.subTest(option=option):
                judged = ControlledJudge(choice(option))
                result = OpportunityJudge(judged).assess(request())
                self.assertEqual((result["status"], result["dimension"], result["next_step"]), (status, dimension, next_step))
                self.assertEqual(result["decision"], option)
                self.assertTrue(result["model_called"])
                self.assertFalse(result["candidate_executed"] or result["registry_changed"] or result["deployment_changed"])

    def test_outcome_and_evidence_provenance_are_retained_not_transformed(self):
        for outcome in ("accepted", "failed", "unknown"):
            with self.subTest(outcome=outcome):
                result = OpportunityJudge(ControlledJudge(choice("quality"))).assess(request(outcome=outcome))
                self.assertEqual(result["observed_outcome"], outcome)
        empty = request()
        empty["evidence"] = []
        judged = ControlledJudge(choice("insufficient_evidence"))
        result = OpportunityJudge(judged).assess(empty)
        self.assertEqual(result["evidence_refs"], [])
        self.assertEqual(judged.calls[0]["state"]["request"]["evidence"], [])

    def test_low_confidence_or_tied_distribution_defers_without_erasing_judgment(self):
        tie = {item: 1 / len(OPTIONS) for item in OPTIONS}
        for selected in (choice("quality", confidence=0.79), choice("quality", probabilities=tie)):
            with self.subTest(confidence=selected.confidence, probabilities=selected.probabilities):
                result = OpportunityJudge(ControlledJudge(selected)).assess(request())
                self.assertEqual((result["status"], result["decision"], result["next_step"]),
                                 ("deferred", "uncertain", "review_uncertainty"))
                self.assertEqual(result["judgment"]["option"], "quality")
                self.assertFalse(result["candidate_executed"])

    def test_shared_choice_contract_rejects_malformed_judgments(self):
        malformed = ChoiceResult("quality", {"quality": 1.0}, 0.95, 0.01, "jev-1.13.0", 1, 1)
        self.assert_code("provider_invalid_judgement", lambda: OpportunityJudge(ControlledJudge(malformed)).assess(request()))

    def test_invalid_or_sensitive_input_never_reaches_judge(self):
        bad_values = []
        malformed = request()
        malformed["extra"] = True
        bad_values.append(malformed)
        malformed = request()
        malformed["schema_version"] = True
        bad_values.append(malformed)
        malformed = request()
        malformed["evidence"] = [{"id": "same", "summary": "A"}, {"id": "same", "summary": "B"}]
        bad_values.append(malformed)
        malformed = request()
        malformed["task_family"] = "https://example.invalid"
        bad_values.append(malformed)
        malformed = request()
        malformed["objective"] = "x" * 2001
        bad_values.append(malformed)
        malformed = request()
        malformed["evidence"] = [{"id": str(i), "summary": "x"} for i in range(17)]
        bad_values.append(malformed)
        malformed = request()
        malformed["objective"] = "o" * 2000
        malformed["observation"] = "n" * 4000
        malformed["evidence"] = [{"id": f"e{i}", "summary": "x" * 4000} for i in range(16)]
        bad_values.append(malformed)
        judge = ControlledJudge(choice("quality"))
        for value in bad_values:
            self.assert_code("invalid_opportunity_request", lambda value=value: OpportunityJudge(judge).assess(value))
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-configured-key-abc"}):
            sensitive = request()
            sensitive["objective"] = "Compare using test-configured-key-abc"
            self.assert_code("provider_sensitive_input", lambda: OpportunityJudge(judge).assess(sensitive))
        sensitive = request()
        sensitive["observation"] = "password=synthetic-not-a-real-password"
        self.assert_code("provider_sensitive_input", lambda: OpportunityJudge(judge).assess(sensitive))
        self.assertEqual(judge.calls, [])
        self.assertEqual(MAX_REQUEST_BYTES, 65_536)

    def test_caller_mutation_during_judgment_cannot_change_snapshot_or_provenance(self):
        original = request()
        expected_fingerprint = fingerprint(original)
        judge = ControlledJudge(choice("reuse"), mutate=original)
        result = OpportunityJudge(judge).assess(original)
        self.assertEqual(judge.calls[0]["state"]["request"]["evidence"][0]["summary"],
                         "The observed run completed and was accepted.")
        self.assertEqual(result["evidence_refs"], ["run:001"])
        self.assertEqual(result["request_fingerprint"], expected_fingerprint)
        self.assertNotIn("objective", result)
        self.assertNotIn("observation", result)
        self.assertNotIn("summary", result)
        self.assertEqual(original["evidence"][0]["summary"], "caller changed during call")

    def test_service_exception_is_propagated_without_retry(self):
        failure = EvolutionError("provider_timeout", "synthetic timeout")
        judge = ControlledJudge(failure)
        with self.assertRaises(EvolutionError) as caught:
            OpportunityJudge(judge).assess(request())
        self.assertIs(caught.exception, failure)
        self.assertEqual(len(judge.calls), 1)

    def test_elapsed_deadline_and_result_size_are_enforced(self):
        slow = ControlledJudge(choice("quality"), delay=0.02)
        self.assert_code("provider_timeout", lambda: OpportunityJudge(slow).assess(request(), timeout_seconds=0.01))
        tiny = ControlledJudge(choice("quality"))
        self.assert_code("provider_output_too_large", lambda: OpportunityJudge(tiny).assess(request(), max_output_bytes=1))
        self.assertEqual(len(tiny.calls), 1)


if __name__ == "__main__":
    unittest.main()
