"""Split-assessment policy gates, not evidence of live model correctness."""

from __future__ import annotations

import json
from pathlib import Path
import time
import unittest

from abs_evolution.contracts import BatchJudgment, ChoiceAnswer, NoulAnswer
from abs_evolution.errors import EvolutionError
from abs_evolution.opportunity_assessment import OpportunityAssessment


REQUEST = Path(__file__).resolve().parents[1] / "examples" / "optimization" / "request.json"


class ControlledBatchJudge:
    def __init__(self, result, *, delay=0):
        self.result, self.delay = result, delay
        self.calls = 0

    def judge_batch(self, *_args, **_kwargs):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def judgment(option="worth_comparing", *, confidence=0.9, probability=0.9, dimensions=(0.2, 0.8, 0.9, 0.5)):
    others = (1 - probability) / 2
    probabilities = {name: probability if name == option else others
                     for name in ("worth_comparing", "no_opportunity", "insufficient_evidence")}
    return BatchJudgment(
        answers={
            "opportunity": ChoiceAnswer(option, probabilities, confidence),
            **{name: NoulAnswer(value) for name, value in zip(("quality", "efficiency", "method", "reuse"), dimensions)},
        },
        duration_seconds=0.01, model="jev-1.13.0", input_tokens=200, output_tokens=100,
    )


class OpportunityAssessmentTests(unittest.TestCase):
    def test_presence_and_independent_dimension_gates_preserve_uncertainty_and_conflict(self):
        cases = [
            (judgment(), "assessed", "worth_comparing", "compare"),
            (judgment(confidence=0.79), "deferred", "uncertain", "review_uncertainty"),
            (judgment(probability=0.79), "deferred", "uncertain", "review_uncertainty"),
            (judgment("insufficient_evidence"), "deferred", "insufficient_evidence", "gather_evidence"),
            (judgment("no_opportunity"), "deferred", "no_opportunity", "review_conflict"),
            (judgment("no_opportunity", dimensions=(0.1, 0.2, 0.3, 0.4)), "assessed", "no_opportunity", "retain_baseline"),
            (judgment(dimensions=(0.1, 0.2, 0.3, 0.4)), "deferred", "worth_comparing", "review_dimensions"),
        ]
        tied = judgment(confidence=1)
        tied.answers["opportunity"] = ChoiceAnswer("worth_comparing", {
            "worth_comparing": 0.5, "no_opportunity": 0.5, "insufficient_evidence": 0.0,
        }, 1.0)
        cases.append((tied, "deferred", "uncertain", "review_uncertainty"))
        for result, status, decision, next_step in cases:
            with self.subTest(status=status, decision=decision, next_step=next_step):
                judge = ControlledBatchJudge(result)
                assessor = OpportunityAssessment(judge, min_probability=0.5) if result is tied else OpportunityAssessment(judge)
                output = assessor.assess(json.loads(REQUEST.read_text()))
                self.assertEqual((output["status"], output["decision"], output["next_step"]), (status, decision, next_step))
                self.assertEqual(judge.calls, 1)
                for name, answer in result.answers.items():
                    if name != "opportunity":
                        self.assertEqual(output["dimensions"][name]["probability_yes"], answer.probability_yes)
                self.assertEqual(output["dimensions"]["efficiency"]["status"],
                                 "supported" if result.answers["efficiency"].probability_yes == 0.8 else "not_supported")

    def test_failed_or_expired_batch_has_no_retry_and_never_returns_partial_assessment(self):
        failure = EvolutionError("provider_unavailable", "Controlled unavailable provider")
        cases = [
            (ControlledBatchJudge(failure), {}, "provider_unavailable"),
            (ControlledBatchJudge(judgment(), delay=0.02), {"timeout_seconds": 0.01}, "provider_timeout"),
            (ControlledBatchJudge(judgment()), {"max_output_bytes": 1}, "provider_output_too_large"),
        ]
        for judge, kwargs, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(EvolutionError) as caught:
                    OpportunityAssessment(judge).assess(json.loads(REQUEST.read_text()), **kwargs)
                self.assertEqual(caught.exception.code, code)
                self.assertEqual(judge.calls, 1)

    def test_caller_cannot_bypass_typed_batch_validation(self):
        malformed = judgment()
        malformed.answers["reuse"] = NoulAnswer(True)
        judge = ControlledBatchJudge(malformed)
        with self.assertRaises(EvolutionError) as caught:
            OpportunityAssessment(judge).assess(json.loads(REQUEST.read_text()))
        self.assertEqual(caught.exception.code, "provider_invalid_judgement")


if __name__ == "__main__":
    unittest.main()
