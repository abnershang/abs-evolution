from __future__ import annotations

import copy
import os
import time
import unittest
from unittest import mock

from abs_evolution.contracts import ChoiceResult, ProposalResult, fingerprint, validated_routing
from abs_evolution.errors import EvolutionError
from abs_evolution.hybrid import HybridProposalProvider


def request() -> dict:
    return {
        "schema_version": 1,
        "objective": "Correct the training failure without changing passing behavior.",
        "baseline_id": "b" * 64,
        "files": {"skills/solver.py": "def solve(value): return value\n"},
        "training_examples": [
            {"id": "failed", "input": {"n": 1}, "expected": 2, "actual": 1, "passed": False},
            {"id": "passed", "input": {"keep": True}, "expected": True, "actual": True, "passed": True},
        ],
        "allowed_mutation_paths": ["skills/*"],
        "max_changes": 2,
    }


def proposal(content: str = "def solve(value): return value + 1\n") -> dict:
    return {
        "summary": "Correct the observed training failure while preserving the passing example.",
        "kind": "skill",
        "changes": [{"path": "skills/solver.py", "content": content}],
    }


def choice(option: str, options: list[str], *, confidence: float = 0.92) -> ChoiceResult:
    residual = 0.08 / max(len(options) - 1, 1)
    probabilities = {key: (0.92 if key == option else residual) for key in options}
    if len(options) == 1:
        probabilities[option] = 1.0
    return ChoiceResult(
        option=option,
        probabilities=probabilities,
        confidence=confidence,
        duration_seconds=0.02,
        model="jev-1.13.0",
        input_tokens=64,
        output_tokens=12,
    )


class RecordingJudge:
    def __init__(
        self,
        result: ChoiceResult | Exception,
        *,
        delay: float = 0,
        mutate_state: bool = False,
        original_to_mutate: dict | None = None,
    ) -> None:
        self.result = result
        self.delay = delay
        self.mutate_state = mutate_state
        self.original_to_mutate = original_to_mutate
        self.calls: list[dict] = []

    def choose(self, state, *, instructions, criteria, timeout_seconds, max_output_bytes):
        self.calls.append({
            "state": state,
            "instructions": instructions,
            "criteria": criteria,
            "timeout_seconds": timeout_seconds,
            "max_output_bytes": max_output_bytes,
        })
        if self.mutate_state:
            state["request"]["objective"] = "unjudged mutation"
            if state["candidates"]:
                state["candidates"][0]["proposal"]["summary"] = "unjudged candidate mutation"
        if self.original_to_mutate is not None:
            self.original_to_mutate["objective"] = "caller mutation during judgment"
        if self.delay:
            time.sleep(self.delay)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class RecordingGenerator:
    def __init__(self, result: ProposalResult | Exception | None = None) -> None:
        self.result = result or ProposalResult(
            value=proposal(),
            duration_seconds=0.1,
            input_tokens=120,
            output_tokens=44,
            model="openai/test-model",
            generation_id="gen-test-123",
            cost_credits=0.00042,
        )
        self.calls: list[dict] = []

    def propose(self, req, *, timeout_seconds, max_output_bytes):
        self.calls.append({
            "request": req,
            "timeout_seconds": timeout_seconds,
            "max_output_bytes": max_output_bytes,
        })
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class HybridProposalProviderTests(unittest.TestCase):
    def assert_code(self, code: str, call) -> EvolutionError:
        with self.assertRaises(EvolutionError) as caught:
            call()
        self.assertEqual(caught.exception.code, code)
        return caught.exception

    def make(self, option: str, *, candidates=None, generator=None, confidence=0.92, judge=None):
        options = [*(f"reuse_{item['id']}" for item in candidates or []), "generate", "insufficient_evidence"]
        selected_judge = judge or RecordingJudge(choice(option, options, confidence=confidence))
        provider = HybridProposalProvider(
            selected_judge,
            generator,
            candidates=candidates,
        )
        return provider, selected_judge

    def test_confident_reuse_returns_exact_candidate_without_generator_usage(self) -> None:
        candidate = {"id": "known-fix", "baseline_id": request()["baseline_id"], "proposal": proposal()}
        generator = RecordingGenerator()
        provider, judge = self.make("reuse_known-fix", candidates=[candidate], generator=generator)

        result = provider.propose(request(), timeout_seconds=3, max_output_bytes=8192)

        self.assertTrue(provider.requires_hard_resource_limits)
        self.assertEqual(result.value, candidate["proposal"])
        self.assertIsNone(result.model)
        self.assertIsNone(result.input_tokens)
        self.assertIsNone(result.output_tokens)
        self.assertIsNone(result.cost_credits)
        self.assertEqual(generator.calls, [])
        call = judge.calls[0]
        self.assertEqual(set(call["state"]), {"request", "candidates"})
        self.assertEqual(call["state"]["candidates"], [candidate])
        self.assertEqual(set(call["criteria"]), {"reuse_known-fix", "generate", "insufficient_evidence"})
        trace = validated_routing(result.routing)
        self.assertEqual(trace["route"], "reuse")
        self.assertEqual(trace["selected_candidate_id"], "known-fix")
        self.assertEqual(trace["judge"]["input_tokens"], 64)
        self.assertEqual(trace["generator"], {
            "called": False,
            "model": None,
            "input_tokens": None,
            "output_tokens": None,
            "cost_credits": None,
            "duration_seconds": None,
            "generation_id": None,
        })
        self.assertNotIn("proposal", trace)
        self.assertNotIn("input", trace)

        # A passing task can still have a stated continuous-improvement goal.
        # The stubbed typed choice exercises the reuse path only; it does not
        # establish that a live model interpreted the rubric correctly.
        successful_request = request()
        successful_request["objective"] = "Explore a quality and efficiency improvement for this successful task."
        for example in successful_request["training_examples"]:
            example["expected"] = copy.deepcopy(example["actual"])
            example["passed"] = True
        optimization_provider, optimization_judge = self.make(
            "reuse_known-fix", candidates=[candidate], generator=generator,
        )
        optimized = optimization_provider.propose(
            successful_request, timeout_seconds=3, max_output_bytes=8192,
        )
        self.assertTrue(all(item["passed"] for item in successful_request["training_examples"]))
        self.assertEqual(optimized.value, candidate["proposal"])
        self.assertEqual(validated_routing(optimized.routing)["route"], "reuse")
        self.assertEqual(generator.calls, [])
        self.assertEqual(len(optimization_judge.calls), 1)

    def test_confident_generate_calls_generator_once_with_original_request_and_remaining_budget(self) -> None:
        req = request()
        original = copy.deepcopy(req)
        candidate = {"id": "known-fix", "baseline_id": req["baseline_id"], "proposal": proposal()}
        candidates = [candidate]
        generator = RecordingGenerator()
        judge = RecordingJudge(
            choice("generate", ["reuse_known-fix", "generate", "insufficient_evidence"]),
            delay=0.02,
            mutate_state=True,
            original_to_mutate=req,
        )
        provider = HybridProposalProvider(judge, generator, candidates=candidates)

        result = provider.propose(req, timeout_seconds=2, max_output_bytes=8192)

        self.assertEqual(len(judge.calls), 1)
        self.assertEqual(judge.calls[0]["state"]["request"]["objective"], "unjudged mutation")
        self.assertEqual(judge.calls[0]["state"]["candidates"][0]["proposal"]["summary"], "unjudged candidate mutation")
        self.assertEqual(len(generator.calls), 1)
        self.assertIsNot(generator.calls[0]["request"], req)
        self.assertEqual(generator.calls[0]["request"], original)
        self.assertEqual(req["objective"], "caller mutation during judgment")
        self.assertGreater(generator.calls[0]["timeout_seconds"], 0)
        self.assertLess(generator.calls[0]["timeout_seconds"], 2)
        self.assertEqual(generator.calls[0]["max_output_bytes"], 8192)
        self.assertEqual(result.value, proposal())
        self.assertEqual(result.model, "openai/test-model")
        self.assertEqual(result.input_tokens, 120)
        trace = validated_routing(result.routing)
        self.assertEqual(trace["route"], "generate")
        self.assertEqual(trace["candidate_pool_fingerprint"], fingerprint(candidates))
        self.assertEqual(trace["judge"]["input_tokens"], 64)
        self.assertEqual(trace["generator"]["input_tokens"], 120)
        self.assertEqual(trace["generator"]["cost_credits"], 0.00042)
        self.assertEqual(trace["generator"]["generation_id"], "gen-test-123")
        self.assertNotIn("objective", trace)

    def test_uncertain_low_confidence_and_incomplete_choice_block_without_fallback(self) -> None:
        generator = RecordingGenerator()
        for option, confidence, code in (
            ("insufficient_evidence", 0.99, "provider_blocked_uncertain"),
            ("generate", 0.79, "provider_blocked_uncertain"),
        ):
            provider, judge = self.make(option, generator=generator, confidence=confidence)
            self.assert_code(
                code,
                lambda provider=provider: provider.propose(request(), timeout_seconds=2, max_output_bytes=8192),
            )
            self.assertEqual(generator.calls, [])
            self.assertEqual(len(judge.calls), 1)

        incomplete = ChoiceResult(
            option="generate",
            probabilities={"generate": 1.0},
            confidence=1.0,
            duration_seconds=0.01,
            model="jev-1.13.0",
            input_tokens=None,
            output_tokens=None,
        )
        provider, judge = self.make("generate", generator=generator, judge=RecordingJudge(incomplete))
        self.assert_code(
            "provider_invalid_judgement",
            lambda: provider.propose(request(), timeout_seconds=2, max_output_bytes=8192),
        )
        self.assertEqual(generator.calls, [])
        self.assertEqual(len(judge.calls), 1)

    def test_generate_without_configured_generator_is_an_explicit_block(self) -> None:
        provider, judge = self.make("generate")
        self.assert_code(
            "provider_blocked_generation_unavailable",
            lambda: provider.propose(request(), timeout_seconds=2, max_output_bytes=8192),
        )
        self.assertEqual(len(judge.calls), 1)

    def test_tied_route_is_uncertain_and_invalid_judge_metadata_blocks_before_generation(self) -> None:
        options = ["generate", "insufficient_evidence"]
        tied = ChoiceResult(
            option="generate", probabilities={"generate": 0.5, "insufficient_evidence": 0.5},
            confidence=0.5, duration_seconds=0.01,
        )
        generator = RecordingGenerator()
        provider, judge = self.make("generate", generator=generator, judge=RecordingJudge(tied))
        self.assert_code(
            "provider_blocked_uncertain",
            lambda: provider.propose(request(), timeout_seconds=2, max_output_bytes=8192),
        )
        self.assertEqual(generator.calls, [])
        self.assertEqual(len(judge.calls), 1)

        malformed = ChoiceResult(
            option="generate", probabilities={"generate": 0.9, "insufficient_evidence": 0.1},
            confidence=0.9, duration_seconds=0.01, model="invalid model id",
        )
        provider, judge = self.make("generate", generator=generator, judge=RecordingJudge(malformed))
        self.assert_code(
            "provider_invalid_judgement",
            lambda: provider.propose(request(), timeout_seconds=2, max_output_bytes=8192),
        )
        self.assertEqual(generator.calls, [])
        self.assertEqual(len(judge.calls), 1)

    def test_judge_service_failure_and_generator_failure_do_not_fallback_or_retry(self) -> None:
        judge = RecordingJudge(EvolutionError("provider_timeout", "synthetic timeout"))
        generator = RecordingGenerator()
        provider = HybridProposalProvider(judge, generator)
        self.assert_code(
            "provider_timeout",
            lambda: provider.propose(request(), timeout_seconds=2, max_output_bytes=8192),
        )
        self.assertEqual(generator.calls, [])
        self.assertEqual(len(judge.calls), 1)

        failing_generator = RecordingGenerator(EvolutionError("provider_timeout", "synthetic timeout"))
        provider, judge = self.make("generate", generator=failing_generator)
        self.assert_code(
            "provider_timeout",
            lambda: provider.propose(request(), timeout_seconds=2, max_output_bytes=8192),
        )
        self.assertEqual(len(judge.calls), 1)
        self.assertEqual(len(failing_generator.calls), 1)

    def test_all_candidates_are_validated_before_judge_call(self) -> None:
        valid = {"id": "one", "baseline_id": request()["baseline_id"], "proposal": proposal()}
        invalid_cases = [
            [{**valid, "extra": "not accepted"}],
            [valid, copy.deepcopy(valid)],
            [{**valid, "baseline_id": "c" * 64}],
            [{**valid, "proposal": proposal(request()["files"]["skills/solver.py"])}],
            [{**valid, "proposal": {**proposal(), "summary": "password: synthetic-placeholder"}}],
        ]
        for candidates in invalid_cases:
            with self.subTest(candidates=candidates):
                provider, judge = self.make("generate", candidates=candidates)
                with self.assertRaises(EvolutionError):
                    provider.propose(request(), timeout_seconds=2, max_output_bytes=8192)
                self.assertEqual(judge.calls, [])

    def test_nested_secret_and_literal_configured_keys_are_rejected_before_judgment(self) -> None:
        provider, judge = self.make("generate")
        sensitive = request()
        sensitive["training_examples"][0]["input"] = {"clientSecret": "synthetic-placeholder"}
        self.assert_code(
            "provider_sensitive_input",
            lambda: provider.propose(sensitive, timeout_seconds=2, max_output_bytes=8192),
        )
        self.assertEqual(judge.calls, [])

        with mock.patch.dict(os.environ, {
            "TYPESAFE_API_KEY": "synthetic-typesafe-key-12345",
            "OPENROUTER_API_KEY": "synthetic-openrouter-key-12345",
        }):
            for field, key in (("objective", "TYPESAFE_API_KEY"), ("input", "OPENROUTER_API_KEY")):
                contaminated = request()
                if field == "objective":
                    contaminated["objective"] += " " + os.environ[key]
                else:
                    contaminated["training_examples"][0]["input"] = {"example": os.environ[key]}
                provider, judge = self.make("generate")
                self.assert_code(
                    "provider_sensitive_input",
                    lambda provider=provider, contaminated=contaminated: provider.propose(
                        contaminated, timeout_seconds=2, max_output_bytes=8192,
                    ),
                )
                self.assertEqual(judge.calls, [])

    def test_routing_serialization_rejects_unbounded_or_inconsistent_metadata(self) -> None:
        provider, _judge = self.make("generate", generator=RecordingGenerator())
        result = provider.propose(request(), timeout_seconds=2, max_output_bytes=8192)
        trace = validated_routing(result.routing)
        trace["generator"]["input_tokens"] = -1
        self.assert_code("invalid_routing_trace", lambda: validated_routing(trace))

        result = provider.propose(request(), timeout_seconds=2, max_output_bytes=8192)
        trace = validated_routing(result.routing)
        trace["request_fingerprint"] = "not-a-fingerprint"
        self.assert_code("invalid_routing_trace", lambda: validated_routing(trace))


if __name__ == "__main__":
    unittest.main()
