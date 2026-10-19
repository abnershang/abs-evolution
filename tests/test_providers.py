from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from abs_evolution.errors import EvolutionError
from abs_evolution.providers import CodexProvider, FileProposalProvider


def request() -> dict:
    return {
        "schema_version": 1,
        "objective": "Fix the solver while preserving passing behavior.",
        "baseline_id": "a" * 64,
        "files": {"skills/solver.py": "def solve(value): return value\n"},
        "training_examples": [
            {
                "id": "failed-case",
                "input": {"n": 1},
                "expected": 2,
                "actual": 1,
                "passed": False,
            },
            {
                "id": "passed-case",
                "input": {"holdout": "ordinary nested input key"},
                "expected": {"result": True},
                "actual": {"result": True},
                "passed": True,
            },
        ],
        "allowed_mutation_paths": ["skills/*"],
        "max_changes": 2,
    }


def proposal(content: str = "def solve(value): return value + 1\n") -> dict:
    return {
        "summary": "Correct the observed training failure without changing passing cases.",
        "kind": "skill",
        "changes": [{"path": "skills/solver.py", "content": content}],
    }


class ProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.proposal_path = self.base / "proposal.json"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_proposal(self, value: object) -> None:
        self.proposal_path.write_text(json.dumps(value), encoding="utf-8")

    def assert_code(self, code: str, call) -> EvolutionError:
        with self.assertRaises(EvolutionError) as caught:
            call()
        self.assertEqual(caught.exception.code, code)
        return caught.exception

    def load(self, *, req: dict | None = None, limit: int = 8192):
        return FileProposalProvider(self.proposal_path).propose(
            request() if req is None else req,
            timeout_seconds=2,
            max_output_bytes=limit,
        )

    def test_file_provider_returns_strict_proposal_without_claiming_model_usage(self) -> None:
        expected = proposal()
        self.write_proposal(expected)

        result = self.load()

        self.assertEqual(result.value, expected)
        self.assertGreaterEqual(result.duration_seconds, 0)
        self.assertIsNone(result.input_tokens)
        self.assertIsNone(result.output_tokens)

    def test_training_examples_accept_both_outcomes_and_nested_keys_are_data(self) -> None:
        self.write_proposal(proposal())

        result = self.load()

        self.assertEqual(len(result.value["changes"]), 1)
        self.assertTrue(request()["training_examples"][1]["passed"])
        self.assertIn("holdout", request()["training_examples"][1]["input"])

    def test_request_schema_rejects_unknown_keys_and_inconsistent_outcomes(self) -> None:
        self.write_proposal(proposal())
        unknown = request()
        unknown["holdout"] = []
        self.assert_code("invalid_provider_request", lambda: self.load(req=unknown))

        inconsistent = request()
        inconsistent["training_examples"][1]["passed"] = False
        self.assert_code("invalid_provider_request", lambda: self.load(req=inconsistent))

    def test_request_rejects_sensitive_paths_and_key_material(self) -> None:
        self.write_proposal(proposal())
        credential_path = request()
        credential_path["files"] = {"skills/.env": "safe-looking text"}
        self.assert_code("provider_sensitive_path", lambda: self.load(req=credential_path))

        credential_text = request()
        credential_text["files"] = {"skills/config.py": 'api_key = "123456789abcdef"'}
        self.assert_code("provider_sensitive_input", lambda: self.load(req=credential_text))

    def test_file_provider_rejects_unknown_output_fields_and_duplicate_paths(self) -> None:
        unknown = proposal()
        unknown["extra"] = "not allowed"
        self.write_proposal(unknown)
        self.assert_code("invalid_proposal", self.load)

        duplicate = proposal()
        duplicate["changes"].append(dict(duplicate["changes"][0], content="different"))
        self.write_proposal(duplicate)
        self.assert_code("invalid_proposal", self.load)

    def test_file_provider_rejects_unsafe_disallowed_and_noop_paths(self) -> None:
        unsafe = proposal()
        unsafe["changes"][0]["path"] = "skills/../../outside.py"
        self.write_proposal(unsafe)
        self.assert_code("invalid_proposal", self.load)

        disallowed = proposal()
        disallowed["changes"][0]["path"] = "README.md"
        self.write_proposal(disallowed)
        self.assert_code("invalid_proposal", self.load)

        noop = proposal(content=request()["files"]["skills/solver.py"])
        self.write_proposal(noop)
        self.assert_code("provider_noop", self.load)

        missing_delete = proposal()
        missing_delete["changes"][0] = {"path": "skills/new.py", "content": None}
        self.write_proposal(missing_delete)
        self.assert_code("provider_noop", self.load)

    def test_file_provider_rejects_duplicate_json_keys_invalid_json_and_output_overflow(self) -> None:
        self.proposal_path.write_text(
            '{"summary":"a","summary":"b","kind":"skill","changes":[]}',
            encoding="utf-8",
        )
        self.assert_code("invalid_proposal_file", self.load)

        self.proposal_path.write_text('{"summary":NaN}', encoding="utf-8")
        self.assert_code("invalid_proposal_file", self.load)

        self.write_proposal(proposal())
        self.assert_code("provider_output_limit", lambda: self.load(limit=16))

    def test_file_provider_refuses_symlink_proposal_file(self) -> None:
        self.write_proposal(proposal())
        linked = self.base / "linked.json"
        try:
            linked.symlink_to(self.proposal_path)
        except (OSError, NotImplementedError):
            self.skipTest("Symlinks are unavailable on this filesystem")
        self.assert_code(
            "invalid_proposal_file",
            lambda: FileProposalProvider(linked).propose(
                request(), timeout_seconds=2, max_output_bytes=8192,
            ),
        )

    def test_codex_provider_is_blocked_without_a_verified_zero_tool_policy(self) -> None:
        exception = self.assert_code(
            "provider_blocked",
            lambda: CodexProvider(model="gpt-6-luna").propose(
                request(), timeout_seconds=30, max_output_bytes=8192,
            ),
        )
        self.assertIn("verifiable zero-tool policy", exception.message)
        self.assertIn("offline", exception.message)

    def test_provider_call_limits_are_validated(self) -> None:
        self.write_proposal(proposal())
        self.assert_code("invalid_provider_limits", lambda: self.load(limit=0))
        self.assert_code(
            "invalid_provider_limits",
            lambda: FileProposalProvider(self.proposal_path).propose(
                request(), timeout_seconds=float("inf"), max_output_bytes=8192,
            ),
        )


if __name__ == "__main__":
    unittest.main()
