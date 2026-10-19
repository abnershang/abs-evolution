"""Explicit local commands. No scheduler, plugin installation, or host edits."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import stat
import sys
import tempfile

from .contracts import Policy
from .errors import EvolutionError
from .opportunities import summarize


def _read_json(path: Path, limit: int = 2_097_152):
    if path.stat().st_size > limit:
        raise EvolutionError("input_too_large", "Input JSON exceeds the command limit")
    def nonfinite(value):
        raise ValueError(f"Nonfinite value: {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=nonfinite)


def _provider_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--provider", choices=("file", "codex", "openrouter", "hybrid"), required=True)
    command.add_argument("--proposal-file", type=Path)
    command.add_argument("--model", help="Explicit OpenRouter model; in hybrid mode enables new-content generation")
    command.add_argument("--max-output-tokens", type=int, default=None,
                         help="OpenRouter completion-token cap, default 2048")
    command.add_argument("--candidate-pool", type=Path,
                         help="Hybrid only: reviewed JSON list of baseline-bound candidate proposals")
    command.add_argument("--jev-model", help="Hybrid only: pinned JEV version, default jev-1.13.0")
    command.add_argument("--min-choice-confidence", type=float, default=None,
                         help="Hybrid only: provisional routing threshold, default 0.8")
    command.add_argument("--min-choice-probability", type=float, default=None,
                         help="Hybrid only: provisional selected-option threshold, default 0.8")
    command.add_argument("--allow-external", action="store_true",
                         help="Hybrid only: assert approval to send reviewed inputs to TypeSafe and any configured generator")


def _read_candidate_pool(path: Path):
    from .providers import _strict_json_loads

    limit = 4 * 1024 * 1024
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | os.O_NONBLOCK)
    except OSError as exc:
        raise EvolutionError("invalid_candidate_pool", "Candidate pool must be an available regular file") from exc
    try:
        with os.fdopen(fd, "rb") as handle:
            metadata = os.fstat(handle.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise EvolutionError("invalid_candidate_pool", "Candidate pool must be a regular file")
            if metadata.st_size > limit:
                raise EvolutionError("input_too_large", "Candidate pool exceeds the command limit")
            raw = handle.read(limit + 1)
        if len(raw) > limit:
            raise EvolutionError("input_too_large", "Candidate pool exceeds the command limit")
        return _strict_json_loads(raw.decode("utf-8", "strict"))
    except (UnicodeError, ValueError, RecursionError, OSError) as exc:
        raise EvolutionError("invalid_candidate_pool", "Candidate pool must contain bounded strict UTF-8 JSON") from exc


def _proposal_provider(args: argparse.Namespace):
    from .providers import CodexProvider, FileProposalProvider
    hybrid_options = (
        args.candidate_pool, args.jev_model,
        args.min_choice_confidence, args.min_choice_probability,
    )
    if args.provider == "hybrid":
        if args.proposal_file is not None:
            raise EvolutionError("invalid_arguments", "--proposal-file is only valid for the file provider")
        if not args.allow_external:
            raise EvolutionError(
                "provider_blocked_external_not_allowed",
                "Hybrid routing requires --allow-external for the reviewed provider inputs",
            )
        if args.max_output_tokens is not None and not args.model:
            raise EvolutionError("invalid_arguments", "A completion-token cap requires a configured --model")
        from .hybrid import HybridProposalProvider
        from .jev import JevJudge
        candidates = _read_candidate_pool(args.candidate_pool) if args.candidate_pool is not None else []
        generator = None
        if args.model:
            from .openrouter import OpenRouterProvider
            generator = OpenRouterProvider(
                model=args.model,
                max_output_tokens=args.max_output_tokens if args.max_output_tokens is not None else 2048,
            )
        return HybridProposalProvider(
            JevJudge(model=args.jev_model if args.jev_model is not None else "jev-1.13.0"),
            generator,
            candidates=candidates,
            min_confidence=args.min_choice_confidence if args.min_choice_confidence is not None else 0.8,
            min_probability=args.min_choice_probability if args.min_choice_probability is not None else 0.8,
        )
    if any(value is not None for value in hybrid_options) or args.allow_external:
        raise EvolutionError("invalid_arguments", "JEV routing options are only valid for the hybrid provider")
    if args.provider == "file":
        if args.proposal_file is None:
            raise EvolutionError("proposal_required", "The file provider requires --proposal-file")
        if args.model is not None or args.max_output_tokens is not None:
            raise EvolutionError("invalid_arguments", "Model options do not apply to the file provider")
        return FileProposalProvider(args.proposal_file)
    if args.proposal_file is not None:
        raise EvolutionError("invalid_arguments", "--proposal-file is only valid for the file provider")
    if args.provider == "codex":
        if args.max_output_tokens is not None:
            raise EvolutionError("invalid_arguments", "--max-output-tokens is only valid for OpenRouter")
        return CodexProvider(model=args.model)
    if not args.model:
        raise EvolutionError("model_required", "OpenRouter requires an explicit --model provider/model ID")
    from .openrouter import OpenRouterProvider
    return OpenRouterProvider(
        model=args.model,
        max_output_tokens=args.max_output_tokens if args.max_output_tokens is not None else 2048,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Isolated local ABS evolution experiments")
    result.add_argument("--state", type=Path, help="Explicit local state directory; never an installed ABS path")
    commands = result.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Create a new isolated registry")
    init.add_argument("--policy", type=Path)
    commands.add_parser("status", help="Read local registry state")
    commands.add_parser("doctor", help="Probe the local execution sandbox without model calls")

    snapshots = commands.add_parser("snapshot").add_subparsers(dest="action", required=True)
    capture = snapshots.add_parser("import", help="Copy an explicit source into immutable storage")
    capture.add_argument("source", type=Path)
    capture.add_argument("--kind", choices=("skill", "tool", "workflow", "memory", "optimizer"), default="skill")
    snapshots.add_parser("list")
    show = snapshots.add_parser("show")
    show.add_argument("snapshot_id")

    channels = commands.add_parser("channel").add_subparsers(dest="action", required=True)
    bootstrap = channels.add_parser("bootstrap", help="Select the initial local baseline, without deployment")
    bootstrap.add_argument("snapshot_id")
    bootstrap.add_argument("--channel", default="stable")
    promote = channels.add_parser("promote", help="Promote existing engine evidence, locally only")
    promote.add_argument("evaluation_id")
    promote.add_argument("--expected-current", required=True)
    promote.add_argument("--channel", default="stable")
    rollback = channels.add_parser("rollback", help="Return future leases to a previously accepted version")
    rollback.add_argument("snapshot_id")
    rollback.add_argument("--expected-current", required=True)
    rollback.add_argument("--channel", default="stable")

    tasks = commands.add_parser("task").add_subparsers(dest="action", required=True)
    begin = tasks.add_parser("begin", help="Pin a full snapshot for an explicitly registered task")
    begin.add_argument("task_id")
    begin.add_argument("--channel", default="stable")
    end = tasks.add_parser("end")
    end.add_argument("task_id")
    tasks.add_parser("list")

    receipts = commands.add_parser("receipt").add_subparsers(dest="action", required=True)
    record = receipts.add_parser("record", help="Store a compact opt-in receipt, never a chat transcript")
    record.add_argument("file", type=Path)
    receipt_list = receipts.add_parser("list")
    receipt_list.add_argument("--limit", type=int, default=100)
    opportunities = commands.add_parser("opportunities", help="Find repeated failure tags for investigation")
    opportunities.add_argument("--min-occurrences", type=int, default=2)
    opportunities.add_argument("--limit", type=int, default=1000)

    policies = commands.add_parser("policy").add_subparsers(dest="action", required=True)
    policies.add_parser("show")
    policy_set = policies.add_parser("set", help="Explicitly update local operator policy; invalidates old evidence")
    policy_set.add_argument("file", type=Path)

    propose = commands.add_parser("propose", help="Generate validated proposal data without executing or promoting it")
    propose.add_argument("--request", type=Path, required=True,
                         help="Reviewed provider-request JSON, with training data only")
    _provider_arguments(propose)
    propose.add_argument("--timeout-seconds", type=float, default=60)
    propose.add_argument("--max-output-bytes", type=int, default=262144)

    evolve = commands.add_parser("evolve", help="Run one bounded, isolated experiment")
    evolve.add_argument("--suite", type=Path, required=True)
    _provider_arguments(evolve)
    evolve.add_argument("--channel", default="stable")
    evolve.add_argument("--promote", action="store_true", help="Allow evidence-gated local registry promotion; never installs ABS")
    evolve.add_argument(
        "--allow-shared-resources", action="store_true",
        help="Explicitly allow a supervised local run without hard RAM or aggregate scratch quotas; not for unattended use",
    )
    return result


def dispatch(args: argparse.Namespace):
    if args.command == "propose":
        provider = _proposal_provider(args)
        proposal = provider.propose(
            _read_json(args.request, 4 * 1024 * 1024),
            timeout_seconds=args.timeout_seconds,
            max_output_bytes=args.max_output_bytes,
        )
        output = {
            "status": "proposed",
            "provider": args.provider,
            "requested_model": args.model,
            "proposal": proposal.value,
            "duration_seconds": proposal.duration_seconds,
            "usage": {
                "input_tokens": proposal.input_tokens,
                "output_tokens": proposal.output_tokens,
                "model": proposal.model,
                "generation_id": proposal.generation_id,
                "cost_credits": proposal.cost_credits,
            },
            "candidate_executed": False,
            "registry_changed": False,
            "deployment_changed": False,
        }
        if proposal.routing is not None:
            from .contracts import validated_routing
            output["routing"] = validated_routing(proposal.routing)
        return output
    if args.command == "doctor":
        from .sandbox import Sandbox
        with tempfile.TemporaryDirectory(prefix="abs-evolution-probe-") as raw:
            scratch = Path(raw).resolve()
            sandbox = Sandbox()
            probe = sandbox.run(
                [sys.executable, "-I", "-S", "-c", "print('sandbox-ready')"],
                cwd=scratch, read_paths=[scratch], write_paths=[scratch],
                timeout_seconds=5, max_output_bytes=4096,
            )
            data = asdict(probe)
            data["model_called"] = False
            data["deployment_changed"] = False
            data["backend"] = sandbox.backend_identity
            data["hard_resource_limits"] = sandbox.hard_resource_limits
            data["unattended_ready"] = False
            return data
    if args.state is None:
        raise EvolutionError("state_required", "Supply --state with a separate local state directory")
    from .registry import Registry
    if args.command == "init":
        policy = Policy.from_dict(_read_json(args.policy)) if args.policy else Policy()
        registry = Registry.initialize(args.state, policy)
        return registry.status()
    registry = Registry(args.state)
    if args.command == "status":
        return registry.status()
    if args.command == "snapshot":
        if args.action == "import":
            return registry.import_snapshot(args.source, kind=args.kind, provenance={"origin": "explicit-local-import"})
        if args.action == "list":
            return registry.list_snapshots()
        registry.snapshot_path(args.snapshot_id)
        return registry.snapshot_info(args.snapshot_id)
    if args.command == "channel":
        if args.action == "bootstrap":
            return registry.bootstrap(args.snapshot_id, channel=args.channel)
        if args.action == "promote":
            return registry.promote(args.evaluation_id, expected_current=args.expected_current, channel=args.channel)
        return registry.rollback(args.snapshot_id, expected_current=args.expected_current, channel=args.channel)
    if args.command == "task":
        if args.action == "begin":
            return registry.begin_task(args.task_id, channel=args.channel)
        if args.action == "end":
            return registry.end_task(args.task_id)
        return registry.active_tasks()
    if args.command == "receipt":
        if args.action == "record":
            return registry.record_receipt(_read_json(args.file, registry.policy().max_receipt_bytes))
        return registry.list_receipts(limit=args.limit)
    if args.command == "opportunities":
        return summarize(registry.list_receipts(limit=args.limit), min_occurrences=args.min_occurrences)
    if args.command == "policy":
        if args.action == "show":
            policy = registry.policy()
            return {"policy": policy.to_dict(), "digest": policy.digest}
        policy = Policy.from_dict(_read_json(args.file))
        registry.update_policy(policy)
        return {"policy": policy.to_dict(), "digest": policy.digest}
    if args.command == "evolve":
        from .engine import EvolutionEngine
        from .sandbox import Sandbox
        provider = _proposal_provider(args)
        return EvolutionEngine(registry, Sandbox(), provider).run(
            args.suite, auto_promote=args.promote, channel=args.channel,
            allow_shared_resources=args.allow_shared_resources,
        )
    raise EvolutionError("unknown_command", "Unknown command")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        value = dispatch(args)
        print(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))
        if isinstance(value, dict) and value.get("status") in {"blocked", "failed", "quality_failed", "infrastructure_error", "timed_out", "output_limit"}:
            return 2
        return 0
    except EvolutionError as exc:
        print(json.dumps({"error": {"code": exc.code, "message": exc.message}}, ensure_ascii=False), file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(json.dumps({"error": {"code": "invalid_input_or_io", "message": str(exc)}}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
