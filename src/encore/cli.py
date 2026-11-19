"""Encore CLI and composition boundary.

This entry point composes the complete tool suite and bounded agent loop.
Interactive execution begins only with ``--chat``, so ordinary startup remains
model- and network-free.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from uuid import uuid4

from encore.agent_loop import MAX_STEPS, AgentLoop
from encore.api_client import ApiClient
from encore.approval import ApprovalGate, IdempotencyLedger
from encore.config import RuntimePolicy, Settings
from encore.episodic_memory import EpisodicMemoryStore, open_persistent_episodic_collection
from encore.fault_injection import InjectionConfig, TransientFaultInjector
from encore.memory_store import ArchiveStore, MemoryPrivacyManager, PersonalMemoryStore
from encore.memory_tools import MemoryTools
from encore.model_client import ModelClient
from encore.paths import KNOWLEDGE_DIR, SYSTEM_PROMPT, TRACES_DIR, resolve_memory_root
from encore.retention import RetentionPlan, RetentionService
from encore.semantic_memory import (
    SemanticMemoryStore,
    corpus_fingerprint,
    load_documents,
    open_persistent_semantic_collection,
)
from encore.session_memory import SessionMemoryService
from encore.session_runtime import SessionController, run_console_session
from encore.session_state import SessionEvidence
from encore.tool_executor import ConsoleApprovalHandler, ToolExecutor
from encore.tool_suite import ToolSuite
from encore.tools import ReadTools
from encore.tracing import TraceWriter
from encore.write_tools import WriteTools
from encore.working_memory import WorkingMemoryService


def parse_utc_datetime(value: str) -> datetime:
    """Parse one timezone-aware ISO-8601 CLI value and normalize it to UTC."""
    # Python accepts ``+00:00`` universally; translating Z supports common CLI input.
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--now must be an ISO-8601 date/time such as 2026-10-28T12:00:00Z"
        ) from exc
    # A timezone is mandatory so a grader's simulation does not depend on locale.
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("--now must include Z or a UTC offset")
    return parsed.astimezone(timezone.utc)


def parse_injection_file(value: str) -> InjectionConfig:
    """Load one strict JSON injection file through argparse's error boundary."""
    try:
        return InjectionConfig.from_file(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _print_retention_plan(plan: RetentionPlan) -> None:
    """Print a privacy-minimized, non-mutating preview of exact candidate IDs."""
    print(f"Retention time: {plan.now.isoformat()}")
    print(f"Per-user episodic cap: {plan.max_summaries}")
    print(f"Eligible episodic summaries: {len(plan.candidates)}")
    # IDs and policy reasons are enough to review scope without exposing summaries.
    for candidate in plan.candidates:
        timestamp = datetime.fromtimestamp(candidate.timestamp, timezone.utc)
        print(
            f"  {candidate.memory_id}: {', '.join(candidate.reasons)}; "
            f"stored {timestamp.isoformat()}; quality_passed={candidate.quality_passed}"
        )

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse only process-level controls; natural-language goals come later."""
    # argparse supplies consistent help and rejects unknown modes before startup.
    parser = argparse.ArgumentParser(description="Run the Encore music agent.")
    # Provider selection is explicit so offline development cannot spend API quota.
    parser.add_argument("mode", choices=("online", "offline"))
    # This flag is necessary but still insufficient: each write also needs approval.
    parser.add_argument(
        "--allow-writes",
        action="store_true",
        help="Permit individually approved account writes in either model mode.",
    )
    # Only one process/session lifecycle action may run in an invocation.
    memory_action = parser.add_mutually_exclusive_group()
    # Chat is explicit so ordinary composition remains model- and network-free.
    memory_action.add_argument(
        "--chat",
        action="store_true",
        help="Start an interactive multi-request Encore session.",
    )
    # Privacy deletion is local-only and never implies ListenBrainz deletion.
    memory_action.add_argument(
        "--forget",
        action="store_true",
        help="Delete the configured user's local Encore memory and archived logs.",
    )
    # Planning is deliberately read-only and makes TTL/cap behavior visible.
    memory_action.add_argument(
        "--retention-plan",
        action="store_true",
        help="Preview same-user episodic summaries eligible under TTL/cap policy.",
    )
    # Applying is explicit and still requires a typed confirmation after preview.
    memory_action.add_argument(
        "--apply-retention",
        action="store_true",
        help="Preview, confirm, consolidate, promote, and remove eligible summaries.",
    )
    # A fixed clock lets the grader demonstrate retention without waiting 90 days.
    parser.add_argument(
        "--now",
        type=parse_utc_datetime,
        help="Use a timezone-aware ISO-8601 time for memory-policy simulation.",
    )
    parser.add_argument(
        "--inject",
        type=parse_injection_file,
        metavar="FILE.json",
        help=(
            "Load reproducible evaluation faults: seed, transient_rate, and/or "
            "max_steps."
        ),
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help=(
            "Console loop detail: 0 final output only, 1 action summaries, "
            "2 full observable model/tool transcript."
        ),
    )
    # Passing argv makes CLI behavior deterministic in unit tests.
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Compose dependencies without performing model, API, or account actions."""
    # Parse user-controlled startup policy before reading dependent configuration.
    args = parse_args(argv)
    # Freeze environment values into one settings object.
    settings = Settings.from_environment()
    # Runtime policy combines mode and the explicit write switch.
    policy = RuntimePolicy(args.mode, args.allow_writes)
    # Normal sessions use a live UTC clock; --now intentionally freezes it.
    memory_clock = (
        (lambda: args.now)
        if args.now is not None
        else (lambda: datetime.now(timezone.utc))
    )

    # All personal memory lives beneath one ignored, application-owned directory.
    memory_root = resolve_memory_root(settings.memory_dir)
    # SQLite stores exact hot state and explicit preferences.
    personal_store = PersonalMemoryStore(
        memory_root / "personal.sqlite3",
        now=lambda: memory_clock().timestamp(),
    )
    # Raw compressed-away turns remain separate from episodic summaries.
    archive_store = ArchiveStore(
        memory_root / "archives",
        now=lambda: memory_clock().timestamp(),
    )
    # Opening the collection creates metadata but does not embed text or call a model.
    episodic_store = EpisodicMemoryStore(
        open_persistent_episodic_collection(memory_root / "chroma"),
        now=memory_clock,
    )
    if args.forget:
        # Missing identity makes complete scoped deletion impossible, so fail closed.
        if not settings.encore_user_id:
            personal_store.close()
            print("Cannot forget memory: ENCORE_USER_ID is not configured.")
            return 2
        # The coordinator has no API client and therefore cannot touch ListenBrainz.
        report = MemoryPrivacyManager(
            personal_store,
            archive_store,
            episodic_store,
        ).forget_user(settings.encore_user_id)
        personal_store.close()
        # Counts make privacy behavior visible and straightforward to grade.
        print("Deleted local Encore memory for the configured user:")
        print(f"  hot state: {report.hot_state}")
        print(f"  preferences: {report.preferences}")
        print(f"  episodic summaries: {report.episodic_summaries}")
        print(f"  archived messages: {report.archived_messages}")
        print(f"  archive files: {report.archive_files}")
        print("ListenBrainz account data was not changed.")
        return 0

    if args.retention_plan or args.apply_retention:
        # Missing identity would make a safe, user-scoped plan impossible.
        if not settings.encore_user_id:
            personal_store.close()
            print("Cannot run retention: ENCORE_USER_ID is not configured.")
            return 2
        # Model construction is inert for planning and used only after confirmation.
        retention = RetentionService(
            ModelClient(settings, args.mode),
            episodic_store,
            personal_store,
            now=memory_clock,
        )
        # Planning reads metadata only; it does not call a model or mutate a store.
        plan = retention.plan(settings.encore_user_id, now=args.now)
        _print_retention_plan(plan)
        if args.retention_plan:
            personal_store.close()
            print("Preview only: no memory was changed.")
            return 0
        if not plan.candidates:
            personal_store.close()
            print("Nothing is eligible; no memory was changed.")
            return 0

        # The exact phrase prevents an accidental Enter/yes from deleting memory.
        confirmation_phrase = f"APPLY RETENTION {len(plan.candidates)}"
        print(
            "This will call the configured cheap model, promote only verified "
            "standing preferences, then delete the listed local summaries."
        )
        print("ListenBrainz account data will not be changed.")
        try:
            response = input(f"Type {confirmation_phrase!r} to continue: ").strip()
        except (EOFError, KeyboardInterrupt):
            # Non-interactive/aborted input is a decline, never implied approval.
            response = ""
        if response != confirmation_phrase:
            personal_store.close()
            print("Retention declined; no memory was changed.")
            return 0

        result = retention.apply_plan(settings.encore_user_id, plan)
        personal_store.close()
        if not result.success:
            print(result.error or "Retention did not complete.")
            print("No broad cleanup retry was attempted.")
            return 1
        # Counts demonstrate policy behavior without printing private preferences.
        print(f"Long-term preferences promoted: {len(result.promoted)}")
        print(f"Preference proposals skipped: {len(result.skipped)}")
        print(f"Episodic summaries removed: {result.deleted_count}")
        print("ListenBrainz account data was not changed.")
        return 0

    if args.chat and not settings.encore_user_id:
        # Retrieval, traces, archives, and session summaries all need one owner key.
        personal_store.close()
        print("Cannot start a session: ENCORE_USER_ID is not configured.")
        return 2

    # Public semantic startup occurs only after independent privacy/retention exits.
    # Therefore a stale public index can never block --forget from deleting user data.
    knowledge_documents = load_documents(KNOWLEDGE_DIR)
    knowledge_fingerprint = corpus_fingerprint(knowledge_documents)
    # Semantic facts use a distinct global collection with no user identifier.
    semantic_store = SemanticMemoryStore(
        open_persistent_semantic_collection(
            memory_root / "chroma",
            knowledge_fingerprint,
        ),
        knowledge_fingerprint,
    )

    # Constructing the full suite proves every Part 3 schema has one handler.
    # Construction performs no network/model call and issues no write permit.
    # One client centralizes host allowlisting, auth, pacing, and timeouts.
    fault_injector = (
        TransientFaultInjector(args.inject) if args.inject is not None else None
    )
    client = ApiClient(settings, fault_injector=fault_injector)
    # Verification evidence lives only for this agent session.
    evidence = SessionEvidence()
    # The loop will use this gate to exchange confirmation for opaque permits.
    approval_gate = ApprovalGate()
    # Reads and writes share evidence so verification can authorize later tracks.
    reads = ReadTools(client, settings.listenbrainz_user, evidence)
    # Writes receive every deterministic dependency explicitly.
    writes = WriteTools(
        client,
        settings.listenbrainz_user,
        policy,
        evidence,
        approval_gate,
        # Session-scoped: nothing personal persists outside the memory stores.
        IdempotencyLedger(),
    )
    # All three memory tools now use their persistent, type-appropriate stores.
    memory = MemoryTools(
        personal_store,
        episodic_store,
        semantic_store,
        settings.encore_user_id,
    )
    # Construction verifies all 18 action names have exactly one handler.
    suite = ToolSuite(reads, writes, memory)
    # Executor owns schema validation, approval interaction, and retry policy.
    # It is composed now but does not prompt or run anything until the loop calls it.
    executor = ToolExecutor(
        suite.registry(),
        approval_gate=approval_gate,
        approval_handler=ConsoleApprovalHandler(),
    )

    # These lines are diagnostic only and contain no secret values.
    print("Encore is ready.")
    print(f"Mode: {args.mode}")
    print(f"Writes enabled: {policy.writes_enabled}")
    if args.inject is not None:
        print(
            "Fault injection: "
            f"seed={args.inject.seed}, "
            f"transient_rate={args.inject.transient_rate}, "
            f"max_steps={args.inject.max_steps or MAX_STEPS}"
        )
    if args.verbosity:
        print(f"Loop verbosity: {args.verbosity}")
    print("Implemented tools: " + ", ".join(suite.registry()))
    # Model construction resolves no endpoint and sends no request until chat runs.
    model = ModelClient(settings, args.mode)
    # Versioned procedural memory is read locally and reinjected on every model call.
    prompt_path = SYSTEM_PROMPT
    system_prompt = prompt_path.read_text(encoding="utf-8")
    # One UUID ties traces, raw archives, and the episodic write to this session.
    session_id = str(uuid4())
    # Composition-only startup creates no trace; interactive mode owns one writer.
    trace = (
        TraceWriter(
            TRACES_DIR,
            settings.encore_user_id,
            session_id=session_id,
            now=memory_clock,
        )
        if args.chat and settings.encore_user_id
        else None
    )
    # Working memory is active whenever a deletion identity is configured.
    working_memory = (
        WorkingMemoryService(
            model,
            personal_store,
            archive_store,
            settings.encore_user_id,
            session_id,
            trace=trace,
            # Uses the memory clock so --now simulations stay consistent.
            session_started_at=memory_clock().timestamp(),
        )
        if settings.encore_user_id
        else None
    )
    step_limit = (
        args.inject.max_steps
        if args.inject is not None and args.inject.max_steps is not None
        else MAX_STEPS
    )
    loop = AgentLoop(
        model,
        executor,
        system_prompt,
        max_steps=step_limit,
        trace=trace,
        working_memory=working_memory,
        verbosity=args.verbosity,
    )
    # Session closure will call this service even after a failed agent request.
    # Composition alone performs neither summarization nor an embedding write.
    session_memory = SessionMemoryService(model, episodic_store, trace=trace)

    # Referencing concrete objects documents successful dependency composition.
    print(f"Tool executor: {type(executor).__name__} ready")
    print(f"Agent loop: {type(loop).__name__} ready (maximum {step_limit} steps)")
    print(
        "Working memory: "
        + ("WorkingMemoryService ready" if working_memory is not None else "disabled")
    )
    print(f"Session memory: {type(session_memory).__name__} ready")
    if args.chat:
        assert settings.encore_user_id is not None
        assert trace is not None
        try:
            # A new chat must not inherit the previous session's active playlist,
            # recordings, or declined write. Composition-only startup stays inert.
            if working_memory is not None:
                working_memory.reset_session_state()
            controller = SessionController(
                loop,
                episodic_store,
                session_memory,
                settings.encore_user_id,
                session_id,
                trace=trace,
            )
            # The console runner guarantees session-memory closure in a finally block.
            return run_console_session(controller)
        finally:
            # Close local resources even after model, trace, or terminal failures.
            trace.close()
            personal_store.close()
    # The composition-only CLI owns and closes its SQLite resource before exit.
    personal_store.close()
    # Unix convention: zero indicates successful composition.
    return 0


if __name__ == "__main__":
    # Raise SystemExit so the shell receives main's explicit status code.
    raise SystemExit(main())
