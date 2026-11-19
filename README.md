# Encore

Encore is a bounded music agent for ListenBrainz and MusicBrainz. It can inspect
listening history, discover recordings, create and edit ListenBrainz playlists,
export them to Spotify or Apple Music, and remember information across sessions.

The central rule is:

> The model proposes; deterministic Python validates and executes.

## Run it

From the repository root, create an environment, install Encore in editable
mode, and copy the safe configuration template:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env
```

Fill in `.env`, then run:

```bash
# Complete offline test suite
python -m pytest -q

# Check composition without starting a chat
python -m encore offline

# Offline models with per-operation account-write approval enabled
python -m encore offline --chat --allow-writes

# Online chat, read-only account access
python -m encore online --chat

# Online chat with per-operation account-write approval enabled
python -m encore online --chat --allow-writes

# Reproducible evaluation fault injection
python -m encore online --chat \
  --inject evaluation/injections/transient_seeded.json

# Show concise route, action, result, approval, compression, and final events
python -m encore online --chat --verbosity 1

# Show the full observable model/tool transcript and payloads
python -m encore online --chat --verbosity 2
```

Every account write still displays its exact proposed change and requires
approval. `--allow-writes` never bypasses that approval.

`online` and `offline` select only the model provider. Write authorization is an
independent startup choice: either mode is read-only without `--allow-writes`,
and either mode may perform individually approved writes with the flag.

`--inject` accepts a strict JSON object containing an integer `seed`, a
`transient_rate` from 0 to 1, and an optional `max_steps` from 1 to 12. Injected
ListenBrainz 429 and MusicBrainz 503 responses occur before network dispatch and
travel through the normal loop-owned retry path.

Verbosity `0` is the default final-answer interface. Level `1` prints compact
ReAct action/observation descriptions. Level `2` additionally prints complete
model function-call messages, arguments, and `ToolResult` payloads. Because the
model is forced to select functions, it does not return a private prose scratchpad;
level `2` shows all reasoning-loop data the application actually receives, but
never prints system prompts, credentials, or authorization headers.

If a request uses all 12 action steps, the console prints a deterministic summary
of completed and unresolved tool calls and asks whether to continue thinking.
Typing `yes` starts another independently bounded 12-step segment on the same
objective; any other response preserves the `budget_exceeded` result. This keeps
the runaway guard intact while allowing explicit user-directed continuation.

Additional memory commands:

```bash
# Delete only the configured user's local Encore memory
python -m encore online --forget

# Preview retention without changing anything
python -m encore offline --retention-plan \
  --now 2027-02-01T12:00:00Z

# Preview, explicitly confirm, then apply retention
python -m encore online --apply-retention \
  --now 2027-02-01T12:00:00Z
```

## Request flow

1. `SessionController` retrieves up to three relevant past sessions on the first
   request and carries bounded conversation between requests.
2. The cheap router selects one or more domains: listening, catalog, discovery,
   library, memory, or none.
3. Python derives all tools in those domains from the single tool catalog.
4. `AgentLoop` permits one proposed tool call per step, for at most 12 steps.
   It rejects exact repeats, stops after three consecutive malformed decisions
   or three failed uses of one tool, and preserves completed tool evidence in a
   fallback answer instead of spending the remaining budget in a loop.
5. `ToolExecutor` validates arguments, obtains approval for writes, and retries
   only transient read failures.
6. The tool returns the shared `ToolResult` contract.
7. A forced expensive-tier `final_answer` call produces the user response.
8. Python reconciles the final status with observed execution evidence.

The five possible statuses are `answered`, `answered_low_confidence`, `stuck`,
`budget_exceeded`, and `policy_halt`.

## The four memory types

| Type | Implementation | Purpose |
|---|---|---|
| Working | bounded model context plus SQLite hot state | Current conversation, active playlist, recent recordings, artist references, and current preferences |
| Episodic | user-filtered Chroma collection | Quality-checked summaries of previous sessions |
| Semantic | separate Chroma collection plus BM25 rank fusion | ListenBrainz/MusicBrainz documentation and music terminology |
| Procedural | `src/encore/prompts/system_v4.md` | Role, action protocol, stale-memory rules, and tool-use guidance |

Working-memory compression runs at 5,600 estimated conversation tokens or after
eight tool results. It replaces old messages with one compact summary and only
the tool evidence still needed. Exact hot state remains outside the generated
summary and is reinjected on every call.

At session close, one forced cheap-model call creates a six-field episodic
summary. A second independent call audits every claim. Python applies the 0.85
quality threshold and stores failed audits with lower retrieval weight and a
verify-before-use annotation.

Episodic retrieval uses:

```text
score = (0.80 × semantic similarity + 0.20 × recency) × quality weight
```

Near-duplicates merge at cosine similarity 0.93. Records have a 90-day TTL and a
50-summary per-user cap.

## Semantic retrieval choice

Production uses heading-aligned chunks, the pinned Chroma MiniLM embedding, and
reciprocal-rank fusion of dense and BM25 rankings. The measured experiment is
kept separately in `semantic_evaluation.py`; runtime retrieval remains in
`semantic_memory.py`.

Recorded results over 28 questions were:

| Chunking and retrieval | Top-3 recall |
|---|---:|
| Fixed 256-token windows, vector | 24/28 |
| Fixed windows, hybrid | 27/28 |
| Section-aware, vector | 28/28 |
| Section-aware, hybrid | 28/28 |

Run the experiment and build commands with:

```bash
python scripts/evaluate_semantic.py
python scripts/build_semantic_index.py
```

Run the frozen 30-request evaluation plus two additional trials for each of five
variance requests with the configured OpenRouter models:

```bash
python scripts/run_agent_evaluation.py \
  --mode online --output-name online_final
```

The runner disables account writes by default, isolates personal memory per
trial, applies case-specific fault fixtures, and incrementally writes results.
Each run writes to the git-ignored `data/runs/<output-name>/` directory:
`results.csv`, full answer review evidence, and a manifest with model aliases,
prompt/data hashes, source fingerprint, Python, and package versions. Add
`--resume` after an interrupted run, or use
`--base-only --ids H01 H02` for a small diagnostic batch.

For an offline run, omit `--mode online` and choose a separate output name:

```bash
python scripts/run_agent_evaluation.py --output-name offline_check
```

To rerun selected account-write cases with local models, keep them separate from
the read-only baseline and enable the evaluator's case-scoped approval policy:

```bash
python scripts/run_agent_evaluation.py \
  --base-only --allow-writes --ids H08 H09 H10 F04 F08
```

The runner approves only the write types predefined for each selected case,
always declines `delete_playlist`, and declines every unexpected write proposal.

## Account-write safety

- Model-provider mode does not control writes; `--allow-writes` is required.
- Every write requires a deterministic preview and explicit approval.
- Approval produces an opaque permit bound to the exact tool and arguments.
- Playlist recordings must be verified during the current session.
- Playlist edits fetch current ownership and state before writing.
- Playlist creation reuses an exact marked title instead of duplicating it.
- Writes are sequential, stop on partial failure, and are never blindly retried.
- Deletion is limited to owned playlists beginning with `[ListenBrainz]`.
- Spotify and Apple Music playlists are created only through ListenBrainz export;
  Encore cannot place them into service folders or inspect those libraries.

## Repository layout

```text
src/encore/          the encore package
  cli.py             CLI and dependency composition (python -m encore)
  paths.py           shared locations for resources and runtime data
  prompts/           versioned system prompts (procedural memory)
  knowledge_base/    public documents indexed as semantic memory
tests/               offline pytest suite
scripts/             evaluation and index-building commands
evaluation/          versioned evaluation inputs: requests, scenarios, fault fixtures
data/                git-ignored runtime state: memory/, traces/, runs/
```

`data/` is created on first use and never committed. Set `ENCORE_MEMORY_DIR`
to an absolute path to keep personal memory somewhere else.

## Important modules

All modules live in `src/encore/`.

| File | Responsibility |
|---|---|
| `cli.py` | CLI and dependency composition |
| `agent_loop.py` | Bounded request controller and final status |
| `router.py` | Domain classification and safe fallback |
| `tool_catalog.py` | Tool names, domains, descriptions, and JSON Schemas |
| `tool_executor.py` | Validation, approval interaction, and read retries |
| `tools.py` / `write_tools.py` | API read and account-write handlers |
| `approval.py` | Exact previews, permits, and session idempotency |
| `working_memory.py` | Context ordering, hot state, and compression |
| `episodic_memory.py` / `session_memory.py` | Past-session storage and quality checks |
| `semantic_memory.py` | Selected production knowledge retrieval |
| `semantic_evaluation.py` | Chunking and retrieval comparison experiment |
| `memory_store.py` / `retention.py` | SQLite state, privacy deletion, TTL/cap retention |
| `tracing.py` | Sanitized rubric-facing JSONL traces |
| `structured_output.py` | Shared strict parser for forced model calls |

## Privacy and observability

Traces record model identifiers, tiers, token counts, timing, routing, tool names,
sanitized arguments, approval outcomes, retries, and final status. They omit raw
prompts, model responses, tool payloads, user identity, headers, and credentials.

`--forget` deletes the configured user's hot state, preferences, episodic
summaries, and archived compressed turns. Semantic documentation is public and
is retained. The deletion coordinator has no music-service client, so it cannot
alter ListenBrainz or Spotify data.

## Current limitations

- Re-exporting a playlist in a later session may create another external-service
  playlist because ListenBrainz does not expose a reliable cross-session export
  idempotency key.
- `find_playlists` searches ListenBrainz staging playlists, not the Spotify or
  Apple Music library.
- Retention is an explicit CLI operation rather than an automatic scheduler.
- The submitted online evaluation kept account writes disabled, so its write
  cases measure safe stopping rather than successful account mutation.
