# System prompt changelog

## v4

- Added measured no-repeat and early-finalization rules for successful reads.
- Added an explicit one-pass playlist workflow from discovery through export.
- Added clarification guidance for underspecified recommendations.
- Added resource-absence guidance to prevent repeated equivalent searches.

## v3

- Defaulted playlist creation and editing to the Spotify export workflow unless
  the user explicitly requests Apple Music or ListenBrainz only.
- Clarified the ListenBrainz staging, MBID verification, and per-write approval
  sequence, while keeping feedback writes scoped to ListenBrainz.
- Added rules against duplicate preference writes and user-facing memory internals.

## v2

- Added the one-tool-per-step and explicit `final_answer` action protocol.
- Added failure honesty and a rule against silently repeating declined writes.
- Kept approval, retry, and termination authority in deterministic application code.

## v1

- Added the initial role, deterministic safety boundaries, stale-memory rule,
  and distinction between episodic and semantic retrieval.
