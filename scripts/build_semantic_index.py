"""Build Encore's reviewed semantic index without replacing existing records.

This command writes only generated public-document chunks to the local Chroma
directory. It never calls a music API and never reads or modifies user memory.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Running a script directly does not put src/ on sys.path; add it so the
# encore package imports work without an editable install.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from encore.config import Settings
from encore.paths import KNOWLEDGE_DIR, resolve_memory_root
from encore.semantic_memory import (
    build_semantic_index,
    corpus_fingerprint,
    load_default_tokenizer,
    load_documents,
    open_persistent_semantic_collection,
    section_chunks,
)


def main() -> int:
    """Chunk the versioned corpus and add it once to the compatible collection."""
    settings = Settings.from_environment()
    knowledge_root = KNOWLEDGE_DIR
    documents = load_documents(knowledge_root)
    fingerprint = corpus_fingerprint(documents)

    # Relative memory configuration resolves under the git-ignored data/ directory.
    memory_root = resolve_memory_root(settings.memory_dir)
    collection = open_persistent_semantic_collection(
        memory_root / "chroma",
        fingerprint,
    )
    # Tokenizer warmup may materialize the pinned local model on the first run.
    tokenizer = load_default_tokenizer()
    chunks = section_chunks(documents, tokenizer)
    result = build_semantic_index(collection, chunks, fingerprint)

    action = "added" if result.added else "already verified"
    print(f"Semantic index {action}: {result.chunk_count} section-aware chunks.")
    print(f"Corpus fingerprint: {fingerprint}")
    print("No user memory or music-service account data was changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
