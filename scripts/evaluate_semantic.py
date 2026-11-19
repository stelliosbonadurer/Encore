"""Run Encore's semantic-memory experiment and print reproducible JSON.

The first invocation may download Chroma's pinned MiniLM artifact.  This script
does not create a persistent index, call a music API, or mutate user data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Running a script directly does not put src/ on sys.path; add it so the
# encore package imports work without an editable install.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from encore.episodic_memory import DEFAULT_EMBEDDING_MODEL_ID
from encore.paths import EVALUATION_DIR, KNOWLEDGE_DIR
from encore.semantic_evaluation import (
    HybridRetriever,
    evaluate_retriever,
    fixed_chunks,
    load_evaluation_questions,
    validate_answer_spans,
)
from encore.semantic_memory import (
    load_default_tokenizer,
    load_documents,
    section_chunks,
)


def main() -> int:
    """Evaluate two chunkers under vector-only and hybrid retrieval."""
    # All experiment inputs are version-controlled beside the application.
    knowledge_root = KNOWLEDGE_DIR
    documents = load_documents(knowledge_root)
    questions = load_evaluation_questions(EVALUATION_DIR / "semantic_questions.json")
    # Bad labels fail before any embedding work can produce misleading metrics.
    validate_answer_spans(documents, questions)

    # The embedding function and tokenizer come from the same pinned artifact.
    embedder = DefaultEmbeddingFunction()
    tokenizer = load_default_tokenizer(embedder)
    # Build both accepted candidate policies from the identical source documents.
    candidates = (
        fixed_chunks(documents, tokenizer),
        section_chunks(documents, tokenizer),
    )

    results = []
    for chunks in candidates:
        # Chunk vectors are reused for both retrieval policies in this strategy.
        retriever = HybridRetriever(chunks, embedder)
        results.append(evaluate_retriever(retriever, questions, retrieval="vector"))
        results.append(evaluate_retriever(retriever, questions, retrieval="hybrid"))

    # JSON makes the exact output easy to inspect and later record via reviewable patch.
    output = {
        "embedding_model_id": DEFAULT_EMBEDDING_MODEL_ID,
        "document_count": len(documents),
        "question_count": len(questions),
        "results": [result.to_dict() for result in results],
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
