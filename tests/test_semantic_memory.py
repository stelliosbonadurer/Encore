"""Offline tests for semantic chunking, ranking, and evaluation labels."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pytest

from encore.paths import EVALUATION_DIR, KNOWLEDGE_DIR
from encore.semantic_evaluation import (
    FIXED_STRATEGY,
    HybridRetriever,
    EvaluationQuestion,
    evaluate_retriever,
    fixed_chunks,
    load_evaluation_questions,
    validate_answer_spans,
)
from encore.semantic_memory import (
    MODEL_TOKEN_LIMIT,
    SECTION_STRATEGY,
    BM25Index,
    KnowledgeDocument,
    KnowledgeChunk,
    SemanticIndexMismatch,
    SemanticMemoryStore,
    build_semantic_index,
    corpus_fingerprint,
    chunk_by_tokens,
    load_documents,
    lexical_terms,
    section_chunks,
    expected_semantic_metadata,
)


class FakeCollection:
    """In-memory Chroma substitute for persistence and hybrid-search tests."""

    def __init__(self, fingerprint: str) -> None:
        self.metadata = expected_semantic_metadata(fingerprint)
        self.records: dict[str, tuple[str, dict]] = {}
        self.dense_order: list[str] = []

    def count(self) -> int:
        return len(self.records)

    def add(self, *, ids: list[str], documents: list[str], metadatas: list[dict]) -> None:
        for chunk_id, document, metadata in zip(ids, documents, metadatas):
            self.records[chunk_id] = (document, metadata)
        self.dense_order = list(ids)

    def get(self, *, include: list[str]) -> dict:
        ids = list(self.records)
        result: dict = {"ids": ids}
        if "documents" in include:
            result["documents"] = [self.records[item][0] for item in ids]
        if "metadatas" in include:
            result["metadatas"] = [self.records[item][1] for item in ids]
        return result

    def query(self, **_: object) -> dict:
        return {"ids": [self.dense_order], "distances": [[0.1] * len(self.dense_order)]}


@dataclass(frozen=True)
class FakeEncoding:
    """Minimal tokenizers.Encoding substitute with visible word offsets."""

    ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    word_ids: tuple[int | None, ...]


class FakeTokenizer:
    """Whitespace/punctuation tokenizer that adds two MiniLM-like boundaries."""

    _pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def encode(self, text: str, add_special_tokens: bool = True) -> FakeEncoding:
        """Return deterministic offsets and one word ID per matched token."""
        matches = list(self._pattern.finditer(text))
        ids = list(range(10, 10 + len(matches)))
        offsets = [match.span() for match in matches]
        word_ids: list[int | None] = list(range(len(matches)))
        if add_special_tokens:
            # Boundary offsets are conventional empty offsets and never sliced.
            ids = [1, *ids, 2]
            offsets = [(0, 0), *offsets, (0, 0)]
            word_ids = [None, *word_ids, None]
        return FakeEncoding(tuple(ids), tuple(offsets), tuple(word_ids))


def keyword_embedder(texts: list[str]) -> list[list[float]]:
    """Map three visible concepts to small deterministic test vectors."""
    vectors = []
    for text in texts:
        lowered = text.casefold()
        vectors.append(
            [
                float("spotify" in lowered or "streaming" in lowered),
                float("recording" in lowered or "track" in lowered),
                float("shoegaze" in lowered or "vocals" in lowered),
            ]
        )
    return vectors


def test_fixed_windows_stay_within_limit_and_overlap() -> None:
    """Adjacent fixed chunks repeat context without exceeding the chosen size."""
    tokenizer = FakeTokenizer()
    text = " ".join(f"word{index}" for index in range(40))
    chunks = chunk_by_tokens(tokenizer, text, chunk_tokens=12, overlap=3)
    assert len(chunks) > 1
    assert all(len(tokenizer.encode(chunk).ids) <= 12 for chunk in chunks)
    # The last three words of the first ten-token body begin the next window.
    assert chunks[0].split()[-3:] == chunks[1].split()[:3]


def test_invalid_overlap_is_rejected() -> None:
    """An overlap as large as capacity could otherwise prevent progress."""
    with pytest.raises(ValueError, match="overlap"):
        chunk_by_tokens(FakeTokenizer(), "some words", chunk_tokens=8, overlap=6)


def test_section_chunks_keep_heading_and_stable_ids() -> None:
    """Human headings remain attached and unchanged content yields the same ID."""
    tokenizer = FakeTokenizer()
    document = KnowledgeDocument(
        "guide.md",
        "# Guide\n\nIntro.\n\n## Radio modes\n\nEasy stays familiar.\n",
    )
    first = section_chunks([document], tokenizer)
    second = section_chunks([document], tokenizer)
    radio = next(chunk for chunk in first if chunk.section == "Radio modes")
    assert radio.text.startswith("## Radio modes")
    assert radio.strategy == SECTION_STRATEGY
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]


def test_oversized_section_repeats_heading_and_respects_model_limit() -> None:
    """Subsections remain meaningful even after a long section is divided."""
    tokenizer = FakeTokenizer()
    body = " ".join(f"term{index}" for index in range(400))
    chunks = section_chunks(
        [KnowledgeDocument("long.md", f"# Long guide\n\n{body}")],
        tokenizer,
    )
    assert len(chunks) > 1
    assert all(chunk.text.startswith("# Long guide") for chunk in chunks)
    assert all(chunk.token_count <= MODEL_TOKEN_LIMIT for chunk in chunks)


def test_fixed_strategy_labels_every_chunk() -> None:
    """Experiment metadata always identifies the baseline policy."""
    chunks = fixed_chunks(
        [KnowledgeDocument("tiny.md", "# Tiny\n\nA small document.")],
        FakeTokenizer(),
    )
    assert chunks
    assert {chunk.strategy for chunk in chunks} == {FIXED_STRATEGY}


def test_bm25_prioritizes_exact_technical_identifier() -> None:
    """Lexical ranking handles API paths that dense semantics may blur."""
    chunks = section_chunks(
        [
            KnowledgeDocument("general.md", "# General\n\nUse the account service."),
            KnowledgeDocument(
                "api.md",
                "# Endpoint\n\nCall /1/user/create for this exact operation.",
            ),
        ],
        FakeTokenizer(),
    )
    result = BM25Index(chunks).search("Which action uses /1/user/create?", 1)
    assert result[0].chunk.source == "api.md"


def test_lexical_terms_keep_paths_and_normalize_simple_inflections() -> None:
    """BM25 preserves identifiers while aligning ordinary word variants."""
    terms = lexical_terms("Requests exceeded /1/explore/lb-radio while running")
    assert "request" in terms
    assert "exceed" in terms
    assert "/1/explore/lb-radio" in terms
    assert "run" in terms


def test_vector_and_hybrid_evaluation_report_exact_answer_rank() -> None:
    """The evaluator counts a hit only when returned evidence contains the label."""
    chunks = section_chunks(
        [
            KnowledgeDocument("streaming.md", "# Link\n\nSpotify is a streaming service."),
            KnowledgeDocument("catalog.md", "# Meaning\n\nA track is one release appearance."),
        ],
        FakeTokenizer(),
    )
    questions = [EvaluationQuestion("Which streaming link is supported?", "Spotify")]
    retriever = HybridRetriever(chunks, keyword_embedder)
    vector = evaluate_retriever(retriever, questions, retrieval="vector")
    hybrid = evaluate_retriever(retriever, questions, retrieval="hybrid")
    assert vector.hits_at_1 == 1
    assert hybrid.hits_at_3 == 1
    assert vector.longest_chunk_tokens <= MODEL_TOKEN_LIMIT


def test_versioned_corpus_has_enough_valid_original_labels() -> None:
    """The checked-in experiment satisfies its minimum corpus/evaluation sizes."""
    documents = load_documents(KNOWLEDGE_DIR)
    questions = load_evaluation_questions(EVALUATION_DIR / "semantic_questions.json")
    glossary = next(document.text for document in documents if document.source == "music_glossary.md")
    # Level-two headings count the original concepts; the level-one title is excluded.
    assert len(re.findall(r"^## ", glossary, flags=re.MULTILINE)) >= 30
    assert len(questions) >= 20
    validate_answer_spans(documents, questions)


def test_semantic_build_is_additive_once_then_verified_noop() -> None:
    """The builder never overwrites an existing compatible reviewed corpus."""
    documents = [KnowledgeDocument("one.md", "# One\n\nPublic fact.")]
    fingerprint = corpus_fingerprint(documents)
    chunks = section_chunks(documents, FakeTokenizer())
    collection = FakeCollection(fingerprint)
    first = build_semantic_index(collection, chunks, fingerprint)
    second = build_semantic_index(collection, chunks, fingerprint)
    assert first.added is True
    assert second.added is False
    assert collection.count() == len(chunks)


def test_semantic_build_refuses_unexpected_existing_records() -> None:
    """Unexpected additions require review rather than automatic replacement."""
    documents = [KnowledgeDocument("one.md", "# One\n\nPublic fact.")]
    fingerprint = corpus_fingerprint(documents)
    chunks = section_chunks(documents, FakeTokenizer())
    collection = FakeCollection(fingerprint)
    collection.records["unexpected"] = ("other", {})
    with pytest.raises(SemanticIndexMismatch, match="automatic replacement"):
        build_semantic_index(collection, chunks, fingerprint)


def test_persistent_semantic_store_combines_dense_and_exact_path_ranks() -> None:
    """A technical path match can outrank a dense-first generic chunk."""
    documents = [KnowledgeDocument("api.md", "# API\n\nFacts.")]
    fingerprint = corpus_fingerprint(documents)
    collection = FakeCollection(fingerprint)
    general = KnowledgeChunk(
        "general", "general.md", "General", SECTION_STRATEGY, 8,
        "A general account request explanation.",
    )
    endpoint = KnowledgeChunk(
        "endpoint", "api.md", "Endpoint", SECTION_STRATEGY, 8,
        "Call /1/explore/lb-radio for radio exploration.",
    )
    build_semantic_index(collection, [general, endpoint], fingerprint)
    # Pretend dense retrieval preferred the generic explanation.
    collection.dense_order = [general.chunk_id, endpoint.chunk_id]
    store = SemanticMemoryStore(collection, fingerprint)
    results = store.search_knowledge("Use /1/explore/lb-radio", 1)
    assert results[0]["id"] == endpoint.chunk_id
    assert results[0]["source"] == "api.md"
