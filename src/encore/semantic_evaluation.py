"""Reproducible semantic-retrieval experiment kept outside runtime retrieval."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from encore.semantic_memory import (
    RRF_K,
    BM25Index,
    Embedder,
    KnowledgeChunk,
    KnowledgeDocument,
    TextTokenizer,
    chunk_by_tokens,
    make_chunk,
)

FIXED_STRATEGY = "fixed_256_overlap_32"


@dataclass(frozen=True)
class EvaluationQuestion:
    question: str
    answer: str


@dataclass(frozen=True)
class RetrievalHit:
    chunk: KnowledgeChunk
    score: float


@dataclass(frozen=True)
class QuestionResult:
    question: str
    answer: str
    first_relevant_rank: int | None
    top_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationResult:
    strategy: str
    retrieval: str
    question_count: int
    chunk_count: int
    longest_chunk_tokens: int
    hits_at_1: int
    hits_at_3: int
    questions: tuple[QuestionResult, ...]

    @property
    def recall_at_1(self) -> float:
        return self.hits_at_1 / self.question_count if self.question_count else 0.0

    @property
    def recall_at_3(self) -> float:
        return self.hits_at_3 / self.question_count if self.question_count else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "retrieval": self.retrieval,
            "question_count": self.question_count,
            "chunk_count": self.chunk_count,
            "longest_chunk_tokens": self.longest_chunk_tokens,
            "hits_at_1": self.hits_at_1,
            "recall_at_1": round(self.recall_at_1, 6),
            "hits_at_3": self.hits_at_3,
            "recall_at_3": round(self.recall_at_3, 6),
            "misses_at_3": [
                {
                    "question": item.question,
                    "answer": item.answer,
                    "top_chunk_ids": list(item.top_chunk_ids),
                }
                for item in self.questions
                if item.first_relevant_rank is None or item.first_relevant_rank > 3
            ],
        }


def normalize_text(text: str) -> str:
    return " ".join(text.split()).casefold()


def load_evaluation_questions(path: Path) -> list[EvaluationQuestion]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("evaluation questions must be a non-empty JSON array")
    questions: list[EvaluationQuestion] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict) or set(item) != {"question", "answer"}:
            raise ValueError(f"evaluation question {index} has invalid fields")
        question, answer = item["question"], item["answer"]
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"evaluation question {index} is empty")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"evaluation answer {index} is empty")
        questions.append(EvaluationQuestion(question.strip(), answer.strip()))
    return questions


def validate_answer_spans(
    documents: Sequence[KnowledgeDocument],
    questions: Sequence[EvaluationQuestion],
) -> None:
    corpus = normalize_text("\n".join(document.text for document in documents))
    missing = [item.answer for item in questions if normalize_text(item.answer) not in corpus]
    if missing:
        raise ValueError("answers absent from corpus: " + "; ".join(missing))


def fixed_chunks(
    documents: Sequence[KnowledgeDocument],
    tokenizer: TextTokenizer,
) -> list[KnowledgeChunk]:
    """Build the assignment baseline: 256-token windows with 32-token overlap."""
    chunks: list[KnowledgeChunk] = []
    for document in documents:
        for text in chunk_by_tokens(tokenizer, document.text):
            chunks.append(
                make_chunk(
                    tokenizer,
                    source=document.source,
                    section="Document window",
                    strategy=FIXED_STRATEGY,
                    text=text,
                )
            )
    return chunks


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions do not match")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


class HybridRetriever:
    """Evaluation retriever for vector-only and hybrid comparisons."""

    def __init__(self, chunks: Sequence[KnowledgeChunk], embedder: Embedder) -> None:
        if not chunks:
            raise ValueError("retrieval requires at least one chunk")
        self.chunks = tuple(chunks)
        self.embedder = embedder
        vectors = embedder([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise ValueError("embedder returned the wrong number of vectors")
        self._vectors = [tuple(float(value) for value in vector) for vector in vectors]
        self._bm25 = BM25Index(chunks)

    def _vector_scores(self, query: str) -> list[float]:
        query_vectors = self.embedder([query])
        if len(query_vectors) != 1:
            raise ValueError("embedder must return exactly one query vector")
        query_vector = tuple(float(value) for value in query_vectors[0])
        return [_cosine_similarity(query_vector, vector) for vector in self._vectors]

    def vector_search(self, query: str, limit: int = 3) -> list[RetrievalHit]:
        if limit < 1:
            raise ValueError("limit must be positive")
        ranked = sorted(
            enumerate(self._vector_scores(query)),
            key=lambda item: (item[1], self.chunks[item[0]].chunk_id),
            reverse=True,
        )
        return [RetrievalHit(self.chunks[index], score) for index, score in ranked[:limit]]

    def hybrid_search(self, query: str, limit: int = 3) -> list[RetrievalHit]:
        if limit < 1:
            raise ValueError("limit must be positive")
        vector_scores = self._vector_scores(query)
        lexical_scores = self._bm25.scores(query)
        vector_order = sorted(
            range(len(self.chunks)),
            key=lambda index: (vector_scores[index], self.chunks[index].chunk_id),
            reverse=True,
        )
        lexical_order = sorted(
            (index for index, score in enumerate(lexical_scores) if score > 0.0),
            key=lambda index: (lexical_scores[index], self.chunks[index].chunk_id),
            reverse=True,
        )
        vector_rank = {index: rank for rank, index in enumerate(vector_order, start=1)}
        lexical_rank = {index: rank for rank, index in enumerate(lexical_order, start=1)}
        fused = [
            1.0 / (RRF_K + vector_rank[index])
            + (1.0 / (RRF_K + lexical_rank[index]) if index in lexical_rank else 0.0)
            for index in range(len(self.chunks))
        ]
        ranked = sorted(
            range(len(self.chunks)),
            key=lambda index: (fused[index], self.chunks[index].chunk_id),
            reverse=True,
        )
        return [RetrievalHit(self.chunks[index], fused[index]) for index in ranked[:limit]]


def evaluate_retriever(
    retriever: HybridRetriever,
    questions: Sequence[EvaluationQuestion],
    *,
    retrieval: str,
) -> EvaluationResult:
    if not questions:
        raise ValueError("evaluation requires at least one question")
    if retrieval not in {"vector", "hybrid"}:
        raise ValueError("retrieval must be 'vector' or 'hybrid'")
    search = retriever.vector_search if retrieval == "vector" else retriever.hybrid_search
    rows: list[QuestionResult] = []
    hits_at_1 = hits_at_3 = 0
    for item in questions:
        hits = search(item.question, min(3, len(retriever.chunks)))
        answer = normalize_text(item.answer)
        ranks = [
            rank
            for rank, hit in enumerate(hits, start=1)
            if answer in normalize_text(hit.chunk.text)
        ]
        first_rank = ranks[0] if ranks else None
        hits_at_1 += int(first_rank == 1)
        hits_at_3 += int(first_rank is not None and first_rank <= 3)
        rows.append(
            QuestionResult(
                item.question,
                item.answer,
                first_rank,
                tuple(hit.chunk.chunk_id for hit in hits),
            )
        )
    return EvaluationResult(
        retriever.chunks[0].strategy,
        retrieval,
        len(questions),
        len(retriever.chunks),
        max(chunk.token_count for chunk in retriever.chunks),
        hits_at_1,
        hits_at_3,
        tuple(rows),
    )
