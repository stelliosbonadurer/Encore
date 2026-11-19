"""Selected section-aware hybrid retrieval for Encore's knowledge corpus.

The comparison baseline and recall metrics live in ``semantic_evaluation.py``;
this module contains only shared corpus preparation and production retrieval.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from encore.episodic_memory import DEFAULT_EMBEDDING_MODEL_ID

# MiniLM accepts at most 256 tokens, including its two boundary tokens.
MODEL_TOKEN_LIMIT = 256
# The tokenizer adds one [CLS] and one [SEP] token around each input.
SPECIAL_TOKEN_COUNT = 2
# The fixed baseline shares 32 tokens across adjacent windows.
FIXED_OVERLAP = 32
# Reciprocal-rank fusion uses this conventional damping constant.
RRF_K = 60
# This name records the selected production chunking strategy.
SECTION_STRATEGY = "section_256"
# Semantic facts live separately from user-scoped episodic summaries.
SEMANTIC_COLLECTION_NAME = "encore_semantic_v1"
# Stored metadata interpretation changes only with an explicit schema bump.
SEMANTIC_SCHEMA_VERSION = 1

# Words, numbers, paths, and identifier-like strings are useful lexical clues.
LEXICAL_TOKEN_PATTERN = re.compile(r"/?[a-z0-9]+(?:[-_./:][a-z0-9]+)*")
# Very common function words add lexical noise but little topic evidence.
LEXICAL_STOP_WORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "do", "does",
        "for", "from", "how", "i", "in", "is", "it", "of", "on", "or",
        "that", "the", "this", "to", "what", "when", "which", "with",
    }
)
# Markdown headings define the human-authored semantic boundaries in this corpus.
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


class TokenEncoding(Protocol):
    """The subset of a Hugging Face tokenizers Encoding used by chunking."""

    # Integer token identifiers are needed only for exact length checks.
    ids: Sequence[int]
    # Character offsets allow slices to preserve the original readable text.
    offsets: Sequence[tuple[int, int]]
    # Equal neighboring IDs identify subword pieces of one source word.
    word_ids: Sequence[int | None]


class TextTokenizer(Protocol):
    """A production or test tokenizer capable of returning offset metadata."""

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> TokenEncoding: ...


# An embedder receives a batch and returns one vector for each input string.
Embedder = Callable[[list[str]], Sequence[Sequence[float]]]


class SemanticCollection(Protocol):
    """The narrow Chroma surface used by persistent semantic retrieval."""

    # Metadata pins the corpus, chunker, schema, and vector artifact.
    metadata: Mapping[str, Any] | None

    def count(self) -> int: ...

    def add(self, **kwargs: Any) -> None: ...

    def get(self, **kwargs: Any) -> dict[str, Any]: ...

    def query(self, **kwargs: Any) -> dict[str, Any]: ...


class SemanticIndexMismatch(RuntimeError):
    """Raised instead of silently mixing or replacing incompatible knowledge."""


@dataclass(frozen=True)
class KnowledgeDocument:
    """One version-controlled Markdown source in the public corpus."""

    # The file name is a transparent citation label, not an arbitrary URL.
    source: str
    # Text remains unchanged so exact-answer evaluation is reproducible.
    text: str


@dataclass(frozen=True)
class KnowledgeChunk:
    """One bounded, identifiable retrieval unit derived from a document."""

    # A content-derived ID stays stable across runs and insertion order changes.
    chunk_id: str
    # Source identifies the document from which the evidence was taken.
    source: str
    # Section is the closest Markdown heading known to the chunker.
    section: str
    # Strategy records the experimental policy that created this chunk.
    strategy: str
    # Token count includes the embedding model's special boundary tokens.
    token_count: int
    # Text is the only evidence returned to the agent.
    text: str

    def to_record(self, *, score: float | None = None) -> dict[str, Any]:
        """Return the small JSON-compatible shape used by the memory tool."""
        # Required provenance accompanies every piece of returned evidence.
        record: dict[str, Any] = {
            "id": self.chunk_id,
            "source": self.source,
            "title": self.section,
            "text": self.text,
            "strategy": self.strategy,
        }
        # Evaluation and production search may expose a comparable fused score.
        if score is not None:
            record["score"] = round(float(score), 8)
        return record


@dataclass(frozen=True)
class RankedChunk:
    """One lexical retrieval result, primarily useful for diagnostics."""

    chunk: KnowledgeChunk
    score: float


def load_documents(folder: Path) -> list[KnowledgeDocument]:
    """Load only top-level Markdown files in deterministic filename order."""
    # The evaluation JSON is excluded because labels must not enter retrieval.
    paths = sorted(path for path in Path(folder).glob("*.md") if path.is_file())
    if not paths:
        raise ValueError(f"no Markdown knowledge documents found in {folder}")
    return [
        KnowledgeDocument(path.name, path.read_text(encoding="utf-8"))
        for path in paths
    ]


def corpus_fingerprint(
    documents: Sequence[KnowledgeDocument],
    *,
    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL_ID,
) -> str:
    """Hash exact sources plus policy identity for stale-index detection."""
    digest = hashlib.sha256()
    # Policy fields ensure a chunking/model change cannot reuse old vectors.
    for value in (
        str(SEMANTIC_SCHEMA_VERSION),
        SECTION_STRATEGY,
        "hybrid_rrf",
        embedding_model_id,
    ):
        digest.update(value.encode("utf-8"))
        digest.update(b"\x00")
    # Sorting makes the fingerprint independent of filesystem enumeration order.
    for document in sorted(documents, key=lambda item: item.source):
        digest.update(document.source.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(document.text.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def expected_semantic_metadata(
    fingerprint: str,
    *,
    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL_ID,
) -> dict[str, Any]:
    """Return the exact immutable-by-policy Chroma collection metadata."""
    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise ValueError("corpus fingerprint must be a SHA-256 hex digest")
    return {
        "hnsw:space": "cosine",
        "schema_version": SEMANTIC_SCHEMA_VERSION,
        "embedding_model_id": embedding_model_id,
        "corpus_fingerprint": fingerprint,
        "chunking_strategy": SECTION_STRATEGY,
        "retrieval_strategy": "hybrid_rrf",
    }


def validate_semantic_metadata(
    metadata: Mapping[str, Any] | None,
    fingerprint: str,
    *,
    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL_ID,
) -> None:
    """Refuse an existing collection with stale corpus or vector identity."""
    actual = dict(metadata or {})
    expected = expected_semantic_metadata(
        fingerprint,
        embedding_model_id=embedding_model_id,
    )
    # hnsw:space is provider-owned after creation; the other fields are ours.
    for key in (
        "schema_version",
        "embedding_model_id",
        "corpus_fingerprint",
        "chunking_strategy",
        "retrieval_strategy",
    ):
        if actual.get(key) != expected[key]:
            raise SemanticIndexMismatch(
                f"semantic index metadata mismatch for {key}; "
                "build a new reviewed index instead of mixing vectors"
            )


def open_persistent_semantic_collection(
    path: Path,
    fingerprint: str,
    *,
    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL_ID,
) -> SemanticCollection:
    """Open or create the semantic collection without embedding corpus text."""
    # Local imports keep unit tests and unrelated CLI commands lightweight.
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    storage = Path(path)
    storage.mkdir(mode=0o700, parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(storage),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=SEMANTIC_COLLECTION_NAME,
        metadata=expected_semantic_metadata(
            fingerprint,
            embedding_model_id=embedding_model_id,
        ),
        embedding_function=DefaultEmbeddingFunction(),
    )
    # get_or_create preserves old metadata, so validation must follow the open.
    validate_semantic_metadata(
        collection.metadata,
        fingerprint,
        embedding_model_id=embedding_model_id,
    )
    return collection


def load_default_tokenizer(embedder: Embedder | None = None) -> TextTokenizer:
    """Load the exact tokenizer packaged with Chroma's pinned MiniLM artifact.

    Calling the embedding function once intentionally materializes the model on
    first use.  Evaluation scripts call this function explicitly; ordinary CLI
    composition does not, so startup never causes a surprise download.
    """
    # Imports stay local so offline unit tests need no model files.
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
    from tokenizers import Tokenizer

    # Use the caller's exact embedding function when one was already created.
    active_embedder = embedder or DefaultEmbeddingFunction()
    # One harmless local phrase triggers Chroma's documented lazy artifact setup.
    active_embedder(["Encore tokenizer warmup"])
    # Chroma publishes the tokenizer beside the pinned ONNX model files.
    tokenizer_path = os.path.join(
        ONNXMiniLM_L6_V2.DOWNLOAD_PATH,
        ONNXMiniLM_L6_V2.EXTRACTED_FOLDER_NAME,
        "tokenizer.json",
    )
    tokenizer = Tokenizer.from_file(tokenizer_path)
    # We must observe true lengths rather than silently truncate during evaluation.
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def token_count(tokenizer: TextTokenizer, text: str) -> int:
    """Count exactly what the embedding model receives, including boundaries."""
    return len(tokenizer.encode(text).ids)


def chunk_by_tokens(
    tokenizer: TextTokenizer,
    text: str,
    *,
    chunk_tokens: int = MODEL_TOKEN_LIMIT,
    overlap: int = FIXED_OVERLAP,
) -> list[str]:
    """Slice original text into whole-word token windows within the model limit."""
    if chunk_tokens <= SPECIAL_TOKEN_COUNT:
        raise ValueError("chunk_tokens must leave room for content tokens")
    # Content capacity excludes MiniLM's automatically added boundary tokens.
    content_capacity = chunk_tokens - SPECIAL_TOKEN_COUNT
    if overlap < 0 or overlap >= content_capacity:
        raise ValueError("overlap must be non-negative and smaller than content capacity")
    # Special tokens are excluded because their offsets do not map to source text.
    encoding = tokenizer.encode(text, add_special_tokens=False)
    offsets = list(encoding.offsets)
    word_ids = list(encoding.word_ids)
    token_total = len(offsets)
    if token_total == 0:
        return []

    def continues_word(index: int) -> bool:
        """Return whether a boundary would split two pieces of the same word."""
        return (
            0 < index < token_total
            and word_ids[index] is not None
            and word_ids[index] == word_ids[index - 1]
        )

    def punctuation_only(index: int) -> bool:
        """Avoid beginning a following chunk with detached punctuation."""
        begin, end = offsets[index]
        return not any(character.isalnum() for character in text[begin:end])

    chunks: list[str] = []
    start = 0
    while start < token_total:
        # First choose the largest possible endpoint under the content capacity.
        end = min(start + content_capacity, token_total)
        # Back up when that endpoint bisects a tokenizer word.
        while end > start + 1 and continues_word(end):
            end -= 1
        # Character offsets retain punctuation, Markdown, and human readability.
        piece = text[offsets[start][0] : offsets[end - 1][1]].strip()
        # A re-tokenization safety check handles unusual URLs or huge words.
        while end - start > 1 and token_count(tokenizer, piece) > chunk_tokens:
            end -= 1
            piece = text[offsets[start][0] : offsets[end - 1][1]].strip()
        if not piece:
            raise ValueError("tokenizer produced an empty non-terminal chunk")
        chunks.append(piece)
        if end >= token_total:
            break
        # Advance at least one token even when overlap settings are pathological.
        following = max(end - overlap, start + 1)
        # Shift forward until the next chunk starts at a readable boundary.
        while following < end and (
            continues_word(following) or punctuation_only(following)
        ):
            following += 1
        start = following
    return chunks


def _stable_chunk_id(source: str, strategy: str, section: str, text: str) -> str:
    """Derive a compact deterministic ID from provenance and exact content."""
    material = "\x1f".join((source, strategy, section, text)).encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()[:20]
    return f"knowledge-{digest}"


def make_chunk(
    tokenizer: TextTokenizer,
    *,
    source: str,
    section: str,
    strategy: str,
    text: str,
) -> KnowledgeChunk:
    """Construct one chunk only after enforcing the embedding limit."""
    count = token_count(tokenizer, text)
    if count > MODEL_TOKEN_LIMIT:
        raise ValueError(f"chunk exceeds {MODEL_TOKEN_LIMIT} tokens: {source}/{section}")
    return KnowledgeChunk(
        chunk_id=_stable_chunk_id(source, strategy, section, text),
        source=source,
        section=section,
        strategy=strategy,
        token_count=count,
        text=text,
    )


def _markdown_sections(text: str) -> list[tuple[str, str]]:
    """Split Markdown at headings while preserving each heading in its section."""
    matches = list(HEADING_PATTERN.finditer(text))
    if not matches:
        return [("Document", text.strip())] if text.strip() else []
    sections: list[tuple[str, str]] = []
    # Preserve any meaningful preamble before the first heading.
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("Preamble", preamble))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        title = match.group(2).strip()
        section_text = text[match.start() : end].strip()
        sections.append((title, section_text))
    return sections


def section_chunks(
    documents: Sequence[KnowledgeDocument],
    tokenizer: TextTokenizer,
) -> list[KnowledgeChunk]:
    """Build heading-aware chunks, subchunking only sections over 256 tokens."""
    chunks: list[KnowledgeChunk] = []
    for document in documents:
        for section, section_text in _markdown_sections(document.text):
            if token_count(tokenizer, section_text) <= MODEL_TOKEN_LIMIT:
                pieces = [section_text]
            else:
                # Each oversized piece repeats the heading to retain local meaning.
                heading_line = section_text.splitlines()[0].strip()
                body = "\n".join(section_text.splitlines()[1:]).strip()
                heading_tokens = token_count(tokenizer, heading_line) - SPECIAL_TOKEN_COUNT
                body_limit = MODEL_TOKEN_LIMIT - max(heading_tokens, 0)
                if body_limit <= SPECIAL_TOKEN_COUNT:
                    raise ValueError(f"heading alone is too long: {document.source}/{section}")
                body_pieces = chunk_by_tokens(
                    tokenizer,
                    body,
                    chunk_tokens=body_limit,
                    overlap=min(FIXED_OVERLAP, body_limit - SPECIAL_TOKEN_COUNT - 1),
                )
                pieces = [f"{heading_line}\n\n{piece}" for piece in body_pieces]
            for piece in pieces:
                chunks.append(
                    make_chunk(
                        tokenizer,
                        source=document.source,
                        section=section,
                        strategy=SECTION_STRATEGY,
                        text=piece,
                    )
                )
    return chunks


def lexical_terms(text: str) -> list[str]:
    """Tokenize transparently for BM25 while retaining technical identifiers."""
    terms: list[str] = []
    for raw_term in LEXICAL_TOKEN_PATTERN.findall(text.casefold()):
        # Endpoint paths and compound identifiers must remain exact.
        if any(character in raw_term for character in "-_./:"):
            terms.append(raw_term)
            continue
        if raw_term in LEXICAL_STOP_WORDS:
            continue
        # Small transparent suffix normalization aligns request/requests and
        # exceed/exceeded/exceeding without adding a stemming dependency.
        term = raw_term
        if len(term) > 5 and term.endswith("ing"):
            term = term[:-3]
            # English often doubles a final consonant before "ing" (running).
            if len(term) > 2 and term[-1] == term[-2] and term[-1] not in "aeiou":
                term = term[:-1]
        elif len(term) > 4 and term.endswith("ed"):
            term = term[:-2]
        elif len(term) > 3 and term.endswith("s") and not term.endswith("ss"):
            term = term[:-1]
        terms.append(term)
    return terms


class BM25Index:
    """Small dependency-free BM25 index over the fixed semantic corpus."""

    def __init__(
        self,
        chunks: Sequence[KnowledgeChunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not chunks:
            raise ValueError("BM25 requires at least one chunk")
        # A tuple protects ranking order from caller mutation.
        self.chunks = tuple(chunks)
        self.k1 = k1
        self.b = b
        # Term frequencies are computed once because the corpus is versioned.
        self._frequencies = [Counter(lexical_terms(chunk.text)) for chunk in chunks]
        self._lengths = [sum(frequencies.values()) for frequencies in self._frequencies]
        self._average_length = sum(self._lengths) / len(self._lengths)
        # Document frequencies count presence, not repetitions in one document.
        self._document_frequencies: Counter[str] = Counter()
        for frequencies in self._frequencies:
            self._document_frequencies.update(frequencies.keys())

    def scores(self, query: str) -> list[float]:
        """Calculate standard positive-IDF BM25 scores for every chunk."""
        query_terms = list(dict.fromkeys(lexical_terms(query)))
        document_count = len(self.chunks)
        scores: list[float] = []
        for frequencies, length in zip(self._frequencies, self._lengths):
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                document_frequency = self._document_frequencies[term]
                inverse_document_frequency = math.log(
                    1.0 + (document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                # Exact technical identifiers are the main reason this project
                # adds BM25; give paths/compound IDs twice an ordinary term's weight.
                identifier_weight = (
                    2.0 if any(character in term for character in "-_./:") else 1.0
                )
                normalization = frequency + self.k1 * (
                    1.0 - self.b + self.b * length / self._average_length
                )
                score += identifier_weight * inverse_document_frequency * (
                    frequency * (self.k1 + 1.0) / normalization
                )
            scores.append(score)
        return scores

    def search(self, query: str, limit: int) -> list[RankedChunk]:
        ranked = sorted(
            enumerate(self.scores(query)),
            key=lambda item: (item[1], self.chunks[item[0]].chunk_id),
            reverse=True,
        )
        return [
            RankedChunk(self.chunks[index], score)
            for index, score in ranked[:limit]
            if score > 0.0
        ]

@dataclass(frozen=True)
class SemanticBuildResult:
    """Describe an additive first build or a verified no-op repeat build."""

    chunk_count: int
    added: bool


def build_semantic_index(
    collection: SemanticCollection,
    chunks: Sequence[KnowledgeChunk],
    fingerprint: str,
    *,
    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL_ID,
) -> SemanticBuildResult:
    """Populate one empty compatible collection; never delete or overwrite it."""
    if not chunks:
        raise ValueError("cannot build an empty semantic index")
    validate_semantic_metadata(
        collection.metadata,
        fingerprint,
        embedding_model_id=embedding_model_id,
    )
    expected_ids = {chunk.chunk_id for chunk in chunks}
    if len(expected_ids) != len(chunks):
        raise ValueError("semantic chunk IDs are not unique")
    existing_count = collection.count()
    if existing_count:
        # Exact equality makes repeated builds safe without an upsert or deletion.
        existing = collection.get(include=[])
        existing_ids = {str(item) for item in existing.get("ids", [])}
        if existing_ids != expected_ids:
            raise SemanticIndexMismatch(
                "semantic collection is non-empty but does not exactly match "
                "the reviewed corpus; automatic replacement is disabled"
            )
        return SemanticBuildResult(chunk_count=len(chunks), added=False)
    # Chroma embeds this one reviewed batch with its configured pinned function.
    collection.add(
        ids=[chunk.chunk_id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        metadatas=[
            {
                "source": chunk.source,
                "section": chunk.section,
                "strategy": chunk.strategy,
                "token_count": chunk.token_count,
            }
            for chunk in chunks
        ],
    )
    return SemanticBuildResult(chunk_count=len(chunks), added=True)


class SemanticMemoryStore:
    """Query the persistent section index with the evaluated hybrid policy."""

    def __init__(
        self,
        collection: SemanticCollection,
        fingerprint: str,
        *,
        embedding_model_id: str = DEFAULT_EMBEDDING_MODEL_ID,
    ) -> None:
        validate_semantic_metadata(
            collection.metadata,
            fingerprint,
            embedding_model_id=embedding_model_id,
        )
        self.collection = collection
        self.fingerprint = fingerprint
        # BM25 needs the small public corpus in memory; no user data is involved.
        stored = collection.get(include=["documents", "metadatas"])
        ids = list(stored.get("ids", []))
        documents = list(stored.get("documents", []) or [])
        metadatas = list(stored.get("metadatas", []) or [])
        if not (len(ids) == len(documents) == len(metadatas)):
            raise RuntimeError("semantic collection returned incomplete stored records")
        chunks: list[KnowledgeChunk] = []
        for chunk_id, text, metadata in zip(ids, documents, metadatas):
            if not isinstance(metadata, Mapping):
                raise RuntimeError("semantic chunk metadata is malformed")
            chunks.append(
                KnowledgeChunk(
                    chunk_id=str(chunk_id),
                    source=str(metadata["source"]),
                    section=str(metadata["section"]),
                    strategy=str(metadata["strategy"]),
                    token_count=int(metadata["token_count"]),
                    text=str(text),
                )
            )
        # Stable ID order makes lexical tie behavior reproducible after reload.
        self.chunks = tuple(sorted(chunks, key=lambda chunk: chunk.chunk_id))
        self._by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        self._bm25 = BM25Index(self.chunks) if self.chunks else None

    def search_knowledge(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Return evaluated hybrid results with source and section provenance."""
        if not query.strip() or limit < 1 or not self.chunks or self._bm25 is None:
            return []
        # The corpus is only 63 chunks, so a complete dense ranking stays simple
        # and makes production RRF exactly match the measured evaluation policy.
        result = self.collection.query(
            query_texts=[query],
            n_results=len(self.chunks),
            include=["distances"],
        )
        nested_ids = result.get("ids") or [[]]
        dense_ids = [str(item) for item in nested_ids[0]]
        if set(dense_ids) != set(self._by_id):
            raise RuntimeError("semantic query did not return the complete indexed corpus")
        lexical_scores = self._bm25.scores(query)
        lexical_order = sorted(
            (
                index
                for index, score in enumerate(lexical_scores)
                if score > 0.0
            ),
            key=lambda index: (lexical_scores[index], self.chunks[index].chunk_id),
            reverse=True,
        )
        dense_rank = {chunk_id: rank for rank, chunk_id in enumerate(dense_ids, start=1)}
        lexical_rank = {
            self.chunks[index].chunk_id: rank
            for rank, index in enumerate(lexical_order, start=1)
        }
        fused_scores = {
            chunk_id: 1.0 / (RRF_K + dense_rank[chunk_id])
            + (
                1.0 / (RRF_K + lexical_rank[chunk_id])
                if chunk_id in lexical_rank
                else 0.0
            )
            for chunk_id in dense_ids
        }
        ranked_ids = sorted(
            dense_ids,
            key=lambda chunk_id: (fused_scores[chunk_id], chunk_id),
            reverse=True,
        )
        return [
            self._by_id[chunk_id].to_record(score=fused_scores[chunk_id])
            for chunk_id in ranked_ids[:limit]
        ]
