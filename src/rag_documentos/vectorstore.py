"""Vector stores.

* ``ChromaStore``: persistent ChromaDB collection with cosine distance (default).
* ``NumpyStore``: exact brute-force cosine search; zero extra dependencies. Optionally
  persisted to disk. Ideal for tests, demos and small corpora.
"""

from __future__ import annotations

import io
import json
import threading
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .embeddings import l2_normalize
from .fsutil import atomic_write_bytes, atomic_write_text


@dataclass(frozen=True)
class StoredChunk:
    id: str
    doc_id: str
    filename: str
    page: int
    index: int
    text: str


@dataclass(frozen=True)
class SearchHit:
    chunk: StoredChunk
    score: float  # cosine similarity, higher is better


class VectorStore(Protocol):
    def add(self, chunks: Sequence[StoredChunk], vectors: np.ndarray) -> None: ...

    def search(
        self, vector: np.ndarray, k: int, doc_ids: Sequence[str] | None = None
    ) -> list[SearchHit]: ...

    def delete_document(self, doc_id: str) -> int: ...

    def count(self) -> int: ...


def _check_batch(chunks: Sequence[StoredChunk], vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or len(chunks) != len(vectors):
        raise ValueError("vectors must be a 2-D array with one row per chunk")
    return vectors


class NumpyStore:
    """Exact cosine search over an in-memory matrix, optionally persisted to ``persist_dir``."""

    def __init__(self, persist_dir: Path | None = None) -> None:
        self._dir = Path(persist_dir) if persist_dir else None
        self._vectors = np.empty((0, 0), dtype=np.float32)
        self._chunks: list[StoredChunk] = []
        self._lock = threading.RLock()
        if self._dir is not None:
            self._dir.mkdir(parents=True, exist_ok=True)
            self._load()

    def _load(self) -> None:
        assert self._dir is not None
        vectors_path, chunks_path = self._dir / "vectors.npy", self._dir / "chunks.json"
        if vectors_path.exists() and chunks_path.exists():
            self._vectors = np.load(vectors_path)
            items = json.loads(chunks_path.read_text(encoding="utf-8"))
            self._chunks = [StoredChunk(**item) for item in items]

    def _save(self) -> None:
        if self._dir is None:
            return
        buffer = io.BytesIO()
        np.save(buffer, self._vectors)
        atomic_write_bytes(self._dir / "vectors.npy", buffer.getvalue())
        payload = json.dumps([asdict(c) for c in self._chunks], ensure_ascii=False)
        atomic_write_text(self._dir / "chunks.json", payload)

    def add(self, chunks: Sequence[StoredChunk], vectors: np.ndarray) -> None:
        vectors = l2_normalize(_check_batch(chunks, vectors))
        if len(chunks) == 0:
            return
        with self._lock:
            if not self._chunks:
                self._vectors = vectors.copy()
            elif vectors.shape[1] != self._vectors.shape[1]:
                raise ValueError("embedding dimension differs from the stored vectors")
            else:
                self._vectors = np.vstack([self._vectors, vectors])
            self._chunks.extend(chunks)
            self._save()

    def search(
        self, vector: np.ndarray, k: int, doc_ids: Sequence[str] | None = None
    ) -> list[SearchHit]:
        with self._lock:
            if not self._chunks or k <= 0:
                return []
            if doc_ids is not None:
                allowed = set(doc_ids)
                if not allowed:
                    return []
            else:
                allowed = None
            query = l2_normalize(np.asarray(vector, dtype=np.float32).reshape(-1))
            if query.shape[0] != self._vectors.shape[1]:
                raise ValueError("query dimension differs from the stored vectors")
            scores = self._vectors @ query
            if allowed is not None:
                mask = np.fromiter(
                    (c.doc_id in allowed for c in self._chunks),
                    dtype=bool,
                    count=len(self._chunks),
                )
                scores = np.where(mask, scores, -np.inf)
            order = np.argsort(-scores)[:k]
            return [
                SearchHit(chunk=self._chunks[i], score=float(scores[i]))
                for i in order
                if np.isfinite(scores[i])
            ]

    def delete_document(self, doc_id: str) -> int:
        with self._lock:
            keep = [i for i, c in enumerate(self._chunks) if c.doc_id != doc_id]
            removed = len(self._chunks) - len(keep)
            if removed:
                self._chunks = [self._chunks[i] for i in keep]
                self._vectors = (
                    self._vectors[keep] if keep else np.empty((0, 0), dtype=np.float32)
                )
                self._save()
            return removed

    def count(self) -> int:
        with self._lock:
            return len(self._chunks)


class ChromaStore:
    """ChromaDB-backed store using cosine distance."""

    _BATCH = 500

    def __init__(self, persist_dir: Path, collection: str = "rag_documentos") -> None:
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as exc:
            raise RuntimeError(
                "chromadb is not installed. Run: pip install 'rag-documentos[chroma]'"
            ) from exc
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: Sequence[StoredChunk], vectors: np.ndarray) -> None:
        vectors = l2_normalize(_check_batch(chunks, vectors))
        for start in range(0, len(chunks), self._BATCH):
            batch = chunks[start : start + self._BATCH]
            self._collection.add(
                ids=[c.id for c in batch],
                embeddings=vectors[start : start + len(batch)].tolist(),
                documents=[c.text for c in batch],
                metadatas=[
                    {
                        "doc_id": c.doc_id,
                        "filename": c.filename,
                        "page": c.page,
                        "index": c.index,
                    }
                    for c in batch
                ],
            )

    def search(
        self, vector: np.ndarray, k: int, doc_ids: Sequence[str] | None = None
    ) -> list[SearchHit]:
        total = self._collection.count()
        if total == 0 or k <= 0:
            return []
        if doc_ids is not None:
            allowed = list(dict.fromkeys(doc_ids))
            if not allowed:
                return []
            where: dict | None = {"doc_id": {"$in": allowed}}
            matching = len(self._collection.get(where=where, include=[])["ids"])
            if matching == 0:
                return []
            n_results = min(k, matching)
        else:
            where = None
            n_results = min(k, total)

        query = l2_normalize(np.asarray(vector, dtype=np.float32).reshape(-1))
        result = self._collection.query(
            query_embeddings=[query.tolist()],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[SearchHit] = []
        for chunk_id, text, meta, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            strict=True,
        ):
            chunk = StoredChunk(
                id=chunk_id,
                doc_id=str(meta["doc_id"]),
                filename=str(meta["filename"]),
                page=int(meta["page"]),
                index=int(meta["index"]),
                text=text,
            )
            hits.append(SearchHit(chunk=chunk, score=1.0 - float(distance)))
        return hits

    def delete_document(self, doc_id: str) -> int:
        ids = self._collection.get(where={"doc_id": doc_id}, include=[])["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def count(self) -> int:
        return int(self._collection.count())


def get_vector_store(kind: str, chroma_dir: Path, numpy_dir: Path) -> VectorStore:
    if kind == "chroma":
        return ChromaStore(chroma_dir)
    if kind == "numpy":
        return NumpyStore(numpy_dir)
    raise ValueError(f"Unknown vector store: {kind!r}")
