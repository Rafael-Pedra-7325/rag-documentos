"""Text embedding backends.

* ``SentenceTransformerEmbedder``: multilingual neural model that runs locally (default).
* ``HashingEmbedder``: dependency-free lexical embedder, used for offline demos and tests.
"""

from __future__ import annotations

import re
import unicodedata
import zlib
from collections import Counter
from collections.abc import Sequence
from math import log
from typing import Protocol

import numpy as np


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale vectors (1-D or 2-D) to unit length, so dot product equals cosine similarity."""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(n, dim)`` array of unit vectors."""

    def embed_query(self, text: str) -> np.ndarray:
        """Return a ``(dim,)`` unit vector."""


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


_STOPWORDS = frozenset(
    _strip_accents(word)
    for word in (
        "a o as os um uma uns umas de do da dos das em no na nos nas por para com sem sob "
        "sobre e ou que se ao aos à às é são foi ser ter tem há como mais mas quando qual "
        "quais quanto quantos quantas the an of to in on for with and or is are was be it "
        "this that at by from"
    ).split()
)


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", _strip_accents(text.lower()))
    # crude plural stripping so "dias" and "dia" share a feature
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if w not in _STOPWORDS]


class HashingEmbedder:
    """Bag of (hashed) words and bigrams. Lexical, deterministic and needs no model download."""

    name = "hashing"

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> np.ndarray:
        tokens = _tokens(text)
        features = Counter(tokens)
        features.update(f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False))
        vector = np.zeros(self.dim, dtype=np.float32)
        for feature, count in features.items():
            digest = zlib.crc32(feature.encode("utf-8"))  # stable across processes
            sign = 1.0 if (digest >> 16) & 1 else -1.0
            vector[digest % self.dim] += sign * (1.0 + log(count))
        return vector

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return l2_normalize(np.stack([self._embed_one(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return l2_normalize(self._embed_one(text))


class SentenceTransformerEmbedder:
    """Local multilingual embeddings via ``sentence-transformers``.

    E5 models expect ``query:`` / ``passage:`` prefixes; they are added automatically.
    """

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run: pip install 'rag-documentos[embeddings]'"
            ) from exc
        self.name = model_name
        self._batch_size = batch_size
        self._model = SentenceTransformer(model_name)
        is_e5 = "e5" in model_name.lower()
        self._query_prefix = "query: " if is_e5 else ""
        self._passage_prefix = "passage: " if is_e5 else ""
        self.dim = int(len(self._model.encode(["dimension probe"])[0]))

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return self._encode([self._passage_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self._query_prefix + text])[0]


def get_embedder(backend: str, model_name: str) -> Embedder:
    if backend == "hashing":
        return HashingEmbedder()
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name)
    raise ValueError(f"Unknown embedding backend: {backend!r}")
