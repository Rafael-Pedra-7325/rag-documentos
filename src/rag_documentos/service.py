"""The RAG pipeline: ingest PDFs, retrieve relevant chunks, generate cited answers."""

from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .chunking import chunk_pages
from .config import Settings
from .embeddings import Embedder, get_embedder
from .llm import LLM, NOT_FOUND_MESSAGE, Passage, get_llm
from .pdf import NoTextError, extract_pages
from .registry import DocumentInfo, DocumentRegistry
from .vectorstore import StoredChunk, VectorStore, get_vector_store

_CITATION = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


def cited_refs(answer: str) -> set[int]:
    """Numbers of the passages the answer cites, e.g. ``[1]``, ``[2][3]`` or ``[1, 2]``."""
    refs: set[int] = set()
    for match in _CITATION.finditer(answer):
        refs.update(int(n) for n in re.findall(r"\d+", match.group(1)))
    return refs


@dataclass(frozen=True)
class Source:
    ref: int
    doc_id: str
    filename: str
    page: int
    score: float
    text: str
    cited: bool  # whether the answer actually references this passage


@dataclass(frozen=True)
class Answer:
    answer: str
    sources: list[Source]
    model: str


@dataclass(frozen=True)
class IngestResult:
    document: DocumentInfo
    warnings: list[str] = field(default_factory=list)
    replaced: bool = False


class RAGService:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        store: VectorStore,
        registry: DocumentRegistry,
        llm: LLM,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.registry = registry
        self.llm = llm
        self._write_lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings: Settings) -> RAGService:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            settings=settings,
            embedder=get_embedder(settings.embedding_backend, settings.embedding_model),
            store=get_vector_store(settings.vector_store, settings.chroma_dir, settings.numpy_dir),
            registry=DocumentRegistry(settings.registry_path),
            llm=get_llm(settings.llm_provider, settings.llm_model, settings.openai_base_url),
        )

    # ------------------------------------------------------------------ ingestion
    def ingest_pdf(self, data: bytes, filename: str) -> IngestResult:
        """Parse, chunk, embed and index a PDF. Re-uploading identical bytes replaces it."""
        filename = Path(filename).name or "documento.pdf"
        parsed = extract_pages(data)
        pieces = chunk_pages(parsed.pages, self.settings.chunk_size, self.settings.chunk_overlap)
        if not pieces:
            raise NoTextError("O PDF não tem texto suficiente para ser indexado.")

        doc_id = hashlib.sha256(data).hexdigest()[:16]
        vectors = self.embedder.embed_documents([c.text for c in pieces])  # slow, no lock held
        stored = [
            StoredChunk(
                id=f"{doc_id}:{c.index}",
                doc_id=doc_id,
                filename=filename,
                page=c.page,
                index=c.index,
                text=c.text,
            )
            for c in pieces
        ]
        info = DocumentInfo(
            doc_id=doc_id,
            filename=filename,
            pages=parsed.total_pages,
            chunks=len(stored),
            size_bytes=len(data),
            uploaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        with self._write_lock:
            replaced = self.registry.get(doc_id) is not None
            if replaced:
                self.store.delete_document(doc_id)
            self.store.add(stored, vectors)
            self.registry.put(info)

        warnings: list[str] = []
        if parsed.empty_pages:
            shown = ", ".join(str(n) for n in parsed.empty_pages[:10])
            more = "…" if len(parsed.empty_pages) > 10 else ""
            warnings.append(
                f"{len(parsed.empty_pages)} página(s) sem texto extraível "
                f"(PDF escaneado?): {shown}{more}"
            )
        return IngestResult(document=info, warnings=warnings, replaced=replaced)

    def list_documents(self) -> list[DocumentInfo]:
        return self.registry.list()

    def delete_document(self, doc_id: str) -> bool:
        with self._write_lock:
            removed = self.store.delete_document(doc_id)
            existed = self.registry.remove(doc_id)
        return existed or removed > 0

    # ------------------------------------------------------------------ querying
    def ask(
        self,
        question: str,
        top_k: int | None = None,
        doc_ids: list[str] | None = None,
    ) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("A pergunta não pode ser vazia.")

        k = top_k or self.settings.top_k
        hits = self.store.search(self.embedder.embed_query(question), k, doc_ids)
        hits = [h for h in hits if h.score >= self.settings.min_score]
        if not hits:
            return Answer(answer=NOT_FOUND_MESSAGE, sources=[], model=self.llm.name)

        passages = [
            Passage(ref=i, filename=h.chunk.filename, page=h.chunk.page, text=h.chunk.text)
            for i, h in enumerate(hits, start=1)
        ]
        text = self.llm.generate(question, passages)
        cited = cited_refs(text)
        sources = [
            Source(
                ref=i,
                doc_id=h.chunk.doc_id,
                filename=h.chunk.filename,
                page=h.chunk.page,
                score=round(h.score, 4),
                text=h.chunk.text,
                cited=i in cited,
            )
            for i, h in enumerate(hits, start=1)
        ]
        return Answer(answer=text, sources=sources, model=self.llm.name)
