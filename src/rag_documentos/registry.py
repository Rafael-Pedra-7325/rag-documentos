"""A tiny JSON-backed registry of the documents that have been indexed."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from .fsutil import atomic_write_text


@dataclass
class DocumentInfo:
    doc_id: str
    filename: str
    pages: int  # total pages in the PDF
    chunks: int
    size_bytes: int
    uploaded_at: str  # ISO-8601, UTC


class DocumentRegistry:
    """Keeps document metadata; in memory only when ``path`` is ``None``."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = Path(path) if path else None
        self._docs: dict[str, DocumentInfo] = {}
        self._lock = threading.RLock()
        if self._path is not None and self._path.exists():
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._docs = {item["doc_id"]: DocumentInfo(**item) for item in raw}

    def _save(self) -> None:
        if self._path is not None:
            payload = json.dumps([asdict(d) for d in self._docs.values()], ensure_ascii=False)
            atomic_write_text(self._path, payload)

    def put(self, info: DocumentInfo) -> None:
        with self._lock:
            self._docs[info.doc_id] = info
            self._save()

    def get(self, doc_id: str) -> DocumentInfo | None:
        with self._lock:
            return self._docs.get(doc_id)

    def remove(self, doc_id: str) -> bool:
        with self._lock:
            existed = self._docs.pop(doc_id, None) is not None
            if existed:
                self._save()
            return existed

    def list(self) -> list[DocumentInfo]:
        with self._lock:
            return sorted(self._docs.values(), key=lambda d: d.uploaded_at, reverse=True)
