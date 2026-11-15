"""Retrieval evaluation: does the right page show up in the top-k results?"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .service import RAGService


@dataclass(frozen=True)
class EvalItem:
    question: str
    page: int  # page that contains the answer


@dataclass
class EvalReport:
    total: int
    k: int
    hit_at_1: float
    hit_at_k: float
    mrr: float
    misses: list[dict] = field(default_factory=list)


def load_items(path: str | Path) -> list[EvalItem]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EvalItem(question=item["question"], page=int(item["page"])) for item in raw]


def evaluate_retrieval(service: RAGService, items: list[EvalItem], k: int = 3) -> EvalReport:
    """Hit@1, Hit@k and MRR, judged by page number of the retrieved chunks."""
    hits_1 = hits_k = 0
    reciprocal_ranks: list[float] = []
    misses: list[dict] = []
    for item in items:
        results = service.store.search(service.embedder.embed_query(item.question), k)
        pages = [r.chunk.page for r in results]
        if item.page in pages:
            rank = pages.index(item.page) + 1
            hits_k += 1
            hits_1 += rank == 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
            misses.append({"question": item.question, "expected": item.page, "got": pages})
    total = len(items) or 1
    return EvalReport(
        total=len(items),
        k=k,
        hit_at_1=hits_1 / total,
        hit_at_k=hits_k / total,
        mrr=sum(reciprocal_ranks) / total,
        misses=misses,
    )
