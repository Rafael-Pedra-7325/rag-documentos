from __future__ import annotations

import io
import textwrap
from pathlib import Path

import pytest

from rag_documentos.config import Settings
from rag_documentos.embeddings import HashingEmbedder
from rag_documentos.llm import ExtractiveLLM
from rag_documentos.registry import DocumentRegistry
from rag_documentos.service import RAGService
from rag_documentos.vectorstore import NumpyStore

ROOT = Path(__file__).resolve().parent.parent

PAGES = [
    "As férias dos colaboradores são de trinta dias corridos após doze meses de trabalho. "
    "A solicitação de férias deve ser feita com quarenta e cinco dias de antecedência.",
    "O auxílio internet do trabalho remoto é de cem reais por mês. "
    "É obrigatório usar a VPN corporativa ao acessar sistemas internos.",
    "O vale-refeição é de quarenta e cinco reais por dia útil. "
    "O reembolso de despesas de viagem ocorre em até trinta dias.",
]


def make_pdf(pages: list[str]) -> bytes:
    """Build a small text PDF, one string per page (an empty string gives a blank page)."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    for text in pages:
        y = 800
        for line in textwrap.wrap(text, 90):
            pdf.drawString(50, y, line)
            y -= 14
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


@pytest.fixture
def sample_pdf() -> bytes:
    return make_pdf(PAGES)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        embedding_backend="hashing",
        vector_store="numpy",
        llm_provider="extractive",
        chunk_size=300,
        chunk_overlap=60,
    )


@pytest.fixture
def service(settings: Settings) -> RAGService:
    return RAGService(
        settings=settings,
        embedder=HashingEmbedder(),
        store=NumpyStore(),
        registry=DocumentRegistry(),
        llm=ExtractiveLLM(),
    )
