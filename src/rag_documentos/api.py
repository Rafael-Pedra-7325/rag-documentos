"""FastAPI application: REST endpoints plus a small web UI served at ``/``."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from . import __version__
from .config import Settings
from .llm import LLMError
from .pdf import PDFError
from .service import RAGService

STATIC_DIR = Path(__file__).parent / "static"
SAMPLE_PDF = Path(__file__).resolve().parents[2] / "examples" / "manual-colaborador.pdf"


# --------------------------------------------------------------------------- schemas
class DocumentOut(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chunks: int
    size_bytes: int
    uploaded_at: str


class UploadResponse(BaseModel):
    document: DocumentOut
    warnings: list[str]
    replaced: bool


class QueryRequest(BaseModel):
    question: str = Field(
        min_length=3, max_length=2000, examples=["Quantos dias de férias eu tenho?"]
    )
    top_k: int | None = Field(default=None, ge=1, le=20, description="Trechos a recuperar")
    doc_ids: list[str] | None = Field(
        default=None, description="Restringe a busca a estes documentos"
    )


class SourceOut(BaseModel):
    ref: int
    doc_id: str
    filename: str
    page: int
    score: float
    text: str
    cited: bool


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    model: str


# --------------------------------------------------------------------------- routes
router = APIRouter(prefix="/api")


def get_service(request: Request) -> RAGService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Serviço ainda não inicializado.")
    return service


ServiceDep = Annotated[RAGService, Depends(get_service)]


@router.get("/health", tags=["meta"])
def health(service: ServiceDep) -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "embedding_model": service.embedder.name,
        "llm": service.llm.name,
        "documents": len(service.list_documents()),
    }


@router.post("/documents", response_model=UploadResponse, status_code=201, tags=["documentos"])
def upload_document(
    file: Annotated[UploadFile, File(description="Arquivo PDF")], service: ServiceDep
) -> dict:
    """Indexa um PDF. Reenviar o mesmo arquivo substitui a versão anterior."""
    filename = file.filename or "documento.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Envie um arquivo com extensão .pdf.")

    limit = service.settings.max_upload_mb * 1024 * 1024
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(
            status_code=413, detail=f"Arquivo maior que {service.settings.max_upload_mb} MB."
        )
    if b"%PDF" not in data[:1024]:
        raise HTTPException(status_code=415, detail="O arquivo não parece ser um PDF válido.")

    try:
        result = service.ingest_pdf(data, filename)
    except PDFError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return asdict(result)


@router.get("/documents", response_model=list[DocumentOut], tags=["documentos"])
def list_documents(service: ServiceDep) -> list[dict]:
    return [asdict(doc) for doc in service.list_documents()]


@router.delete("/documents/{doc_id}", status_code=204, tags=["documentos"])
def delete_document(doc_id: str, service: ServiceDep) -> Response:
    if not service.delete_document(doc_id):
        raise HTTPException(status_code=404, detail="Documento não encontrado.")
    return Response(status_code=204)


@router.post("/query", response_model=QueryResponse, tags=["perguntas"])
def query(body: QueryRequest, service: ServiceDep) -> dict:
    """Responde com base nos documentos indexados, citando arquivo e página."""
    try:
        answer = service.ask(body.question, top_k=body.top_k, doc_ids=body.doc_ids)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return asdict(answer)


# --------------------------------------------------------------------------- app
@asynccontextmanager
async def _lifespan(app: FastAPI):
    if getattr(app.state, "service", None) is None:
        settings = Settings.from_env()
        service = RAGService.from_settings(settings)
        if settings.seed_sample_pdf and not service.list_documents() and SAMPLE_PDF.exists():
            service.ingest_pdf(SAMPLE_PDF.read_bytes(), SAMPLE_PDF.name)
        app.state.service = service
    yield


def create_app(service: RAGService | None = None) -> FastAPI:
    """Build the app. Pass a ``service`` to inject dependencies (used by the tests)."""
    app = FastAPI(
        title="RAG Documentos",
        version=__version__,
        description="Pergunte aos seus PDFs: busca vetorial + LLM, citando a página de origem.",
        lifespan=_lifespan,
    )
    if service is not None:
        app.state.service = service
    app.include_router(router)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
