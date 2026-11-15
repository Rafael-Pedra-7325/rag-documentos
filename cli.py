"""Command line interface: ``rag-docs ingest | ask | list | delete | eval``."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from .config import Settings
from .embeddings import get_embedder
from .evaluation import evaluate_retrieval, load_items
from .llm import ExtractiveLLM
from .pdf import PDFError
from .registry import DocumentRegistry
from .service import RAGService
from .vectorstore import NumpyStore


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-docs", description="Pergunte aos seus PDFs pela linha de comando."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="indexa um ou mais PDFs")
    ingest.add_argument("paths", nargs="+", type=Path)

    ask = sub.add_parser("ask", help="faz uma pergunta sobre os documentos indexados")
    ask.add_argument("question")
    ask.add_argument("-k", "--top-k", type=int, default=None)

    sub.add_parser("list", help="lista os documentos indexados")

    delete = sub.add_parser("delete", help="remove um documento do índice")
    delete.add_argument("doc_id")

    evaluate = sub.add_parser("eval", help="mede a qualidade da recuperação (Hit@k, MRR)")
    evaluate.add_argument("--pdf", type=Path, required=True)
    evaluate.add_argument("--questions", type=Path, required=True)
    evaluate.add_argument("-k", type=int, default=3)
    evaluate.add_argument("--embedding", choices=["sentence-transformers", "hashing"])
    return parser


def _cmd_ingest(service: RAGService, args: argparse.Namespace) -> int:
    status = 0
    for path in args.paths:
        try:
            result = service.ingest_pdf(path.read_bytes(), path.name)
        except (OSError, PDFError) as exc:
            print(f"✗ {path}: {exc}", file=sys.stderr)
            status = 1
            continue
        doc = result.document
        verb = "reindexado" if result.replaced else "indexado"
        print(f"✓ {doc.filename} {verb}: {doc.pages} páginas, {doc.chunks} trechos [{doc.doc_id}]")
        for warning in result.warnings:
            print(f"  ! {warning}")
    return status


def _cmd_ask(service: RAGService, args: argparse.Namespace) -> int:
    answer = service.ask(args.question, top_k=args.top_k)
    print(answer.answer)
    if answer.sources:
        print("\nFontes:")
        for source in answer.sources:
            mark = "*" if source.cited else " "
            print(f" {mark}[{source.ref}] {source.filename}, p. {source.page} ({source.score:.3f})")
    return 0


def _cmd_list(service: RAGService, _: argparse.Namespace) -> int:
    documents = service.list_documents()
    if not documents:
        print("Nenhum documento indexado.")
    for doc in documents:
        print(f"{doc.doc_id}  {doc.filename}  ({doc.pages} págs., {doc.chunks} trechos)")
    return 0


def _cmd_delete(service: RAGService, args: argparse.Namespace) -> int:
    if service.delete_document(args.doc_id):
        print("Documento removido.")
        return 0
    print("Documento não encontrado.", file=sys.stderr)
    return 1


def _cmd_eval(settings: Settings, args: argparse.Namespace) -> int:
    backend = args.embedding or settings.embedding_backend
    service = RAGService(
        settings=replace(settings, embedding_backend=backend),
        embedder=get_embedder(backend, settings.embedding_model),
        store=NumpyStore(),  # ephemeral, so every run is reproducible
        registry=DocumentRegistry(),
        llm=ExtractiveLLM(),
    )
    service.ingest_pdf(args.pdf.read_bytes(), args.pdf.name)
    report = evaluate_retrieval(service, load_items(args.questions), k=args.k)
    print(f"Embedder: {service.embedder.name}  |  perguntas: {report.total}")
    print(
        f"Hit@1: {report.hit_at_1:.0%}   Hit@{report.k}: {report.hit_at_k:.0%}"
        f"   MRR: {report.mrr:.3f}"
    )
    for miss in report.misses:
        print(f"  ✗ {miss['question']} (esperado p. {miss['expected']}, obtido {miss['got']})")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    settings = Settings.from_env()
    if args.command == "eval":
        return _cmd_eval(settings, args)
    service = RAGService.from_settings(settings)
    handlers = {"ingest": _cmd_ingest, "ask": _cmd_ask, "list": _cmd_list, "delete": _cmd_delete}
    return handlers[args.command](service, args)


if __name__ == "__main__":
    raise SystemExit(main())
