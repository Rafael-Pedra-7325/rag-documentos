from dataclasses import replace

from rag_documentos.evaluation import evaluate_retrieval, load_items
from tests.conftest import ROOT


def test_hashing_embedder_retrieves_the_right_page_on_the_sample_pdf(service):
    service.settings = replace(service.settings, chunk_size=900, chunk_overlap=150)
    pdf = ROOT / "examples" / "manual-colaborador.pdf"
    service.ingest_pdf(pdf.read_bytes(), pdf.name)
    items = load_items(ROOT / "examples" / "eval_questions.json")

    report = evaluate_retrieval(service, items, k=3)

    assert report.total == 13
    assert report.hit_at_k >= 0.8, report.misses
    assert report.mrr >= 0.6
