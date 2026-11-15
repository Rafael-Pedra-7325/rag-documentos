import pytest

from rag_documentos.llm import NOT_FOUND_MESSAGE, Passage
from rag_documentos.pdf import NoTextError, PDFError
from rag_documentos.service import cited_refs
from tests.conftest import make_pdf


def test_ingest_registers_document(service, sample_pdf):
    result = service.ingest_pdf(sample_pdf, "manual.pdf")
    assert result.document.filename == "manual.pdf"
    assert result.document.pages == 3
    assert result.document.chunks >= 3
    assert result.replaced is False and result.warnings == []
    assert [d.doc_id for d in service.list_documents()] == [result.document.doc_id]


@pytest.mark.parametrize(
    "question,expected_page",
    [
        ("Qual o prazo para solicitar as férias?", 1),
        ("Quanto é o auxílio internet?", 2),
        ("Preciso de VPN?", 2),
        ("Qual o valor do vale-refeição?", 3),
    ],
)
def test_answers_point_to_the_right_page(service, sample_pdf, question, expected_page):
    service.ingest_pdf(sample_pdf, "manual.pdf")
    answer = service.ask(question)
    assert answer.sources[0].page == expected_page
    assert answer.sources[0].filename == "manual.pdf"


def test_reuploading_the_same_pdf_does_not_duplicate_chunks(service, sample_pdf):
    first = service.ingest_pdf(sample_pdf, "manual.pdf")
    count = service.store.count()
    second = service.ingest_pdf(sample_pdf, "manual-renamed.pdf")
    assert second.replaced is True
    assert second.document.doc_id == first.document.doc_id
    assert service.store.count() == count
    assert len(service.list_documents()) == 1


def test_delete_document_removes_chunks_and_registry_entry(service, sample_pdf):
    doc_id = service.ingest_pdf(sample_pdf, "manual.pdf").document.doc_id
    assert service.delete_document(doc_id) is True
    assert service.store.count() == 0
    assert service.list_documents() == []
    assert service.delete_document(doc_id) is False


def test_ask_on_empty_index_says_not_found(service):
    answer = service.ask("Qualquer coisa?")
    assert answer.answer == NOT_FOUND_MESSAGE
    assert answer.sources == []


def test_blank_question_is_rejected(service):
    with pytest.raises(ValueError):
        service.ask("   ")


def test_filtering_by_document(service):
    one = service.ingest_pdf(make_pdf(["Documento um fala sobre férias e descanso."]), "um.pdf")
    two = service.ingest_pdf(make_pdf(["Documento dois fala sobre férias e viagens."]), "dois.pdf")
    answer = service.ask("férias", doc_ids=[two.document.doc_id])
    assert {s.doc_id for s in answer.sources} == {two.document.doc_id}
    assert one.document.doc_id != two.document.doc_id


def test_warns_about_pages_without_text(service):
    pdf = make_pdf(["Texto com conteúdo suficiente para indexar.", ""])
    result = service.ingest_pdf(pdf, "x.pdf")
    assert len(result.warnings) == 1 and "2" in result.warnings[0]


def test_pdf_without_text_is_rejected(service):
    with pytest.raises(NoTextError):
        service.ingest_pdf(make_pdf(["", ""]), "scan.pdf")
    with pytest.raises(PDFError):
        service.ingest_pdf(b"nao e pdf", "x.pdf")


def test_sources_are_marked_as_cited_when_the_llm_cites_them(service, sample_pdf):
    class CitingLLM:
        name = "fake"

        def generate(self, question: str, passages: list[Passage]) -> str:
            return f"Resposta baseada no primeiro trecho [{passages[0].ref}]."

    service.llm = CitingLLM()
    service.ingest_pdf(sample_pdf, "manual.pdf")
    answer = service.ask("Qual o prazo para solicitar as férias?")
    assert [s.cited for s in answer.sources][0] is True
    assert not any(s.cited for s in answer.sources[1:])
    assert answer.model == "fake"


def test_min_score_filters_weak_matches(service, sample_pdf):
    from dataclasses import replace

    service.settings = replace(service.settings, min_score=0.99)
    service.ingest_pdf(sample_pdf, "manual.pdf")
    assert service.ask("Qual o prazo para solicitar as férias?").sources == []


@pytest.mark.parametrize(
    "text,expected",
    [
        ("veja [1] e [3]", {1, 3}),
        ("fontes [1, 2]", {1, 2}),
        ("[2][4]", {2, 4}),
        ("sem citação", set()),
    ],
)
def test_cited_refs(text, expected):
    assert cited_refs(text) == expected
