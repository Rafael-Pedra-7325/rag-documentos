import pytest

from rag_documentos.pdf import NoTextError, PDFError, clean_text, extract_pages
from tests.conftest import PAGES, make_pdf


def test_extract_pages_returns_text_with_page_numbers(sample_pdf):
    parsed = extract_pages(sample_pdf)
    assert parsed.total_pages == 3
    assert [p.number for p in parsed.pages] == [1, 2, 3]
    assert "trinta dias corridos" in parsed.pages[0].text
    assert "VPN" in parsed.pages[1].text


def test_blank_pages_are_reported():
    parsed = extract_pages(make_pdf([PAGES[0], "", PAGES[2]]))
    assert parsed.empty_pages == [2]
    assert [p.number for p in parsed.pages] == [1, 3]


def test_pdf_without_any_text_raises():
    with pytest.raises(NoTextError):
        extract_pages(make_pdf(["", ""]))


def test_garbage_bytes_raise_pdf_error():
    with pytest.raises(PDFError):
        extract_pages(b"isto nao e um pdf")


def test_clean_text_joins_hyphenated_words_and_collapses_whitespace():
    raw = "Uma pala-\nvra quebrada   e linhas\nque continuam.\n\nNovo parágrafo."
    assert clean_text(raw) == "Uma palavra quebrada e linhas que continuam.\n\nNovo parágrafo."
