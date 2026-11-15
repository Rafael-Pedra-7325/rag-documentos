import pytest

from rag_documentos.chunking import chunk_pages, chunk_text, split_units
from rag_documentos.pdf import Page

TEXT = (
    "A primeira frase fala de férias. A segunda frase fala de reembolso. "
    "A terceira frase fala de segurança. A quarta frase fala de benefícios. "
    "A quinta frase fala de licenças. A sexta frase fala de trabalho remoto."
)


def test_chunks_respect_size_limit():
    chunks = chunk_text(TEXT, size=120, overlap=30)
    assert len(chunks) > 1
    assert all(len(c) <= 120 for c in chunks)


def test_consecutive_chunks_overlap():
    chunks = chunk_text(TEXT, size=120, overlap=60)
    for previous, current in zip(chunks, chunks[1:], strict=False):
        last_sentence = previous.split(". ")[-1]
        assert last_sentence in current


def test_no_text_is_lost():
    chunks = chunk_text(TEXT, size=120, overlap=0)
    assert " ".join(chunks) == TEXT


def test_long_sentence_is_hard_split():
    long_sentence = "palavra " * 100
    assert all(len(u) <= 50 for u in split_units(long_sentence.strip(), 50))


def test_giant_token_is_split():
    assert all(len(c) <= 40 for c in chunk_text("x" * 200, size=40, overlap=0))


def test_chunks_keep_page_numbers_and_global_index():
    pages = [Page(1, TEXT), Page(3, TEXT)]
    chunks = chunk_pages(pages, size=120, overlap=30)
    assert {c.page for c in chunks} == {1, 3}
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].page == 1 and chunks[-1].page == 3


def test_tiny_fragments_are_dropped():
    assert chunk_pages([Page(1, "12")], size=120, overlap=0) == []


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_parameters(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("texto qualquer", size=size, overlap=overlap)
