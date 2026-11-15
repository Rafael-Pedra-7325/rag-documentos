import numpy as np

from rag_documentos.embeddings import HashingEmbedder


def test_vectors_are_unit_length_and_deterministic():
    embedder = HashingEmbedder()
    first = embedder.embed_documents(["férias de trinta dias", "auxílio internet"])
    second = embedder.embed_documents(["férias de trinta dias", "auxílio internet"])
    assert first.shape == (2, embedder.dim)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-5)
    assert np.array_equal(first, second)


def test_related_text_scores_higher_than_unrelated():
    embedder = HashingEmbedder()
    docs = embedder.embed_documents(
        ["O prazo para solicitar férias é de 45 dias", "O vale-refeição é pago por dia útil"]
    )
    scores = docs @ embedder.embed_query("Qual o prazo das férias?")
    assert scores[0] > scores[1]


def test_accents_and_case_do_not_matter():
    embedder = HashingEmbedder()
    a = embedder.embed_query("FÉRIAS")
    b = embedder.embed_query("ferias")
    assert np.allclose(a, b)


def test_empty_batch():
    assert HashingEmbedder().embed_documents([]).shape == (0, 1024)
