import numpy as np
import pytest

from rag_documentos.vectorstore import NumpyStore, StoredChunk


def _chunk(doc_id: str, index: int, text: str = "texto") -> StoredChunk:
    return StoredChunk(
        id=f"{doc_id}:{index}", doc_id=doc_id, filename=f"{doc_id}.pdf",
        page=index + 1, index=index, text=text,
    )


VECTORS = np.array([[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]], dtype=np.float32)


def _store_with_two_docs(store: NumpyStore) -> NumpyStore:
    store.add([_chunk("a", 0), _chunk("a", 1)], VECTORS[:2])
    store.add([_chunk("b", 0)], VECTORS[2:])
    return store


def test_search_orders_by_cosine_similarity():
    store = _store_with_two_docs(NumpyStore())
    hits = store.search(np.array([1, 0, 0]), k=3)
    assert [h.chunk.id for h in hits] == ["a:0", "b:0", "a:1"]
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_search_can_be_restricted_to_documents():
    store = _store_with_two_docs(NumpyStore())
    hits = store.search(np.array([1, 0, 0]), k=5, doc_ids=["b"])
    assert [h.chunk.doc_id for h in hits] == ["b"]


def test_delete_document():
    store = _store_with_two_docs(NumpyStore())
    assert store.delete_document("a") == 2
    assert store.count() == 1
    assert store.delete_document("a") == 0
    assert [h.chunk.doc_id for h in store.search(np.array([1, 0, 0]), k=5)] == ["b"]


def test_empty_store_returns_nothing():
    assert NumpyStore().search(np.array([1, 0, 0]), k=3) == []


def test_dimension_mismatch_is_rejected():
    store = _store_with_two_docs(NumpyStore())
    with pytest.raises(ValueError):
        store.add([_chunk("c", 0)], np.ones((1, 5), dtype=np.float32))


def test_persistence_round_trip(tmp_path):
    _store_with_two_docs(NumpyStore(tmp_path))
    reloaded = NumpyStore(tmp_path)
    assert reloaded.count() == 3
    assert reloaded.search(np.array([0, 1, 0]), k=1)[0].chunk.id == "a:1"
    reloaded.delete_document("a")
    assert NumpyStore(tmp_path).count() == 1
