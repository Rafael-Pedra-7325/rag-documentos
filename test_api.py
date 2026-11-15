import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from rag_documentos.api import create_app  # noqa: E402


@pytest.fixture
def client(service):
    return TestClient(create_app(service))


def _upload(client, pdf: bytes, name: str = "manual.pdf"):
    return client.post("/api/documents", files={"file": (name, pdf, "application/pdf")})


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["documents"] == 0


def test_index_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200 and "RAG Documentos" in response.text


def test_upload_list_query_delete_flow(client, sample_pdf):
    uploaded = _upload(client, sample_pdf)
    assert uploaded.status_code == 201
    doc_id = uploaded.json()["document"]["doc_id"]

    assert [d["doc_id"] for d in client.get("/api/documents").json()] == [doc_id]

    answer = client.post("/api/query", json={"question": "Qual o prazo para solicitar as férias?"})
    assert answer.status_code == 200
    body = answer.json()
    assert body["sources"][0]["page"] == 1
    assert body["sources"][0]["filename"] == "manual.pdf"

    assert client.delete(f"/api/documents/{doc_id}").status_code == 204
    assert client.delete(f"/api/documents/{doc_id}").status_code == 404


def test_rejects_non_pdf_extension(client):
    response = client.post("/api/documents", files={"file": ("notas.txt", b"oi", "text/plain")})
    assert response.status_code == 415


def test_rejects_fake_pdf(client):
    assert _upload(client, b"isto nao e um pdf").status_code == 415


def test_rejects_corrupted_pdf(client):
    assert _upload(client, b"%PDF-1.4 corrompido").status_code == 422


def test_query_validation(client):
    assert client.post("/api/query", json={"question": "a"}).status_code == 422
