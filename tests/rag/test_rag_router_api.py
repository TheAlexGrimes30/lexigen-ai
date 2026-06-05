from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from backend.modules.rag.router import router


class FakeRAGApplicationService:
    """Fake RAG application service for router API tests."""

    def __init__(self, *, answer: str = "fake answer", health_payload: dict | None = None) -> None:
        """Initializes fake response data and call tracking."""
        self.answer = answer
        self.health_payload = health_payload or {
            "status": "ready",
            "ready": True,
            "error": None,
        }
        self.received_queries: list[str] = []

    def health(self) -> dict:
        """Returns a predefined health payload."""
        return self.health_payload

    async def ask(self, query: str) -> str:
        """Stores the query and returns a predefined answer."""
        self.received_queries.append(query)
        return self.answer


class BrokenRAGApplicationService(FakeRAGApplicationService):
    """Fake RAG service that raises during query processing."""

    async def ask(self, query: str) -> str:
        """Raises a deterministic runtime error."""
        raise RuntimeError("rag unavailable")


def build_test_app(monkeypatch, service=None) -> FastAPI:
    """Builds a test FastAPI app with a patched RAG service."""

    app = FastAPI()
    app.include_router(router)

    monkeypatch.setattr(
        "backend.modules.rag.router.rag_app_service",
        service or FakeRAGApplicationService(),
    )

    return app


@pytest.mark.asyncio
async def test_rag_health_returns_service_health(monkeypatch):
    """Verifies that GET /api/rag/health returns service health data."""

    service = FakeRAGApplicationService(
        health_payload={
            "status": "loading",
            "ready": False,
            "error": None,
        }
    )
    app = build_test_app(monkeypatch, service=service)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/rag/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "loading",
        "ready": False,
        "error": None,
    }


@pytest.mark.asyncio
async def test_rag_query_returns_answer(monkeypatch):
    """Verifies that POST /api/rag/query returns a RAG answer."""

    service = FakeRAGApplicationService(answer="RAG answer")
    app = build_test_app(monkeypatch, service=service)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/rag/query",
            json={"query": "What is credit law?"},
        )

    assert response.status_code == 200
    assert response.json() == {"answer": "RAG answer"}
    assert service.received_queries == ["What is credit law?"]


@pytest.mark.asyncio
async def test_rag_query_rejects_empty_query(monkeypatch):
    """Verifies that an empty query fails request validation."""

    app = build_test_app(monkeypatch)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/rag/query",
            json={"query": ""},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_rag_query_rejects_missing_query(monkeypatch):
    """Verifies that a missing query field fails request validation."""

    app = build_test_app(monkeypatch)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/rag/query",
            json={},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_rag_query_propagates_service_error(monkeypatch):
    """Verifies that router returns HTTP 500 when the RAG service fails."""

    app = build_test_app(monkeypatch, service=BrokenRAGApplicationService())

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/rag/query",
            json={"query": "question"},
        )

    assert response.status_code == 500
