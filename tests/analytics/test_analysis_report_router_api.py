from io import BytesIO
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient

from backend.db.database import get_db
from backend.modules.analytics.router import router
from backend.modules.auth.dependencies import get_current_user


DOCX_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument."
    "wordprocessingml.document"
)


class FakeAnalysisReportService:
    """Fake report service for analytics API tests."""

    def __init__(self, buffer: BytesIO | None = None) -> None:
        self.buffer = buffer or BytesIO(b"fake-docx-content")
        self.calls = []

    async def build_user_docx_report(self, db, analysis_id, user_id):
        self.calls.append(
            {
                "db": db,
                "analysis_id": analysis_id,
                "user_id": user_id,
            }
        )
        self.buffer.seek(0)
        return self.buffer


def build_test_app(monkeypatch, service=None, user_id=None) -> FastAPI:
    """Build isolated FastAPI app with overridden dependencies."""

    app = FastAPI()
    app.include_router(router)

    fake_db = object()
    fake_user = SimpleNamespace(id=user_id or uuid4())

    async def fake_get_db():
        yield fake_db

    async def fake_get_current_user():
        return fake_user

    app.dependency_overrides[get_db] = fake_get_db
    app.dependency_overrides[get_current_user] = fake_get_current_user

    fake_service = service or FakeAnalysisReportService()

    monkeypatch.setattr(
        "backend.modules.analytics.router.analysis_report_service",
        fake_service,
    )

    return app


@pytest.mark.asyncio
async def test_download_analysis_result_returns_docx_response(monkeypatch):
    """Endpoint returns DOCX stream with correct headers."""

    analysis_id = uuid4()
    user_id = uuid4()
    service = FakeAnalysisReportService(buffer=BytesIO(b"docx-bytes"))
    app = build_test_app(
        monkeypatch=monkeypatch,
        service=service,
        user_id=user_id,
    )

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get(
            f"/api/analysis-results/{analysis_id}/download"
        )

    assert response.status_code == 200
    assert response.content == b"docx-bytes"
    assert response.headers["content-type"] == DOCX_MEDIA_TYPE
    assert response.headers["content-disposition"] == (
        "attachment; filename=analysis_result.docx"
    )
    assert service.calls == [
        {
            "db": service.calls[0]["db"],
            "analysis_id": analysis_id,
            "user_id": user_id,
        }
    ]


@pytest.mark.asyncio
async def test_download_analysis_result_passes_db_analysis_id_and_current_user(monkeypatch):
    """Endpoint passes dependency values to analysis report service."""

    analysis_id = uuid4()
    user_id = uuid4()
    service = FakeAnalysisReportService()
    app = build_test_app(
        monkeypatch=monkeypatch,
        service=service,
        user_id=user_id,
    )

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get(
            f"/api/analysis-results/{analysis_id}/download"
        )

    assert response.status_code == 200
    assert len(service.calls) == 1
    assert service.calls[0]["analysis_id"] == analysis_id
    assert service.calls[0]["user_id"] == user_id
    assert service.calls[0]["db"] is not None


@pytest.mark.asyncio
async def test_download_analysis_result_returns_404_from_service(monkeypatch):
    """Endpoint propagates service 404 for missing or foreign analysis result."""

    class NotFoundService:
        async def build_user_docx_report(self, db, analysis_id, user_id):
            raise HTTPException(
                status_code=404,
                detail="Результат анализа не найден",
            )

    app = build_test_app(
        monkeypatch=monkeypatch,
        service=NotFoundService(),
    )

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get(
            f"/api/analysis-results/{uuid4()}/download"
        )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Результат анализа не найден",
    }


@pytest.mark.asyncio
async def test_download_analysis_result_requires_auth_when_dependency_fails(monkeypatch):
    """Endpoint returns auth error when current user dependency rejects request."""

    app = FastAPI()
    app.include_router(router)

    async def fake_get_db():
        yield object()

    async def unauthorized_user():
        raise HTTPException(
            status_code=401,
            detail="Not authenticated",
        )

    app.dependency_overrides[get_db] = fake_get_db
    app.dependency_overrides[get_current_user] = unauthorized_user

    monkeypatch.setattr(
        "backend.modules.analytics.router.analysis_report_service",
        FakeAnalysisReportService(),
    )

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get(
            f"/api/analysis-results/{uuid4()}/download"
        )

    assert response.status_code == 401
    assert response.json() == {
        "detail": "Not authenticated",
    }


@pytest.mark.asyncio
async def test_download_analysis_result_rejects_invalid_uuid(monkeypatch):
    """FastAPI validation rejects invalid analysis_id path parameter."""

    app = build_test_app(monkeypatch=monkeypatch)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/api/analysis-results/not-a-uuid/download"
        )

    assert response.status_code == 422
