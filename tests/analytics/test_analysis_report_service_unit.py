from io import BytesIO
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException
from docx import Document

from backend.modules.analytics.builders import DocxAnalysisReportBuilder
from backend.modules.analytics.service import AnalysisReportService


class FakeAnalysisResultRepository:
    """Fake repository for analysis report service unit tests."""

    def __init__(self, result=None) -> None:
        self.result = result
        self.calls = []

    async def get_user_analysis_result(self, db, analysis_id, user_id):
        self.calls.append(
            {
                "db": db,
                "analysis_id": analysis_id,
                "user_id": user_id,
            }
        )
        return self.result


class FakeDocxBuilder:
    """Fake DOCX builder for service unit tests."""

    def __init__(self) -> None:
        self.calls = []
        self.buffer = BytesIO(b"fake-docx")

    def build(self, text: str) -> BytesIO:
        self.calls.append(text)
        self.buffer.seek(0)
        return self.buffer


def read_docx_paragraphs(buffer: BytesIO) -> list[str]:
    """Read paragraph texts from generated DOCX buffer."""

    buffer.seek(0)
    document = Document(buffer)
    return [paragraph.text for paragraph in document.paragraphs]


def test_docx_analysis_report_builder_returns_docx_buffer_with_heading_and_text():
    """Builder creates a valid DOCX file with expected heading and paragraphs."""

    builder = DocxAnalysisReportBuilder()

    buffer = builder.build("Первый вывод\n\nВторой вывод")

    assert isinstance(buffer, BytesIO)
    assert buffer.tell() == 0

    paragraphs = read_docx_paragraphs(buffer)

    assert "Результат анализа договора" in paragraphs
    assert "Первый вывод" in paragraphs
    assert "Второй вывод" in paragraphs


def test_docx_analysis_report_builder_skips_empty_lines():
    """Builder skips blank lines and keeps non-empty lines."""

    builder = DocxAnalysisReportBuilder()

    buffer = builder.build("\n\nРиск 1\n   \nРекомендация 1\n")

    paragraphs = read_docx_paragraphs(buffer)

    assert "Риск 1" in paragraphs
    assert "Рекомендация 1" in paragraphs
    assert "" not in [text for text in paragraphs if text != ""]


@pytest.mark.asyncio
async def test_build_user_docx_report_builds_report_for_existing_result():
    """Service loads user's analysis result and delegates DOCX generation to builder."""

    analysis_id = uuid4()
    user_id = uuid4()
    result = SimpleNamespace(summary="Краткий анализ договора")

    repository = FakeAnalysisResultRepository(result=result)
    builder = FakeDocxBuilder()
    service = AnalysisReportService(
        repository=repository,
        docx_builder=builder,
    )

    buffer = await service.build_user_docx_report(
        db="db-session",
        analysis_id=analysis_id,
        user_id=user_id,
    )

    assert buffer.read() == b"fake-docx"
    assert builder.calls == ["Краткий анализ договора"]
    assert repository.calls == [
        {
            "db": "db-session",
            "analysis_id": analysis_id,
            "user_id": user_id,
        }
    ]


@pytest.mark.asyncio
async def test_build_user_docx_report_raises_404_when_result_not_found():
    """Service raises HTTP 404 when analysis result does not belong to user or does not exist."""

    service = AnalysisReportService(
        repository=FakeAnalysisResultRepository(result=None),
        docx_builder=FakeDocxBuilder(),
    )

    with pytest.raises(HTTPException) as exc_info:
        await service.build_user_docx_report(
            db="db-session",
            analysis_id=uuid4(),
            user_id=uuid4(),
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Результат анализа не найден"


def test_build_docx_delegates_directly_to_builder():
    """Service direct DOCX method delegates text to builder."""

    builder = FakeDocxBuilder()
    service = AnalysisReportService(
        repository=FakeAnalysisResultRepository(),
        docx_builder=builder,
    )

    buffer = service.build_docx("Текст анализа")

    assert buffer.read() == b"fake-docx"
    assert builder.calls == ["Текст анализа"]
