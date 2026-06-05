import uuid
from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile

from backend.db import MessageRole
from backend.modules.messages.document_analysis import DocumentUploadValidator, DocumentAnalysisHandler


class FakeMessagesRepository:
    """Fake repository for document analysis unit tests."""

    def __init__(self) -> None:
        """Initializes repository call tracking."""

        self.added_entities = []
        self.committed_entities = []

    async def add_and_flush(self, db, entity):
        """Stores an entity and assigns an id when necessary."""

        self.added_entities.append(entity)
        if getattr(entity, "id", None) is None:
            entity.id = uuid.uuid4()
        return entity

    async def commit_and_refresh_many(self, db, entities):
        """Stores committed entities."""

        self.committed_entities = list(entities)


class FakeDocumentParser:
    """Fake document parser for upload tests."""

    def __init__(self, text: str) -> None:
        """Stores text returned by the parser."""

        self.text = text

    async def extract_text_from_upload(self, file: UploadFile) -> str:
        """Returns predefined extracted text."""

        return self.text


def make_upload(filename: str = "contract.pdf") -> UploadFile:
    """Builds an UploadFile instance for tests."""

    return UploadFile(
        filename=filename,
        file=BytesIO(b"fake"),
        headers={"content-type": "application/pdf"},
    )


def test_document_upload_validator_allows_docx_and_pdf():
    """Verifies that DOCX and PDF files pass validation."""

    validator = DocumentUploadValidator()

    validator.validate(make_upload("contract.docx"))
    validator.validate(make_upload("contract.pdf"))


def test_document_upload_validator_rejects_unsupported_extension():
    """Verifies that unsupported file extensions return HTTP 400."""

    validator = DocumentUploadValidator()

    with pytest.raises(HTTPException) as exc_info:
        validator.validate(make_upload("contract.txt"))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Можно загрузить только DOCX или PDF"


@pytest.mark.asyncio
async def test_create_turn_creates_document_messages_and_analysis(monkeypatch):
    """Verifies that a valid upload creates document, analysis, and messages."""

    async def fake_analyze_contract(text: str) -> str:
        """Returns a fake contract analysis."""

        return f"Analysis for {text}"

    monkeypatch.setattr(
        "backend.modules.messages.document_analysis.rag_app_service.analyze_contract",
        fake_analyze_contract,
    )

    repository = FakeMessagesRepository()
    handler = DocumentAnalysisHandler(
        repository=repository,
        parser=FakeDocumentParser("contract text"),
        validator=DocumentUploadValidator(),
    )
    chat_id = uuid.uuid4()
    user_id = uuid.uuid4()

    user_message, assistant_message = await handler.create_turn(
        db=None,
        chat_id=chat_id,
        user_id=user_id,
        content="Analyze this",
        file=make_upload("contract.pdf"),
    )

    assert user_message.role == MessageRole.user
    assert user_message.content == "Analyze this\nФайл: contract.pdf"
    assert assistant_message.role == MessageRole.assistant
    assert assistant_message.content == "Analysis for contract text"
    assert assistant_message.chat_document_id is not None
    assert assistant_message.analysis_result_id is not None
    assert len(repository.added_entities) == 4
    assert repository.committed_entities == [user_message, assistant_message]


@pytest.mark.asyncio
async def test_create_turn_uses_default_user_content_for_empty_message(monkeypatch):
    """Verifies that empty content is replaced with a document upload message."""

    async def fake_analyze_contract(text: str) -> str:
        """Returns a fake contract analysis."""

        return "Analysis"

    monkeypatch.setattr(
        "backend.modules.messages.document_analysis.rag_app_service.analyze_contract",
        fake_analyze_contract,
    )

    handler = DocumentAnalysisHandler(
        repository=FakeMessagesRepository(),
        parser=FakeDocumentParser("contract text"),
        validator=DocumentUploadValidator(),
    )

    user_message, _ = await handler.create_turn(
        db=None,
        chat_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        content="   ",
        file=make_upload("contract.pdf"),
    )

    assert user_message.content == "Документ отправлен на анализ: contract.pdf"


@pytest.mark.asyncio
async def test_create_turn_rejects_empty_extracted_text():
    """Verifies that empty extracted text returns HTTP 400."""

    handler = DocumentAnalysisHandler(
        repository=FakeMessagesRepository(),
        parser=FakeDocumentParser(""),
        validator=DocumentUploadValidator(),
    )

    with pytest.raises(HTTPException) as exc_info:
        await handler.create_turn(
            db=None,
            chat_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            content="Analyze",
            file=make_upload("contract.pdf"),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Не удалось извлечь текст из документа"


@pytest.mark.asyncio
async def test_create_turn_creates_system_message_when_analysis_fails(monkeypatch):
    """Verifies that analysis failures are converted into system messages."""

    async def fake_analyze_contract(text: str) -> str:
        """Raises a fake RAG analysis error."""

        raise RuntimeError("analysis failed")

    monkeypatch.setattr(
        "backend.modules.messages.document_analysis.rag_app_service.analyze_contract",
        fake_analyze_contract,
    )

    handler = DocumentAnalysisHandler(
        repository=FakeMessagesRepository(),
        parser=FakeDocumentParser("contract text"),
        validator=DocumentUploadValidator(),
    )

    _, assistant_message = await handler.create_turn(
        db=None,
        chat_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        content="Analyze",
        file=make_upload("contract.pdf"),
    )

    assert assistant_message.role == MessageRole.system
    assert "Система не смогла проанализировать документ" in assistant_message.content
    assert "analysis failed" in assistant_message.content
