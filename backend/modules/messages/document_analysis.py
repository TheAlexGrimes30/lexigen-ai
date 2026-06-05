from pathlib import Path
from uuid import UUID

from fastapi import HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.logger_config import get_logger
from backend.db import AnalysisResult, ChatDocument
from backend.db.enums import MessageRole
from backend.db.messages import Message
from backend.modules.messages.interfaces import BaseDocumentParser, BaseMessagesRepository
from backend.modules.rag.service import rag_app_service

logger = get_logger(__name__)


class DocumentUploadValidator:
    """Валидатор загружаемых документов."""

    def validate(
        self,
        file: UploadFile,
    ) -> None:
        """Проверяет расширение загружаемого файла."""

        filename = file.filename or ""
        suffix = Path(filename).suffix.lower()

        logger.info(
            "Document upload validation started: filename=%s, suffix=%s",
            filename,
            suffix,
        )

        if suffix not in {".docx", ".pdf"}:
            logger.warning(
                "Document upload validation failed: unsupported suffix=%s",
                suffix,
            )

            raise HTTPException(
                status_code=400,
                detail="Можно загрузить только DOCX или PDF",
            )

        logger.info(
            "Document upload validation passed: filename=%s",
            filename,
        )


class DocumentAnalysisHandler:
    """Обработчик создания сообщений и сущностей анализа документа."""

    def __init__(
        self,
        repository: BaseMessagesRepository,
        parser: BaseDocumentParser,
        validator: DocumentUploadValidator,
    ) -> None:
        """Инициализирует обработчик анализа документов."""
        self.repository = repository
        self.parser = parser
        self.validator = validator

    async def create_turn(
        self,
        db: AsyncSession,
        chat_id: UUID,
        user_id: UUID,
        content: str,
        file: UploadFile,
    ) -> tuple[Message, Message]:
        """Создаёт пользовательское сообщение, анализ документа и ответ ассистента."""

        logger.info(
            "Document analysis turn creation started: chat_id=%s, user_id=%s, filename=%s",
            chat_id,
            user_id,
            file.filename,
        )

        try:
            self.validator.validate(file)

            original_filename = file.filename or "document"

            extracted_text = await self.parser.extract_text_from_upload(file)

            if not extracted_text:
                logger.warning(
                    "Document text extraction failed: chat_id=%s, user_id=%s, filename=%s",
                    chat_id,
                    user_id,
                    original_filename,
                )

                raise HTTPException(
                    status_code=400,
                    detail="Не удалось извлечь текст из документа",
                )

            logger.info(
                "Document text extracted successfully: chat_id=%s, user_id=%s, text_length=%s",
                chat_id,
                user_id,
                len(extracted_text),
            )

            chat_document = ChatDocument(
                chat_id=chat_id,
                uploaded_by=user_id,
                filename=original_filename,
                original_filename=original_filename,
                mime_type=file.content_type,
                extracted_text=extracted_text,
            )

            await self.repository.add_and_flush(db, chat_document)

            user_message = await self._create_user_message(
                db=db,
                chat_id=chat_id,
                user_id=user_id,
                content=content,
                original_filename=original_filename,
                chat_document=chat_document,
            )

            analysis_text, assistant_role = await self._analyze_document(
                extracted_text
            )

            chat_document.summary = analysis_text

            analysis_result = AnalysisResult(
                chat_id=chat_id,
                document_id=chat_document.id,
                generated_by_user_id=user_id,
                summary=analysis_text,
                risks_found=None,
                recommendations=None,
            )

            await self.repository.add_and_flush(db, analysis_result)

            assistant_message = Message(
                chat_id=chat_id,
                user_id=user_id,
                role=assistant_role,
                content=analysis_text,
                chat_document_id=chat_document.id,
                analysis_result_id=analysis_result.id,
            )

            await self.repository.add_and_flush(db, assistant_message)

            await self.repository.commit_and_refresh_many(
                db=db,
                entities=[
                    user_message,
                    assistant_message,
                ],
            )

            logger.info(
                "Document analysis turn created successfully: chat_id=%s, user_id=%s, assistant_role=%s",
                chat_id,
                user_id,
                assistant_role.value,
            )

            return user_message, assistant_message

        except HTTPException:
            raise

        except Exception:
            logger.exception(
                "Failed to create document analysis turn: chat_id=%s, user_id=%s",
                chat_id,
                user_id,
            )
            raise

    async def _create_user_message(
        self,
        db: AsyncSession,
        chat_id: UUID,
        user_id: UUID,
        content: str,
        original_filename: str,
        chat_document: ChatDocument,
    ) -> Message:
        """Создаёт пользовательское сообщение с информацией о файле."""

        logger.info(
            "Creating document user message: chat_id=%s, user_id=%s, filename=%s",
            chat_id,
            user_id,
            original_filename,
        )

        user_content = (content or "").strip()

        if not user_content:
            user_content = f"Документ отправлен на анализ: {original_filename}"
        else:
            user_content = f"{user_content}\nФайл: {original_filename}"

        user_message = Message(
            chat_id=chat_id,
            user_id=user_id,
            role=MessageRole.user,
            content=user_content,
            chat_document_id=chat_document.id,
        )

        await self.repository.add_and_flush(db, user_message)

        logger.info(
            "Document user message created: chat_id=%s, user_id=%s",
            chat_id,
            user_id,
        )

        return user_message

    async def _analyze_document(
        self,
        extracted_text: str,
    ) -> tuple[str, MessageRole]:
        """Выполняет RAG-анализ документа и возвращает текст с ролью."""

        logger.info(
            "RAG document analysis started: text_length=%s",
            len(extracted_text),
        )

        try:
            analysis_text = await rag_app_service.analyze_contract(
                extracted_text
            )

            logger.info("RAG document analysis completed successfully")

            return analysis_text, MessageRole.assistant

        except Exception as exc:
            logger.exception("RAG document analysis failed")

            return (
                f"Система не смогла проанализировать документ: {exc}",
                MessageRole.system,
            )
