from pathlib import Path
from uuid import UUID

from fastapi import HTTPException, UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import AnalysisResult, ChatDocument
from backend.db.enums import MessageRole, UserRole
from backend.db.messages import Message
from backend.db.users import User
from backend.modules.chats.service import chats_service
from backend.modules.messages.schema import MessageResponse
from backend.modules.rag.service import rag_app_service
from backend.modules.subscriptions.service import subscriptions_service
from backend.parsers.document_parser import ContractDocumentParser


class MessagesService:
    """Сервис сообщений чата с поддержкой анализа DOCX/PDF без сохранения файла на диск."""

    def __init__(self):
        self.document_parser = ContractDocumentParser()

    async def list_messages(
        self,
        db: AsyncSession,
        chat_id: UUID,
    ) -> list[Message]:
        stmt = (
            select(Message)
            .where(Message.chat_id == chat_id)
            .order_by(Message.created_at.asc())
        )

        result = await db.execute(stmt)

        return list(result.scalars().all())

    async def create_message_turn_response(
        self,
        db: AsyncSession,
        chat_id: UUID,
        content: str,
        file: UploadFile | None,
        current_user: User,
    ) -> list[MessageResponse]:
        chat = await chats_service.get_chat(
            db,
            chat_id,
            current_user,
        )

        if not chat:
            raise HTTPException(
                status_code=404,
                detail="Чат не найден",
            )

        if file is not None:
            await self._ensure_document_analysis_allowed(
                db=db,
                current_user=current_user,
            )

            user_message, assistant_message = await self.create_document_analysis_turn(
                db=db,
                chat_id=chat_id,
                user_id=chat.user_id,
                content=content,
                file=file,
            )
        else:
            user_text = (content or "").strip()

            if not user_text:
                raise HTTPException(
                    status_code=400,
                    detail="Сообщение не может быть пустым",
                )

            user_message, assistant_message = await self.create_dialog_turn(
                db=db,
                chat_id=chat_id,
                user_id=chat.user_id,
                user_text=user_text,
            )

        return [
            MessageResponse.model_validate(user_message),
            MessageResponse.model_validate(assistant_message),
        ]

    async def _ensure_document_analysis_allowed(
        self,
        db: AsyncSession,
        current_user: User,
    ) -> None:
        """
        Правила:
        - администратор может анализировать без подписки без ограничений;
        - пользователь с активной подпиской может анализировать без ограничений;
        - пользователь без подписки может загрузить документ только один раз.
        """

        if current_user.role == UserRole.admin:
            return

        has_subscription = await subscriptions_service.user_has_paid_subscription(
            db,
            current_user.id,
        )

        if has_subscription:
            return

        used_uploads = int(
            await db.scalar(
                select(func.count(ChatDocument.id)).where(
                    ChatDocument.uploaded_by == current_user.id,
                )
            )
            or 0
        )

        if used_uploads >= 1:
            raise HTTPException(
                status_code=403,
                detail=(
                    "Без подписки доступна только одна загрузка документа. "
                    "Оформите Basic, Pro или Enterprise в личном кабинете."
                ),
            )

    async def create_dialog_turn(
        self,
        db: AsyncSession,
        chat_id: UUID,
        user_id: UUID,
        user_text: str,
    ) -> tuple[Message, Message]:
        user_message = Message(
            chat_id=chat_id,
            user_id=user_id,
            role=MessageRole.user,
            content=user_text,
        )

        db.add(user_message)
        await db.flush()

        try:
            assistant_text = await rag_app_service.ask(user_text)
            assistant_role = MessageRole.assistant
        except Exception as exc:
            assistant_text = f"Система не смогла получить RAG-ответ: {exc}"
            assistant_role = MessageRole.system

        assistant_message = Message(
            chat_id=chat_id,
            user_id=user_id,
            role=assistant_role,
            content=assistant_text,
        )

        db.add(assistant_message)

        await db.commit()
        await db.refresh(user_message)
        await db.refresh(assistant_message)

        return user_message, assistant_message

    async def create_document_analysis_turn(
        self,
        db: AsyncSession,
        chat_id: UUID,
        user_id: UUID,
        content: str,
        file: UploadFile,
    ) -> tuple[Message, Message]:
        self._validate_document(file)

        original_filename = file.filename or "document"

        extracted_text = await self.document_parser.extract_text_from_upload(file)

        if not extracted_text:
            raise HTTPException(
                status_code=400,
                detail="Не удалось извлечь текст из документа",
            )

        chat_document = ChatDocument(
            chat_id=chat_id,
            uploaded_by=user_id,
            filename=original_filename,
            original_filename=original_filename,
            mime_type=file.content_type,
            extracted_text=extracted_text,
        )

        db.add(chat_document)
        await db.flush()

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

        db.add(user_message)
        await db.flush()

        try:
            analysis_text = await rag_app_service.analyze_contract(extracted_text)
            assistant_role = MessageRole.assistant
        except Exception as exc:
            analysis_text = f"Система не смогла проанализировать документ: {exc}"
            assistant_role = MessageRole.system

        chat_document.summary = analysis_text

        analysis_result = AnalysisResult(
            chat_id=chat_id,
            document_id=chat_document.id,
            generated_by_user_id=user_id,
            summary=analysis_text,
            risks_found=None,
            recommendations=None,
        )

        db.add(analysis_result)
        await db.flush()

        assistant_message = Message(
            chat_id=chat_id,
            user_id=user_id,
            role=assistant_role,
            content=analysis_text,
            chat_document_id=chat_document.id,
            analysis_result_id=analysis_result.id,
        )

        db.add(assistant_message)

        await db.commit()
        await db.refresh(user_message)
        await db.refresh(assistant_message)

        return user_message, assistant_message

    async def list_messages_response(
        self,
        db: AsyncSession,
        chat_id: UUID,
        current_user: User,
    ) -> list[MessageResponse]:
        chat = await chats_service.get_chat(
            db,
            chat_id,
            current_user,
        )

        if not chat:
            raise HTTPException(
                status_code=404,
                detail="Чат не найден",
            )

        messages = await self.list_messages(
            db,
            chat_id,
        )

        return [
            MessageResponse.model_validate(message)
            for message in messages
        ]

    async def create_system_error_response(
        self,
        db: AsyncSession,
        chat_id: UUID,
        error_text: str,
        current_user: User,
    ) -> MessageResponse:
        chat = await chats_service.get_chat(
            db,
            chat_id,
            current_user,
        )

        if not chat:
            raise HTTPException(
                status_code=404,
                detail="Чат не найден",
            )

        system_message = Message(
            chat_id=chat_id,
            user_id=chat.user_id,
            role=MessageRole.system,
            content=error_text,
        )

        db.add(system_message)

        await db.commit()
        await db.refresh(system_message)

        return MessageResponse.model_validate(system_message)

    def _validate_document(self, file: UploadFile) -> None:
        suffix = Path(file.filename or "").suffix.lower()

        if suffix not in {".docx", ".pdf"}:
            raise HTTPException(
                status_code=400,
                detail="Можно загрузить только DOCX или PDF",
            )


messages_service = MessagesService()
