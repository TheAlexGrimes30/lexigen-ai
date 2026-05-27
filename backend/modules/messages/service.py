from uuid import UUID

from fastapi import HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.messages import Message
from backend.db.users import User
from backend.modules.chats.service import chats_service
from backend.modules.messages.dialog import DialogTurnHandler
from backend.modules.messages.document_analysis import (
    DocumentAnalysisHandler,
    DocumentUploadValidator,
)
from backend.modules.messages.interfaces import (
    BaseDocumentAnalysisPolicy,
    BaseMessagesRepository,
    BaseMessagesService,
)
from backend.modules.messages.mapper import MessageMapper
from backend.modules.messages.policies import DocumentAnalysisPolicy
from backend.modules.messages.repository import MessagesRepository
from backend.modules.messages.schema import MessageResponse
from backend.parsers.document_parser import ContractDocumentParser


class MessagesService(BaseMessagesService):
    """Сервис сообщений чата с поддержкой анализа DOCX/PDF без сохранения файла на диск."""

    def __init__(
        self,
        repository: BaseMessagesRepository,
        mapper: MessageMapper,
        dialog_handler: DialogTurnHandler,
        document_handler: DocumentAnalysisHandler,
        document_policy: BaseDocumentAnalysisPolicy,
    ) -> None:
        """Инициализирует сервис сообщений."""
        self.repository = repository
        self.mapper = mapper
        self.dialog_handler = dialog_handler
        self.document_handler = document_handler
        self.document_policy = document_policy

    async def list_messages(
        self,
        db: AsyncSession,
        chat_id: UUID,
    ) -> list[Message]:
        """Возвращает список сообщений чата."""
        return await self.repository.list_by_chat(
            db=db,
            chat_id=chat_id,
        )

    async def create_message_turn_response(
        self,
        db: AsyncSession,
        chat_id: UUID,
        content: str,
        file: UploadFile | None,
        current_user: User,
    ) -> list[MessageResponse]:
        """Создаёт пару сообщений пользователя и системы или ассистента."""
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
            await self.document_policy.ensure_allowed(
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

        return self.mapper.to_response_list(
            [
                user_message,
                assistant_message,
            ]
        )

    async def create_dialog_turn(
        self,
        db: AsyncSession,
        chat_id: UUID,
        user_id: UUID,
        user_text: str,
    ) -> tuple[Message, Message]:
        """Создаёт обычный диалоговый turn без файла."""
        return await self.dialog_handler.create_turn(
            db=db,
            chat_id=chat_id,
            user_id=user_id,
            user_text=user_text,
        )

    async def create_document_analysis_turn(
        self,
        db: AsyncSession,
        chat_id: UUID,
        user_id: UUID,
        content: str,
        file: UploadFile,
    ) -> tuple[Message, Message]:
        """Создаёт turn анализа документа."""
        return await self.document_handler.create_turn(
            db=db,
            chat_id=chat_id,
            user_id=user_id,
            content=content,
            file=file,
        )

    async def list_messages_response(
        self,
        db: AsyncSession,
        chat_id: UUID,
        current_user: User,
    ) -> list[MessageResponse]:
        """Возвращает DTO списка сообщений чата текущего пользователя."""
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
            db=db,
            chat_id=chat_id,
        )

        return self.mapper.to_response_list(messages)

    async def create_system_error_response(
        self,
        db: AsyncSession,
        chat_id: UUID,
        error_text: str,
        current_user: User,
    ) -> MessageResponse:
        """Создаёт системное сообщение об ошибке."""
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

        from backend.db.enums import MessageRole
        from backend.db.messages import Message

        system_message = Message(
            chat_id=chat_id,
            user_id=chat.user_id,
            role=MessageRole.system,
            content=error_text,
        )

        await self.repository.add_and_flush(db, system_message)

        await self.repository.commit_and_refresh_many(
            db=db,
            entities=[system_message],
        )

        return self.mapper.to_response(system_message)

    async def _ensure_document_analysis_allowed(
        self,
        db: AsyncSession,
        current_user: User,
    ) -> None:
        """Проверяет право пользователя на анализ документа."""
        await self.document_policy.ensure_allowed(
            db=db,
            current_user=current_user,
        )

    def _validate_document(
        self,
        file: UploadFile,
    ) -> None:
        """Проверяет расширение документа."""
        self.document_handler.validator.validate(file)


_messages_repository = MessagesRepository()

messages_service = MessagesService(
    repository=_messages_repository,
    mapper=MessageMapper(),
    dialog_handler=DialogTurnHandler(
        repository=_messages_repository,
    ),
    document_handler=DocumentAnalysisHandler(
        repository=_messages_repository,
        parser=ContractDocumentParser(),
        validator=DocumentUploadValidator(),
    ),
    document_policy=DocumentAnalysisPolicy(
        repository=_messages_repository,
    ),
)
