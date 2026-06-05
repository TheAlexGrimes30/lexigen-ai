import uuid
from datetime import datetime, timezone
from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile

from backend.db import Chat, Message, MessageRole, User, UserRole
from backend.modules.messages.mapper import MessageMapper
from backend.modules.messages.service import MessagesService


class FakeMessagesRepository:
    """Fake repository for message service unit tests."""

    def __init__(self, messages: list[Message] | None = None) -> None:
        """Initializes fake repository state."""

        self.messages = messages or []
        self.added_entities = []
        self.committed_entities = []

    async def list_by_chat(self, db, chat_id):
        """Returns messages for the requested chat."""

        return [message for message in self.messages if message.chat_id == chat_id]

    async def add_and_flush(self, db, entity):
        """Adds an entity to the fake repository."""

        self.added_entities.append(entity)
        if getattr(entity, "id", None) is None:
            entity.id = uuid.uuid4()
        if getattr(entity, "created_at", None) is None:
            entity.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        return entity

    async def commit_and_refresh_many(self, db, entities):
        """Stores entities committed by the service."""

        self.committed_entities = list(entities)


class FakeDialogHandler:
    """Fake dialog turn handler for service unit tests."""

    def __init__(self, user_message: Message, assistant_message: Message) -> None:
        """Stores messages returned by the fake handler."""

        self.user_message = user_message
        self.assistant_message = assistant_message
        self.calls = []

    async def create_turn(self, db, chat_id, user_id, user_text):
        """Returns predefined dialog messages."""

        self.calls.append((chat_id, user_id, user_text))
        return self.user_message, self.assistant_message


class FakeDocumentHandler:
    """Fake document analysis handler for service unit tests."""

    def __init__(self, user_message: Message, assistant_message: Message) -> None:
        """Stores messages returned by the fake handler."""

        self.user_message = user_message
        self.assistant_message = assistant_message
        self.calls = []

    async def create_turn(self, db, chat_id, user_id, content, file):
        """Returns predefined document analysis messages."""

        self.calls.append((chat_id, user_id, content, file))
        return self.user_message, self.assistant_message


class FakeDocumentPolicy:
    """Fake document analysis policy for service unit tests."""

    def __init__(self) -> None:
        """Initializes policy call tracking."""

        self.calls = []

    async def ensure_allowed(self, db, current_user):
        """Records that policy validation was requested."""

        self.calls.append(current_user.id)


def make_user() -> User:
    """Builds a user model for service unit tests."""

    return User(
        id=uuid.uuid4(),
        name="Test User",
        email="user@test.com",
        password_hash="hash",
        role=UserRole.user,
    )


def make_chat(user: User) -> Chat:
    """Builds a chat model for service unit tests."""

    chat = Chat(
        id=uuid.uuid4(),
        title="Test chat",
        user_id=user.id,
    )
    chat.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return chat


def make_message(
    chat_id: uuid.UUID,
    user_id: uuid.UUID,
    role: MessageRole,
    content: str,
) -> Message:
    """Builds a message model for service unit tests."""

    message = Message(
        id=uuid.uuid4(),
        chat_id=chat_id,
        user_id=user_id,
        role=role,
        content=content,
    )
    message.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return message


def make_service(
    repository: FakeMessagesRepository,
    dialog_handler: FakeDialogHandler | None = None,
    document_handler: FakeDocumentHandler | None = None,
    document_policy: FakeDocumentPolicy | None = None,
) -> MessagesService:
    """Builds a MessagesService with fake dependencies."""

    fallback_user = uuid.uuid4()
    fallback_chat = uuid.uuid4()
    return MessagesService(
        repository=repository,
        mapper=MessageMapper(),
        dialog_handler=dialog_handler
        or FakeDialogHandler(
            make_message(fallback_chat, fallback_user, MessageRole.user, "u"),
            make_message(fallback_chat, fallback_user, MessageRole.assistant, "a"),
        ),
        document_handler=document_handler
        or FakeDocumentHandler(
            make_message(fallback_chat, fallback_user, MessageRole.user, "u"),
            make_message(fallback_chat, fallback_user, MessageRole.assistant, "a"),
        ),
        document_policy=document_policy or FakeDocumentPolicy(),
    )


@pytest.mark.asyncio
async def test_list_messages_response_returns_chat_messages(monkeypatch):
    """Verifies that messages for an owned chat are returned as DTOs."""

    user = make_user()
    chat = make_chat(user)
    message = make_message(chat.id, user.id, MessageRole.user, "Hello")
    repository = FakeMessagesRepository(messages=[message])
    service = make_service(repository)

    async def fake_get_chat(db, chat_id, current_user):
        """Returns an owned chat."""

        return chat

    monkeypatch.setattr(
        "backend.modules.messages.service.chats_service.get_chat",
        fake_get_chat,
    )

    result = await service.list_messages_response(
        db=None,
        chat_id=chat.id,
        current_user=user,
    )

    assert len(result) == 1
    assert result[0].id == message.id
    assert result[0].content == "Hello"


@pytest.mark.asyncio
async def test_list_messages_response_raises_404_for_missing_chat(monkeypatch):
    """Verifies that listing messages for a missing chat returns HTTP 404."""

    async def fake_get_chat(db, chat_id, current_user):
        """Returns no chat."""

        return None

    monkeypatch.setattr(
        "backend.modules.messages.service.chats_service.get_chat",
        fake_get_chat,
    )

    service = make_service(FakeMessagesRepository())

    with pytest.raises(HTTPException) as exc_info:
        await service.list_messages_response(
            db=None,
            chat_id=uuid.uuid4(),
            current_user=make_user(),
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Чат не найден"


@pytest.mark.asyncio
async def test_create_message_turn_response_creates_dialog_turn(monkeypatch):
    """Verifies that text messages create a dialog turn."""

    user = make_user()
    chat = make_chat(user)
    user_message = make_message(chat.id, user.id, MessageRole.user, "Hello")
    assistant_message = make_message(chat.id, user.id, MessageRole.assistant, "Answer")
    dialog_handler = FakeDialogHandler(user_message, assistant_message)
    service = make_service(FakeMessagesRepository(), dialog_handler=dialog_handler)

    async def fake_get_chat(db, chat_id, current_user):
        """Returns an owned chat."""

        return chat

    monkeypatch.setattr(
        "backend.modules.messages.service.chats_service.get_chat",
        fake_get_chat,
    )

    result = await service.create_message_turn_response(
        db=None,
        chat_id=chat.id,
        content="  Hello  ",
        file=None,
        current_user=user,
    )

    assert [item.content for item in result] == ["Hello", "Answer"]
    assert dialog_handler.calls == [(chat.id, user.id, "Hello")]


@pytest.mark.asyncio
async def test_create_message_turn_response_rejects_empty_text(monkeypatch):
    """Verifies that empty text without file returns HTTP 400."""

    user = make_user()
    chat = make_chat(user)
    service = make_service(FakeMessagesRepository())

    async def fake_get_chat(db, chat_id, current_user):
        """Returns an owned chat."""

        return chat

    monkeypatch.setattr(
        "backend.modules.messages.service.chats_service.get_chat",
        fake_get_chat,
    )

    with pytest.raises(HTTPException) as exc_info:
        await service.create_message_turn_response(
            db=None,
            chat_id=chat.id,
            content="   ",
            file=None,
            current_user=user,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Сообщение не может быть пустым"


@pytest.mark.asyncio
async def test_create_message_turn_response_creates_document_turn(monkeypatch):
    """Verifies that file uploads use document policy and handler."""

    user = make_user()
    chat = make_chat(user)
    user_message = make_message(chat.id, user.id, MessageRole.user, "Document")
    assistant_message = make_message(chat.id, user.id, MessageRole.assistant, "Analysis")
    document_handler = FakeDocumentHandler(user_message, assistant_message)
    document_policy = FakeDocumentPolicy()
    service = make_service(
        FakeMessagesRepository(),
        document_handler=document_handler,
        document_policy=document_policy,
    )
    file = UploadFile(filename="contract.pdf", file=BytesIO(b"data"))

    async def fake_get_chat(db, chat_id, current_user):
        """Returns an owned chat."""

        return chat

    monkeypatch.setattr(
        "backend.modules.messages.service.chats_service.get_chat",
        fake_get_chat,
    )

    result = await service.create_message_turn_response(
        db=None,
        chat_id=chat.id,
        content="Analyze",
        file=file,
        current_user=user,
    )

    assert [item.content for item in result] == ["Document", "Analysis"]
    assert document_policy.calls == [user.id]
    assert document_handler.calls == [(chat.id, user.id, "Analyze", file)]


@pytest.mark.asyncio
async def test_create_system_error_response_persists_system_message(monkeypatch):
    """Verifies that system error responses are persisted and mapped."""

    user = make_user()
    chat = make_chat(user)
    repository = FakeMessagesRepository()
    service = make_service(repository)

    async def fake_get_chat(db, chat_id, current_user):
        """Returns an owned chat."""

        return chat

    monkeypatch.setattr(
        "backend.modules.messages.service.chats_service.get_chat",
        fake_get_chat,
    )

    result = await service.create_system_error_response(
        db=None,
        chat_id=chat.id,
        error_text="System failed",
        current_user=user,
    )

    assert result.role == MessageRole.system.value
    assert result.content == "System failed"
    assert len(repository.added_entities) == 1
    assert repository.committed_entities == repository.added_entities
