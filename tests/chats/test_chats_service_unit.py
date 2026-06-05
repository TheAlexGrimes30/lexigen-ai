import uuid
from datetime import timezone, datetime

import pytest

from backend.db import Chat, UserRole, User
from backend.modules.chats.mapper import ChatMapper
from backend.modules.chats.schema import ChatUpdateRequest, ChatCreateRequest
from backend.modules.chats.service import ChatsService


class FakeChatsRepository:
    """Fake chats repository for service unit tests."""

    def __init__(self, chats: list[Chat] | None = None) -> None:
        """Initializes the fake repository with optional chat records."""

        self.chats = chats or []
        self.list_by_user_called_with: uuid.UUID | None = None
        self.get_by_id_and_user_called_with: tuple[uuid.UUID, uuid.UUID] | None = None
        self.added_chat: Chat | None = None
        self.saved_chat: Chat | None = None
        self.deleted_chat: Chat | None = None

    async def list_by_user(self, db, user_id: uuid.UUID) -> list[Chat]:
        """Returns all chats that belong to the requested user."""

        self.list_by_user_called_with = user_id
        return [chat for chat in self.chats if chat.user_id == user_id]

    async def get_by_id_and_user(self, db, chat_id: uuid.UUID, user_id: uuid.UUID) -> Chat | None:
        """Returns a matching chat by id and owner."""

        self.get_by_id_and_user_called_with = (chat_id, user_id)
        for chat in self.chats:
            if chat.id == chat_id and chat.user_id == user_id:
                return chat
        return None

    async def add(self, db, chat: Chat) -> Chat:
        """Stores a chat in memory and returns it."""

        if chat.id is None:
            chat.id = uuid.uuid4()
        chat.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.chats.append(chat)
        self.added_chat = chat
        return chat

    async def save(self, db, chat: Chat) -> Chat:
        """Stores the last saved chat and returns it."""

        self.saved_chat = chat
        return chat

    async def delete(self, db, chat: Chat) -> None:
        """Removes a chat from the in-memory list."""

        self.deleted_chat = chat
        self.chats = [existing for existing in self.chats if existing.id != chat.id]


def make_user(user_id: uuid.UUID | None = None) -> User:
    """Builds a user model for chats service tests."""

    user = User(
        id=user_id or uuid.uuid4(),
        name="Test User",
        email="user@test.com",
        password_hash="hash",
        role=UserRole.user,
    )
    user.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return user


def make_chat(user_id: uuid.UUID, title: str = "Test chat") -> Chat:
    """Builds a chat model for chats service tests."""

    chat = Chat(
        id=uuid.uuid4(),
        user_id=user_id,
        title=title,
    )
    chat.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return chat


def build_service(repository: FakeChatsRepository) -> ChatsService:
    """Builds a chats service with fake dependencies."""

    return ChatsService(
        repository=repository,
        mapper=ChatMapper(),
    )


@pytest.mark.asyncio
async def test_list_chats_returns_only_current_user_chats():
    """Verifies that the service lists only chats owned by the current user."""

    current_user = make_user()
    other_user = make_user()
    owned_chat = make_chat(current_user.id, "Owned chat")
    foreign_chat = make_chat(other_user.id, "Foreign chat")
    repository = FakeChatsRepository([owned_chat, foreign_chat])
    service = build_service(repository)

    result = await service.list_chats(db=None, current_user=current_user)

    assert result == [owned_chat]
    assert repository.list_by_user_called_with == current_user.id


@pytest.mark.asyncio
async def test_create_chat_assigns_current_user_as_owner():
    """Verifies that a new chat is created for the current user."""

    current_user = make_user()
    repository = FakeChatsRepository()
    service = build_service(repository)

    result = await service.create_chat(
        db=None,
        title="New chat",
        current_user=current_user,
    )

    assert result.title == "New chat"
    assert result.user_id == current_user.id
    assert repository.added_chat is result


@pytest.mark.asyncio
async def test_get_chat_returns_owned_chat():
    """Verifies that an owned chat can be loaded by id."""

    current_user = make_user()
    chat = make_chat(current_user.id)
    repository = FakeChatsRepository([chat])
    service = build_service(repository)

    result = await service.get_chat(
        db=None,
        chat_id=chat.id,
        current_user=current_user,
    )

    assert result == chat
    assert repository.get_by_id_and_user_called_with == (chat.id, current_user.id)


@pytest.mark.asyncio
async def test_get_chat_returns_none_for_foreign_chat():
    """Verifies that a chat owned by another user is not returned."""

    current_user = make_user()
    foreign_user = make_user()
    foreign_chat = make_chat(foreign_user.id)
    repository = FakeChatsRepository([foreign_chat])
    service = build_service(repository)

    result = await service.get_chat(
        db=None,
        chat_id=foreign_chat.id,
        current_user=current_user,
    )

    assert result is None


@pytest.mark.asyncio
async def test_update_chat_strips_title_and_saves_chat():
    """Verifies that updating a chat trims the title and saves the model."""

    current_user = make_user()
    chat = make_chat(current_user.id, "Old title")
    repository = FakeChatsRepository([chat])
    service = build_service(repository)

    result = await service.update_chat(
        db=None,
        chat_id=chat.id,
        title="  Updated title  ",
        current_user=current_user,
    )

    assert result == chat
    assert chat.title == "Updated title"
    assert repository.saved_chat == chat


@pytest.mark.asyncio
async def test_update_chat_returns_none_when_chat_does_not_exist():
    """Verifies that updating a missing chat returns None."""

    current_user = make_user()
    repository = FakeChatsRepository()
    service = build_service(repository)

    result = await service.update_chat(
        db=None,
        chat_id=uuid.uuid4(),
        title="Updated title",
        current_user=current_user,
    )

    assert result is None
    assert repository.saved_chat is None


@pytest.mark.asyncio
async def test_delete_chat_deletes_existing_owned_chat():
    """Verifies that deleting an owned chat returns True."""

    current_user = make_user()
    chat = make_chat(current_user.id)
    repository = FakeChatsRepository([chat])
    service = build_service(repository)

    result = await service.delete_chat(
        db=None,
        chat_id=chat.id,
        current_user=current_user,
    )

    assert result is True
    assert repository.deleted_chat == chat
    assert repository.chats == []


@pytest.mark.asyncio
async def test_delete_chat_returns_false_when_chat_does_not_exist():
    """Verifies that deleting a missing chat returns False."""

    current_user = make_user()
    repository = FakeChatsRepository()
    service = build_service(repository)

    result = await service.delete_chat(
        db=None,
        chat_id=uuid.uuid4(),
        current_user=current_user,
    )

    assert result is False
    assert repository.deleted_chat is None


@pytest.mark.asyncio
async def test_list_chats_response_returns_dto_list():
    """Verifies that list_chats_response maps chats to response DTOs."""

    current_user = make_user()
    chat = make_chat(current_user.id)
    repository = FakeChatsRepository([chat])
    service = build_service(repository)

    result = await service.list_chats_response(db=None, current_user=current_user)

    assert len(result) == 1
    assert result[0].id == chat.id
    assert result[0].title == chat.title


@pytest.mark.asyncio
async def test_create_chat_response_returns_dto():
    """Verifies that create_chat_response returns a chat response DTO."""

    current_user = make_user()
    repository = FakeChatsRepository()
    service = build_service(repository)
    payload = ChatCreateRequest(title="Created chat")

    result = await service.create_chat_response(
        db=None,
        payload=payload,
        current_user=current_user,
    )

    assert result.title == "Created chat"
    assert result.user_id == current_user.id


@pytest.mark.asyncio
async def test_update_chat_response_returns_dto_for_existing_chat():
    """Verifies that update_chat_response returns a DTO for existing chats."""

    current_user = make_user()
    chat = make_chat(current_user.id, "Old title")
    repository = FakeChatsRepository([chat])
    service = build_service(repository)
    payload = ChatUpdateRequest(title="New title")

    result = await service.update_chat_response(
        db=None,
        chat_id=chat.id,
        payload=payload,
        current_user=current_user,
    )

    assert result is not None
    assert result.title == "New title"


@pytest.mark.asyncio
async def test_update_chat_response_returns_none_for_missing_chat():
    """Verifies that update_chat_response returns None for missing chats."""

    current_user = make_user()
    repository = FakeChatsRepository()
    service = build_service(repository)
    payload = ChatUpdateRequest(title="New title")

    result = await service.update_chat_response(
        db=None,
        chat_id=uuid.uuid4(),
        payload=payload,
        current_user=current_user,
    )

    assert result is None


@pytest.mark.asyncio
async def test_delete_chat_response_returns_deleted_status():
    """Verifies that delete_chat_response returns deleted status."""

    current_user = make_user()
    chat = make_chat(current_user.id)
    repository = FakeChatsRepository([chat])
    service = build_service(repository)

    result = await service.delete_chat_response(
        db=None,
        chat_id=chat.id,
        current_user=current_user,
    )

    assert result == {"status": "deleted"}


@pytest.mark.asyncio
async def test_delete_chat_response_returns_not_found_status():
    """Verifies that delete_chat_response returns not_found status for missing chats."""

    current_user = make_user()
    repository = FakeChatsRepository()
    service = build_service(repository)

    result = await service.delete_chat_response(
        db=None,
        chat_id=uuid.uuid4(),
        current_user=current_user,
    )

    assert result == {"status": "not_found"}