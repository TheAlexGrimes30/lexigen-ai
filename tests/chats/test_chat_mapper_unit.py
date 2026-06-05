import uuid
from datetime import datetime, timezone

from backend.db import Chat
from backend.modules.chats.mapper import ChatMapper
from backend.modules.chats.schema import ChatResponse


def make_chat() -> Chat:
    """Builds a chat model for mapper unit tests."""

    chat = Chat(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        title="Mapped chat",
    )
    chat.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return chat


def test_to_response_maps_chat_model_to_dto():
    """Verifies that a chat ORM model is converted to ChatResponse."""

    chat = make_chat()
    mapper = ChatMapper()

    result = mapper.to_response(chat)

    assert isinstance(result, ChatResponse)
    assert result.id == chat.id
    assert result.user_id == chat.user_id
    assert result.title == chat.title
    assert result.created_at == chat.created_at


def test_to_response_list_maps_all_chat_models_to_dtos():
    """Verifies that a chat list is converted to response DTOs."""

    chats = [make_chat(), make_chat()]
    mapper = ChatMapper()

    result = mapper.to_response_list(chats)

    assert len(result) == 2
    assert [item.id for item in result] == [chat.id for chat in chats]


def test_to_response_list_returns_empty_list_for_empty_input():
    """Verifies that an empty chat list maps to an empty response list."""

    mapper = ChatMapper()

    result = mapper.to_response_list([])

    assert result == []
