import uuid
from datetime import datetime, timezone

from backend.db import Message, MessageRole
from backend.modules.messages.mapper import MessageMapper
from backend.modules.messages.schema import MessageResponse


def make_message(content: str | None = "Hello") -> Message:
    """Builds a message model for mapper unit tests."""

    message = Message(
        id=uuid.uuid4(),
        chat_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        role=MessageRole.user,
        content=content,
    )
    message.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)

    return message


def test_to_response_maps_message_model_to_dto():
    """Verifies that a message ORM model is converted to MessageResponse."""

    message = make_message()
    mapper = MessageMapper()

    result = mapper.to_response(message)

    assert isinstance(result, MessageResponse)
    assert result.id == message.id
    assert result.chat_id == message.chat_id
    assert result.user_id == message.user_id
    assert result.role == message.role.value
    assert result.content == message.content
    assert result.created_at == message.created_at


def test_to_response_maps_nullable_content_and_links():
    """Verifies that optional content and relation identifiers are mapped."""

    message = make_message(content=None)
    message.chat_document_id = uuid.uuid4()
    message.analysis_result_id = uuid.uuid4()
    mapper = MessageMapper()

    result = mapper.to_response(message)

    assert result.content is None
    assert result.chat_document_id == message.chat_document_id
    assert result.analysis_result_id == message.analysis_result_id


def test_to_response_list_maps_all_messages_to_dtos():
    """Verifies that a message list is converted to response DTOs."""

    messages = [make_message("First"), make_message("Second")]
    mapper = MessageMapper()

    result = mapper.to_response_list(messages)

    assert len(result) == 2
    assert [item.id for item in result] == [message.id for message in messages]


def test_to_response_list_returns_empty_list_for_empty_input():
    """Verifies that an empty message list maps to an empty response list."""

    mapper = MessageMapper()

    result = mapper.to_response_list([])

    assert result == []
