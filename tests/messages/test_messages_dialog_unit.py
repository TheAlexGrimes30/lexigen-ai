import uuid

import pytest

from backend.db import MessageRole
from backend.modules.messages.dialog import DialogTurnHandler


class FakeMessagesRepository:
    """Fake repository for dialog turn unit tests."""

    def __init__(self) -> None:
        """Initializes repository call tracking."""

        self.added_entities = []
        self.committed_entities = []

    async def add_and_flush(self, db, entity):
        """Stores an entity passed for insertion."""

        self.added_entities.append(entity)
        if entity.id is None:
            entity.id = uuid.uuid4()
        return entity

    async def commit_and_refresh_many(self, db, entities):
        """Stores entities passed to commit and refresh."""

        self.committed_entities = list(entities)


@pytest.mark.asyncio
async def test_create_turn_creates_user_and_assistant_messages(monkeypatch):
    """Verifies that a successful RAG call creates user and assistant messages."""

    async def fake_ask(text: str) -> str:
        """Returns a fake RAG answer."""

        return f"Answer for {text}"

    monkeypatch.setattr(
        "backend.modules.messages.dialog.rag_app_service.ask",
        fake_ask,
    )

    repository = FakeMessagesRepository()
    handler = DialogTurnHandler(repository=repository)
    chat_id = uuid.uuid4()
    user_id = uuid.uuid4()

    user_message, assistant_message = await handler.create_turn(
        db=None,
        chat_id=chat_id,
        user_id=user_id,
        user_text="Question",
    )

    assert user_message.role == MessageRole.user
    assert user_message.content == "Question"
    assert assistant_message.role == MessageRole.assistant
    assert assistant_message.content == "Answer for Question"
    assert repository.added_entities == [user_message, assistant_message]
    assert repository.committed_entities == [user_message, assistant_message]


@pytest.mark.asyncio
async def test_create_turn_creates_system_message_when_rag_fails(monkeypatch):
    """Verifies that RAG failures are converted into system messages."""

    async def fake_ask(text: str) -> str:
        """Raises a fake RAG error."""

        raise RuntimeError("RAG unavailable")

    monkeypatch.setattr(
        "backend.modules.messages.dialog.rag_app_service.ask",
        fake_ask,
    )

    repository = FakeMessagesRepository()
    handler = DialogTurnHandler(repository=repository)

    _, assistant_message = await handler.create_turn(
        db=None,
        chat_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        user_text="Question",
    )

    assert assistant_message.role == MessageRole.system
    assert "Система не смогла получить RAG-ответ" in assistant_message.content
    assert "RAG unavailable" in assistant_message.content
