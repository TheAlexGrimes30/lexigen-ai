import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.db import Base, Chat, ChatDocument, Message, MessageRole, User, UserRole
from backend.db.analysis_result import AnalysisResult
from backend.modules.messages.repository import MessagesRepository


@pytest.fixture
async def db_session():
    """Creates an isolated in-memory SQLite database for repository tests."""

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        future=True,
    )

    async with engine.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: Base.metadata.create_all(
                bind=sync_conn,
                tables=[
                    User.__table__,
                    Chat.__table__,
                    ChatDocument.__table__,
                    AnalysisResult.__table__,
                ],
            )
        )

        await conn.execute(
            text(
                """
                CREATE TABLE messages (
                    id CHAR(32) PRIMARY KEY,
                    user_id CHAR(32) NOT NULL,
                    chat_id CHAR(32) NOT NULL,
                    role VARCHAR(9) NOT NULL,
                    content TEXT,
                    chat_document_id CHAR(32),
                    analysis_result_id CHAR(32),
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL
                )
                """
            )
        )

    session_factory = async_sessionmaker(
        bind=engine,
        expire_on_commit=False,
    )

    async with session_factory() as session:
        yield session

    await engine.dispose()


def make_user(email: str = "user@test.com") -> User:
    """Builds a user model for repository integration tests."""

    return User(
        id=uuid.uuid4(),
        name=email.split("@")[0],
        email=email,
        password_hash="hash",
        role=UserRole.user,
    )


def make_chat(user_id: uuid.UUID, title: str = "Test chat") -> Chat:
    """Builds a chat model for repository integration tests."""

    return Chat(
        id=uuid.uuid4(),
        user_id=user_id,
        title=title,
    )


def make_message(
    user_id: uuid.UUID,
    chat_id: uuid.UUID,
    content: str,
    created_at: datetime | None = None,
) -> Message:
    """Builds a message model for repository integration tests."""

    message = Message(
        id=uuid.uuid4(),
        user_id=user_id,
        chat_id=chat_id,
        role=MessageRole.user,
        content=content,
    )
    if created_at is not None:
        message.created_at = created_at
    return message


def make_document(user_id: uuid.UUID, chat_id: uuid.UUID, filename: str) -> ChatDocument:
    """Builds a chat document model for repository integration tests."""

    return ChatDocument(
        id=uuid.uuid4(),
        chat_id=chat_id,
        uploaded_by=user_id,
        filename=filename,
        original_filename=filename,
        mime_type="application/pdf",
        extracted_text="text",
    )


@pytest.mark.asyncio
async def test_repository_list_by_chat_returns_empty_list(db_session):
    """Verifies that list_by_chat returns an empty list for a new chat."""

    repository = MessagesRepository()
    user = make_user()
    chat = make_chat(user.id)

    db_session.add_all([user, chat])
    await db_session.commit()

    result = await repository.list_by_chat(db_session, chat.id)

    assert result == []


@pytest.mark.asyncio
async def test_repository_list_by_chat_returns_messages_in_created_order(db_session):
    """Verifies that messages are listed by ascending creation time."""

    repository = MessagesRepository()
    user = make_user()
    chat = make_chat(user.id)
    other_chat = make_chat(user.id, "Other chat")

    db_session.add_all([user, chat, other_chat])
    await db_session.flush()

    second = make_message(
        user.id,
        chat.id,
        "Second",
        datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    first = make_message(
        user.id,
        chat.id,
        "First",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    foreign = make_message(
        user.id,
        other_chat.id,
        "Foreign",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    db_session.add_all([second, first, foreign])
    await db_session.commit()

    result = await repository.list_by_chat(db_session, chat.id)

    assert [message.content for message in result] == ["First", "Second"]


@pytest.mark.asyncio
async def test_repository_count_user_documents_counts_only_user_documents(db_session):
    """Verifies that document counting is scoped to the user."""

    repository = MessagesRepository()
    user = make_user("user@test.com")
    other_user = make_user("other@test.com")
    chat = make_chat(user.id)
    other_chat = make_chat(other_user.id)

    db_session.add_all([user, other_user, chat, other_chat])
    await db_session.flush()

    db_session.add_all(
        [
            make_document(user.id, chat.id, "one.pdf"),
            make_document(user.id, chat.id, "two.pdf"),
            make_document(other_user.id, other_chat.id, "other.pdf"),
        ]
    )
    await db_session.commit()

    result = await repository.count_user_documents(db_session, user.id)

    assert result == 2


@pytest.mark.asyncio
async def test_repository_add_and_flush_assigns_database_defaults(db_session):
    """Verifies that add_and_flush inserts an entity without committing."""

    repository = MessagesRepository()
    user = make_user()
    chat = make_chat(user.id)
    db_session.add_all([user, chat])
    await db_session.flush()

    message = make_message(user.id, chat.id, "Pending")

    result = await repository.add_and_flush(db_session, message)

    assert result is message
    assert message.created_at is not None


@pytest.mark.asyncio
async def test_repository_commit_and_refresh_many_persists_entities(db_session):
    """Verifies that commit_and_refresh_many commits and refreshes entities."""

    repository = MessagesRepository()
    user = make_user()
    chat = make_chat(user.id)
    db_session.add_all([user, chat])
    await db_session.flush()

    message = make_message(user.id, chat.id, "Committed")
    await repository.add_and_flush(db_session, message)

    await repository.commit_and_refresh_many(db_session, [message])

    result = await repository.list_by_chat(db_session, chat.id)

    assert len(result) == 1
    assert result[0].id == message.id
    assert result[0].content == "Committed"
