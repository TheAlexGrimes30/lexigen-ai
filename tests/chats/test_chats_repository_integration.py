import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.db import Base, Chat, User, UserRole
from backend.db.analysis_result import AnalysisResult
from backend.db.chat_documents import ChatDocument
from backend.modules.chats.repository import ChatsRepository


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


def make_chat(user_id: uuid.UUID, title: str) -> Chat:
    """Builds a chat model for repository integration tests."""

    return Chat(
        id=uuid.uuid4(),
        user_id=user_id,
        title=title,
    )


@pytest.mark.asyncio
async def test_repository_list_by_user_returns_empty_list_for_new_user(db_session):
    """Verifies that list_by_user returns an empty list when no chats exist."""

    repository = ChatsRepository()
    user = make_user()

    db_session.add(user)
    await db_session.commit()

    result = await repository.list_by_user(db_session, user.id)

    assert result == []


@pytest.mark.asyncio
async def test_repository_list_by_user_returns_only_owned_chats(db_session):
    """Verifies that list_by_user excludes chats owned by other users."""

    repository = ChatsRepository()
    user = make_user("user@test.com")
    other_user = make_user("other@test.com")

    db_session.add_all([user, other_user])
    await db_session.flush()

    owned_chat = make_chat(user.id, "Owned chat")
    foreign_chat = make_chat(other_user.id, "Foreign chat")

    db_session.add_all([owned_chat, foreign_chat])
    await db_session.commit()

    result = await repository.list_by_user(db_session, user.id)

    assert [chat.id for chat in result] == [owned_chat.id]


@pytest.mark.asyncio
async def test_repository_get_by_id_and_user_returns_owned_chat(db_session):
    """Verifies that get_by_id_and_user returns a chat matching id and owner."""

    repository = ChatsRepository()
    user = make_user()

    db_session.add(user)
    await db_session.flush()

    chat = make_chat(user.id, "Owned chat")

    db_session.add(chat)
    await db_session.commit()

    result = await repository.get_by_id_and_user(
        db_session,
        chat.id,
        user.id,
    )

    assert result is not None
    assert result.id == chat.id


@pytest.mark.asyncio
async def test_repository_get_by_id_and_user_returns_none_for_foreign_chat(
    db_session,
):
    """Verifies that get_by_id_and_user hides chats owned by another user."""

    repository = ChatsRepository()
    user = make_user("user@test.com")
    other_user = make_user("other@test.com")

    db_session.add_all([user, other_user])
    await db_session.flush()

    foreign_chat = make_chat(other_user.id, "Foreign chat")

    db_session.add(foreign_chat)
    await db_session.commit()

    result = await repository.get_by_id_and_user(
        db_session,
        foreign_chat.id,
        user.id,
    )

    assert result is None


@pytest.mark.asyncio
async def test_repository_add_persists_chat(db_session):
    """Verifies that add persists a chat and refreshes it."""

    repository = ChatsRepository()
    user = make_user()

    db_session.add(user)
    await db_session.flush()

    chat = make_chat(user.id, "Created chat")

    result = await repository.add(db_session, chat)

    assert result.id == chat.id
    assert result.created_at is not None


@pytest.mark.asyncio
async def test_repository_save_persists_updated_title(db_session):
    """Verifies that save persists chat title changes."""

    repository = ChatsRepository()
    user = make_user()

    db_session.add(user)
    await db_session.flush()

    chat = make_chat(user.id, "Old title")

    db_session.add(chat)
    await db_session.commit()

    chat.title = "New title"

    await repository.save(db_session, chat)

    result = await repository.get_by_id_and_user(
        db_session,
        chat.id,
        user.id,
    )

    assert result is not None
    assert result.title == "New title"


@pytest.mark.asyncio
async def test_repository_delete_removes_chat(db_session):
    """Verifies that delete removes a chat from the database."""

    repository = ChatsRepository()
    user = make_user()

    db_session.add(user)
    await db_session.flush()

    chat = make_chat(user.id, "Deleted chat")

    db_session.add(chat)
    await db_session.commit()

    await repository.delete(db_session, chat)

    result = await repository.get_by_id_and_user(
        db_session,
        chat.id,
        user.id,
    )

    assert result is None