import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.db import Base, Subscription, SubscriptionPlan, User, UserRole
from backend.modules.subscriptions.repository import SubscriptionsRepository


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
                    Subscription.__table__,
                ],
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


def make_subscription(
    user_id: uuid.UUID,
    plan: SubscriptionPlan = SubscriptionPlan.basic,
    is_active: bool = True,
) -> Subscription:
    """Builds a subscription model for repository integration tests."""

    return Subscription(
        id=uuid.uuid4(),
        user_id=user_id,
        plan_name=plan,
        is_active=is_active,
    )


@pytest.mark.asyncio
async def test_get_active_by_user_id_returns_none_without_subscription(db_session):
    """Verifies that no active subscription returns None."""

    repository = SubscriptionsRepository()
    user = make_user()
    db_session.add(user)
    await db_session.commit()

    result = await repository.get_active_by_user_id(db_session, user.id)

    assert result is None


@pytest.mark.asyncio
async def test_get_active_by_user_id_returns_active_subscription(db_session):
    """Verifies that active subscription is returned for user."""

    repository = SubscriptionsRepository()
    user = make_user()
    db_session.add(user)
    await db_session.flush()

    subscription = make_subscription(user.id, SubscriptionPlan.pro, is_active=True)
    db_session.add(subscription)
    await db_session.commit()

    result = await repository.get_active_by_user_id(db_session, user.id)

    assert result is not None
    assert result.id == subscription.id
    assert result.plan_name == SubscriptionPlan.pro


@pytest.mark.asyncio
async def test_get_active_by_user_id_ignores_inactive_subscription(db_session):
    """Verifies that inactive subscriptions are ignored."""

    repository = SubscriptionsRepository()
    user = make_user()
    db_session.add(user)
    await db_session.flush()

    subscription = make_subscription(user.id, SubscriptionPlan.basic, is_active=False)
    db_session.add(subscription)
    await db_session.commit()

    result = await repository.get_active_by_user_id(db_session, user.id)

    assert result is None


@pytest.mark.asyncio
async def test_delete_by_user_id_removes_user_subscriptions_only(db_session):
    """Verifies that deleting by user id removes only that user's subscriptions."""

    repository = SubscriptionsRepository()
    user = make_user("user@test.com")
    other_user = make_user("other@test.com")
    db_session.add_all([user, other_user])
    await db_session.flush()

    user_subscription = make_subscription(user.id, SubscriptionPlan.basic)
    other_subscription = make_subscription(other_user.id, SubscriptionPlan.pro)
    db_session.add_all([user_subscription, other_subscription])
    await db_session.commit()

    await repository.delete_by_user_id(db_session, user.id)
    await repository.commit(db_session)

    result = (
        await db_session.execute(
            select(Subscription).order_by(Subscription.user_id)
        )
    ).scalars().all()

    assert [item.id for item in result] == [other_subscription.id]


@pytest.mark.asyncio
async def test_add_adds_subscription_to_session(db_session):
    """Verifies that add stores a subscription in the current transaction."""

    repository = SubscriptionsRepository()
    user = make_user()
    db_session.add(user)
    await db_session.flush()

    subscription = make_subscription(user.id, SubscriptionPlan.enterprise)

    result = await repository.add(db_session, subscription)
    await repository.commit(db_session)

    stored = await repository.get_active_by_user_id(db_session, user.id)

    assert result is subscription
    assert stored is not None
    assert stored.plan_name == SubscriptionPlan.enterprise


@pytest.mark.asyncio
async def test_commit_persists_pending_subscription(db_session):
    """Verifies that commit persists pending subscription changes."""

    repository = SubscriptionsRepository()
    user = make_user()
    db_session.add(user)
    await db_session.flush()

    subscription = make_subscription(user.id)
    db_session.add(subscription)

    await repository.commit(db_session)

    stored = await repository.get_active_by_user_id(db_session, user.id)

    assert stored is not None
    assert stored.id == subscription.id
