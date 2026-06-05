import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.db import Base, Subscription, SubscriptionPlan, User, UserRole
from backend.modules.subscriptions.mapper import SubscriptionMapper
from backend.modules.subscriptions.plans import SubscriptionPlansProvider
from backend.modules.subscriptions.repository import SubscriptionsRepository
from backend.modules.subscriptions.service import SubscriptionsService


@pytest.fixture
async def db_session():
    """Creates an isolated in-memory SQLite database for service integration tests."""

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


def build_service() -> SubscriptionsService:
    """Builds a real subscriptions service for integration tests."""

    plans_provider = SubscriptionPlansProvider()

    return SubscriptionsService(
        repository=SubscriptionsRepository(),
        plans_provider=plans_provider,
        mapper=SubscriptionMapper(plans_provider),
    )


def make_user(email: str = "user@test.com") -> User:
    """Builds a user model for service integration tests."""

    return User(
        id=uuid.uuid4(),
        name=email.split("@")[0],
        email=email,
        password_hash="hash",
        role=UserRole.user,
    )


@pytest.mark.asyncio
async def test_set_user_subscription_persists_selected_plan(db_session):
    """Verifies that setting subscription persists selected plan."""

    service = build_service()
    user = make_user()
    db_session.add(user)
    await db_session.commit()

    result = await service.set_user_subscription(
        db=db_session,
        user=user,
        plan=SubscriptionPlan.pro,
    )

    stored = await db_session.scalar(
        select(Subscription).where(Subscription.user_id == user.id)
    )

    assert result["plan"] == SubscriptionPlan.pro
    assert stored is not None
    assert stored.plan_name == SubscriptionPlan.pro
    assert stored.is_active is True


@pytest.mark.asyncio
async def test_set_user_subscription_replaces_previous_subscription(db_session):
    """Verifies that changing a plan replaces previous subscriptions."""

    service = build_service()
    user = make_user()
    db_session.add(user)
    await db_session.commit()

    await service.set_user_subscription(
        db=db_session,
        user=user,
        plan=SubscriptionPlan.basic,
    )
    await service.set_user_subscription(
        db=db_session,
        user=user,
        plan=SubscriptionPlan.enterprise,
    )

    rows = (
        await db_session.execute(
            select(Subscription).where(Subscription.user_id == user.id)
        )
    ).scalars().all()

    assert len(rows) == 1
    assert rows[0].plan_name == SubscriptionPlan.enterprise


@pytest.mark.asyncio
async def test_user_has_paid_subscription_reflects_database_state(db_session):
    """Verifies that paid subscription check reflects stored subscription state."""

    service = build_service()
    user = make_user()
    db_session.add(user)
    await db_session.commit()

    before = await service.user_has_paid_subscription(db_session, user.id)

    await service.set_user_subscription(
        db=db_session,
        user=user,
        plan=SubscriptionPlan.basic,
    )

    after = await service.user_has_paid_subscription(db_session, user.id)

    assert before is False
    assert after is True
