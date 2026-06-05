import uuid

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine, AsyncSession

from backend.db import Base, Subscription, SubscriptionPlan, User, UserRole
from backend.modules.admin.repository import AdminAnalyticsRepository
from backend.modules.admin.service import AdminService


class FakeAdminAnalyticsRepository:
    """Fake repository for AdminService unit tests."""

    def __init__(self, total_users: int, rows: list[tuple[object, int]]) -> None:
        self.total_users = total_users
        self.rows = rows
        self.count_users_called = False
        self.count_users_by_active_subscription_called = False

    async def count_users(self, db: AsyncSession):
        self.count_users_called = True
        return self.total_users

    async def count_users_by_active_subscription(self, db: AsyncSession):
        self.count_users_by_active_subscription_called = True
        return self.rows


@pytest.mark.asyncio
async def test_service_counts_without_subscription():
    repository = FakeAdminAnalyticsRepository(
        total_users=10,
        rows=[
            (SubscriptionPlan.basic, 3),
            (SubscriptionPlan.pro, 2),
        ],
    )
    service = AdminService(repository=repository)

    result = await service.get_users_by_subscription_analytics(db=None)

    assert result == {
        "total_users": 10,
        "without_subscription": 5,
        "by_plan": {
            SubscriptionPlan.basic.value: 3,
            SubscriptionPlan.pro.value: 2,
            SubscriptionPlan.enterprise.value: 0,
        },
    }
    assert repository.count_users_called is True
    assert repository.count_users_by_active_subscription_called is True


@pytest.mark.asyncio
async def test_service_never_returns_negative_without_subscription():
    repository = FakeAdminAnalyticsRepository(
        total_users=2,
        rows=[
            (SubscriptionPlan.basic, 3),
            (SubscriptionPlan.pro, 2),
        ],
    )
    service = AdminService(repository=repository)

    result = await service.get_users_by_subscription_analytics(db=None)

    assert result["total_users"] == 2
    assert result["without_subscription"] == 0


@pytest.mark.asyncio
async def test_service_returns_zeroes_when_no_subscriptions():
    repository = FakeAdminAnalyticsRepository(total_users=4, rows=[])
    service = AdminService(repository=repository)

    result = await service.get_users_by_subscription_analytics(db=None)

    assert result == {
        "total_users": 4,
        "without_subscription": 4,
        "by_plan": {
            SubscriptionPlan.basic.value: 0,
            SubscriptionPlan.pro.value: 0,
            SubscriptionPlan.enterprise.value: 0,
        },
    }


def test_service_build_empty_plan_map_contains_expected_paid_plans():
    service = AdminService(repository=FakeAdminAnalyticsRepository(0, []))

    result = service._build_empty_plan_map()

    assert result == {
        SubscriptionPlan.basic.value: 0,
        SubscriptionPlan.pro.value: 0,
        SubscriptionPlan.enterprise.value: 0,
    }


@pytest.fixture
async def db_session():
    """Creates isolated in-memory SQLite DB with only required tables."""

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


def make_user(email: str, role: UserRole = UserRole.user) -> User:
    username = email.split("@")[0]
    return User(
        id=uuid.uuid4(),
        name=username,
        email=email,
        password_hash="test_password_hash",
        role=role,
    )


def make_subscription(
    user_id: uuid.UUID,
    plan_name: SubscriptionPlan,
    is_active: bool,
) -> Subscription:
    return Subscription(
        id=uuid.uuid4(),
        user_id=user_id,
        plan_name=plan_name,
        is_active=is_active,
    )


@pytest.mark.asyncio
async def test_repository_count_users_returns_zero_for_empty_database(db_session):
    repository = AdminAnalyticsRepository()

    result = await repository.count_users(db_session)

    assert result == 0


@pytest.mark.asyncio
async def test_repository_count_users_counts_registered_users(db_session):
    repository = AdminAnalyticsRepository()

    db_session.add_all(
        [
            make_user("user1@test.com"),
            make_user("user2@test.com"),
        ]
    )
    await db_session.commit()

    result = await repository.count_users(db_session)

    assert result == 2


@pytest.mark.asyncio
async def test_repository_groups_only_active_subscriptions_by_plan(db_session):
    repository = AdminAnalyticsRepository()

    user_1 = make_user("user1@test.com")
    user_2 = make_user("user2@test.com")
    user_3 = make_user("user3@test.com")

    db_session.add_all([user_1, user_2, user_3])
    await db_session.flush()

    db_session.add_all(
        [
            make_subscription(user_1.id, SubscriptionPlan.basic, True),
            make_subscription(user_2.id, SubscriptionPlan.pro, True),
            make_subscription(user_3.id, SubscriptionPlan.enterprise, False),
        ]
    )
    await db_session.commit()

    result = await repository.count_users_by_active_subscription(db_session)

    assert sorted(result, key=lambda item: item[0].value) == sorted(
        [
            (SubscriptionPlan.basic, 1),
            (SubscriptionPlan.pro, 1),
        ],
        key=lambda item: item[0].value,
    )


@pytest.mark.asyncio
async def test_repository_counts_distinct_users_for_duplicate_active_subscriptions(db_session):
    repository = AdminAnalyticsRepository()

    user = make_user("user@test.com")
    db_session.add(user)
    await db_session.flush()

    db_session.add_all(
        [
            make_subscription(user.id, SubscriptionPlan.basic, True),
            make_subscription(user.id, SubscriptionPlan.basic, True),
        ]
    )
    await db_session.commit()

    result = await repository.count_users_by_active_subscription(db_session)

    assert result == [(SubscriptionPlan.basic, 1)]
