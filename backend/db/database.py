from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from backend.app.config import settings

engine = create_async_engine(settings.DATABASE_URL, echo=True)

AsyncSession = async_sessionmaker( bind=engine, expire_on_commit=False )

async def get_db():
    async with AsyncSession() as session:
        yield session