from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, Depends
from sqlalchemy import text
from backend.db.database import engine, AsyncSession, get_db
from backend.db.base import Base

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        yield await engine.dispose()

app = FastAPI( title="LexigenAI",
               description="AI assistant for credit law",
               version="1.0.0",
               lifespan=lifespan )

@app.get("/")
async def root():
    return {"message": "LexigenAI backend is running"}

@app.get("/health/db")
async def health_db(db: AsyncSession = Depends(get_db)):
    result = await db.execute(text("SELECT 1"))
    return { "status": "ok", "db_response": result.scalar() }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
)