import asyncio
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from backend.db.database import engine, AsyncSession, get_db
from backend.modules.auth.router import router as auth_router
from backend.modules.chats.router import router as chats_router
from backend.modules.messages.router import router as messages_router
from backend.modules.rag.router import router as rag_router
from backend.modules.rag.service import rag_app_service
from backend.modules.analytics.router import router as analysis_results_router
from backend.modules.admin.router import router as admin_router
from backend.modules.subscriptions.router import router as subscriptions_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(rag_app_service.startup())
    yield
    await engine.dispose()

app = FastAPI( title="LexigenAI",
               description="AI assistant for credit law",
               version="1.0.0",
               lifespan=lifespan
             )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(chats_router)
app.include_router(messages_router)
app.include_router(rag_router)
app.include_router(analysis_results_router)
app.include_router(admin_router)
app.include_router(subscriptions_router)


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
