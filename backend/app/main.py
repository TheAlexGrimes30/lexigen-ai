import uvicorn
from fastapi import FastAPI

app = FastAPI(
    title="LexigenAI",
    description="AI assistant for credit law",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {"message": "LexigenAI backend is running"}


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )