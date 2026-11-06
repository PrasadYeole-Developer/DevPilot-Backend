"""DevPilot FastAPI application entry point."""

from fastapi import FastAPI

from app.api.routes import chat, health, investigate, repositories
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="AI Developer Assistant that understands a GitHub repository via RAG.",
    version="0.1.0",
)

app.include_router(health.router)
app.include_router(repositories.router)
app.include_router(chat.router)
app.include_router(investigate.router)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": f"{settings.app_name} is running"}
