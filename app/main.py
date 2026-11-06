"""DevPilot FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import chat, conversations, health, investigate, proposals, repositories
from app.core.config import get_settings
from app.core.mongodb import MongoDBConnectionError, mongodb_manager

logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # MongoDB is optional (Milestone 7): if MONGODB_URI isn't set, or the
    # connection attempt fails, the app still starts — /chat and /investigate
    # keep working exactly as in Milestones 1-6 (fully ephemeral), and only
    # the new /api/v1/conversations endpoints return 503 until it's fixed.
    # This is a deliberate graceful-degradation choice: Mongo backs a new,
    # optional capability, not the app's core functionality.
    if settings.mongodb_uri:
        try:
            mongodb_manager.connect(
                uri=settings.mongodb_uri,
                db_name=settings.mongodb_database,
                connect_timeout_ms=settings.mongodb_connect_timeout_ms,
                server_selection_timeout_ms=settings.mongodb_server_selection_timeout_ms,
            )
        except MongoDBConnectionError as exc:
            logger.warning(
                "MongoDB connection failed at startup (%s). Conversation persistence "
                "and long-term memory are unavailable; chat/investigate still work "
                "without persistence.",
                exc,
            )
    else:
        logger.info("MONGODB_URI not set — conversation persistence is disabled.")

    yield

    mongodb_manager.close()


app = FastAPI(
    title=settings.app_name,
    description="AI Developer Assistant that understands a GitHub repository via RAG.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(repositories.router)
app.include_router(chat.router)
app.include_router(investigate.router)
app.include_router(conversations.router)
app.include_router(proposals.router)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": f"{settings.app_name} is running"}
