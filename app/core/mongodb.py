"""
Centralized MongoDB client lifecycle (Milestone 7).

One `pymongo.MongoClient` is created once, at FastAPI startup (via the app's
lifespan context manager in app.main), and reused by everything that needs
Mongo — the LangGraph checkpointer, the long-term memory store, and the
conversation-metadata service. It is closed once at shutdown.

pymongo's MongoClient is internally thread-safe and already pools
connections, so "reuse instead of creating a new client per request" just
means: construct it once here, never inside a route handler.

The client itself is synchronous (pymongo's async driver is a separate,
newer API that langgraph-checkpoint-mongodb/langgraph-store-mongodb do not
use) — both of those packages wrap this sync client's calls in a thread-pool
executor for their async methods (`aget`, `aput`, etc.), which is the
supported pattern for keeping sync-only drivers off the event loop. We follow
the same pattern for the conversation-metadata CRUD in
app.services.conversation_store.
"""

import logging

from pymongo import MongoClient
from pymongo.errors import PyMongoError

logger = logging.getLogger(__name__)


class MongoDBConnectionError(Exception):
    """Raised when the MongoDB client cannot be created, or its startup ping fails."""


class MongoDBManager:
    def __init__(self) -> None:
        self._client: MongoClient | None = None
        self._db_name: str | None = None

    def connect(
        self,
        uri: str,
        db_name: str,
        connect_timeout_ms: int = 5000,
        server_selection_timeout_ms: int = 5000,
    ) -> None:
        """Idempotent: a second call while already connected is a no-op."""
        if self._client is not None:
            return

        try:
            client: MongoClient = MongoClient(
                uri,
                connectTimeoutMS=connect_timeout_ms,
                serverSelectionTimeoutMS=server_selection_timeout_ms,
            )
            # Startup connection verification — fail fast and clearly rather
            # than discovering a bad URI on the first real request.
            client.admin.command("ping")
        except PyMongoError as exc:
            raise MongoDBConnectionError(f"Failed to connect to MongoDB: {exc}") from exc

        self._client = client
        self._db_name = db_name
        logger.info("Connected to MongoDB database %r", db_name)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            logger.info("MongoDB connection closed")
        self._client = None
        self._db_name = None

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    @property
    def client(self) -> MongoClient:
        if self._client is None:
            raise MongoDBConnectionError(
                "MongoDB is not configured or failed to connect at startup. "
                "Set MONGODB_URI in your .env to use conversation persistence."
            )
        return self._client

    @property
    def database(self):
        return self.client[self._db_name]


# Module-level singleton, mirroring app.ingestion.store.repository_store's
# pattern elsewhere in this codebase — initialized by app.main's lifespan,
# consumed by routes/services.
mongodb_manager = MongoDBManager()
