"""
Milestone 7 factory functions for Mongo-backed services, all built on the
single shared client in app.core.mongodb.mongodb_manager.

Centralizing construction here (rather than inlining it in every route) keeps
collection names consistent in one place and gives routes a single call site
that raises a clear MongoDBConnectionError when Mongo isn't configured —
mongodb_manager.client/.database already raise that, so these factories
propagate it for free.
"""

from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.store.mongodb import MongoDBStore

from app.core.config import Settings
from app.core.mongodb import mongodb_manager
from app.services.conversation_store import ConversationStore
from app.services.memory_store import LongTermMemoryService
from app.services.proposal_store import ProposalAuditStore, ProposalStore

CHECKPOINT_COLLECTION = "checkpoints"
CHECKPOINT_WRITES_COLLECTION = "checkpoint_writes"
CONVERSATIONS_COLLECTION = "conversations"
MEMORIES_COLLECTION = "memories"
PROPOSALS_COLLECTION = "proposals"
PROPOSAL_AUDIT_COLLECTION = "proposal_audit"


def get_checkpointer(settings: Settings) -> MongoDBSaver:
    return MongoDBSaver(
        client=mongodb_manager.client,
        db_name=settings.mongodb_database,
        checkpoint_collection_name=CHECKPOINT_COLLECTION,
        writes_collection_name=CHECKPOINT_WRITES_COLLECTION,
    )


def get_conversation_store(settings: Settings) -> ConversationStore:
    collection = mongodb_manager.database[CONVERSATIONS_COLLECTION]
    return ConversationStore(collection)


def get_memory_service(settings: Settings) -> LongTermMemoryService:
    collection = mongodb_manager.database[MEMORIES_COLLECTION]
    store = MongoDBStore(collection=collection)
    return LongTermMemoryService(store)


def get_proposal_store(settings: Settings) -> ProposalStore:
    collection = mongodb_manager.database[PROPOSALS_COLLECTION]
    return ProposalStore(collection)


def get_proposal_audit_store(settings: Settings) -> ProposalAuditStore:
    collection = mongodb_manager.database[PROPOSAL_AUDIT_COLLECTION]
    return ProposalAuditStore(collection)
