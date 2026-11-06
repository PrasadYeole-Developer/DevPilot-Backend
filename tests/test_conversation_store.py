"""Offline tests for ConversationStore (the `conversations` metadata collection)."""

import pytest

from app.services.conversation_store import ConversationStore

pytestmark = pytest.mark.anyio


async def test_create_and_get_conversation(fake_conversations_collection):
    store = ConversationStore(fake_conversations_collection)

    document = await store.create(
        thread_id="t1", session_id="session-a", repository_id="acme/webapp", workflow="chat", title="My chat"
    )
    assert document["_id"] == "t1"

    fetched = await store.get("t1")
    assert fetched["session_id"] == "session-a"
    assert fetched["repository_id"] == "acme/webapp"
    assert fetched["title"] == "My chat"
    assert fetched["created_at"] == fetched["updated_at"]


async def test_get_missing_conversation_returns_none(fake_conversations_collection):
    store = ConversationStore(fake_conversations_collection)
    assert await store.get("does-not-exist") is None


async def test_list_for_session_scopes_by_session_and_supports_pagination(fake_conversations_collection):
    store = ConversationStore(fake_conversations_collection)
    for i in range(5):
        await store.create(
            thread_id=f"t{i}", session_id="session-a", repository_id="acme/webapp", workflow="chat"
        )
    await store.create(thread_id="other", session_id="session-b", repository_id="acme/webapp", workflow="chat")

    docs, total = await store.list_for_session(session_id="session-a", limit=2, offset=0)
    assert total == 5
    assert len(docs) == 2
    assert all(d["session_id"] == "session-a" for d in docs)

    docs_page_2, _ = await store.list_for_session(session_id="session-a", limit=2, offset=2)
    assert len(docs_page_2) == 2
    assert {d["_id"] for d in docs} .isdisjoint({d["_id"] for d in docs_page_2})


async def test_list_for_session_filters_by_repository_and_workflow(fake_conversations_collection):
    store = ConversationStore(fake_conversations_collection)
    await store.create(thread_id="t1", session_id="s", repository_id="acme/webapp", workflow="chat")
    await store.create(thread_id="t2", session_id="s", repository_id="acme/other", workflow="chat")
    await store.create(thread_id="t3", session_id="s", repository_id="acme/webapp", workflow="investigation")

    docs, total = await store.list_for_session(session_id="s", repository_id="acme/webapp", workflow="chat")
    assert total == 1
    assert docs[0]["_id"] == "t1"


async def test_touch_updates_timestamp_and_optional_status(fake_conversations_collection):
    store = ConversationStore(fake_conversations_collection)
    await store.create(thread_id="t1", session_id="s", repository_id="r", workflow="investigation")
    original = await store.get("t1")

    await store.touch("t1", status="complete")
    updated = await store.get("t1")

    assert updated["status"] == "complete"
    assert updated["updated_at"] >= original["updated_at"]


async def test_delete_conversation(fake_conversations_collection):
    store = ConversationStore(fake_conversations_collection)
    await store.create(thread_id="t1", session_id="s", repository_id="r", workflow="chat")

    assert await store.delete("t1") is True
    assert await store.get("t1") is None
    assert await store.delete("t1") is False  # already gone
