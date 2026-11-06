"""Offline tests for LongTermMemoryService (long-term, cross-conversation memory)."""

import pytest

from app.services.memory_store import LongTermMemoryService, MemoryStoreError

pytestmark = pytest.mark.anyio


async def test_remember_and_recall(fake_long_term_store):
    service = LongTermMemoryService(fake_long_term_store)

    await service.remember(
        session_id="session-a",
        repository_id="acme/webapp",
        content="deactivate_user() never revokes existing sessions (app/auth/deactivate.py)",
        kind="confirmed_finding",
        source="investigation:t1",
    )

    memories = await service.recall(session_id="session-a", repository_id="acme/webapp")
    assert len(memories) == 1
    assert memories[0]["kind"] == "confirmed_finding"
    assert "deactivate_user" in memories[0]["content"]
    assert memories[0]["source"] == "investigation:t1"


async def test_duplicate_memory_does_not_create_a_second_entry(fake_long_term_store):
    service = LongTermMemoryService(fake_long_term_store)
    content = "Sessions are validated only by checking token presence (app/auth/session.py)"

    await service.remember("session-a", "acme/webapp", content, "confirmed_finding", "investigation:t1")
    await service.remember("session-a", "acme/webapp", content, "confirmed_finding", "investigation:t2")

    memories = await service.recall(session_id="session-a", repository_id="acme/webapp")
    assert len(memories) == 1
    # The second write's provenance wins (it's an upsert/refresh, not a no-op).
    assert memories[0]["source"] == "investigation:t2"


async def test_different_content_creates_separate_memories(fake_long_term_store):
    service = LongTermMemoryService(fake_long_term_store)
    await service.remember("session-a", "acme/webapp", "Finding A", "confirmed_finding", "investigation:t1")
    await service.remember("session-a", "acme/webapp", "Finding B", "confirmed_finding", "investigation:t1")

    memories = await service.recall(session_id="session-a", repository_id="acme/webapp")
    assert len(memories) == 2


@pytest.mark.parametrize(
    "secret_like_content",
    [
        "API_KEY=sk-abc123xyz789secretvalue",
        "mongodb+srv://dbuser:supersecret@cluster0.mongodb.net/devpilot",
        "password: Tr0ub4dor&3",
        "AIzaSyD-fakekeyfakekeyfakekeyfakekeyfake12",
        "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.fake.token",
    ],
)
async def test_remember_rejects_content_containing_an_actual_secret_value(fake_long_term_store, secret_like_content):
    service = LongTermMemoryService(fake_long_term_store)
    with pytest.raises(MemoryStoreError):
        await service.remember("session-a", "acme/webapp", secret_like_content, "confirmed_finding", "investigation:t1")

    assert await service.recall(session_id="session-a", repository_id="acme/webapp") == []


@pytest.mark.parametrize(
    "benign_content",
    [
        "Sessions are validated only by checking token presence (app/auth/session.py)",
        "deactivate_user() sets is_active=False but does not revoke the session token",
        "The app stores a password hash, not a plaintext password, in the users table",
        "Config loads a secret key for signing cookies from an environment variable",
    ],
)
async def test_remember_allows_benign_mentions_of_security_concepts(fake_long_term_store, benign_content):
    """Naming a security concept (token/password/secret) is fine — only actual secret
    *values* must be blocked, or every legitimate auth-related finding would be unstorable."""
    service = LongTermMemoryService(fake_long_term_store)
    key = await service.remember("session-a", "acme/webapp", benign_content, "confirmed_finding", "investigation:t1")
    assert key


async def test_memories_are_isolated_by_session_and_repository(fake_long_term_store):
    service = LongTermMemoryService(fake_long_term_store)
    await service.remember("session-a", "acme/webapp", "Finding for A/webapp", "confirmed_finding", "src")
    await service.remember("session-b", "acme/webapp", "Finding for B/webapp", "confirmed_finding", "src")
    await service.remember("session-a", "acme/other-repo", "Finding for A/other-repo", "confirmed_finding", "src")

    a_webapp = await service.recall(session_id="session-a", repository_id="acme/webapp")
    assert [m["content"] for m in a_webapp] == ["Finding for A/webapp"]

    b_webapp = await service.recall(session_id="session-b", repository_id="acme/webapp")
    assert [m["content"] for m in b_webapp] == ["Finding for B/webapp"]

    a_other = await service.recall(session_id="session-a", repository_id="acme/other-repo")
    assert [m["content"] for m in a_other] == ["Finding for A/other-repo"]


async def test_forget_removes_a_memory(fake_long_term_store):
    service = LongTermMemoryService(fake_long_term_store)
    key = await service.remember("session-a", "acme/webapp", "Temporary note", "note", "manual")
    await service.forget(session_id="session-a", repository_id="acme/webapp", key=key)

    assert await service.recall(session_id="session-a", repository_id="acme/webapp") == []
