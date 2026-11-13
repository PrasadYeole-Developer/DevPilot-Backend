"""
Tests for GET /api/v1/repositories/current — the read-only restoration
endpoint the frontend calls on load to recover repository state after a
browser refresh.

Dependencies are monkeypatched at the route-module level, matching the
convention established in test_api_contract.py.
"""

import app.api.routes.repositories as repositories_module
from fastapi.testclient import TestClient

from app.ingestion.file_filter import SkippedEntry
from app.ingestion.store import IngestedRepository, RepositoryStore, RetrievedFile
from app.main import app


class _ExplodingService:
    """Fails loudly if the /current route ever tries to construct it —
    proof that the endpoint performs no ingestion, embedding, or Pinecone work."""

    def __init__(self, *args, **kwargs):
        raise AssertionError(
            "GET /api/v1/repositories/current must not construct this service"
        )


def _populated_store() -> RepositoryStore:
    store = RepositoryStore()
    store.save(
        IngestedRepository(
            owner="PrasadYeole-Developer",
            repo="FrameVerse-Frontend",
            default_branch="main",
            files=[
                RetrievedFile(path="src/App.tsx", content="export default function App() {}", size=42, sha="sha1"),
                RetrievedFile(path="src/index.tsx", content="// entry", size=10, sha="sha2"),
            ],
            total_files_discovered=30,
            eligible_files=25,
            skipped_files=[SkippedEntry(path="node_modules/x.js", reason="excluded_directory")],
            skipped_files_truncated=False,
            tree_truncated=False,
            errors=[],
        )
    )
    return store


def test_current_repository_returns_metadata_when_present(monkeypatch):
    monkeypatch.setattr(repositories_module, "GitHubClient", _ExplodingService)
    monkeypatch.setattr(repositories_module, "EmbeddingService", _ExplodingService)
    monkeypatch.setattr(repositories_module, "PineconeService", _ExplodingService)
    monkeypatch.setattr(repositories_module, "repository_store", _populated_store())

    client = TestClient(app)
    response = client.get("/api/v1/repositories/current")

    assert response.status_code == 200
    body = response.json()
    assert body["repository"] is not None

    repo = body["repository"]
    assert repo["owner"] == "PrasadYeole-Developer"
    assert repo["repository"] == "FrameVerse-Frontend"
    assert repo["repository_id"] == "PrasadYeole-Developer/FrameVerse-Frontend"
    assert repo["default_branch"] == "main"
    # retrieved_files is derived from the actual stored files list, not echoed blindly.
    assert repo["retrieved_files"] == 2
    assert repo["total_files_discovered"] == 30
    assert repo["eligible_files"] == 25
    assert repo["skipped_files"] == [{"path": "node_modules/x.js", "reason": "excluded_directory"}]
    assert repo["skipped_files_truncated"] is False
    assert repo["tree_truncated"] is False
    assert repo["errors"] == []


def test_current_repository_returns_null_when_absent(monkeypatch):
    monkeypatch.setattr(repositories_module, "GitHubClient", _ExplodingService)
    monkeypatch.setattr(repositories_module, "EmbeddingService", _ExplodingService)
    monkeypatch.setattr(repositories_module, "PineconeService", _ExplodingService)
    monkeypatch.setattr(repositories_module, "repository_store", RepositoryStore())

    client = TestClient(app)
    response = client.get("/api/v1/repositories/current")

    assert response.status_code == 200
    assert response.json() == {"repository": None}


def test_current_repository_does_not_mutate_store(monkeypatch):
    """Calling /current repeatedly must not change what get_current() returns."""
    store = _populated_store()
    monkeypatch.setattr(repositories_module, "GitHubClient", _ExplodingService)
    monkeypatch.setattr(repositories_module, "EmbeddingService", _ExplodingService)
    monkeypatch.setattr(repositories_module, "PineconeService", _ExplodingService)
    monkeypatch.setattr(repositories_module, "repository_store", store)

    client = TestClient(app)
    client.get("/api/v1/repositories/current")
    client.get("/api/v1/repositories/current")

    current = store.get_current()
    assert current is not None
    assert current.owner == "PrasadYeole-Developer"
    assert len(current.files) == 2
