"""
Offline tests for Milestone 7's MongoDB configuration and connection manager.

No real MongoDB is used — pymongo.MongoClient is mocked, since these tests
are about *our* connect/close/error-handling logic, not pymongo's wire
protocol.
"""

from unittest.mock import MagicMock, patch

import pytest
from pymongo.errors import PyMongoError

from app.core.config import Settings
from app.core.mongodb import MongoDBConnectionError, MongoDBManager


def test_mongodb_uri_defaults_to_none():
    settings = Settings(google_api_key=None, pinecone_api_key=None, _env_file=None)
    assert settings.mongodb_uri is None
    assert settings.mongodb_database == "devpilot"


def test_mongodb_manager_not_connected_raises_clear_error():
    manager = MongoDBManager()
    assert manager.is_connected is False
    with pytest.raises(MongoDBConnectionError):
        _ = manager.client


@patch("app.core.mongodb.MongoClient")
def test_mongodb_manager_connects_and_pings(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    manager = MongoDBManager()
    manager.connect(uri="mongodb://fake", db_name="devpilot_test")

    assert manager.is_connected is True
    mock_client.admin.command.assert_called_once_with("ping")
    assert manager.client is mock_client
    assert manager.database is mock_client["devpilot_test"]


@patch("app.core.mongodb.MongoClient")
def test_mongodb_manager_connect_is_idempotent(mock_client_cls):
    mock_client_cls.return_value = MagicMock()
    manager = MongoDBManager()

    manager.connect(uri="mongodb://fake", db_name="devpilot_test")
    manager.connect(uri="mongodb://fake-again", db_name="other_db")  # should be a no-op

    assert mock_client_cls.call_count == 1


@patch("app.core.mongodb.MongoClient")
def test_mongodb_manager_connect_failure_raises_clear_error(mock_client_cls):
    mock_client = MagicMock()
    mock_client.admin.command.side_effect = PyMongoError("no route to host")
    mock_client_cls.return_value = mock_client

    manager = MongoDBManager()
    with pytest.raises(MongoDBConnectionError):
        manager.connect(uri="mongodb://unreachable", db_name="devpilot_test")

    assert manager.is_connected is False


@patch("app.core.mongodb.MongoClient")
def test_mongodb_manager_close_resets_state(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    manager = MongoDBManager()
    manager.connect(uri="mongodb://fake", db_name="devpilot_test")
    manager.close()

    assert manager.is_connected is False
    mock_client.close.assert_called_once()
    with pytest.raises(MongoDBConnectionError):
        _ = manager.client


@patch("app.core.mongodb.MongoClient")
def test_mongodb_manager_close_without_connect_is_safe(mock_client_cls):
    manager = MongoDBManager()
    manager.close()  # must not raise
    assert manager.is_connected is False
