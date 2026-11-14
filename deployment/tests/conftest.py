"""Pytest configuration and session-scoped fixtures for API deployment tests."""

import pytest
from fastapi.testclient import TestClient
from deployment.api.main import app


@pytest.fixture(scope="session")
def client():
    """Session-scoped TestClient entering lifespan once for the entire test suite."""
    with TestClient(app) as test_client:
        yield test_client
