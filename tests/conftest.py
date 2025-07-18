"""Pytest configuration and fixtures."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest

# Set test environment
os.environ["OPENROUTER_API_KEY"] = "test-key"
os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["JSON_LOGGING"] = "false"


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_env(monkeypatch, temp_dir):
    """Mock environment variables for testing."""
    monkeypatch.setenv("DATA_DIR", str(temp_dir / "data"))
    monkeypatch.setenv("CACHE_DIR", str(temp_dir / "cache"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


@pytest.fixture
def sample_article():
    """Sample article for testing."""
    from datetime import datetime, timezone
    from aisafety_news.ingest.sources import Article
    
    return Article(
        title="Test AI Safety Article",
        url="https://example.com/test-article",
        content="This is a test article about AI safety and governance.",
        published_date=datetime.now(timezone.utc),
        source="example.com",
        category="Technology"
    )


@pytest.fixture
def sample_articles():
    """Sample articles list for testing."""
    from datetime import datetime, timezone
    from aisafety_news.ingest.sources import Article
    
    return [
        Article(
            title="EU AI Regulation Update",
            url="https://example.com/eu-ai",
            content="The EU has updated its AI regulation framework.",
            published_date=datetime.now(timezone.utc),
            source="example.com",
            category="Policy"
        ),
        Article(
            title="OpenAI Safety Research",
            url="https://example.com/openai-safety",
            content="OpenAI publishes new safety research findings.",
            published_date=datetime.now(timezone.utc),
            source="openai.com",
            category="Research"
        )
    ]
