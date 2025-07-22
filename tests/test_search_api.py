"""Test Search API feed functionality."""

import pytest
import json
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from aisafety_news.ingest.sources import (
    SearchAPIAdapter,
    ExaSearchAPIAdapter,
    Article,
)
from aisafety_news.config import get_model_config, get_settings


class TestSearchAPIFeeds:
    """Test Search API feed functionality."""

    @pytest.mark.asyncio
    async def test_exa_search_adapter_instantiation(self):
        """Test that the Exa Search API adapter can be instantiated."""
        model_config = get_model_config()

        # Create a test source config
        test_source = model_config.get_search_sources()[0]

        adapter = SearchAPIAdapter(test_source)
        assert isinstance(adapter, SearchAPIAdapter)
        assert adapter.api == 'exa'

    @pytest.mark.asyncio
    @patch('aisafety_news.ingest.sources.Exa')
    async def test_exa_search_api_mocked(self, mock_exa):
        """Test Exa Search API adapter with a mocked response."""
        settings = get_settings()
        settings.exa_api_key = "test_key"
        model_config = get_model_config()
        test_source = model_config.get_search_sources()[0]

        # Mock the Exa client's search_and_contents method
        mock_results = {
            "results": [
                {
                    "title": "Test Article",
                    "url": "https://example.com/test",
                    "published_date": "2025-07-18T12:00:00Z",
                    "text": "This is a test article content.",
                }
            ]
        }
        
        # Configure the mock to return an object with a 'results' attribute
        mock_instance = mock_exa.return_value
        from types import SimpleNamespace
        mock_instance.search_and_contents.return_value = SimpleNamespace(results=[SimpleNamespace(**r) for r in mock_results['results']])

        adapter = SearchAPIAdapter(test_source)
        start_date = datetime.now(timezone.utc) - timedelta(days=7)
        articles = await adapter.fetch_articles(start_date, limit=1)

        assert len(articles) == 1
        article = articles[0]
        assert isinstance(article, Article)
        assert article['title'] == "Test Article"
        assert article['content'] == "This is a test article content."
