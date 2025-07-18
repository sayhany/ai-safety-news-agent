"""News source registry and adapter framework."""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import aiohttp
import feedparser
from selectolax.parser import HTMLParser

from ..config import get_model_config, get_settings
from ..logging import get_logger, log_processing_stage, PerformanceLogger
from ..utils import (
    extract_domain, normalize_url, is_valid_url, 
    parse_date_string, clean_text, AsyncSemaphore
)

logger = get_logger(__name__)


class Article(Dict[str, Any]):
    """Article data structure with validation."""
    
    def __init__(self, **kwargs):
        # Required fields
        required_fields = ['title', 'url', 'content', 'published_date', 'source']
        for field in required_fields:
            if field not in kwargs:
                raise ValueError(f"Missing required field: {field}")
        
        # Set defaults for optional fields
        defaults = {
            'description': '',
            'author': '',
            'tags': [],
            'language': 'en',
            'content_hash': '',
            'scraped_at': datetime.now(timezone.utc).isoformat(),
        }
        
        # Merge with defaults
        for key, default_value in defaults.items():
            kwargs.setdefault(key, default_value)
        
        super().__init__(**kwargs)
    
    @property
    def domain(self) -> str:
        """Get domain from URL."""
        return extract_domain(self['url'])
    
    @property
    def published_datetime(self) -> Optional[datetime]:
        """Get published date as datetime object."""
        if isinstance(self['published_date'], datetime):
            return self['published_date']
        elif isinstance(self['published_date'], str):
            return parse_date_string(self['published_date'])
        return None


class SourceAdapter(ABC):
    """Abstract base class for news source adapters."""
    
    def __init__(self, source_config: Dict[str, Any]):
        self.config = source_config
        self.url = source_config['url']
        self.domain = source_config['domain']
        self.category = source_config.get('category', 'general')
        self.priority = source_config.get('priority', 0.5)
        
        self.settings = get_settings()
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        timeout = aiohttp.ClientTimeout(total=30)
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={'User-Agent': self.settings.user_agent}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def fetch_articles(
        self, 
        start_date: datetime, 
        limit: Optional[int] = None
    ) -> List[Article]:
        """Fetch articles from the source.
        
        Args:
            start_date: Fetch articles published after this date
            limit: Maximum number of articles to fetch
            
        Returns:
            List of Article objects
        """
        pass
    
    async def _fetch_url(self, url: str) -> str:
        """Fetch URL content with error handling.
        
        Args:
            url: URL to fetch
            
        Returns:
            Response text
            
        Raises:
            aiohttp.ClientError: If request fails
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")
        
        logger.debug("Fetching URL", url=url, domain=self.domain)
        
        async with self.session.get(url) as response:
            response.raise_for_status()
            content = await response.text()
            
            logger.debug(
                "URL fetched successfully",
                url=url,
                status=response.status,
                content_length=len(content)
            )
            
            return content
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime."""
        return parse_date_string(date_str)
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content."""
        return clean_text(content)


class RSSAdapter(SourceAdapter):
    """RSS feed adapter."""
    
    async def fetch_articles(
        self, 
        start_date: datetime, 
        limit: Optional[int] = None
    ) -> List[Article]:
        """Fetch articles from RSS feed."""
        try:
            # Fetch RSS content
            rss_content = await self._fetch_url(self.url)
            
            # Parse RSS feed
            feed = feedparser.parse(rss_content)
            
            if not feed.entries:
                logger.warning("No entries found in RSS feed", url=self.url)
                return []
            
            articles = []
            for entry in feed.entries:
                try:
                    article = await self._parse_rss_entry(entry, start_date)
                    if article:
                        articles.append(article)
                        
                        if limit and len(articles) >= limit:
                            break
                            
                except Exception as e:
                    logger.warning(
                        "Failed to parse RSS entry",
                        url=self.url,
                        entry_title=getattr(entry, 'title', 'Unknown'),
                        error=str(e)
                    )
                    continue
            
            logger.info(
                "RSS articles fetched",
                source=self.domain,
                total_entries=len(feed.entries),
                parsed_articles=len(articles)
            )
            
            return articles
            
        except Exception as e:
            logger.error(
                "Failed to fetch RSS feed",
                url=self.url,
                error=str(e)
            )
            return []
    
    async def _parse_rss_entry(
        self, 
        entry: Any, 
        start_date: datetime
    ) -> Optional[Article]:
        """Parse individual RSS entry."""
        # Extract basic fields
        title = getattr(entry, 'title', '').strip()
        link = getattr(entry, 'link', '').strip()
        description = getattr(entry, 'description', '').strip()
        
        if not title or not link:
            return None
        
        # Parse publication date
        published_date = None
        for date_field in ['published', 'updated', 'pubDate']:
            if hasattr(entry, date_field):
                date_str = getattr(entry, date_field)
                published_date = self._parse_date(date_str)
                if published_date:
                    break
        
        if not published_date:
            published_date = datetime.now(timezone.utc)
        
        # Filter by date
        if published_date < start_date:
            return None
        
        # Get full article content
        content = await self._fetch_article_content(link, description)
        
        # Extract author
        author = getattr(entry, 'author', '')
        
        # Extract tags
        tags = []
        if hasattr(entry, 'tags'):
            tags = [tag.term for tag in entry.tags if hasattr(tag, 'term')]
        
        return Article(
            title=self._clean_content(title),
            url=normalize_url(link),
            content=content,
            description=self._clean_content(description),
            published_date=published_date,
            source=self.domain,
            author=author,
            tags=tags,
            category=self.category
        )
    
    async def _fetch_article_content(self, url: str, fallback_content: str) -> str:
        """Fetch full article content from URL."""
        try:
            # Try to fetch full article
            html_content = await self._fetch_url(url)
            
            # Parse HTML and extract main content
            parser = HTMLParser(html_content)
            
            # Try common content selectors
            content_selectors = [
                'article',
                '.article-content',
                '.post-content',
                '.entry-content',
                '.content',
                'main',
                '.main-content'
            ]
            
            content = ""
            for selector in content_selectors:
                elements = parser.css(selector)
                if elements:
                    content = elements[0].text(strip=True)
                    break
            
            # Fallback to body if no specific content found
            if not content:
                body = parser.css_first('body')
                if body:
                    content = body.text(strip=True)
            
            # Clean and validate content
            content = self._clean_content(content)
            
            # Use fallback if content is too short
            if len(content) < 100:
                content = fallback_content
            
            return content
            
        except Exception as e:
            logger.debug(
                "Failed to fetch article content, using fallback",
                url=url,
                error=str(e)
            )
            return self._clean_content(fallback_content)


class HTMLAdapter(SourceAdapter):
    """HTML page adapter for sites without RSS."""
    
    def __init__(self, source_config: Dict[str, Any]):
        super().__init__(source_config)
        self.article_selector = source_config.get('article_selector', 'article')
        self.title_selector = source_config.get('title_selector', 'h1, h2, .title')
        self.content_selector = source_config.get('content_selector', '.content, .article-body')
        self.date_selector = source_config.get('date_selector', 'time, .date')
    
    async def fetch_articles(
        self, 
        start_date: datetime, 
        limit: Optional[int] = None
    ) -> List[Article]:
        """Fetch articles from HTML page."""
        try:
            html_content = await self._fetch_url(self.url)
            parser = HTMLParser(html_content)
            
            # Find article elements
            article_elements = parser.css(self.article_selector)
            
            articles = []
            for element in article_elements:
                try:
                    article = await self._parse_html_article(element, start_date)
                    if article:
                        articles.append(article)
                        
                        if limit and len(articles) >= limit:
                            break
                            
                except Exception as e:
                    logger.warning(
                        "Failed to parse HTML article",
                        url=self.url,
                        error=str(e)
                    )
                    continue
            
            logger.info(
                "HTML articles fetched",
                source=self.domain,
                total_elements=len(article_elements),
                parsed_articles=len(articles)
            )
            
            return articles
            
        except Exception as e:
            logger.error(
                "Failed to fetch HTML page",
                url=self.url,
                error=str(e)
            )
            return []
    
    async def _parse_html_article(
        self, 
        element: Any, 
        start_date: datetime
    ) -> Optional[Article]:
        """Parse individual HTML article element."""
        # Extract title
        title_elem = element.css_first(self.title_selector)
        title = title_elem.text(strip=True) if title_elem else ""
        
        # Extract link
        link_elem = element.css_first('a[href]')
        link = link_elem.attributes.get('href', '') if link_elem else ""
        
        if not title or not link:
            return None
        
        # Make link absolute
        if link.startswith('/'):
            link = urljoin(self.url, link)
        
        # Extract content
        content_elem = element.css_first(self.content_selector)
        content = content_elem.text(strip=True) if content_elem else ""
        
        # Extract date
        date_elem = element.css_first(self.date_selector)
        date_str = ""
        if date_elem:
            # Try datetime attribute first
            date_str = date_elem.attributes.get('datetime', '')
            if not date_str:
                date_str = date_elem.text(strip=True)
        
        published_date = self._parse_date(date_str) if date_str else datetime.now(timezone.utc)
        
        # Filter by date
        if published_date < start_date:
            return None
        
        return Article(
            title=self._clean_content(title),
            url=normalize_url(link),
            content=self._clean_content(content),
            published_date=published_date,
            source=self.domain,
            category=self.category
        )


class SourceRegistry:
    """Registry for managing news sources and their adapters."""
    
    def __init__(self):
        self.settings = get_settings()
        self.model_config = get_model_config()
        self.sources: List[Dict[str, Any]] = []
        self.adapters: Dict[str, SourceAdapter] = {}
        self._load_sources()
    
    def _load_sources(self):
        """Load sources from configuration."""
        try:
            self.sources = self.model_config.get_approved_sources()
            logger.info("Loaded sources", count=len(self.sources))
        except Exception as e:
            logger.error("Failed to load sources", error=str(e))
            self.sources = []
    
    def create_adapter(self, source_config: Dict[str, Any]) -> SourceAdapter:
        """Create appropriate adapter for source."""
        adapter_type = source_config.get('type', 'rss')
        
        if adapter_type == 'rss':
            return RSSAdapter(source_config)
        elif adapter_type == 'html':
            return HTMLAdapter(source_config)
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")
    
    async def fetch_all_articles(
        self, 
        start_date: datetime, 
        limit_per_source: Optional[int] = None
    ) -> List[Article]:
        """Fetch articles from all sources."""
        if not self.sources:
            logger.warning("No sources configured")
            return []
        
        # Create semaphore for concurrency control
        semaphore = AsyncSemaphore(
            global_limit=self.settings.global_parallel,
            domain_limits={source['domain']: self.settings.max_per_domain 
                          for source in self.sources}
        )
        
        async def fetch_source(source_config: Dict[str, Any]) -> List[Article]:
            """Fetch articles from a single source."""
            domain = source_config['domain']
            
            async with semaphore:
                try:
                    adapter = self.create_adapter(source_config)
                    async with adapter:
                        with PerformanceLogger(f"fetch_{domain}", logger):
                            articles = await adapter.fetch_articles(
                                start_date, 
                                limit=limit_per_source
                            )
                            
                            logger.info(
                                **log_processing_stage(
                                    stage=f"fetch_{domain}",
                                    input_count=1,
                                    output_count=len(articles)
                                )
                            )
                            
                            return articles
                            
                except Exception as e:
                    logger.error(
                        "Failed to fetch from source",
                        domain=domain,
                        error=str(e)
                    )
                    return []
        
        # Fetch from all sources concurrently
        with PerformanceLogger("fetch_all_sources", logger):
            tasks = [fetch_source(source) for source in self.sources]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect all articles
        all_articles = []
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)
            elif isinstance(result, Exception):
                logger.error("Source fetch failed", error=str(result))
        
        logger.info(
            **log_processing_stage(
                stage="fetch_all_sources",
                input_count=len(self.sources),
                output_count=len(all_articles)
            )
        )
        
        return all_articles


# Global registry instance
_registry = None


def get_source_registry() -> SourceRegistry:
    """Get global source registry instance."""
    global _registry
    if _registry is None:
        _registry = SourceRegistry()
    return _registry


async def gather_articles(
    start_date: str, 
    mock: bool = False, 
    limit_per_source: Optional[int] = None
) -> List[Article]:
    """Gather articles from all configured sources.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        mock: Use mock data instead of real sources
        limit_per_source: Limit articles per source
        
    Returns:
        List of articles
    """
    if mock:
        return _generate_mock_articles(start_date)
    
    # Parse start date
    start_dt = parse_date_string(start_date)
    if not start_dt:
        raise ValueError(f"Invalid start date: {start_date}")
    
    # Ensure timezone
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    
    # Fetch articles
    registry = get_source_registry()
    articles = await registry.fetch_all_articles(start_dt, limit_per_source)
    
    return articles


def _generate_mock_articles(start_date: str) -> List[Article]:
    """Generate mock articles for testing."""
    mock_articles = [
        Article(
            title="EU Proposes New AI Safety Regulations",
            url="https://example.com/eu-ai-safety",
            content="The European Union has proposed comprehensive new regulations for AI safety, including mandatory risk assessments for high-risk AI systems and strict oversight requirements.",
            published_date=datetime.now(timezone.utc),
            source="example.com",
            category="Policy"
        ),
        Article(
            title="OpenAI Releases Safety Framework for Large Language Models",
            url="https://example.com/openai-safety",
            content="OpenAI has published a new safety framework outlining best practices for developing and deploying large language models, with emphasis on alignment and robustness testing.",
            published_date=datetime.now(timezone.utc),
            source="example.com",
            category="Technology"
        ),
        Article(
            title="NIST Updates AI Risk Management Guidelines",
            url="https://example.com/nist-ai-risk",
            content="The National Institute of Standards and Technology has updated its AI Risk Management Framework to include new guidance on algorithmic bias detection and mitigation strategies.",
            published_date=datetime.now(timezone.utc),
            source="nist.gov",
            category="Policy"
        )
    ]
    
    return mock_articles


if __name__ == "__main__":
    # Test the source registry
    async def test_sources():
        articles = await gather_articles("2025-07-18", mock=True)
        print(f"Fetched {len(articles)} articles")
        for article in articles:
            print(f"- {article['title']} ({article['source']})")
    
    asyncio.run(test_sources())
