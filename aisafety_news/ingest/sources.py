"""News source registry and adapter framework."""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import aiohttp
import aiohttp
from exa_py import Exa
from selectolax.parser import HTMLParser

from ..config import get_model_config, get_settings
from ..logging import get_logger, log_processing_stage, PerformanceLogger
from ..utils import (
    extract_domain, normalize_url, is_valid_url,
    parse_date_string, clean_text, AsyncSemaphore, retry_async
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
        self.url = getattr(source_config, 'url', None)
        self.domain = getattr(source_config, 'domain', None)
        self.category = getattr(source_config, 'category', 'general')
        self.priority = getattr(source_config, 'priority', 0.5)
        
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
    
    async def _fetch_url(self, url: str, retry_count: int = 3) -> str:
        """Fetch URL content with error handling and retry logic.
        
        Args:
            url: URL to fetch
            retry_count: Number of retry attempts
            
        Returns:
            Response text
            
        Raises:
            aiohttp.ClientError: If request fails after all retries
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")
        
        logger.debug("Fetching URL", url=url, domain=self.domain)
        
        async def fetch_with_session():
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
        
        # Use retry logic from utils
        try:
            return await retry_async(
                fetch_with_session,
                max_retries=retry_count,
                backoff_factor=2.0,
                exceptions=(aiohttp.ClientError, aiohttp.ClientResponseError, asyncio.TimeoutError)
            )
        except Exception as e:
            logger.error(
                "Failed to fetch URL after retries",
                url=url,
                domain=self.domain,
                error=str(e)
            )
            raise
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime."""
        return parse_date_string(date_str)
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content."""
        return clean_text(content)


class SearchAPIAdapter(SourceAdapter):
    """Search API adapter."""

    def __init__(self, source_config: Dict[str, Any]):
        super().__init__(source_config)
        self.api = getattr(source_config, 'api', 'exa')
        self.query = getattr(source_config, 'query', '')

    async def fetch_articles(
        self,
        start_date: datetime,
        limit: Optional[int] = None
    ) -> List[Article]:
        """Fetch articles from a search API."""
        if self.api == 'exa':
            adapter = ExaSearchAPIAdapter(self.config)
        elif self.api == 'bing':
            # Placeholder for Bing adapter
            raise NotImplementedError("Bing Search API adapter not implemented.")
        else:
            raise ValueError(f"Unknown search API: {self.api}")

        async with adapter as a:
            return await a.fetch_articles(start_date, limit)


class ExaSearchAPIAdapter(SearchAPIAdapter):
    """Exa Search API adapter."""

    async def fetch_articles(
        self,
        start_date: datetime,
        limit: Optional[int] = None
    ) -> List[Article]:
        if not self.settings.exa_api_key:
            raise ValueError("Exa Search API key is not set.")

        exa = Exa(api_key=self.settings.exa_api_key)
        
        # Define approved domains for AI safety news
        approved_domains = [
            "reuters.com",
            "apnews.com", 
            "bbc.com",
            "theguardian.com",
            "nytimes.com",
            "techcrunch.com",
            "arstechnica.com",
            "theverge.com",
            "wired.com",
            "wsj.com",
            "npr.org",
            # High-quality AI safety sources
            "anthropic.com",
            "openai.com", 
            "deepmind.com",
            "mit.edu",
            "stanford.edu",
            "berkeley.edu"
        ]
        
        # Run synchronous Exa API call in thread executor
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: exa.search_and_contents(
                self.query,
                num_results=limit or 10,
                start_published_date=start_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
                use_autoprompt=True,
                include_domains=approved_domains
            )
        )

        return self._parse_search_results(results)

    def _parse_search_results(self, data: Any) -> List[Article]:
        articles = []
        for item in data.results:
            try:
                article = Article(
                    title=item.title or "",
                    url=item.url or "",
                    content=item.text or "",
                    published_date=item.published_date or datetime.now(timezone.utc).isoformat(),
                    source=item.url or "",
                )
                articles.append(article)
            except ValueError as e:
                logger.warning(f"Skipping search result due to missing field: {e}")
        return articles


class HTMLAdapter(SourceAdapter):
    """HTML page adapter for sites without RSS."""
    
    def __init__(self, source_config: Dict[str, Any]):
        super().__init__(source_config)
        self.article_selector = getattr(source_config, 'article_selector', 'article')
        self.title_selector = getattr(source_config, 'title_selector', 'h1, h2, .title')
        self.content_selector = getattr(source_config, 'content_selector', '.content, .article-body')
        self.date_selector = getattr(source_config, 'date_selector', 'time, .date')
    
    async def fetch_articles(
        self, 
        start_date: datetime, 
        limit: Optional[int] = None
    ) -> List[Article]:
        """Fetch articles from HTML page."""
        try:
            html_content = await self._fetch_url(str(self.url))
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


class SourceHealthMonitor:
    """Monitor source health and availability."""

    def __init__(self):
        self.source_status: Dict[str, Dict[str, Any]] = {}
        self.failure_threshold = 3
        self.check_interval = 3600  # 1 hour

    def record_success(self, name: str, response_time: float, entry_count: int):
        """Record successful source fetch."""
        self.source_status[name] = {
            'status': 'healthy',
            'last_success': datetime.now(timezone.utc),
            'response_time': response_time,
            'entry_count': entry_count,
            'consecutive_failures': 0,
            'last_error': None,
        }
        logger.debug(
            "Source health: success recorded",
            name=name,
            response_time=response_time,
            entries=entry_count,
        )

    def record_failure(self, name: str, error: str):
        """Record failed source fetch."""
        if name not in self.source_status:
            self.source_status[name] = {
                'status': 'unknown',
                'consecutive_failures': 0,
            }

        self.source_status[name]['consecutive_failures'] += 1
        self.source_status[name]['last_error'] = error
        self.source_status[name]['last_failure'] = datetime.now(timezone.utc)

        if self.source_status[name]['consecutive_failures'] >= self.failure_threshold:
            self.source_status[name]['status'] = 'unhealthy'
            logger.error(
                "Source marked as unhealthy",
                name=name,
                failures=self.source_status[name]['consecutive_failures'],
                error=error,
            )
        else:
            self.source_status[name]['status'] = 'degraded'
            logger.warning(
                "Source experiencing issues",
                name=name,
                failures=self.source_status[name]['consecutive_failures'],
                error=error,
            )

    def get_health_report(self) -> Dict[str, Any]:
        """Get health report for all monitored sources."""
        total = len(self.source_status)
        healthy = sum(1 for s in self.source_status.values() if s['status'] == 'healthy')
        degraded = sum(1 for s in self.source_status.values() if s['status'] == 'degraded')
        unhealthy = sum(1 for s in self.source_status.values() if s['status'] == 'unhealthy')

        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'summary': {
                'total': total,
                'healthy': healthy,
                'degraded': degraded,
                'unhealthy': unhealthy,
            },
            'sources': self.source_status,
        }
        logger.info(
            "Source health report",
            total=total,
            healthy=healthy,
            degraded=degraded,
            unhealthy=unhealthy,
        )
        return report

    def should_skip_source(self, name: str) -> bool:
        """Check if source should be skipped due to poor health."""
        if name not in self.source_status:
            return False
        status = self.source_status[name]
        if status['status'] == 'unhealthy':
            last_failure = status.get('last_failure')
            if last_failure:
                time_since_failure = datetime.now(timezone.utc) - last_failure
                if time_since_failure.total_seconds() < self.check_interval:
                    logger.info(
                        "Skipping unhealthy source",
                        name=name,
                        time_since_failure=time_since_failure.total_seconds(),
                    )
                    return True
        return False

# Global health monitor instance
_health_monitor = SourceHealthMonitor()

def get_source_health_monitor() -> SourceHealthMonitor:
    """Get global source health monitor instance."""
    return _health_monitor


class SourceRegistry:
    """Registry for managing news sources and their adapters."""
    
    def __init__(self):
        self.settings = get_settings()
        self.model_config = get_model_config()
        self.sources: List[Dict[str, Any]] = []
        self.adapters: Dict[str, SourceAdapter] = {}
        self.health_monitor = get_source_health_monitor()
        self._load_sources()
    
    def _load_sources(self):
        """Load sources from configuration."""
        try:
            rss_sources = self.model_config.get_approved_sources()
            search_sources = self.model_config.get_search_sources()
            self.sources = rss_sources + search_sources
            logger.info(
                "Loaded sources",
                rss_count=len(rss_sources),
                search_count=len(search_sources),
                total=len(self.sources),
            )
        except Exception as e:
            logger.error("Failed to load sources", error=str(e))
            self.sources = []
    
    def create_adapter(self, source_config: Dict[str, Any]) -> SourceAdapter:
        """Create appropriate adapter for source."""
        if hasattr(source_config, 'query'):
            adapter_type = 'search'
        elif hasattr(source_config, 'url'):
            adapter_type = getattr(source_config, 'type', 'html')
        else:
            raise ValueError("Invalid source configuration")

        if adapter_type == 'html':
            return HTMLAdapter(source_config)
        elif adapter_type == 'search':
            return SearchAPIAdapter(source_config)
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
            domain_limits={
                source.domain: self.settings.max_per_domain
                for source in self.sources if hasattr(source, 'domain')
            }
        )
        
        async def fetch_source(source_config: Dict[str, Any]) -> List[Article]:
            """Fetch articles from a single source."""
            source_name = getattr(source_config, 'name', getattr(source_config, 'domain', 'unknown'))
            
            async with semaphore:
                try:
                    adapter = self.create_adapter(source_config)
                    async with adapter:
                        with PerformanceLogger(f"fetch_{source_name}", logger):
                            articles = await adapter.fetch_articles(
                                start_date,
                                limit=limit_per_source
                            )
                            
                            logger.info(
                                **log_processing_stage(
                                    stage=f"fetch_{source_name}",
                                    input_count=1,
                                    output_count=len(articles)
                                )
                            )
                            
                            return articles
                            
                except Exception as e:
                    logger.error(
                        "Failed to fetch from source",
                        source=source_name,
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
