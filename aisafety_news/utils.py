"""Utility functions for AI Safety Newsletter Agent."""

import asyncio
import hashlib
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urlparse

import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


def normalize_url(url: str) -> str:
    """Normalize URL for consistent processing.
    
    Args:
        url: Raw URL string
        
    Returns:
        Normalized URL
    """
    # Remove trailing slashes and fragments
    parsed = urlparse(url)
    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"

    if parsed.query:
        normalized += f"?{parsed.query}"

    return normalized


def extract_domain(url: str) -> str:
    """Extract domain from URL.
    
    Args:
        url: URL string
        
    Returns:
        Domain name
    """
    return urlparse(url).netloc.lower()


def is_valid_url(url: str) -> bool:
    """Check if URL is valid.
    
    Args:
        url: URL to validate
        
    Returns:
        True if URL is valid
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def generate_content_hash(content: str) -> str:
    """Generate SHA-256 hash of content.
    
    Args:
        content: Content to hash
        
    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def parse_date_string(date_str: str) -> datetime | None:
    """Parse various date string formats.
    
    Args:
        date_str: Date string to parse
        
    Returns:
        Parsed datetime or None if parsing fails
    """
    if not date_str:
        return None

    date_str = date_str.strip()

    # Try RFC 2822 format first (common in RSS feeds)
    # Example: "Thu, 17 Jul 2025 23:17:14 GMT"
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(date_str)
    except (ValueError, TypeError):
        pass

    # Common date formats
    formats = [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%d %b %Y",
        # Additional RFC-like formats
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S",
    ]

    for fmt in formats:
        try:
            parsed = datetime.strptime(date_str, fmt)
            # Ensure timezone info is set
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed
        except ValueError:
            continue

    logger.warning("Failed to parse date string", date_string=date_str)
    return None


def format_datetime_iso(dt: datetime) -> str:
    """Format datetime as ISO string.
    
    Args:
        dt: Datetime to format
        
    Returns:
        ISO formatted datetime string
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.isoformat()


def calculate_recency_score(
    published_date: datetime,
    reference_date: datetime | None = None,
    max_age_days: int = 30
) -> float:
    """Calculate recency score (0.0 to 1.0).
    
    Args:
        published_date: When article was published
        reference_date: Reference date (defaults to now)
        max_age_days: Maximum age for scoring
        
    Returns:
        Recency score between 0.0 and 1.0
    """
    if reference_date is None:
        reference_date = datetime.now(UTC)

    # Ensure both dates have timezone info
    if published_date.tzinfo is None:
        published_date = published_date.replace(tzinfo=UTC)
    if reference_date.tzinfo is None:
        reference_date = reference_date.replace(tzinfo=UTC)

    age_days = (reference_date - published_date).total_seconds() / 86400

    if age_days < 0:
        return 1.0  # Future dates get max score
    elif age_days > max_age_days:
        return 0.0  # Too old
    else:
        return 1.0 - (age_days / max_age_days)


def clean_text(text: str) -> str:
    """Clean and normalize text content.
    
    Args:
        text: Raw text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())

    # Remove common HTML entities
    html_entities = {
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&#39;': "'",
        '&nbsp;': ' ',
    }

    for entity, replacement in html_entities.items():
        text = text.replace(entity, replacement)

    return text


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix


def chunk_list(items: list[T], chunk_size: int) -> list[list[T]]:
    """Split list into chunks of specified size.
    
    Args:
        items: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


async def retry_async(
    func,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Any:
    """Retry async function with exponential backoff.
    
    Args:
        func: Async function to retry
        max_retries: Maximum number of retries
        backoff_factor: Backoff multiplier
        exceptions: Exceptions to catch and retry
        
    Returns:
        Function result
        
    Raises:
        Last exception if all retries fail
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                delay = backoff_factor ** attempt
                logger.warning(
                    "Retry attempt failed",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    delay=delay,
                    error=str(e)
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "All retry attempts failed",
                    max_retries=max_retries,
                    error=str(e)
                )

    raise last_exception


class RateLimiter:
    """Enhanced rate limiter for API calls with burst protection."""

    def __init__(self, max_calls: int, time_window: float, burst_limit: int | None = None):
        """Initialize rate limiter.
        
        Args:
            max_calls: Maximum calls allowed in time window
            time_window: Time window in seconds
            burst_limit: Maximum calls allowed in rapid succession (defaults to max_calls // 4)
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.burst_limit = burst_limit or max(1, max_calls // 4)
        self.calls: list[float] = []
        self.burst_window = 10.0  # 10 second burst window

    async def acquire(self) -> None:
        """Acquire rate limit permission with burst protection."""
        now = time.time()

        # Remove old calls outside the time window
        self.calls = [call_time for call_time in self.calls
                     if now - call_time < self.time_window]

        # Check burst limit (calls in last 10 seconds)
        recent_calls = [call_time for call_time in self.calls
                       if now - call_time < self.burst_window]

        if len(recent_calls) >= self.burst_limit:
            burst_wait = self.burst_window - (now - min(recent_calls))
            if burst_wait > 0:
                logger.info("Burst rate limit hit, waiting", wait_time=burst_wait)
                await asyncio.sleep(burst_wait)

        # Check regular rate limit
        if len(self.calls) >= self.max_calls:
            # Calculate wait time
            oldest_call = min(self.calls)
            wait_time = self.time_window - (now - oldest_call)
            if wait_time > 0:
                logger.debug("Rate limit hit, waiting", wait_time=wait_time)
                await asyncio.sleep(wait_time)

        # Record this call
        self.calls.append(now)


class AsyncSemaphore:
    """Enhanced async semaphore with domain-specific limits."""

    def __init__(self, global_limit: int, domain_limits: dict[str, int] | None = None):
        """Initialize semaphore.
        
        Args:
            global_limit: Global concurrency limit
            domain_limits: Per-domain concurrency limits
        """
        self.global_semaphore = asyncio.Semaphore(global_limit)
        self.domain_semaphores = {}
        self.domain_limits = domain_limits or {}
        self._current_domain = None

    def get_domain_semaphore(self, domain: str) -> asyncio.Semaphore:
        """Get semaphore for specific domain."""
        if domain not in self.domain_semaphores:
            limit = self.domain_limits.get(domain, 1)
            self.domain_semaphores[domain] = asyncio.Semaphore(limit)
        return self.domain_semaphores[domain]

    async def acquire(self, domain: str | None = None):
        """Acquire semaphore for domain."""
        # Always acquire global semaphore
        await self.global_semaphore.acquire()

        # Acquire domain-specific semaphore if specified
        if domain:
            domain_sem = self.get_domain_semaphore(domain)
            await domain_sem.acquire()
            self._current_domain = domain

    def release(self, domain: str | None = None):
        """Release semaphore for domain."""
        # Use stored domain if not provided
        if domain is None:
            domain = self._current_domain

        # Release domain-specific semaphore first
        if domain and domain in self.domain_semaphores:
            self.domain_semaphores[domain].release()

        # Release global semaphore
        self.global_semaphore.release()
        self._current_domain = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.release()


def safe_filename(filename: str) -> str:
    """Create safe filename by removing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename
    """
    # Remove or replace invalid characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    safe_name = re.sub(r'\s+', '_', safe_name)

    # Limit length
    if len(safe_name) > 200:
        safe_name = safe_name[:200]

    return safe_name


def ensure_directory(path: str | Path, mode: int = 0o700) -> Path:
    """Ensure directory exists with secure permissions.
    
    Args:
        path: Directory path
        mode: Directory permissions (default: 0o700 - owner read/write/execute only)
        
    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True, mode=mode)

    # Ensure permissions are set correctly even if directory already existed
    try:
        path_obj.chmod(mode)
    except (OSError, PermissionError) as e:
        logger.warning("Failed to set directory permissions", path=str(path_obj), mode=oct(mode), error=str(e))

    return path_obj


def file_age_days(file_path: str | Path) -> float:
    """Get file age in days.
    
    Args:
        file_path: Path to file
        
    Returns:
        Age in days
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        return float('inf')

    mtime = path_obj.stat().st_mtime
    age_seconds = time.time() - mtime
    return age_seconds / 86400


def cleanup_old_files(
    directory: str | Path,
    max_age_days: int,
    pattern: str = "*"
) -> int:
    """Clean up old files in directory.
    
    Args:
        directory: Directory to clean
        max_age_days: Maximum file age in days
        pattern: File pattern to match
        
    Returns:
        Number of files deleted
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return 0

    deleted_count = 0
    for file_path in dir_path.glob(pattern):
        if file_path.is_file() and file_age_days(file_path) > max_age_days:
            try:
                file_path.unlink()
                deleted_count += 1
                logger.debug("Deleted old file", file=str(file_path))
            except Exception as e:
                logger.warning("Failed to delete file", file=str(file_path), error=str(e))

    return deleted_count


def create_secure_file(
    file_path: str | Path,
    content: str,
    permissions: int = 0o600
) -> None:
    """Create file with secure permissions.
    
    Args:
        file_path: Path to create
        content: File content
        permissions: File permissions (default: 0o600 - owner read/write only)
    """
    path_obj = Path(file_path)

    # Ensure parent directory exists with secure permissions
    ensure_directory(path_obj.parent)

    # Write file with secure permissions
    with open(path_obj, 'w', encoding='utf-8') as f:
        f.write(content)

    try:
        path_obj.chmod(permissions)
    except (OSError, PermissionError) as e:
        logger.warning("Failed to set file permissions", file=str(path_obj), mode=oct(permissions), error=str(e))


def enforce_https_url(url: str) -> str:
    """Enforce HTTPS for URLs.
    
    Args:
        url: Input URL
        
    Returns:
        HTTPS URL
        
    Raises:
        ValueError: If URL cannot be made secure
    """
    if not url:
        raise ValueError("Empty URL provided")

    parsed = urlparse(url)

    if parsed.scheme == 'http':
        # Convert HTTP to HTTPS
        return url.replace('http://', 'https://', 1)
    elif parsed.scheme == 'https':
        return url
    elif not parsed.scheme:
        # Add HTTPS scheme if missing
        return f'https://{url}'
    else:
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")


def validate_request_size(content_length: int | None, max_size_mb: int) -> bool:
    """Validate request size against limits.
    
    Args:
        content_length: Content length in bytes
        max_size_mb: Maximum allowed size in MB
        
    Returns:
        True if size is acceptable
    """
    if content_length is None:
        return True  # Unknown size, allow

    max_bytes = max_size_mb * 1024 * 1024
    return content_length <= max_bytes
