"""Security utilities and middleware for AI Safety Newsletter Agent."""

from aiohttp.web import Request, Response, middleware

from .logging import get_logger

logger = get_logger(__name__)


def get_security_headers() -> dict[str, str]:
    """Get security headers for HTTP responses.
    
    Returns:
        Dictionary of security headers
    """
    return {
        # Prevent XSS attacks
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',

        # HTTPS enforcement
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains; preload',

        # Content Security Policy
        'Content-Security-Policy': (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "font-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self';"
        ),

        # Referrer policy
        'Referrer-Policy': 'strict-origin-when-cross-origin',

        # Permissions policy
        'Permissions-Policy': (
            'geolocation=(), microphone=(), camera=(), '
            'payment=(), usb=(), magnetometer=(), gyroscope=()'
        ),

        # Cache control for sensitive data
        'Cache-Control': 'no-store, no-cache, must-revalidate, private',
        'Pragma': 'no-cache',
        'Expires': '0'
    }


@middleware
async def security_middleware(request: Request, handler) -> Response:
    """Add security headers to all responses.
    
    Args:
        request: HTTP request
        handler: Request handler
        
    Returns:
        HTTP response with security headers
    """
    # Log security-relevant request info
    logger.debug(
        "Security middleware processing request",
        method=request.method,
        path=request.path,
        remote=request.remote,
        user_agent=request.headers.get('User-Agent', 'Unknown')
    )

    # Check for suspicious patterns
    if _is_suspicious_request(request):
        logger.warning(
            "Suspicious request detected",
            method=request.method,
            path=request.path,
            remote=request.remote,
            user_agent=request.headers.get('User-Agent')
        )
        # Could implement rate limiting or blocking here

    # Process the request
    response = await handler(request)

    # Add security headers
    security_headers = get_security_headers()
    for header_name, header_value in security_headers.items():
        response.headers[header_name] = header_value

    # Add server header obfuscation
    response.headers['Server'] = 'AI-Safety-News-Agent'

    return response


def _is_suspicious_request(request: Request) -> bool:
    """Check if request contains suspicious patterns.
    
    Args:
        request: HTTP request to check
        
    Returns:
        True if request appears suspicious
    """
    suspicious_patterns = [
        # Common attack patterns
        '../', '..\\', '<script', 'javascript:', 'vbscript:',
        'onload=', 'onerror=', 'eval(', 'exec(', 'system(',
        'union select', 'drop table', 'insert into',
        # Path traversal
        '../../../../', '..%2F', '%2e%2e%2f',
        # SQL injection
        "' or '1'='1", '" or "1"="1', 'or 1=1--',
        # XSS
        'alert(', 'confirm(', 'prompt(',
        # Command injection
        ';cat ', '|cat ', '`cat ', '$(cat ',
    ]

    # Check path and query parameters
    full_path = str(request.url)
    path_lower = full_path.lower()

    for pattern in suspicious_patterns:
        if pattern in path_lower:
            return True

    # Check for excessive path length
    if len(request.path) > 1000:
        return True

    # Check for too many parameters
    if len(request.query) > 50:
        return True

    return False


class SecurityValidator:
    """Validate security aspects of data and requests."""

    @staticmethod
    def validate_url_safety(url: str) -> bool:
        """Validate URL for security concerns.
        
        Args:
            url: URL to validate
            
        Returns:
            True if URL appears safe
        """
        if not url:
            return False

        url_lower = url.lower()

        # Block dangerous protocols
        dangerous_protocols = [
            'file://', 'ftp://', 'javascript:', 'data:',
            'vbscript:', 'mailto:', 'tel:', 'sms:'
        ]

        for protocol in dangerous_protocols:
            if url_lower.startswith(protocol):
                logger.warning("Blocked dangerous protocol in URL", url=url, protocol=protocol)
                return False

        # Block local/internal addresses
        local_indicators = [
            'localhost', '127.0.0.1', '0.0.0.0',
            '10.0.0.', '192.168.', '172.16.', '172.31.',
            '169.254.', '::1', 'fc00:', 'fe80:'
        ]

        for indicator in local_indicators:
            if indicator in url_lower:
                logger.warning("Blocked internal/local URL", url=url)
                return False

        return True

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for secure storage.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        import re

        # Remove path components
        filename = filename.split('/')[-1].split('\\')[-1]

        # Remove dangerous characters
        filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)

        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = name[:250] + '.' + ext if ext else name[:255]

        # Ensure it's not empty or just dots
        if not filename or filename.replace('.', '').replace('_', '') == '':
            filename = 'unnamed_file'

        return filename

    @staticmethod
    def validate_content_type(content_type: str, allowed_types: list) -> bool:
        """Validate content type against allowed list.
        
        Args:
            content_type: Content type to validate
            allowed_types: List of allowed content types
            
        Returns:
            True if content type is allowed
        """
        if not content_type:
            return False

        # Extract main type (before semicolon)
        main_type = content_type.split(';')[0].strip().lower()

        return main_type in [t.lower() for t in allowed_types]


def create_secure_response(
    content: str,
    content_type: str = 'application/json',
    status: int = 200,
    additional_headers: dict[str, str] | None = None
) -> Response:
    """Create HTTP response with security headers.
    
    Args:
        content: Response content
        content_type: Content type
        status: HTTP status code
        additional_headers: Additional headers to include
        
    Returns:
        Secure HTTP response
    """
    headers = get_security_headers()

    if additional_headers:
        headers.update(additional_headers)

    response = Response(
        text=content,
        content_type=content_type,
        status=status,
        headers=headers
    )

    return response
