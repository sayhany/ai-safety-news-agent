"""Text processing utilities for AI Safety Newsletter Agent."""

import re
from unicodedata import normalize

from ..logging import get_logger

logger = get_logger(__name__)


def canonical_title(title: str) -> str:
    """Convert title to canonical form for deduplication.
    
    Args:
        title: Article title
        
    Returns:
        Canonical title (lowercase, alphanumeric only)
    """
    if not title:
        return ""

    # Normalize unicode
    title = normalize('NFKD', title)

    # Convert to lowercase
    title = title.lower()

    # Remove punctuation and special characters
    title = re.sub(r'[^\w\s]', '', title)

    # Remove extra whitespace
    title = re.sub(r'\s+', '', title)

    return title


def extract_keywords(text: str, min_length: int = 3) -> set[str]:
    """Extract keywords from text.
    
    Args:
        text: Input text
        min_length: Minimum keyword length
        
    Returns:
        Set of keywords
    """
    if not text:
        return set()

    # Convert to lowercase and normalize
    text = normalize('NFKD', text.lower())

    # Extract words
    words = re.findall(r'\b\w+\b', text)

    # Filter by length and remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'among', 'this', 'that',
        'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'shall', 'it', 'he', 'she',
        'they', 'we', 'you', 'i', 'me', 'him', 'her', 'them', 'us', 'my',
        'your', 'his', 'its', 'our', 'their'
    }

    keywords = {
        word for word in words
        if len(word) >= min_length and word not in stop_words
    }

    return keywords


def clean_html_text(html_text: str) -> str:
    """Clean HTML text content.
    
    Args:
        html_text: HTML text
        
    Returns:
        Cleaned plain text
    """
    if not html_text:
        return ""

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', html_text)

    # Decode HTML entities
    html_entities = {
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&#39;': "'",
        '&nbsp;': ' ',
        '&mdash;': '—',
        '&ndash;': '–',
        '&hellip;': '…',
        '&lsquo;': ''',
        '&rsquo;': ''',
        '&ldquo;': '"',
        '&rdquo;': '"',
    }

    for entity, replacement in html_entities.items():
        text = text.replace(entity, replacement)

    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text.strip())

    return text


def extract_sentences(text: str, max_sentences: int = 5) -> list[str]:
    """Extract sentences from text.
    
    Args:
        text: Input text
        max_sentences: Maximum number of sentences
        
    Returns:
        List of sentences
    """
    if not text:
        return []

    # Simple sentence splitting (can be improved with NLTK)
    sentences = re.split(r'[.!?]+', text)

    # Clean and filter sentences
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 10:  # Minimum sentence length
            cleaned_sentences.append(sentence)

        if len(cleaned_sentences) >= max_sentences:
            break

    return cleaned_sentences


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity using Jaccard index.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score (0.0 to 1.0)
    """
    if not text1 or not text2:
        return 0.0

    # Extract keywords from both texts
    keywords1 = extract_keywords(text1)
    keywords2 = extract_keywords(text2)

    if not keywords1 and not keywords2:
        return 0.0

    # Calculate Jaccard similarity
    intersection = keywords1.intersection(keywords2)
    union = keywords1.union(keywords2)

    if not union:
        return 0.0

    return len(intersection) / len(union)


def contains_ai_safety_keywords(
    text: str,
    primary_keywords: list[str],
    secondary_keywords: list[str],
    primary_weight: float = 1.0,
    secondary_weight: float = 0.5
) -> float:
    """Check if text contains AI safety keywords.
    
    Args:
        text: Text to check
        primary_keywords: Primary AI safety keywords
        secondary_keywords: Secondary AI safety keywords
        primary_weight: Weight for primary keywords
        secondary_weight: Weight for secondary keywords
        
    Returns:
        Relevance score (0.0 to 1.0)
    """
    if not text:
        return 0.0

    text_lower = text.lower()
    score = 0.0
    max_score = 0.0

    # Check primary keywords
    for keyword in primary_keywords:
        max_score += primary_weight
        if keyword.lower() in text_lower:
            score += primary_weight

    # Check secondary keywords
    for keyword in secondary_keywords:
        max_score += secondary_weight
        if keyword.lower() in text_lower:
            score += secondary_weight

    return score / max_score if max_score > 0 else 0.0


def generate_summary_bullets(text: str, max_bullets: int = 3, max_words: int = 35) -> list[str]:
    """Generate bullet point summary from text.
    
    Args:
        text: Input text
        max_bullets: Maximum number of bullets
        max_words: Maximum words per bullet
        
    Returns:
        List of bullet points
    """
    if not text:
        return []

    sentences = extract_sentences(text, max_sentences=max_bullets * 2)
    bullets = []

    for sentence in sentences:
        # Clean and truncate sentence
        words = sentence.split()
        if len(words) > max_words:
            words = words[:max_words]
            sentence = ' '.join(words) + '...'

        bullets.append(sentence)

        if len(bullets) >= max_bullets:
            break

    return bullets


def validate_headline_length(headline: str, max_chars: int = 100) -> bool:
    """Validate headline length.
    
    Args:
        headline: Headline text
        max_chars: Maximum characters allowed
        
    Returns:
        True if headline is valid length
    """
    return len(headline) <= max_chars


def format_headline(headline: str, max_chars: int = 100) -> str:
    """Format and truncate headline.
    
    Args:
        headline: Raw headline
        max_chars: Maximum characters
        
    Returns:
        Formatted headline
    """
    if not headline:
        return ""

    # Clean the headline
    headline = clean_html_text(headline).strip()

    # Truncate if necessary
    if len(headline) > max_chars:
        # Try to truncate at word boundary
        words = headline.split()
        truncated = ""
        for word in words:
            if len(truncated + " " + word) <= max_chars - 3:  # Leave room for "..."
                if truncated:
                    truncated += " "
                truncated += word
            else:
                break

        if truncated:
            headline = truncated + "..."
        else:
            headline = headline[:max_chars - 3] + "..."

    return headline


def extract_domain_from_url(url: str) -> str:
    """Extract domain from URL for source identification.
    
    Args:
        url: URL string
        
    Returns:
        Domain name
    """
    from urllib.parse import urlparse

    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Remove 'www.' prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]

        return domain
    except Exception:
        return ""


def is_government_source(domain: str) -> bool:
    """Check if domain is a government source.
    
    Args:
        domain: Domain name
        
    Returns:
        True if government source
    """
    gov_tlds = {'.gov', '.mil', '.edu'}
    gov_domains = {
        'whitehouse.gov', 'nist.gov', 'fda.gov', 'sec.gov', 'ftc.gov',
        'congress.gov', 'senate.gov', 'house.gov', 'supremecourt.gov',
        'europa.eu', 'ec.europa.eu', 'parliament.uk', 'gov.uk'
    }

    # Check TLD
    for tld in gov_tlds:
        if domain.endswith(tld):
            return True

    # Check specific domains
    return domain in gov_domains


def calculate_readability_score(text: str) -> float:
    """Calculate simple readability score.
    
    Args:
        text: Text to analyze
        
    Returns:
        Readability score (higher is more readable)
    """
    if not text:
        return 0.0

    sentences = len(re.split(r'[.!?]+', text))
    words = len(text.split())

    if sentences == 0 or words == 0:
        return 0.0

    # Simple metric: prefer shorter sentences
    avg_sentence_length = words / sentences

    # Score inversely related to sentence length
    # Optimal range: 15-20 words per sentence
    if 15 <= avg_sentence_length <= 20:
        return 1.0
    elif avg_sentence_length < 15:
        return 0.8 + (avg_sentence_length / 15) * 0.2
    else:
        return max(0.1, 1.0 - (avg_sentence_length - 20) / 30)


if __name__ == "__main__":
    # Test functions
    test_title = "OpenAI Debuts GPT-4o?!!!"
    print(f"Canonical title: {canonical_title(test_title)}")

    test_text = "This is a test article about AI safety and machine learning."
    keywords = extract_keywords(test_text)
    print(f"Keywords: {keywords}")

    bullets = generate_summary_bullets(test_text)
    print(f"Bullets: {bullets}")
