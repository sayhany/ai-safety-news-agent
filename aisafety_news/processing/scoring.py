"""
Weighted scoring system for ranking AI safety news articles.

This module provides configurable scoring based on multiple factors:
- Relevance score (from relevance filtering)
- Source credibility
- Recency
- Content quality indicators
- Social engagement (if available)
"""

import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

from ..config import Settings

logger = logging.getLogger(__name__)


class SourceTier(Enum):
    """Source credibility tiers."""
    TIER_1 = "tier_1"  # Highly credible academic/research sources
    TIER_2 = "tier_2"  # Established tech/AI publications
    TIER_3 = "tier_3"  # General news sources
    TIER_4 = "tier_4"  # Blogs and other sources
    UNKNOWN = "unknown"


@dataclass
class ScoringWeights:
    """Configurable weights for different scoring factors."""
    relevance: float = 0.4
    source_credibility: float = 0.25
    recency: float = 0.15
    content_quality: float = 0.1
    engagement: float = 0.1


@dataclass
class ArticleScore:
    """Complete scoring breakdown for an article."""
    total_score: float
    relevance_score: float
    source_score: float
    recency_score: float
    quality_score: float
    engagement_score: float
    source_tier: SourceTier
    reasoning: str


class ArticleScorer:
    """Weighted scoring system for AI safety news articles."""

    # Source credibility mapping
    SOURCE_TIERS = {
        # Tier 1: Research institutions and academic sources
        'arxiv.org': SourceTier.TIER_1,
        'openai.com': SourceTier.TIER_1,
        'anthropic.com': SourceTier.TIER_1,
        'deepmind.com': SourceTier.TIER_1,
        'ai.googleblog.com': SourceTier.TIER_1,
        'research.facebook.com': SourceTier.TIER_1,
        'microsoft.com/en-us/research': SourceTier.TIER_1,
        'fhi.ox.ac.uk': SourceTier.TIER_1,
        'intelligence.org': SourceTier.TIER_1,
        'humancompatible.ai': SourceTier.TIER_1,
        'alignment.org': SourceTier.TIER_1,
        'lesswrong.com': SourceTier.TIER_1,

        # Tier 2: Established AI/tech publications
        'techcrunch.com': SourceTier.TIER_2,
        'venturebeat.com': SourceTier.TIER_2,
        'wired.com': SourceTier.TIER_2,
        'arstechnica.com': SourceTier.TIER_2,
        'theverge.com': SourceTier.TIER_2,
        'mit.edu': SourceTier.TIER_2,
        'stanford.edu': SourceTier.TIER_2,
        'berkeley.edu': SourceTier.TIER_2,
        'nature.com': SourceTier.TIER_2,
        'science.org': SourceTier.TIER_2,
        'ieee.org': SourceTier.TIER_2,

        # Tier 3: General news sources
        'reuters.com': SourceTier.TIER_3,
        'bbc.com': SourceTier.TIER_3,
        'cnn.com': SourceTier.TIER_3,
        'nytimes.com': SourceTier.TIER_3,
        'washingtonpost.com': SourceTier.TIER_3,
        'theguardian.com': SourceTier.TIER_3,
        'wsj.com': SourceTier.TIER_3,
        'ft.com': SourceTier.TIER_3,
        'economist.com': SourceTier.TIER_3,
        'bloomberg.com': SourceTier.TIER_3,
    }

    # Scoring multipliers for each tier
    TIER_MULTIPLIERS = {
        SourceTier.TIER_1: 1.0,
        SourceTier.TIER_2: 0.8,
        SourceTier.TIER_3: 0.6,
        SourceTier.TIER_4: 0.4,
        SourceTier.UNKNOWN: 0.3,
    }

    def __init__(self, settings: Settings, weights: ScoringWeights | None = None):
        """Initialize scorer with configurable weights."""
        self.settings = settings
        self.weights = weights or self._load_weights_from_config()

        # Scoring parameters
        self.recency_half_life_days = getattr(settings, 'recency_half_life_days', 7)
        self.min_content_length = getattr(settings, 'min_content_length', 100)
        self.max_age_days = getattr(settings, 'max_article_age_days', 30)

    def _load_weights_from_config(self) -> ScoringWeights:
        """Load scoring weights from configuration."""
        # Try to load from settings, fall back to defaults
        try:
            weights_config = getattr(self.settings, 'scoring_weights', {})
            return ScoringWeights(
                relevance=weights_config.get('relevance', 0.4),
                source_credibility=weights_config.get('source_credibility', 0.25),
                recency=weights_config.get('recency', 0.15),
                content_quality=weights_config.get('content_quality', 0.1),
                engagement=weights_config.get('engagement', 0.1),
            )
        except Exception as e:
            logger.warning(f"Failed to load scoring weights from config: {e}")
            return ScoringWeights()

    def _determine_source_tier(self, url: str) -> SourceTier:
        """Determine source credibility tier from URL."""
        if not url:
            return SourceTier.UNKNOWN

        # Extract domain
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower()

            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]

            # Check exact matches first
            if domain in self.SOURCE_TIERS:
                return self.SOURCE_TIERS[domain]

            # Check partial matches for subdomains
            for source_domain, tier in self.SOURCE_TIERS.items():
                if domain.endswith(source_domain):
                    return tier

            # Special cases for academic institutions
            if any(edu_indicator in domain for edu_indicator in ['.edu', '.ac.', 'university']):
                return SourceTier.TIER_2

            # Government sources
            if domain.endswith('.gov'):
                return SourceTier.TIER_2

            return SourceTier.TIER_4

        except Exception as e:
            logger.warning(f"Failed to parse URL {url}: {e}")
            return SourceTier.UNKNOWN

    def _calculate_relevance_score(self, article: dict) -> float:
        """Calculate relevance score (0.0 to 1.0)."""
        # Use existing relevance score if available
        relevance_score = article.get('relevance_score', 0.0)
        return max(0.0, min(1.0, relevance_score))

    def _calculate_source_score(self, article: dict) -> tuple[float, SourceTier]:
        """Calculate source credibility score (0.0 to 1.0)."""
        url = article.get('url', '')
        tier = self._determine_source_tier(url)
        multiplier = self.TIER_MULTIPLIERS[tier]

        # Base score from tier
        base_score = multiplier

        # Bonus for known AI safety sources
        ai_safety_domains = [
            'alignment.org', 'intelligence.org', 'lesswrong.com',
            'anthropic.com', 'openai.com', 'fhi.ox.ac.uk'
        ]

        if any(domain in url.lower() for domain in ai_safety_domains):
            base_score = min(1.0, base_score + 0.1)

        return base_score, tier

    def _calculate_recency_score(self, article: dict) -> float:
        """Calculate recency score using exponential decay (0.0 to 1.0)."""
        published_date = article.get('published_date')
        if not published_date:
            return 0.5  # Default for unknown dates

        try:
            # Parse date
            if isinstance(published_date, str):
                # Use the improved parse_date_string from utils
                from ..utils import parse_date_string
                published_date = parse_date_string(published_date)
                if published_date is None:
                    return 0.5  # Couldn't parse date

            # Ensure both dates have timezone info for comparison
            if published_date.tzinfo is None:
                published_date = published_date.replace(tzinfo=UTC)

            # Calculate age in days using timezone-aware datetime
            now = datetime.now(UTC)
            age_days = (now - published_date).days

            # Articles older than max_age get zero score
            if age_days > self.max_age_days:
                return 0.0

            # Exponential decay based on half-life
            decay_factor = math.exp(-age_days * math.log(2) / self.recency_half_life_days)
            return max(0.0, min(1.0, decay_factor))

        except Exception as e:
            logger.warning(f"Failed to calculate recency for article: {e}")
            return 0.5

    def _calculate_quality_score(self, article: dict) -> float:
        """Calculate content quality score (0.0 to 1.0)."""
        score = 0.0

        # Content length (longer articles generally higher quality)
        content = article.get('content', '')
        content_length = len(content)

        if content_length >= self.min_content_length:
            # Logarithmic scaling for length
            length_score = min(1.0, math.log(content_length / self.min_content_length + 1) / math.log(10))
            score += 0.4 * length_score

        # Title quality (not too short, not too long)
        title = article.get('title', '')
        title_length = len(title)
        if 20 <= title_length <= 150:
            score += 0.2

        # Has summary/description
        if article.get('summary') or article.get('description'):
            score += 0.2

        # Has author information
        if article.get('author'):
            score += 0.1

        # Content quality indicators
        if content:
            # Check for structured content
            if any(indicator in content.lower() for indicator in ['abstract', 'introduction', 'conclusion', 'references']):
                score += 0.1

        return max(0.0, min(1.0, score))

    def _calculate_engagement_score(self, article: dict) -> float:
        """Calculate social engagement score (0.0 to 1.0)."""
        # This is a placeholder - in practice, you'd integrate with social APIs
        engagement_indicators = [
            'shares', 'likes', 'comments', 'retweets', 'upvotes',
            'social_score', 'engagement_count'
        ]

        total_engagement = 0
        for indicator in engagement_indicators:
            value = article.get(indicator, 0)
            if isinstance(value, (int, float)):
                total_engagement += value

        if total_engagement == 0:
            return 0.3  # Default score for no engagement data

        # Logarithmic scaling for engagement
        engagement_score = min(1.0, math.log(total_engagement + 1) / math.log(1000))
        return engagement_score

    def score_article(self, article: dict) -> ArticleScore:
        """Calculate comprehensive score for an article."""
        # Calculate individual scores
        relevance_score = self._calculate_relevance_score(article)
        source_score, source_tier = self._calculate_source_score(article)
        recency_score = self._calculate_recency_score(article)
        quality_score = self._calculate_quality_score(article)
        engagement_score = self._calculate_engagement_score(article)

        # Calculate weighted total
        total_score = (
            self.weights.relevance * relevance_score +
            self.weights.source_credibility * source_score +
            self.weights.recency * recency_score +
            self.weights.content_quality * quality_score +
            self.weights.engagement * engagement_score
        )

        # Generate reasoning
        reasoning_parts = [
            f"Relevance: {relevance_score:.2f}",
            f"Source ({source_tier.value}): {source_score:.2f}",
            f"Recency: {recency_score:.2f}",
            f"Quality: {quality_score:.2f}",
            f"Engagement: {engagement_score:.2f}"
        ]
        reasoning = " | ".join(reasoning_parts)

        return ArticleScore(
            total_score=total_score,
            relevance_score=relevance_score,
            source_score=source_score,
            recency_score=recency_score,
            quality_score=quality_score,
            engagement_score=engagement_score,
            source_tier=source_tier,
            reasoning=reasoning
        )

    def score_articles(self, articles: list[dict]) -> list[dict]:
        """Score and rank a list of articles."""
        logger.info(f"Scoring {len(articles)} articles")

        # Handle empty article list
        if not articles:
            logger.info("No articles to score")
            return []

        # Score all articles
        scored_articles = []
        for article in articles:
            score = self.score_article(article)

            # Add score information to article
            article['total_score'] = score.total_score
            article['score_breakdown'] = {
                'relevance': score.relevance_score,
                'source': score.source_score,
                'recency': score.recency_score,
                'quality': score.quality_score,
                'engagement': score.engagement_score,
            }
            article['source_tier'] = score.source_tier.value
            article['score_reasoning'] = score.reasoning

            scored_articles.append(article)

        # Sort by total score (descending)
        scored_articles.sort(key=lambda x: x['total_score'], reverse=True)

        # Add ranking
        for i, article in enumerate(scored_articles):
            article['rank'] = i + 1

        if scored_articles:
            logger.info(f"Scoring complete. Top score: {scored_articles[0]['total_score']:.3f}")
        else:
            logger.info("Scoring complete. No articles scored.")
        return scored_articles


def score_articles(articles: list[dict], settings: Settings,
                  weights: ScoringWeights | None = None) -> list[dict]:
    """Convenience function for article scoring."""
    scorer = ArticleScorer(settings, weights)
    return scorer.score_articles(articles)
