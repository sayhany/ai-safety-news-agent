"""
Relevance filtering for AI safety news articles.

This module provides both keyword-based and LLM-based relevance filtering
to identify articles related to AI safety, alignment, and governance.
"""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum

from ..config import Settings
from ..models.llm_client import LLMClient
from .text_utils import clean_html_text

logger = logging.getLogger(__name__)


class RelevanceLevel(Enum):
    """Relevance levels for AI safety content."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    IRRELEVANT = "irrelevant"


@dataclass
class RelevanceScore:
    """Relevance scoring result."""
    level: RelevanceLevel
    score: float  # 0.0 to 1.0
    keyword_score: float
    llm_score: float | None
    matched_keywords: list[str]
    reasoning: str | None = None


class RelevanceFilter:
    """AI safety relevance filtering with keyword and LLM-based scoring."""

    # Core AI safety keywords with weights
    CORE_KEYWORDS = {
        # Direct AI safety terms
        "ai safety": 3.0,
        "artificial intelligence safety": 3.0,
        "ai alignment": 3.0,
        "ai risk": 2.5,
        "existential risk": 2.5,
        "superintelligence": 2.5,
        "artificial general intelligence": 2.5,
        "agi": 2.5,
        "ai governance": 2.0,
        "ai ethics": 2.0,
        "responsible ai": 2.0,
        "ai regulation": 2.0,
        "ai policy": 2.0,

        # Technical safety concepts
        "reward hacking": 2.5,
        "mesa optimization": 2.5,
        "inner alignment": 2.5,
        "outer alignment": 2.5,
        "value learning": 2.0,
        "interpretability": 2.0,
        "explainable ai": 2.0,
        "ai transparency": 2.0,
        "robustness": 1.5,
        "adversarial examples": 1.5,
        "distributional shift": 1.5,

        # Organizations and researchers
        "anthropic": 1.5,
        "openai": 1.5,
        "deepmind": 1.5,
        "future of humanity institute": 2.0,
        "center for ai safety": 2.0,
        "machine intelligence research institute": 2.0,
        "miri": 2.0,
        "chai": 2.0,
        "stuart russell": 1.5,
        "yoshua bengio": 1.5,
        "geoffrey hinton": 1.5,
        "yann lecun": 1.5,
        "eliezer yudkowsky": 2.0,
        "nick bostrom": 2.0,

        # Broader AI terms (lower weight)
        "machine learning": 0.5,
        "deep learning": 0.5,
        "neural network": 0.5,
        "artificial intelligence": 0.8,
        "automation": 0.3,
        "algorithm": 0.3,
    }

    # Negative keywords that reduce relevance
    NEGATIVE_KEYWORDS = {
        "cryptocurrency": -1.0,
        "bitcoin": -1.0,
        "blockchain": -0.5,
        "nft": -1.0,
        "gaming": -0.5,
        "entertainment": -0.5,
        "sports": -1.0,
        "fashion": -1.0,
        "celebrity": -1.0,
    }

    def __init__(self, settings: Settings, llm_client: LLMClient | None = None):
        """Initialize relevance filter."""
        self.settings = settings
        self.llm_client = llm_client
        self.keyword_threshold = getattr(settings, 'relevance_keyword_threshold', 1.0)
        self.llm_threshold = getattr(settings, 'relevance_llm_threshold', 0.6)
        self.use_llm = getattr(settings, 'use_llm_relevance', True) and llm_client is not None
        self.llm_only = getattr(settings, 'llm_only_relevance', False) and self.use_llm

        # Compile keyword patterns for efficiency (skip if LLM-only mode)
        if not self.llm_only:
            self._compile_keywords()

    def _compile_keywords(self) -> None:
        """Compile keyword patterns for efficient matching."""
        self.all_keywords = {**self.CORE_KEYWORDS, **self.NEGATIVE_KEYWORDS}
        self.keyword_patterns = {}

        for keyword in self.all_keywords:
            # Create variations for better matching
            variations = [
                keyword.lower(),
                keyword.replace(" ", "-"),
                keyword.replace(" ", "_"),
            ]
            self.keyword_patterns[keyword] = variations

    def _calculate_keyword_score(self, text: str) -> tuple[float, list[str]]:
        """Calculate keyword-based relevance score."""
        text_lower = text.lower()
        matched_keywords = []
        total_score = 0.0

        for keyword, weight in self.all_keywords.items():
            # Check all variations of the keyword
            for pattern in self.keyword_patterns[keyword]:
                if pattern in text_lower:
                    matched_keywords.append(keyword)
                    total_score += weight
                    break  # Don't double-count the same keyword

        # Normalize score with more reasonable scaling
        # Use a smaller divisor to make scores more achievable
        if total_score > 0:
            # Scale based on highest individual keywords rather than all possible
            max_reasonable_score = 10.0  # Achievable with a few high-value keywords
            normalized_score = min(1.0, total_score / max_reasonable_score)
        else:
            normalized_score = 0.0

        return normalized_score, matched_keywords

    async def _calculate_llm_score(self, title: str, content: str) -> tuple[float, str]:
        """Calculate LLM-based relevance score."""
        if not self.llm_client:
            return 0.0, "LLM not available"

        # Truncate content for efficiency and remove problematic characters
        truncated_content = content[:1000] if len(content) > 1000 else content
        # Clean content of problematic characters
        import re
        truncated_content = re.sub(r'[^\w\s\.\,\!\?\-\(\)]', ' ', truncated_content)
        truncated_content = ' '.join(truncated_content.split())  # Normalize whitespace

        prompt = f"""Rate this article's relevance to AI safety on a scale of 0.0 to 1.0.

AI safety includes: AI alignment, AI governance, AI regulation, AI ethics, AI risk assessment, algorithmic bias, AI transparency, responsible AI development, existential risk from AI.

Title: {title}

Content: {truncated_content}

Respond with ONLY a number between 0.0 and 1.0 (like 0.7), nothing else."""

        try:
            llm_response = await self.llm_client.chat(
                messages=prompt,
                max_tokens=10,  # Very short response needed
                temperature=0.0  # More deterministic
            )

            # Parse response more robustly
            response_text = llm_response.content.strip()

            # Extract score using regex
            import re
            # Look for decimal numbers
            match = re.search(r'(\d+\.?\d*)', response_text)
            if match:
                score = float(match.group(1))
                # Handle different scales
                if score > 1.0:
                    if score <= 10.0:  # 0-10 scale
                        score = score / 10.0
                    elif score <= 100.0:  # 0-100 scale
                        score = score / 100.0
                    else:
                        score = 1.0
                score = max(0.0, min(1.0, score))
            else:
                logger.warning(f"Could not parse LLM score from: {response_text}")
                score = 0.0

            reasoning = f"LLM response: {response_text}"
            return score, reasoning

        except Exception as e:
            logger.warning(f"LLM relevance scoring failed: {e}")
            # Return a small positive score for keyword matches as fallback
            return 0.0, f"LLM Error: {str(e)}"

    async def score_relevance(self, title: str, content: str, url: str = "") -> RelevanceScore:
        """Score article relevance using both keyword and LLM methods."""
        # Clean and prepare text
        clean_title = clean_html_text(title)
        clean_content = clean_html_text(content)
        combined_text = f"{clean_title} {clean_content}"

        if self.llm_only:
            # LLM-only mode: Skip keyword processing entirely
            if self.use_llm:
                llm_score, reasoning = await self._calculate_llm_score(clean_title, clean_content)
                final_score = llm_score if llm_score is not None else 0.0
            else:
                final_score = 0.0
                reasoning = "LLM-only mode but no LLM available"
        else:
            # Hybrid mode: Calculate both keyword and LLM scores
            keyword_score, matched_keywords = self._calculate_keyword_score(combined_text)

            # Calculate LLM score if enabled
            llm_score = None
            reasoning = None

            if self.use_llm:
                llm_score, reasoning = await self._calculate_llm_score(clean_title, clean_content)

            # Combine scores with improved logic
            if llm_score is not None and llm_score > 0:
                # If LLM provides a score, use weighted combination favoring LLM
                # 20% keyword (basic filtering), 80% LLM (more accurate AI safety detection)
                final_score = 0.2 * keyword_score + 0.8 * llm_score
            else:
                # If LLM fails or returns 0, use keyword score with small boost for AI mentions
                final_score = keyword_score
                # Small boost if article mentions AI-related terms even if not safety-specific
                text_lower = f"{title} {content}".lower()
                if any(term in text_lower for term in ['artificial intelligence', 'ai ', 'machine learning']):
                    final_score += 0.1  # Small relevance boost

        # Determine relevance level
        if final_score >= 0.8:
            level = RelevanceLevel.HIGH
        elif final_score >= 0.6:
            level = RelevanceLevel.MEDIUM
        elif final_score >= 0.3:
            level = RelevanceLevel.LOW
        else:
            level = RelevanceLevel.IRRELEVANT

        return RelevanceScore(
            level=level,
            score=final_score,
            keyword_score=keyword_score,
            llm_score=llm_score,
            matched_keywords=matched_keywords,
            reasoning=reasoning
        )

    async def filter_articles(self, articles: list[dict]) -> list[dict]:
        """Filter articles by relevance, adding relevance scores."""
        logger.info(f"Filtering {len(articles)} articles for AI safety relevance")

        # Score all articles concurrently
        tasks = []
        for article in articles:
            task = self.score_relevance(
                title=article.get('title', ''),
                content=article.get('content', ''),
                url=article.get('url', '')
            )
            tasks.append(task)

        # Execute scoring sequentially to avoid API issues
        scores = []
        for i, task in enumerate(tasks):
            try:
                logger.debug(f"Processing article {i+1}/{len(tasks)}")
                score = await task
                scores.append(score)
                # Small delay to avoid overwhelming API
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Failed to score article {i+1}: {e}")
                # Create a default score for failed articles
                from . import RelevanceLevel, RelevanceScore
                default_score = RelevanceScore(
                    level=RelevanceLevel.IRRELEVANT,
                    score=0.0,
                    keyword_score=0.0,
                    llm_score=None,
                    matched_keywords=[],
                    reasoning=f"LLM Error: {str(e)}"
                )
                scores.append(default_score)

        # Add scores to articles and filter
        filtered_articles = []
        for article, score in zip(articles, scores, strict=False):
            article['relevance_score'] = score.score
            article['relevance_level'] = score.level.value
            article['matched_keywords'] = score.matched_keywords
            article['llm_reasoning'] = score.reasoning

            # Filter based on minimum threshold
            min_threshold = getattr(self.settings, 'min_relevance_score', 0.2)
            if score.score >= min_threshold:
                filtered_articles.append(article)

        logger.info(f"Filtered to {len(filtered_articles)} relevant articles")
        return filtered_articles


async def filter_relevance(articles: list[dict], settings: Settings,
                          llm_client: LLMClient | None = None) -> list[dict]:
    """Convenience function for relevance filtering."""
    filter_instance = RelevanceFilter(settings, llm_client)
    return await filter_instance.filter_articles(articles)
