"""Content processing module."""

from .dedupe import ArticleDeduplicator, DuplicateGroup, deduplicate_articles
from .relevance import RelevanceFilter, RelevanceLevel, filter_relevance
from .scoring import ArticleScorer, ScoringWeights, SourceTier, score_articles
from .text_utils import canonical_title, clean_html_text, extract_keywords

__all__ = [
    'filter_relevance',
    'RelevanceFilter',
    'RelevanceLevel',
    'deduplicate_articles',
    'ArticleDeduplicator',
    'DuplicateGroup',
    'score_articles',
    'ArticleScorer',
    'ScoringWeights',
    'SourceTier',
    'clean_html_text',
    'canonical_title',
    'extract_keywords',
]
