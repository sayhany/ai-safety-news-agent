"""
Newsletter rendering system for generating Markdown newsletters.

This module provides flexible template-based rendering for AI safety newsletters
with different formats and customization options.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

try:
    from jinja2 import Environment, FileSystemLoader, Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    Environment = FileSystemLoader = Template = None

from .config import Settings

logger = logging.getLogger(__name__)


class NewsletterFormat(Enum):
    """Newsletter format types."""
    DAILY = "daily"
    WEEKLY = "weekly"
    DIGEST = "digest"
    RESEARCH = "research"
    BRIEF = "brief"


@dataclass
class NewsletterConfig:
    """Configuration for newsletter generation."""
    format_type: NewsletterFormat = NewsletterFormat.DAILY
    max_articles: int = 20
    include_summaries: bool = True
    include_key_points: bool = True
    include_implications: bool = True
    include_scores: bool = False
    group_by_category: bool = True
    show_source_tiers: bool = True


@dataclass
class NewsletterMetadata:
    """Newsletter metadata."""
    title: str
    date: datetime
    article_count: int
    top_score: float
    coverage_period: str
    generation_time: datetime


class NewsletterRenderer:
    """Markdown newsletter rendering system."""

    def __init__(self, settings: Settings, template_dir: Path | None = None):
        """Initialize renderer."""
        self.settings = settings

        # Template directory
        if template_dir is None:
            template_dir = Path(__file__).parent.parent / 'templates'
        self.template_dir = Path(template_dir)

        # Initialize Jinja2 environment
        if JINJA2_AVAILABLE:
            self.jinja_env = Environment(
                loader=FileSystemLoader(str(self.template_dir)),
                trim_blocks=True,
                lstrip_blocks=True
            )
            self._register_filters()
        else:
            logger.warning("Jinja2 not available, using fallback rendering")
            self.jinja_env = None

    def _register_filters(self) -> None:
        """Register custom Jinja2 filters."""
        if not self.jinja_env:
            return

        def format_date(date_obj, format_str='%B %d, %Y'):
            """Format date object."""
            if isinstance(date_obj, str):
                try:
                    date_obj = datetime.fromisoformat(date_obj.replace('Z', '+00:00'))
                except ValueError:
                    return date_obj
            return date_obj.strftime(format_str)

        def format_score(score, precision=2):
            """Format score with specified precision."""
            return f"{score:.{precision}f}"

        def truncate_words(text, word_count=50):
            """Truncate text to specified word count."""
            words = text.split()
            if len(words) <= word_count:
                return text
            return ' '.join(words[:word_count]) + '...'

        def source_tier_emoji(tier):
            """Convert source tier to emoji."""
            tier_emojis = {
                'tier_1': '🏆',
                'tier_2': '⭐',
                'tier_3': '📰',
                'tier_4': '📝',
                'unknown': '❓'
            }
            return tier_emojis.get(tier, '❓')

        def relevance_emoji(level):
            """Convert relevance level to emoji."""
            level_emojis = {
                'high': '🔥',
                'medium': '⚡',
                'low': '💡',
                'irrelevant': '❌'
            }
            return level_emojis.get(level, '❓')

        self.jinja_env.filters['format_date'] = format_date
        self.jinja_env.filters['format_score'] = format_score
        self.jinja_env.filters['truncate_words'] = truncate_words
        self.jinja_env.filters['source_tier_emoji'] = source_tier_emoji
        self.jinja_env.filters['relevance_emoji'] = relevance_emoji

    def _categorize_articles(self, articles: list[dict]) -> dict[str, list[dict]]:
        """Categorize articles by topic/theme."""
        categories = {
            'AI Safety Research': [],
            'AI Governance & Policy': [],
            'Technical Developments': [],
            'Industry News': [],
            'Other': []
        }

        # Keywords for categorization
        category_keywords = {
            'AI Safety Research': [
                'ai safety', 'alignment', 'interpretability', 'robustness',
                'adversarial', 'mesa optimization', 'reward hacking',
                'value learning', 'existential risk'
            ],
            'AI Governance & Policy': [
                'regulation', 'policy', 'governance', 'ethics',
                'responsible ai', 'ai act', 'government', 'legislation'
            ],
            'Technical Developments': [
                'model', 'training', 'architecture', 'performance',
                'benchmark', 'dataset', 'algorithm', 'breakthrough'
            ],
            'Industry News': [
                'company', 'funding', 'partnership', 'acquisition',
                'product', 'launch', 'announcement', 'investment'
            ]
        }

        for article in articles:
            title = article.get('title', '').lower()
            content = article.get('content', '').lower()
            text = f"{title} {content}"

            categorized = False
            for category, keywords in category_keywords.items():
                if any(keyword in text for keyword in keywords):
                    categories[category].append(article)
                    categorized = True
                    break

            if not categorized:
                categories['Other'].append(article)

        # Remove empty categories
        return {k: v for k, v in categories.items() if v}

    def _generate_metadata(self, articles: list[dict],
                          config: NewsletterConfig) -> NewsletterMetadata:
        """Generate newsletter metadata."""
        now = datetime.now()

        # Determine title based on format
        format_titles = {
            NewsletterFormat.DAILY: f"AI Safety Daily - {now.strftime('%B %d, %Y')}",
            NewsletterFormat.WEEKLY: f"AI Safety Weekly - Week of {now.strftime('%B %d, %Y')}",
            NewsletterFormat.DIGEST: f"AI Safety Digest - {now.strftime('%B %Y')}",
            NewsletterFormat.RESEARCH: f"AI Safety Research Update - {now.strftime('%B %d, %Y')}",
            NewsletterFormat.BRIEF: f"AI Safety Brief - {now.strftime('%B %d, %Y')}"
        }

        title = format_titles.get(config.format_type, "AI Safety Newsletter")

        # Calculate coverage period
        if articles:
            dates = []
            for article in articles:
                pub_date = article.get('published_date')
                if pub_date:
                    try:
                        if isinstance(pub_date, str):
                            pub_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                        dates.append(pub_date)
                    except ValueError:
                        continue

            if dates:
                min_date = min(dates)
                max_date = max(dates)
                if min_date.date() == max_date.date():
                    coverage_period = min_date.strftime('%B %d, %Y')
                else:
                    coverage_period = f"{min_date.strftime('%B %d')} - {max_date.strftime('%B %d, %Y')}"
            else:
                coverage_period = now.strftime('%B %d, %Y')
        else:
            coverage_period = now.strftime('%B %d, %Y')

        # Get top score
        top_score = max((article.get('total_score', 0.0) for article in articles), default=0.0)

        return NewsletterMetadata(
            title=title,
            date=now,
            article_count=len(articles),
            top_score=top_score,
            coverage_period=coverage_period,
            generation_time=now
        )

    def _render_with_jinja2(self, template_name: str, context: dict[str, Any]) -> str:
        """Render newsletter using Jinja2 template."""
        try:
            template = self.jinja_env.get_template(template_name)
            return template.render(**context)
        except Exception as e:
            logger.error(f"Jinja2 rendering failed: {e}")
            return self._render_fallback(context)

    def _render_fallback(self, context: dict[str, Any]) -> str:
        """Fallback rendering without Jinja2."""
        metadata = context['metadata']
        articles = context['articles']
        config = context['config']

        # Build markdown manually
        lines = [
            f"# {metadata.title}",
            "",
            f"*Generated on {metadata.generation_time.strftime('%B %d, %Y at %I:%M %p')}*",
            f"*Coverage: {metadata.coverage_period}*",
            f"*Articles: {metadata.article_count}*",
            "",
            "---",
            ""
        ]

        if config.group_by_category:
            categories = context.get('categories', {'Articles': articles})
            for category, cat_articles in categories.items():
                if not cat_articles:
                    continue

                lines.extend([
                    f"## {category}",
                    ""
                ])

                for i, article in enumerate(cat_articles[:config.max_articles], 1):
                    lines.extend(self._render_article_fallback(article, config, i))
                    lines.append("")
        else:
            lines.extend([
                "## Articles",
                ""
            ])

            for i, article in enumerate(articles[:config.max_articles], 1):
                lines.extend(self._render_article_fallback(article, config, i))
                lines.append("")

        # Footer
        lines.extend([
            "---",
            "",
            "*Newsletter generated by AI Safety News Agent*",
            f"*{metadata.generation_time.strftime('%Y-%m-%d %H:%M:%S')}*"
        ])

        return '\n'.join(lines)

    def _render_article_fallback(self, article: dict, config: NewsletterConfig, index: int) -> list[str]:
        """Render single article in fallback mode."""
        lines = []

        # Title with optional score and tier
        title = article.get('title', 'Untitled')
        if config.show_source_tiers:
            tier = article.get('source_tier', 'unknown')
            tier_emoji = {'tier_1': '🏆', 'tier_2': '⭐', 'tier_3': '📰', 'tier_4': '📝'}.get(tier, '❓')
            title = f"{tier_emoji} {title}"

        if config.include_scores:
            score = article.get('total_score', 0.0)
            title = f"{title} ({score:.2f})"

        lines.append(f"### {index}. {title}")

        # URL
        url = article.get('url', '')
        if url:
            lines.append(f"🔗 [Read full article]({url})")

        # Summary
        if config.include_summaries and article.get('summary'):
            lines.extend([
                "",
                article['summary']
            ])

        # Key points
        if config.include_key_points and article.get('key_points'):
            lines.extend([
                "",
                "**Key Points:**"
            ])
            for point in article['key_points']:
                lines.append(f"• {point}")

        # Implications
        if config.include_implications and article.get('implications'):
            lines.extend([
                "",
                f"**Implications:** {article['implications']}"
            ])

        return lines

    def render_newsletter(self, articles: list[dict],
                         config: NewsletterConfig | None = None) -> str:
        """Render complete newsletter in Markdown format."""
        if config is None:
            config = NewsletterConfig()

        logger.info(f"Rendering newsletter with {len(articles)} articles")

        # Limit articles
        articles = articles[:config.max_articles]

        # Generate metadata
        metadata = self._generate_metadata(articles, config)

        # Categorize articles if requested
        categories = None
        if config.group_by_category:
            categories = self._categorize_articles(articles)

        # Prepare template context
        context = {
            'metadata': metadata,
            'articles': articles,
            'categories': categories,
            'config': config,
            'now': datetime.now()
        }

        # Render using Jinja2 if available, otherwise fallback
        if JINJA2_AVAILABLE and self.jinja_env:
            template_name = f"{config.format_type.value}_newsletter.md"

            # Try format-specific template, fall back to default
            try:
                return self._render_with_jinja2(template_name, context)
            except Exception:
                try:
                    return self._render_with_jinja2('default_newsletter.md', context)
                except Exception:
                    logger.warning("Template rendering failed, using fallback")
                    return self._render_fallback(context)
        else:
            return self._render_fallback(context)

    def save_newsletter(self, newsletter_content: str,
                       output_path: Path | None = None) -> Path:
        """Save newsletter to file."""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = Path(f"ai_safety_newsletter_{timestamp}.md")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(newsletter_content)

        logger.info(f"Newsletter saved to {output_path}")
        return output_path


def render_newsletter(articles: list[dict], settings: Settings,
                     config: NewsletterConfig | None = None,
                     output_path: Path | None = None) -> str:
    """Convenience function for newsletter rendering."""
    renderer = NewsletterRenderer(settings)
    newsletter = renderer.render_newsletter(articles, config)

    if output_path:
        renderer.save_newsletter(newsletter, output_path)

    return newsletter
