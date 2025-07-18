"""Main orchestrator for AI Safety Newsletter Agent."""

import asyncio
import sys
from datetime import datetime
from typing import List, Optional

import click

from .config import get_settings, validate_config
from .ingest.sources import gather_articles, Article
from .logging import get_logger, setup_logging, PerformanceLogger
from .models.llm_client import create_llm_client
from .utils import parse_date_string

logger = get_logger(__name__)


async def run_pipeline(
    start_date: str, 
    mock: bool = False,
    max_articles: Optional[int] = None
) -> str:
    """Run the complete newsletter generation pipeline.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        mock: Use mock data and clients
        max_articles: Maximum articles to process
        
    Returns:
        Generated newsletter markdown
    """
    settings = get_settings()
    
    logger.info(
        "Starting newsletter generation pipeline",
        start_date=start_date,
        mock=mock,
        max_articles=max_articles or settings.max_articles
    )
    
    try:
        with PerformanceLogger("full_pipeline", logger):
            # Stage 1: Ingestion
            logger.info("Stage 1: Article ingestion")
            with PerformanceLogger("ingestion", logger):
                articles = await gather_articles(
                    start_date=start_date,
                    mock=mock,
                    limit_per_source=10
                )
            
            logger.info(f"Ingested {len(articles)} articles")
            
            if not articles:
                logger.warning("No articles found")
                return "# AI Safety Newsletter\n\nNo articles found for the specified date range."
            
            # Stage 2: Relevance filtering (placeholder)
            logger.info("Stage 2: Relevance filtering")
            with PerformanceLogger("relevance_filtering", logger):
                relevant_articles = await filter_relevant_articles(articles, mock=mock)
            
            logger.info(f"Filtered to {len(relevant_articles)} relevant articles")
            
            # Stage 3: Deduplication (placeholder)
            logger.info("Stage 3: Deduplication")
            with PerformanceLogger("deduplication", logger):
                deduped_articles = deduplicate_articles(relevant_articles)
            
            logger.info(f"Deduplicated to {len(deduped_articles)} unique articles")
            
            # Stage 4: Scoring and ranking (placeholder)
            logger.info("Stage 4: Scoring and ranking")
            with PerformanceLogger("scoring", logger):
                ranked_articles = rank_articles(deduped_articles)
            
            # Limit to max articles
            max_count = max_articles or settings.max_articles
            top_articles = ranked_articles[:max_count]
            
            logger.info(f"Selected top {len(top_articles)} articles")
            
            # Stage 5: Summarization (placeholder)
            logger.info("Stage 5: Summarization")
            with PerformanceLogger("summarization", logger):
                enriched_articles = await summarize_articles(top_articles, mock=mock)
            
            # Stage 6: Newsletter rendering
            logger.info("Stage 6: Newsletter rendering")
            with PerformanceLogger("rendering", logger):
                newsletter = render_newsletter(enriched_articles, start_date)
            
            logger.info("Newsletter generation completed successfully")
            return newsletter
    
    except Exception as e:
        logger.error("Pipeline failed", error=str(e), exc_info=True)
        raise


async def filter_relevant_articles(articles: List[Article], mock: bool = False) -> List[Article]:
    """Filter articles for AI safety relevance.
    
    Args:
        articles: Input articles
        mock: Use mock filtering
        
    Returns:
        Filtered articles
    """
    if mock:
        # Mock filtering: keep articles with AI-related keywords
        ai_keywords = ['ai', 'artificial intelligence', 'machine learning', 'safety', 'ethics', 'governance']
        relevant = []
        
        for article in articles:
            text = f"{article['title']} {article['content']}".lower()
            if any(keyword in text for keyword in ai_keywords):
                relevant.append(article)
        
        return relevant
    
    # TODO: Implement real LLM-based relevance filtering
    return articles


def deduplicate_articles(articles: List[Article]) -> List[Article]:
    """Remove duplicate articles.
    
    Args:
        articles: Input articles
        
    Returns:
        Deduplicated articles
    """
    # Simple deduplication by URL and title
    seen_urls = set()
    seen_titles = set()
    unique_articles = []
    
    for article in articles:
        url = article['url']
        title = article['title'].lower().strip()
        
        if url not in seen_urls and title not in seen_titles:
            seen_urls.add(url)
            seen_titles.add(title)
            unique_articles.append(article)
    
    return unique_articles


def rank_articles(articles: List[Article]) -> List[Article]:
    """Rank articles by importance.
    
    Args:
        articles: Input articles
        
    Returns:
        Ranked articles (highest score first)
    """
    settings = get_settings()
    
    # Simple scoring based on source and recency
    for article in articles:
        score = 0.0
        
        # Source priority
        domain = article.get('source', '')
        if 'gov' in domain:
            score += settings.w_gov
        else:
            score += settings.w_source_priority * 0.5
        
        # Recency (newer articles score higher)
        published_date = article.get('published_date')
        if isinstance(published_date, datetime):
            days_old = (datetime.now() - published_date).days
            recency_score = max(0, 1 - days_old / 30)  # Decay over 30 days
            score += settings.w_recency * recency_score
        
        article['_score'] = score
    
    # Sort by score (descending)
    return sorted(articles, key=lambda x: x.get('_score', 0), reverse=True)


async def summarize_articles(articles: List[Article], mock: bool = False) -> List[Article]:
    """Add summaries to articles.
    
    Args:
        articles: Input articles
        mock: Use mock summarization
        
    Returns:
        Articles with summaries
    """
    if mock:
        # Add mock summaries
        for article in articles:
            article['summary'] = f"**{article['title'][:50]}...**\n\n• Key development in AI safety\n• Implications for governance\n• Industry impact expected"
            article['category'] = 'Technology'
            article['importance'] = 3
        
        return articles
    
    # TODO: Implement real LLM-based summarization
    return articles


def render_newsletter(articles: List[Article], start_date: str) -> str:
    """Render newsletter markdown.
    
    Args:
        articles: Articles to include
        start_date: Newsletter start date
        
    Returns:
        Newsletter markdown
    """
    from datetime import datetime
    
    # Parse start date
    start_dt = parse_date_string(start_date)
    date_str = start_dt.strftime("%B %d, %Y") if start_dt else start_date
    
    # Build newsletter
    lines = [
        "# AI Safety Newsletter",
        f"*{date_str}*",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"This newsletter covers {len(articles)} key developments in AI safety.",
        "",
        "**Highlights:**"
    ]
    
    # Add highlights
    for i, article in enumerate(articles[:3], 1):
        lines.append(f"{i}. {article['title']}")
    
    lines.extend([
        "",
        "---",
        "",
        "## Top Stories",
        ""
    ])
    
    # Add articles
    for i, article in enumerate(articles, 1):
        lines.extend([
            f"### {i}. {article['title']}",
            "",
            article.get('summary', article.get('content', '')[:200] + '...'),
            "",
            f"**Source:** [{article['source']}]({article['url']})",
            "",
            "---" if i < len(articles) else "",
            ""
        ])
    
    lines.extend([
        "---",
        "",
        f"*Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M UTC')} by AI Safety Newsletter Agent*"
    ])
    
    return "\n".join(lines)


@click.command()
@click.argument('start_date', type=click.DateTime(formats=['%Y-%m-%d']))
@click.option('--mock', is_flag=True, help='Use mock data and LLM clients')
@click.option('--max-articles', type=int, help='Maximum articles to include')
@click.option('--output', '-o', type=click.File('w'), default='-', help='Output file (default: stdout)')
@click.option('--log-level', default='INFO', help='Log level')
@click.option('--validate-config', is_flag=True, help='Validate configuration and exit')
def cli(start_date, mock, max_articles, output, log_level, validate_config_flag):
    """AI Safety Newsletter Agent - Generate AI safety newsletters from news sources."""
    
    # Setup logging
    setup_logging(log_level=log_level, json_logging=False)
    
    if validate_config_flag:
        if validate_config():
            click.echo("✅ Configuration is valid")
            sys.exit(0)
        else:
            click.echo("❌ Configuration validation failed")
            sys.exit(1)
    
    # Validate configuration
    if not validate_config():
        click.echo("❌ Configuration validation failed. Use --validate-config for details.")
        sys.exit(1)
    
    # Convert datetime to string
    start_date_str = start_date.strftime('%Y-%m-%d')
    
    try:
        # Run pipeline
        newsletter = asyncio.run(run_pipeline(
            start_date=start_date_str,
            mock=mock,
            max_articles=max_articles
        ))
        
        # Output newsletter
        output.write(newsletter)
        
        if output != sys.stdout:
            click.echo(f"Newsletter written to {output.name}")
    
    except Exception as e:
        logger.error("CLI execution failed", error=str(e))
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
