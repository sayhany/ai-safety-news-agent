import asyncio
import sys

import click

from .config import Settings, get_model_config, get_settings, validate_config
from .ingest.sources import gather_articles
from .logging import PerformanceLogger, get_logger, setup_logging
from .models.llm_client import create_llm_client
from .processing.dedupe import deduplicate_articles as dedupe_articles
from .processing.relevance import filter_relevance
from .processing.scoring import score_articles
from .render import NewsletterConfig
from .render import render_newsletter as render_newsletter_md
from .summarize import SummaryConfig
from .summarize import summarize_articles as llm_summarize_articles
from .ui import init_ui

logger = get_logger(__name__)


async def run_pipeline(
    settings: Settings,
    start_date: str,
    ui = None,
    output_filename: str = None
) -> str:
    """Run the complete newsletter generation pipeline.

    Args:
        settings: The application settings.
        start_date: Start date in YYYY-MM-DD format.
        ui: Optional friendly UI instance.
        output_filename: Output filename for summary display.

    Returns:
        Generated newsletter markdown.
    """
    # Generate default filename if not provided
    if output_filename is None:
        output_filename = f"newsletter_{start_date}.md"

    model_config = get_model_config()
    default_model = model_config.get_llm_route("relevance").primary
    model_used = settings.llm_model_override or default_model

    # Track metrics for final summary
    total_sources = len(model_config.get_search_sources()) + len(model_config.get_approved_sources())

    if ui:
        ui.verbose_log(f"Starting pipeline with start_date={start_date}, mock={settings.mock}, max_articles={settings.max_articles}")
        if settings.llm_model_override:
            ui.show_model_info(model_used)

    try:
        with PerformanceLogger("full_pipeline", logger):
            # Stage 1: Article Ingestion
            if ui:
                with ui.stage("Gathering articles from {}".format(start_date.replace("-", " ")), "📰") as (progress, task):
                    articles = await gather_articles(
                        start_date=start_date,
                        mock=settings.mock,
                        limit_per_source=10,
                    )

                    if progress:
                        ui.complete_progress(progress, task, f"Found {len(articles)} articles total")
            else:
                articles = await gather_articles(
                    start_date=start_date,
                    mock=settings.mock,
                    limit_per_source=10,
                )

            if ui:
                ui.verbose_log(f"Ingested {len(articles)} articles")

            if not articles:
                if ui:
                    ui.warning("No articles found for the specified date range")
                return "# AI Safety Newsletter\n\nNo articles found for the specified date range."

            # Create LLM client once
            llm_client = create_llm_client("summarizer", mock=settings.mock)
            original_count = len(articles)

            # Stage 2: Relevance filtering
            if ui:
                with ui.stage("Filtering for AI safety relevance", "🔍") as (progress, task):
                    if not settings.mock:
                        ui.show_model_info(model_used)
                    articles = await filter_relevance(articles, settings, llm_client)

                    if progress:
                        ui.complete_progress(progress, task, f"{len(articles)} articles passed relevance filter")
            else:
                articles = await filter_relevance(articles, settings, llm_client)

            filtered_count = len(articles)

            # Stage 3: Deduplication
            if ui:
                with ui.stage("Removing duplicates", "🔄") as (progress, task):
                    articles, duplicate_groups = await dedupe_articles(
                        articles, settings, llm_client
                    )

                    duplicates_removed = sum(len(group.duplicates) for group in duplicate_groups)
                    if progress:
                        ui.complete_progress(progress, task, f"{len(articles)} unique articles remain ({duplicates_removed} duplicates removed)")
            else:
                articles, duplicate_groups = await dedupe_articles(
                    articles, settings, llm_client
                )

            # Stage 4: Scoring and Ranking
            if ui:
                with ui.stage("Scoring articles by importance", "⭐") as (progress, task):
                    articles = score_articles(articles, settings)

                    if progress:
                        ui.complete_progress(progress, task, f"All {len(articles)} articles scored")
            else:
                articles = score_articles(articles, settings)

            # Limit to max articles
            top_articles = articles[: settings.max_articles]
            final_count = len(top_articles)

            if ui:
                ui.verbose_log(f"Selected top {final_count} articles")

            # Stage 5: Newsletter Generation
            if ui:
                with ui.stage("Generating newsletter", "📝") as (progress, task):
                    # Summarization
                    summary_config = SummaryConfig()
                    top_articles = await llm_summarize_articles(
                        top_articles, settings, llm_client, summary_config
                    )

                    # Rendering
                    newsletter_config = NewsletterConfig(
                        max_articles=settings.max_articles,
                        include_summaries=True,
                        include_key_points=True,
                        include_implications=True,
                    )
                    newsletter = render_newsletter_md(
                        top_articles, settings, newsletter_config
                    )

                    if progress:
                        ui.complete_progress(progress, task, "Newsletter crafted and formatted")

            else:
                # Non-UI path
                summary_config = SummaryConfig()
                top_articles = await llm_summarize_articles(
                    top_articles, settings, llm_client, summary_config
                )

                newsletter_config = NewsletterConfig(
                    max_articles=settings.max_articles,
                    include_summaries=True,
                    include_key_points=True,
                    include_implications=True,
                )
                newsletter = render_newsletter_md(
                    top_articles, settings, newsletter_config
                )

            # Show final summary
            if ui:
                ui.show_final_summary(
                    total_sources=total_sources,
                    total_articles=original_count,
                    filtered_articles=filtered_count,
                    final_articles=final_count,
                    output_file=output_filename,
                    model_used=model_used
                )

            return newsletter

    except Exception as e:
        logger.error("Pipeline failed", error=str(e), exc_info=True)
        raise


@click.command()
@click.argument("start_date", type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--mock", is_flag=True, help="Use mock data and LLM clients")
@click.option("--max-articles", type=int, help="Maximum articles to include")
@click.option(
    "--output",
    "-o",
    type=click.File("w"),
    default="-",
    help="Output file (default: stdout)",
)
@click.option("--log-level", default="ERROR", help="Log level")
@click.option("--verbose", is_flag=True, help="Show detailed progress information")
@click.option(
    "--validate-config",
    "validate_config_flag",
    is_flag=True,
    help="Validate configuration and exit",
)
@click.option("--api-key", help="Google AI API key to use.")
@click.option("--model", help="LLM model to use (e.g., 'gemini-2.5-flash').")
@click.option(
    "--search-type",
    type=click.Choice(["auto", "neural", "keyword", "fast"]),
    help="Search type for Exa API: auto (default), neural, keyword, or fast"
)
def cli(
    start_date,
    mock,
    max_articles,
    output,
    log_level,
    verbose,
    validate_config_flag,
    api_key,
    model,
    search_type,
):
    """AI Safety Newsletter Agent - Generate AI safety newsletters from news sources."""
    # Set log level to suppress noise unless verbose mode is enabled
    actual_log_level = "INFO" if verbose else log_level
    setup_logging(log_level=actual_log_level, json_logging=False)

    # Suppress ALL logs unless in verbose mode
    if not verbose:
        import logging
        # Set root logger to ERROR to suppress all module logs
        logging.getLogger().setLevel(logging.ERROR)
        # Suppress HTTP request logs
        logging.getLogger("httpx").setLevel(logging.ERROR)
        logging.getLogger("httpcore").setLevel(logging.ERROR)
        logging.getLogger("urllib3").setLevel(logging.ERROR)
        logging.getLogger("aiohttp").setLevel(logging.ERROR)
        # Suppress application module logs
        logging.getLogger("aisafety_news").setLevel(logging.ERROR)

    # Initialize friendly UI
    ui = init_ui(verbose=verbose)
    ui.show_banner()

    try:
        settings = get_settings()
        if mock:
            settings.mock = True
        if api_key:
            settings.google_ai_api_key = api_key
        if model:
            settings.llm_model_override = model
        if max_articles:
            settings.max_articles = max_articles
        if search_type:
            settings.search_type_override = search_type

        if validate_config_flag:
            if validate_config(settings):
                ui.success("Configuration is valid")
                sys.exit(0)
            else:
                ui.error("Configuration validation failed")
                sys.exit(1)

        if not validate_config(settings):
            ui.error("Configuration validation failed. Use --validate-config for details.")
            sys.exit(1)

        start_date_str = start_date.strftime("%Y-%m-%d")
        output_filename = output.name if output.name != "<stdout>" else f"newsletter_{start_date_str}.md"

        newsletter = asyncio.run(run_pipeline(
            settings=settings,
            start_date=start_date_str,
            ui=ui,
            output_filename=output_filename
        ))

        output.write(newsletter)

        if output.name != "<stdout>":
            # Don't show this message as UI already shows final summary
            pass

    except Exception as e:
        logger.error("CLI execution failed", error=str(e))
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
